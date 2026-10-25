# onnxruntime-web-kernels

An unofficial build of [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) 1.30.0 with faster WebGPU
kernels for quantized LLMs, made for [Quick Inference](https://quick-inf.vercel.app), an in-browser chat app. The
kernels live inside the rebuilt runtime: each release carries complete WebAssembly binaries and their JavaScript glue,
as drop-in replacements for the matching files of the `onnxruntime-web` 1.30.0 npm package:

- `ort-wasm-simd-threaded.asyncify.wasm` and `.mjs` (WebGPU, Asyncify)
- `ort-wasm-simd-threaded.jspi.wasm` and `.mjs` (WebGPU, JavaScript Promise Integration)

Each `.wasm` only works with the `.mjs` built with it, so load them as a pair (for example through
`ort.env.wasm.wasmPaths = { mjs, wasm }` with the `onnxruntime-web-use-extern-wasm` builds of the package).

## Changes

`patches/onnxruntime-1.30.0.patch`:

- **2-bit MatMulNBits.** A new WebGPU program for 2-bit weights without zero points
  (`contrib_ops/webgpu/quantization/matmul_nbits_q2_decode.{h,cc}`). ONNX Runtime's general kernel gives each
  128-thread workgroup 8 output rows and copies all of A into workgroup memory first; with 2-bit weights that copy is
  as large as the weights it reads. Here each team of 32 lanes computes 2 rows, every lane keeps its 32 values of A in
  registers for both, and weights are decoded four at a time (`bitcast<vec4<f32>>(codes | 0x4b000000) - (2^23 + 2)`
  gives `q - 2` exactly). Sums are kept in float32. It replaced the general kernel for ternary models
  ([Ternary Bonsai 2](https://huggingface.co/prism-ml)) in Quick Inference with about 30% faster decoding and 55%
  faster prompts, with outputs closer to ONNX Runtime's CPU results.

- **Squared ReLU in WebGPU QMoE** (v1.30.0-qi.2). `activation_type="relu2"` (`max(x, 0)^2`, the name ONNX Runtime's
  CUDA MoE uses), plus `relu`, `silu` and `identity`, which ONNX Runtime 1.30's WebGPU QMoE rejected (it only ran
  SwiGLU). Lets Nemotron-H models store each expert's up projection once instead of twice (as SwiGLU's gate and up
  halves): Nemotron 3 Nano's experts shrink by a third.

- **Fast prompts with blocks of 128** (v1.30.0-qi.3). ONNX Runtime's fast MatMulNBits program for prompts (the "wide
  tile" one) only took blocks of 32 weights; other block sizes fell back to its general kernel. GPTQ and AWQ
  checkpoints usually use blocks of 128, whose bytes are already four 32-weight runs one after another, so the program
  now walks K 32 weights at a time and reads the scale (and zero point) of block `k32 / (block_size / 32)`: any
  multiple of 32 works, with or without zero points, 4 or 8 bits. Results match ONNX Runtime's CPU kernel as closely as
  blocks of 32 do. Muse Glimmer 30B (Red Hat's GPTQ weights, blocks of 128) read prompts 6× faster in Quick Inference
  (5.5 → 33 tokens/s).

- **Faster 4-bit decoding** (v1.30.0-qi.4). A new MatMulNBits program for 4-bit weights and 1 to 8 rows of A
  (decoding, and checking a few guessed tokens at once: `matmul_nbits_q4_decode.{h,cc}`): each team of 32 lanes takes a
  32-weight step of its rows per lane, word by word, turning codes into float16 two at a time (`0x6400 | code` is
  `1024 + code`), with or without zero points; each value of A loaded serves every row of the team, so 4 rows of A cost
  about 1.1-1.3x one. Also a gated form for the fused MLP (`silu(gate) * up`). Qwen3.8 27B (zero points) decodes 54%
  faster than with ONNX Runtime's general kernel, Qwen3.5 0.8B 18%.

- **Mixture-of-experts gates without a bubble sort** (v1.30.0-qi.4). QMoE's top-K was a bubble sort with a workgroup
  barrier per comparison step (about 2,000 barriers per layer for 64 experts, 32,000 for 256); it is now K rounds of a
  parallel argmax (ties to the lower index, as before). The multi-token gate also cleared its per-expert counts from one
  workgroup while others were adding to them; they are now cleared by a dispatch of their own. Up to 8 tokens (checking
  guessed tokens) take the single-token path, k rows per token, instead of a loop over every expert that read the gate's
  counts back to the CPU at each layer.

- **Graph capture replays buffer copies** (v1.30.0-qi.4). A captured graph recorded only compute dispatches, so a
  kernel that fills its output from an input with a buffer copy (ScatterElements, Identity, Reshape, ... when not in
  place) ran that copy only while being captured, and replays read stale data. Copies are now recorded in order with the
  dispatches and repeated on replay. Uploads of weights also wait for the GPU every 256 MB, so their staging buffers
  can't pile up while a big model loads.

- **State windows for linear attention and short convolutions** (v1.30.0-qi.4). `LinearAttention` and
  `CausalConvWithState` take the `state_window` attribute (ONNX Runtime's schema: the states after each of the last W
  tokens, for speculative decoding to roll back to the accepted ones), which only the CPU and CUDA providers implemented.

- **float16 overflows** (v1.30.0-qi.4). Layer norms (SkipLayerNorm, LayerNorm and the norm folded into the fused QKV
  and MLP programs) compute their output in float32: Metal evaluated `value * inv_std_dev * gamma` as
  `value * gamma` first, which overflowed for Qwen3's large residuals. The wide-tile prompt program keeps float32 totals,
  and the RMSNorm fusion (SimplifiedLayerNormFusion) also runs for WebGPU, so a spelled-out RMSNorm doesn't square a
  float16 residual (Qwen2.5).

Everything else is ONNX Runtime 1.30.0 as released.

## Building

```sh
./build.sh            # clones ONNX Runtime v1.30.0 into ./onnxruntime, applies the patches, builds both into ./dist
```

It uses the flags of ONNX Runtime's own WebAssembly release builds. Needs git and Python 3.10 or later; ONNX
Runtime's build script installs Emscripten itself. A first build takes a while and several GB of disk.

## Licences

The patches and scripts here are MIT-licensed (`LICENSE`). ONNX Runtime is MIT-licensed by Microsoft
(`LICENSE-onnxruntime`), and its third-party components are listed in `ThirdPartyNotices.txt`; all three files are
included in every release.
