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
