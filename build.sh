#!/bin/sh
# Builds ONNX Runtime Web's WebGPU binaries (Asyncify and JSPI) with this repository's patches, into ./dist.
#   ./build.sh [ONNX Runtime checkout, default ./onnxruntime (cloned if missing)]
# Needs git and Python 3.10 or later (PYTHON=...); ONNX Runtime's build script installs Emscripten itself.
# JOBS=N sets parallel compile jobs (default 6).
set -eu
ORT_VERSION=1.30.0
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-$HERE/onnxruntime}
PYTHON=${PYTHON:-python3}
JOBS=${JOBS:-6}

if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "v$ORT_VERSION" --recurse-submodules --shallow-submodules https://github.com/microsoft/onnxruntime.git "$SRC"
fi
cd "$SRC"
if [ "$(cat VERSION_NUMBER)" != "$ORT_VERSION" ]; then
  echo "build.sh: expected ONNX Runtime $ORT_VERSION in $SRC" >&2
  exit 1
fi
# Apply each patch once (a checkout already carrying it is left as it is).
for patch in "$HERE"/patches/*.patch; do
  if git apply --reverse --check "$patch" 2>/dev/null; then
    echo "already applied: $(basename "$patch")"
  else
    git apply "$patch"
  fi
done
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
  .venv/bin/pip install -q cmake==3.31.6 ninja flatbuffers numpy
fi
export PATH="$PWD/.venv/bin:$PATH"

# The flags of ONNX Runtime's own WebAssembly release builds (.github/workflows/linux-wasm-ci-build-and-test-workflow.yml).
set -- --parallel "$JOBS" --config Release --skip_submodule_sync --build_wasm --enable_wasm_simd --enable_wasm_threads \
  --use_webgpu --use_webnn --target onnxruntime_webassembly --skip_tests \
  --disable_ml_ops --disable_generation_ops --disable_types string float4 float8 optional sparsetensor \
  --include_ops_by_config onnxruntime/wasm/reduced_types.config --enable_reduced_operator_type_support
.venv/bin/python tools/ci_build/build.py "$@" --build_dir build/wasm_webgpu --enable_wasm_api_exception_catching
.venv/bin/python tools/ci_build/build.py "$@" --build_dir build/wasm_webgpu_jspi --enable_wasm_jspi

OUT="$HERE/dist"
mkdir -p "$OUT"
cp build/wasm_webgpu/Release/ort-wasm-simd-threaded.asyncify.wasm build/wasm_webgpu/Release/ort-wasm-simd-threaded.asyncify.mjs "$OUT"/
cp build/wasm_webgpu_jspi/Release/ort-wasm-simd-threaded.jspi.wasm build/wasm_webgpu_jspi/Release/ort-wasm-simd-threaded.jspi.mjs "$OUT"/
cp LICENSE "$OUT"/LICENSE-onnxruntime
cp ThirdPartyNotices.txt "$OUT"/
cp "$HERE"/LICENSE "$OUT"/LICENSE
cd "$OUT"
shasum -a 256 ./*
