#!/usr/bin/env bash
# Fetch model weights (not stored in git). Depth Anything V2 small, ONNX export by fabio-sim (Apache-2.0
# weights of Depth-Anything-V2-Small; see https://github.com/fabio-sim/Depth-Anything-ONNX).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
URL=https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx
SHA=d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be
if [ ! -f models/depth_anything_v2_vits.onnx ]; then
  curl -L --fail -o models/depth_anything_v2_vits.onnx "$URL"
fi
echo "$SHA  models/depth_anything_v2_vits.onnx" | sha256sum -c -
