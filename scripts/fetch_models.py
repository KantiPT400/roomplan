"""Cross-platform model fetch (Windows/macOS/Linux): same file and checksum as fetch_models.sh.

    python scripts/fetch_models.py
"""
import hashlib, os, urllib.request

URL = "https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vits.onnx"
SHA = "d2b11a11c1d4a12b47608fa65a17ee9a4c605b55ee1730c8e3b526304f2562be"
dst = os.path.join(os.path.dirname(__file__), "..", "models", "depth_anything_v2_vits.onnx")
os.makedirs(os.path.dirname(dst), exist_ok=True)
if not os.path.exists(dst):
    print("downloading", URL)
    urllib.request.urlretrieve(URL, dst)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
assert h == SHA, f"checksum mismatch: {h}"
print(dst, "OK")
