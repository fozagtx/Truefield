#!/usr/bin/env bash
# Download NASA IMS bearing dataset (1.08 GB) and extract Set 2 into data/ims/2nd_test. Data is NOT committed.
# Source: NASA PCoE repository. Citation: J. Lee, H. Qiu, G. Yu, J. Lin, Rexnord Technical Services (2007), IMS, Univ. of Cincinnati.
set -euo pipefail
DATA_DIR="${DATA_DIR:-./data}"
URL="https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"
mkdir -p "$DATA_DIR/ims" && cd "$DATA_DIR/ims"
[ -f bearings.zip ] || curl -L --fail -o bearings.zip "$URL"
unzip -n -q bearings.zip || true
# The archive nests: IMS.7z -> 1st_test.rar / 2nd_test.rar / 3rd_test.rar (layout may vary; handle both).
Z=$(find . -iname 'IMS.7z' | head -1 || true)
if [ -n "$Z" ]; then command -v 7z >/dev/null || { echo "install p7zip-full (apt-get install -y p7zip-full)"; exit 1; }; 7z x -y "$Z" >/dev/null; fi
R=$(find . -iname '2nd_test.rar' | head -1 || true)
if [ -n "$R" ]; then
  if command -v unrar >/dev/null; then unrar x -o+ "$R" >/dev/null; else 7z x -y "$R" >/dev/null; fi
fi
D=$(find . -type d -name '2nd_test' | head -1)
[ -n "$D" ] && [ "$D" != "./2nd_test" ] && mv "$D" ./2nd_test
echo "Set 2 files: $(ls 2nd_test | wc -l) (expected 984)"
echo "Next: python scripts/make_demo_slice.py   # small preprocessed slice for the app container"
