#!/bin/bash
# Untar all flattened chunk tar files into the original directory structure.
# Supports both ittakestwo (480P_*.tar) and robotics (everything else) datasets.

set -e

TAR_DIR="data"
ITT_DST_BASE="data/ittakestwo_release"
ROB_DST_BASE="data/robots_release"

mkdir -p "$ITT_DST_BASE"
mkdir -p "$ROB_DST_BASE"

found_any=false

for tar_file in "$TAR_DIR"/*.tar; do
    if [[ ! -f "$tar_file" ]]; then
        echo "[WARN] No .tar files found in $TAR_DIR"
        exit 0
    fi

    found_any=true
    fname="$(basename "$tar_file")"

    if [[ "$fname" == 480P_* ]]; then
        echo "[UNTAR] $fname -> $ITT_DST_BASE"
        tar -xf "$tar_file" -C "$ITT_DST_BASE"
    else
        echo "[UNTAR] $fname -> $ROB_DST_BASE"
        tar -xf "$tar_file" -C "$ROB_DST_BASE"
    fi
done

if [[ "$found_any" == false ]]; then
    echo "[WARN] No .tar files found in $TAR_DIR"
    exit 0
fi

echo "[DONE] All chunks extracted."
echo "  It Takes Two -> $ITT_DST_BASE"
echo "  Robotics     -> $ROB_DST_BASE"
