#!/usr/bin/env python3
"""Reference tile errors for eovae's opt-in comparison test.

Samples 32 tissue tiles from a slide, scores them with Python ONNX Runtime
using HistoVAE's training preprocessing (RGBA composited onto white,
near-black pixels made white, RGB mapped to [-1, 1]; output (x + 1) * 127.5)
and writes:

    <out>/tile_NN.rgba   raw RGBA of each tile (256x256x4)
    <out>/ref.json       per-tile MAE

Then run, from plugins/eovae:

    EOVAE_ORT_LIB=/path/to/libonnxruntime.so.1 EOVAE_REFERENCE_TILES=<out> \\
        cargo test --release matches_the_python_reference -- --ignored

Needs numpy, Pillow, openslide-python and onnxruntime.
"""

import argparse
import json
import pathlib
import random

import numpy as np
import onnxruntime as ort
import openslide
from PIL import Image


def has_content(arr):
    """HistoVAE's tile filter (pytorch/src/data.py)."""
    arr = arr.astype(np.float32)
    return not (arr.std() < 5 or arr.mean() < 10 or arr.mean() > 245)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("slide")
    parser.add_argument("model", help="HistoVae_reconstruct.onnx (batch 32, 256 px)")
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    slide = openslide.OpenSlide(args.slide)
    width, height = slide.dimensions
    random.seed(args.seed)
    tiles = []
    for _ in range(5000):
        if len(tiles) == 32:
            break
        col, row = random.randrange(width // 256), random.randrange(height // 256)
        img = slide.read_region((col * 256, row * 256), 0, (256, 256))
        rgba = np.array(img)
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        arr = np.array(background)
        dark = (arr[:, :, 0] < 4) & (arr[:, :, 1] < 4) & (arr[:, :, 2] < 4)
        arr[dark] = 255
        if has_content(arr):
            tiles.append((col, row, rgba, arr))
    if len(tiles) < 32:
        raise SystemExit("not enough tissue tiles found")

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inputs = np.stack([t[3] for t in tiles]).astype(np.float32)
    x = (inputs / 255.0 * 2 - 1).transpose(0, 3, 1, 2)
    y = session.run(None, {session.get_inputs()[0].name: x})[0]
    recon = np.clip((y + 1) * 127.5, 0, 255).transpose(0, 2, 3, 1)
    mae = np.abs(recon - inputs).mean(axis=(1, 2, 3)) / 255.0

    records = []
    for i, (col, row, rgba, _) in enumerate(tiles):
        rgba.astype(np.uint8).tofile(args.out / f"tile_{i:02d}.rgba")
        records.append({"col": col, "row": row, "mae": float(mae[i])})
    (args.out / "ref.json").write_text(json.dumps(records, indent=1))
    print(f"mean MAE {float(mae.mean()):.6f} over {len(tiles)} tiles -> {args.out}")


if __name__ == "__main__":
    main()
