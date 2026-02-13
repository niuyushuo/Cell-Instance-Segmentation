import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_mask_as_binary(mask_path, threshold=0):
    mask = Image.open(mask_path).convert("L")
    mask_np = np.array(mask, dtype=np.uint8)
    return (mask_np > threshold).astype(np.uint8)


def make_edt_label(binary_mask, d_max=15.0):
    dist = distance_transform_edt(binary_mask)
    dist_norm = np.clip(dist / float(d_max), 0.0, 1.0).astype(np.float32)
    dist_u8 = (dist_norm * 255.0 + 0.5).astype(np.uint8)
    return dist_u8


def generate_edt_from_mask_dir(mask_dir, out_dir, d_max=15.0, threshold=0, overwrite=False):
    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in mask_dir.iterdir() if p.suffix.lower() in VALID_EXTS]
    files = sorted(files, key=lambda p: natural_key(p.name))

    if not files:
        raise RuntimeError(f"No mask files found in {mask_dir}")

    print(f"Found {len(files)} masks")
    print(f"Input : {mask_dir}")
    print(f"Output: {out_dir}")
    print(f"d_max : {d_max}")

    written = 0
    skipped = 0
    for i, f in enumerate(files, 1):
        out_path = out_dir / f.name
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        fg = load_mask_as_binary(f, threshold=threshold)
        edt_u8 = make_edt_label(fg, d_max=d_max)
        Image.fromarray(edt_u8, mode="L").save(out_path)
        written += 1

        if i <= 3 or i % 100 == 0 or i == len(files):
            print(f"[{i:>5}/{len(files)}] {f.name}")

    print(f"Done. written={written}, skipped={skipped}")


def generate_edt_for_bccd(dataset_root, splits=("train", "test"), out_name="edt", d_max=15.0, threshold=0, overwrite=False):
    dataset_root = Path(dataset_root)
    for split in splits:
        mask_dir = dataset_root / split / "mask"
        out_dir = dataset_root / split / out_name
        if not mask_dir.exists():
            print(f"Skip split={split}: {mask_dir} does not exist")
            continue

        print(f"\n=== split: {split} ===")
        generate_edt_from_mask_dir(
            mask_dir=mask_dir,
            out_dir=out_dir,
            d_max=d_max,
            threshold=threshold,
            overwrite=overwrite,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Generate normalized EDT labels from binary masks.")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="data/BCCD Dataset with mask",
        help="Dataset root containing train/test folders with mask subfolders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to process when using --dataset_root.",
    )
    parser.add_argument(
        "--out_name",
        type=str,
        default="edt",
        help="Output folder name inside each split.",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Process a single mask directory (overrides --dataset_root mode).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for --mask_dir mode.",
    )
    parser.add_argument("--d_max", type=float, default=15.0, help="Distance normalization max.")
    parser.add_argument("--threshold", type=int, default=0, help="Binary foreground threshold.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mask_dir is not None:
        if args.out_dir is None:
            raise ValueError("--out_dir is required when --mask_dir is provided")
        generate_edt_from_mask_dir(
            mask_dir=args.mask_dir,
            out_dir=args.out_dir,
            d_max=args.d_max,
            threshold=args.threshold,
            overwrite=args.overwrite,
        )
    else:
        generate_edt_for_bccd(
            dataset_root=args.dataset_root,
            splits=tuple(args.splits),
            out_name=args.out_name,
            d_max=args.d_max,
            threshold=args.threshold,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
