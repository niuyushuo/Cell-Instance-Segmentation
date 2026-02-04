import argparse
import json
import os
import random
from pathlib import Path


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def build_stem_set(folder):
    stems = set()
    for fname in os.listdir(folder):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in VALID_EXTS:
            stems.add(stem)
    return stems


def get_common_stems(image_dir, mask_dir):
    image_stems = build_stem_set(image_dir)
    mask_stems = build_stem_set(mask_dir)
    common = sorted(image_stems & mask_stems)
    if not common:
        raise RuntimeError(f"No common files found between {image_dir} and {mask_dir}")
    return common


def split_train_val(stems, val_ratio=0.2, seed=8888):
    stems = list(stems)
    rng = random.Random(seed)
    rng.shuffle(stems)

    val_count = max(1, int(round(len(stems) * float(val_ratio))))
    val_count = min(val_count, len(stems) - 1)

    val_stems = sorted(stems[:val_count])
    train_stems = sorted(stems[val_count:])
    return train_stems, val_stems


def main():
    parser = argparse.ArgumentParser(description="Create train/val split file from train images and masks.")
    parser.add_argument("--train_original", type=str, required=True, help="Path to train/original directory.")
    parser.add_argument("--train_mask", type=str, required=True, help="Path to train/mask directory.")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio from train set.")
    parser.add_argument("--seed", type=int, default=8888, help="Random seed for splitting.")
    parser.add_argument(
        "--out_json",
        type=str,
        default="github_uni/datasets/splits/bccd_train_val_split.json",
        help="Output split JSON path.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split JSON.")
    args = parser.parse_args()

    out_path = Path(args.out_json)
    if out_path.exists() and not args.overwrite:
        raise RuntimeError(f"{out_path} already exists. Use --overwrite to replace it.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stems = get_common_stems(args.train_original, args.train_mask)
    train_stems, val_stems = split_train_val(stems, val_ratio=args.val_ratio, seed=args.seed)

    data = {
        "train_original": args.train_original,
        "train_mask": args.train_mask,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "num_total": len(stems),
        "num_train": len(train_stems),
        "num_val": len(val_stems),
        "train_stems": train_stems,
        "val_stems": val_stems,
    }
    out_path.write_text(json.dumps(data, indent=2))

    print(f"Saved split -> {out_path}")
    print(f"total={len(stems)} train={len(train_stems)} val={len(val_stems)}")


if __name__ == "__main__":
    main()
