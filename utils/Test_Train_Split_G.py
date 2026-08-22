"""Create deterministic train/validation/test splits for ISIC-style datasets."""

import argparse
import os
import random
from pathlib import Path

from PIL import Image


def resize_and_copy(image_names, image_dir, mask_dir, output_dir, split, size):
    image_output = output_dir / split / "images"
    mask_output = output_dir / split / "masks"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_name in image_names:
        mask_name = f"{Path(image_name).stem}_Segmentation.png"
        image_source = image_dir / image_name
        mask_source = mask_dir / mask_name
        if not mask_source.exists():
            print(f"Skipping {image_name}: mask not found ({mask_source.name})")
            continue

        with Image.open(image_source) as image:
            image.resize(size, Image.Resampling.BILINEAR).save(image_output / image_name)
        with Image.open(mask_source) as mask:
            mask.resize(size, Image.Resampling.NEAREST).save(mask_output / mask_name)
        copied += 1
    return copied


def split_dataset(dataset_dir, output_dir, train_count=160, test_count=20, seed=42, size=256):
    image_dir = dataset_dir / "images"
    mask_dir = dataset_dir / "masks"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("dataset directory must contain images/ and masks/")

    images = sorted(path.name for path in image_dir.glob("*.jpg"))
    if train_count + test_count > len(images):
        raise ValueError(
            f"requested {train_count + test_count} train/test images, but found {len(images)}"
        )

    random.Random(seed).shuffle(images)
    splits = {
        "train": images[:train_count],
        "test": images[train_count : train_count + test_count],
        "val": images[train_count + test_count :],
    }
    for split, names in splits.items():
        copied = resize_and_copy(
            names,
            image_dir,
            mask_dir,
            output_dir,
            split,
            (size, size),
        )
        print(f"{split}: {copied} image/mask pairs")


def parse_args():
    data_root = os.environ.get("ML_DATA_ROOT")
    default_dataset = Path(data_root) / "isic_2016_1" if data_root else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=default_dataset)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-count", type=int, default=160)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    if args.dataset_dir is None:
        parser.error("set ML_DATA_ROOT or pass --dataset-dir")
    if args.output_dir is None:
        args.output_dir = args.dataset_dir / "split_data"
    return args


if __name__ == "__main__":
    cli_args = parse_args()
    split_dataset(
        cli_args.dataset_dir,
        cli_args.output_dir,
        cli_args.train_count,
        cli_args.test_count,
        cli_args.seed,
        cli_args.size,
    )
