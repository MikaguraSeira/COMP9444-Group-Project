# Medical Image Dataset Augmentation Based on Geometric Transformations
# Reference "A Data Augmentation-Based Framework to Handle Class Imbalance Problem for Alzheimer’s Stage Detection"
"""
python datasetsAugmentation_1.py \
  --data-dir Dataset_BUSI_with_GT \
  --output-dir Dataset_BUSI_Augmentation_1000 \
  --mode balanced \
  --target-count 1000
"""

import argparse
import json
import random
import shutil

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


CLASSES = ('normal', 'benign', 'malignant')
ROTATIONS = (0, 270, 180, 90)
CROPS = (
    "original",
    "crop_right",
    "crop_corner",
    "crop_bottom",
    "crop_left",
    "crop_top",
    "crop_whole",
)


@dataclass(frozen=True)
class Sample:
    image: Path
    masks: tuple[Path, ...]


def is_mask(path: Path):
    """BUSI masks are named as benign (1)_mask.png"""
    return "_mask" in path.stem.lower() # extract "benign (1)_mask" then lower that


def find_masks(image_path: Path): # to tuple[Path, ...]
    # image_path = autodl-tmp/Dataset_BUSI_with_GT/benign/benign (1).png
    prefix = f"{image_path.stem}_mask" # extract "benign (1)" then add "_mask"
    # image_path.parent = autodl-tmp/Dataset_BUSI_with_GT/benign
    return tuple(
        sorted(
            path
            for path in image_path.parent.glob(f"{prefix}*.png")
            if is_mask(path)
        )
    )


def discover_samples(data_dir: Path, class_name: str):
    folder = data_dir / class_name
    if not folder.is_dir():
        return []
    images = sorted(path for path in folder.glob("*.png") if not is_mask(path))
    return [Sample(image=path, masks=find_masks(path)) for path in images]


def crop_box(size: tuple[int, int], crop_name: str, fraction: float):
    """Return a deterministic crop box; named crops retain the named region"""
    width, height = size
    dx = max(1, round(width * fraction))
    dy = max(1, round(height * fraction))

    boxes = {
        "original": (0, 0, width, height),
        "crop_right": (dx, 0, width, height),
        "crop_corner": (dx, dy, width, height),
        "crop_bottom": (0, dy, width, height),
        "crop_left": (0, 0, width - dx, height),
        "crop_top": (0, 0, width, height - dy),
        "crop_whole": (dx, dy, width - dx, height - dy),
    }
    return boxes[crop_name]


def transform(
    image: Image.Image,
    crop_name: str,
    angle: int,
    *,
    crop_fraction: float,
    is_mask_image: bool,
):
    """Apply the same geometry to an image or mask"""
    original_size = image.size
    interpolation = Image.Resampling.NEAREST if is_mask_image else Image.Resampling.BILINEAR
    output = image.crop(crop_box(original_size, crop_name, crop_fraction))
    if output.size != original_size:
        output = output.resize(original_size, interpolation)
    if angle:
        output = output.rotate(angle, resample=interpolation, expand=False, fillcolor=0)
    return output



def variant_name(crop_name: str, angle: int):
    return f"{crop_name}_r{angle:03d}"


def save_variant(
    sample: Sample,
    destination: Path,
    crop_name: str,
    angle: int,
    *,
    crop_fraction: float,
    sequence: int,
):
    destination.mkdir(parents=True, exist_ok=True)
    suffix = variant_name(crop_name, angle)
    output_stem = f"{sample.image.stem}__aug{sequence:05d}__{suffix}"

    with Image.open(sample.image) as source:
        transformed = transform(
            source,
            crop_name,
            angle,
            crop_fraction=crop_fraction,
            is_mask_image=False,
        )
        transformed.save(destination / f"{output_stem}.png")

    for mask_index, mask_path in enumerate(sample.masks, start=1):
        # Preserve the conventional single-mask name; number multiple masks.
        mask_suffix = "_mask" if len(sample.masks) == 1 else f"_mask_{mask_index}"
        with Image.open(mask_path) as source_mask:
            transformed_mask = transform(
                source_mask,
                crop_name,
                angle,
                crop_fraction=crop_fraction,
                is_mask_image=True,
            )
            transformed_mask.save(destination / f"{output_stem}{mask_suffix}.png")


def copy_original(sample: Sample, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sample.image, destination / sample.image.name)
    for mask in sample.masks:
        shutil.copy2(mask, destination / mask.name)


def all_variants():
    """every row in the preview has one rotation"""
    return [(crop_name, angle) for angle in ROTATIONS for crop_name in CROPS]


def make_preview(
    sample: Sample,
    output_path: Path,
    *,
    crop_fraction: float,
):
    column_titles = (
        "Original",
        "Crop Right",
        "Crop Corner",
        "Crop Bottom",
        "Crop Left",
        "Crop Top",
        "Crop Whole",
    )
    with Image.open(sample.image) as source:
        source = source.copy()

    cell_width, cell_height = 220, 170
    left_margin, top_margin = 65, 80
    canvas = Image.new(
        "RGB",
        (left_margin + 7 * cell_width, top_margin + 4 * cell_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = f"28 geometric variants - {sample.image.name}"
    draw.text((left_margin, 12), title, fill="black", font=font)
    for column, column_title in enumerate(column_titles):
        draw.text(
            (left_margin + column * cell_width + 8, 45),
            column_title,
            fill="black",
            font=font,
        )

    for row, angle in enumerate(ROTATIONS):
        draw.text(
            (12, top_margin + row * cell_height + cell_height // 2),
            f"{angle} deg",
            fill="black",
            font=font,
        )
        for column, crop_name in enumerate(CROPS):
            augmented = transform(
                source,
                crop_name,
                angle,
                crop_fraction=crop_fraction,
                is_mask_image=False,
            )
            if augmented.mode not in ("RGB", "L"):
                augmented = augmented.convert("RGB")
            augmented.thumbnail(
                (cell_width - 12, cell_height - 12), Image.Resampling.LANCZOS
            )
            x = left_margin + column * cell_width + (cell_width - augmented.width) // 2
            y = top_margin + row * cell_height + (cell_height - augmented.height) // 2
            canvas.paste(augmented, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def plot_distribution(before: dict[str, int], after: dict[str, int], output_path: Path):
    width, height = 1000, 700
    margin_left, margin_right, margin_top, margin_bottom = 90, 40, 90, 90
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (margin_left, 25),
        "BUSI class distribution before and after augmentation",
        fill="black",
        font=font,
    )
    draw.rectangle((margin_left, 52, margin_left + 18, 70), fill="#4C78A8")
    draw.text((margin_left + 25, 54), "Before", fill="black", font=font)
    draw.rectangle((margin_left + 110, 52, margin_left + 128, 70), fill="#F58518")
    draw.text((margin_left + 135, 54), "After", fill="black", font=font)

    baseline = margin_top + chart_height
    draw.line((margin_left, margin_top, margin_left, baseline), fill="black", width=2)
    draw.line((margin_left, baseline, width - margin_right, baseline), fill="black", width=2)
    maximum = max((*before.values(), *after.values(), 1))
    group_width = chart_width / len(CLASSES)
    bar_width = 75

    for index, label in enumerate(CLASSES):
        center = margin_left + group_width * (index + 0.5)
        for value, offset, color in (
            (before[label], -bar_width, "#4C78A8"),
            (after[label], 0, "#F58518"),
        ):
            bar_height = round(value / maximum * (chart_height - 35))
            x0 = round(center + offset)
            y0 = baseline - bar_height
            draw.rectangle((x0, y0, x0 + bar_width, baseline), fill=color)
            value_text = str(value)
            draw.text(
                (x0 + bar_width // 2 - 8, y0 - 18),
                value_text,
                fill="black",
                font=font,
            )
        draw.text(
            (round(center - 28), baseline + 20),
            label,
            fill="black",
            font=font,
        )

    draw.text(
        (margin_left, height - 25),
        "Number of source/augmented images (masks excluded)",
        fill="black",
        font=font,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def run_balanced(
    samples_by_class: dict[str, list[Sample]],
    output_dir: Path,
    *,
    target_count: int | None,
    crop_fraction: float,
    seed: int,
):
    counts = {name: len(samples) for name, samples in samples_by_class.items()}
    target = target_count if target_count is not None else max(counts.values(), default=0)
    if target < max(counts.values(), default=0):
        raise ValueError(
            "--target-count cannot be smaller than an existing class count; "
            "this script does not delete source samples."
        )

    variants = all_variants()
    rng = random.Random(seed)
    after: dict[str, int] = {}

    for class_name in CLASSES:
        samples = samples_by_class[class_name]
        if not samples:
            raise ValueError(f"No original PNG images found for class: {class_name}")
        destination = output_dir / class_name
        for sample in samples:
            copy_original(sample, destination)

        required = target - len(samples)
        order = list(samples)
        rng.shuffle(order)
        for index in range(required):
            sample = order[index % len(order)]
            crop_name, angle = variants[index % len(variants)]
            save_variant(
                sample,
                destination,
                crop_name,
                angle,
                crop_fraction=crop_fraction,
                sequence=index + 1,
            )
        after[class_name] = target
    return after


def run_full28(
    samples_by_class: dict[str, list[Sample]],
    output_dir: Path,
    *,
    crop_fraction: float,
):
    variants = all_variants()
    after: Counter[str] = Counter()
    for class_name in CLASSES:
        destination = output_dir / class_name
        for sample_index, sample in enumerate(samples_by_class[class_name]):
            for variant_index, (crop_name, angle) in enumerate(variants):
                save_variant(
                    sample,
                    destination,
                    crop_name,
                    angle,
                    crop_fraction=crop_fraction,
                    sequence=sample_index * len(variants) + variant_index + 1,
                )
                after[class_name] += 1
    return dict(after)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("Dataset_BUSI_with_GT"))
    parser.add_argument("--output-dir", type=Path, default=Path("Dataset_BUSI_Augmentation"))
    parser.add_argument(
        "--mode",
        choices=("balanced", "full28"),
        default="balanced",
        help="balanced: exact equal class totals; full28: 28 outputs per source image",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="balanced mode target; default is the largest original class",
    )
    parser.add_argument(
        "--crop-fraction",
        type=float,
        default=0.15,
        help="fraction removed from named edge(s), in (0, 0.5)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--preview-class",
        choices=CLASSES,
        default="benign",
        help="class from which the first sample is used in the 28-panel preview",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.crop_fraction < 0.5:
        raise ValueError("--crop-fraction must be between 0 and 0.5")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir.resolve()}")
    if args.output_dir.resolve() == args.data_dir.resolve():
        raise ValueError("--output-dir must differ from --data-dir")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir.resolve()}. "
            "Use a new/empty directory to avoid stale or overwritten data."
        )

    samples_by_class = {
        class_name: discover_samples(args.data_dir, class_name)
        for class_name in CLASSES
    }
    before = {
        class_name: len(samples_by_class[class_name]) for class_name in CLASSES
    }
    missing_masks = [
        str(sample.image)
        for samples in samples_by_class.values()
        for sample in samples
        if not sample.masks
    ]
    if missing_masks:
        print(
            f"Warning: {len(missing_masks)} source image(s) have no matching mask; "
            "their image variants will still be generated."
        )

    if args.mode == "balanced":
        after = run_balanced(
            samples_by_class,
            args.output_dir,
            target_count=args.target_count,
            crop_fraction=args.crop_fraction,
            seed=args.seed,
        )
    else:
        after = run_full28(
            samples_by_class,
            args.output_dir,
            crop_fraction=args.crop_fraction,
        )

    preview_samples = samples_by_class[args.preview_class]
    if not preview_samples:
        preview_samples = next(
            (samples for samples in samples_by_class.values() if samples), []
        )
    if preview_samples:
        make_preview(
            preview_samples[0],
            args.output_dir / "augmentation_preview_28.png",
            crop_fraction=args.crop_fraction,
        )
    plot_distribution(
        before,
        after,
        args.output_dir / "class_distribution_before_after.png",
    )

    report = {
        "mode": args.mode,
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "crop_fraction": args.crop_fraction,
        "before": before,
        "after": after,
        "source_images_without_masks": missing_masks,
    }
    report_path = args.output_dir / "augmentation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
