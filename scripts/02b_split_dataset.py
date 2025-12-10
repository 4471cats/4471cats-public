"""
Split labeled YOLO dataset into train/val sets.

This script reads from of3t-cats-yolo/labeled/ and creates train/val splits
with stratification by class.

Input:
- of3t-cats-yolo/images/labeled/  - labeled images
- of3t-cats-yolo/labels/labeled/  - YOLO label files

Output:
- of3t-cats-yolo/images/train/    - training images
- of3t-cats-yolo/labels/train/    - training labels
- of3t-cats-yolo/images/val/      - validation images
- of3t-cats-yolo/labels/val/      - validation labels
"""

import argparse
import shutil
from pathlib import Path

from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Constants
DEFAULT_TRAIN_SPLIT = 0.8
WORKSPACE_DIR = Path(__file__).parent.parent
DEFAULT_INPUT_PATH = WORKSPACE_DIR / "datasets" / "of3t-cats-yolo"

CLASS_NAMES = [
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx",
]


def setup_logging(output_dir: Path) -> None:
    """Setup loguru logging to write to a file in the output directory."""
    log_file = output_dir / "split.log"
    _ = logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="1 week",
    )


def get_class_id_from_label(label_path: Path) -> int | None:
    """Extract class ID from a YOLO label file."""
    try:
        with open(label_path, "r") as f:
            content = f.read().strip()
            if content:
                # First value is class_id
                return int(content.split()[0])
    except Exception:
        pass
    return None


def load_labeled_samples(input_path: Path) -> list[tuple[str, int]]:
    """Load all labeled samples and extract their class IDs for stratification."""
    labels_dir = input_path / "labels" / "labeled"
    samples = []

    label_files = list(labels_dir.glob("*.txt"))
    logger.info(f"Found {len(label_files)} label files in {labels_dir}")

    for label_file in label_files:
        class_id = get_class_id_from_label(label_file)
        if class_id is not None:
            image_id = label_file.stem
            samples.append((image_id, class_id))
        else:
            logger.warning(f"Could not extract class from {label_file}")

    logger.info(f"Loaded {len(samples)} labeled samples")
    return samples


def setup_output_directories(input_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create output directories for train/val split."""
    images_train = input_path / "images" / "train"
    images_val = input_path / "images" / "val"
    labels_train = input_path / "labels" / "train"
    labels_val = input_path / "labels" / "val"

    for path in [images_train, images_val, labels_train, labels_val]:
        path.mkdir(parents=True, exist_ok=True)

    return images_train, images_val, labels_train, labels_val


def copy_samples(
    samples: list[tuple[str, int]],
    input_path: Path,
    img_dest: Path,
    label_dest: Path,
    split_name: str,
) -> int:
    """Copy samples to destination directories."""
    copied = 0
    images_src = input_path / "images" / "labeled"
    labels_src = input_path / "labels" / "labeled"

    for image_id, _ in tqdm(samples, desc=f"Copying {split_name}"):
        src_img = images_src / f"{image_id}.jpg"
        src_label = labels_src / f"{image_id}.txt"

        if src_img.exists() and src_label.exists():
            _ = shutil.copy2(src_img, img_dest / f"{image_id}.jpg")
            _ = shutil.copy2(src_label, label_dest / f"{image_id}.txt")
            copied += 1
        else:
            logger.warning(f"Missing files for {image_id}")

    return copied


def update_dataset_config(input_path: Path) -> None:
    """Update data.yaml to reference train/val directories."""
    yaml_content = f"""# YOLO Dataset Configuration
# Generated from Oxford-IIIT Pet Dataset

path: {input_path.name}  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val      # val images (relative to 'path')

# Classes
nc: {len(CLASS_NAMES)}  # number of classes
names: {CLASS_NAMES}  # class names

# Dataset info
# Source: Oxford-IIIT Pet Dataset
# Format: JPG images with trimap-derived bounding boxes
# Classes: {len(CLASS_NAMES)} cat breeds
"""

    yaml_file = input_path / "data.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    logger.info(f"Configuration file updated: {yaml_file}")


def split_dataset(input_path: Path, train_split: float = 0.8) -> None:
    """Split labeled dataset into train/val sets."""
    input_path = Path(input_path)

    # Setup logging
    setup_logging(input_path)

    logger.info(f"Splitting dataset from: {input_path}")
    logger.info(f"Train split ratio: {train_split}")

    # Load labeled samples
    samples = load_labeled_samples(input_path)

    if not samples:
        logger.error("No labeled samples found!")
        return

    # Extract image IDs and class IDs for stratification
    image_ids = [s[0] for s in samples]
    class_ids = [s[1] for s in samples]

    # Split with stratification
    train_ids, val_ids, train_classes, val_classes = train_test_split(
        image_ids,
        class_ids,
        train_size=train_split,
        stratify=class_ids,
        random_state=42,
    )

    train_samples = list(zip(train_ids, train_classes))
    val_samples = list(zip(val_ids, val_classes))

    logger.info(f"Train set size: {len(train_samples)}")
    logger.info(f"Val set size: {len(val_samples)}")

    # Setup output directories
    images_train, images_val, labels_train, labels_val = setup_output_directories(
        input_path
    )

    # Copy samples
    train_copied = copy_samples(
        train_samples, input_path, images_train, labels_train, "train"
    )
    val_copied = copy_samples(val_samples, input_path, images_val, labels_val, "val")

    # Update configuration
    update_dataset_config(input_path)

    logger.info(f"Split completed. Train: {train_copied}, Val: {val_copied}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split labeled YOLO dataset into train/val sets"
    )
    _ = parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to the YOLO dataset (default: {DEFAULT_INPUT_PATH})",
    )
    _ = parser.add_argument(
        "--train-split",
        type=float,
        default=DEFAULT_TRAIN_SPLIT,
        help=f"Fraction of data to use for training (default: {DEFAULT_TRAIN_SPLIT})",
    )
    _ = parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing train/val directories",
    )

    args = parser.parse_args()

    logger.info("📊 Splitting YOLO dataset into train/val sets")
    logger.info(f"Input path: {args.input_path}")
    logger.info(f"Train split: {args.train_split}")

    # Check if input exists
    labeled_dir = args.input_path / "images" / "labeled"
    if not labeled_dir.exists():
        logger.error(f"Labeled directory not found: {labeled_dir}")
        print("❌ Error: Labeled directory not found. Run 02_build_main_dataset.py first.")
        exit(1)

    # Check if output exists
    train_dir = args.input_path / "images" / "train"
    if train_dir.exists() and not args.force:
        logger.warning(f"Train directory already exists: {train_dir}")
        print("❌ Error: Train directory already exists. Use --force to overwrite.")
        exit(1)

    # Clean existing directories if force
    if args.force:
        for subdir in ["train", "val"]:
            for parent in ["images", "labels"]:
                path = args.input_path / parent / subdir
                if path.exists():
                    shutil.rmtree(path)
                    logger.info(f"Removed existing directory: {path}")

    # Split dataset
    split_dataset(args.input_path, args.train_split)
