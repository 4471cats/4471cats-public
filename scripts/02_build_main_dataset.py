"""
Convert Oxford-IIIT Pet Dataset to YOLO format.

This script converts the dataset from the original format (images + trimap annotations)
to YOLO format (images + .txt annotation files).

Output structure:
- of3t-cats-yolo/images/labeled/   - images with successful trimap extraction
- of3t-cats-yolo/labels/labeled/   - YOLO label files (non-empty)
- of3t-cats-yolo/images/unlabeled/ - images where trimap extraction failed
- of3t-cats-yolo/labels/unlabeled/ - empty label files

YOLO format:
- Each image has a corresponding .txt file
- Each line: class_id center_x center_y width height
- All coordinates normalized to [0,1] relative to image dimensions
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from tqdm import tqdm

# Constants
WORKSPACE_DIR = Path(__file__).parent.parent
DEFAULT_DATASET_PATH = WORKSPACE_DIR / "datasets" / "oxford3tpet"
DEFAULT_OUTPUT_PATH = WORKSPACE_DIR / "datasets"
TRIMAP_FOREGROUND = 1
TRIMAP_BACKGROUND = 2
TRIMAP_UNKNOWN = 3

# Hard-coded class mapping for 12 cat breeds
# Maps Oxford dataset class IDs to YOLO class IDs (0-indexed)
CLASS_MAPPING = {
    1: 0,   # Abyssinian
    6: 1,   # Bengal
    7: 2,   # Birman
    8: 3,   # Bombay
    10: 4,  # British_Shorthair
    12: 5,  # Egyptian_Mau
    21: 6,  # Maine_Coon
    24: 7,  # Persian
    27: 8,  # Ragdoll
    28: 9,  # Russian_Blue
    33: 10, # Siamese
    34: 11, # Sphynx
}

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
    log_file = output_dir / "conversion.log"
    _ = logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
        rotation="10 MB",
        retention="1 week",
    )


def load_dataset_list(list_file: str | Path) -> pd.DataFrame:
    """Load the dataset list file and filter for cats only."""
    with open(list_file, "r") as f:
        lines = f.readlines()

    # Filter out comments and extract data
    lines = filter(lambda x: not x.startswith("#"), lines)
    df = pd.DataFrame([line.strip().split() for line in lines])

    df.columns = ["id", "class", "species", "breed"]

    # Convert to appropriate types
    df["class"] = pd.to_numeric(df["class"], errors="coerce")
    df["species"] = pd.to_numeric(df["species"], errors="coerce")
    df["breed"] = pd.to_numeric(df["breed"], errors="coerce")

    # Filter cats only (uppercase first letter indicates cat)
    df = df[df["id"].apply(lambda x: x[0].isupper())]
    return df


class DetectLabel(BaseModel):
    class_id: int
    top: int
    left: int
    right: int
    bottom: int

    image_size: tuple[int, int]

    def to_yolo_format(self) -> str:
        """Convert annotation to YOLO format."""
        center_x = (self.left + self.right) / 2.0 / self.image_size[0]
        center_y = (self.top + self.bottom) / 2.0 / self.image_size[1]

        height = (self.bottom - self.top) / self.image_size[1]
        width = (self.right - self.left) / self.image_size[0]

        return f"{self.class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def setup_output_directories(
    output_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    """Create output directories for YOLO dataset."""
    yolo_path = output_path / "of3t-cats-yolo"
    images_labeled = yolo_path / "images" / "labeled"
    images_unlabeled = yolo_path / "images" / "unlabeled"
    labels_labeled = yolo_path / "labels" / "labeled"
    labels_unlabeled = yolo_path / "labels" / "unlabeled"

    # Create directories
    for path in [
        images_labeled,
        images_unlabeled,
        labels_labeled,
        labels_unlabeled,
    ]:
        path.mkdir(parents=True, exist_ok=True)

    return (
        yolo_path,
        images_labeled,
        images_unlabeled,
        labels_labeled,
        labels_unlabeled,
    )


def load_and_prepare_data(dataset_path: Path) -> pd.DataFrame:
    """Load dataset lists and combine trainval + test data."""
    trainval_file = dataset_path / "annotations" / "trainval.txt"
    test_file = dataset_path / "annotations" / "test.txt"
    trainval_df = load_dataset_list(trainval_file)
    test_df = load_dataset_list(test_file)

    logger.info(f"Found cat images in trainval: {len(trainval_df)}")
    logger.info(f"Found cat images in test: {len(test_df)}")

    # Combine all data
    all_df = pd.concat([trainval_df, test_df], ignore_index=True)
    logger.info(f"Total cat images: {len(all_df)}")

    return all_df


def create_dataset_config(yolo_path: Path) -> None:
    """Create data.yaml configuration file."""
    yaml_content = f"""# YOLO Dataset Configuration
# Generated from Oxford-IIIT Pet Dataset

path: {yolo_path.name}  # dataset root dir
train: images/labeled  # labeled images (relative to 'path')
val: images/labeled    # use same for now, split with 02b script

# Classes
nc: {len(CLASS_NAMES)}  # number of classes
names: {CLASS_NAMES}  # class names

# Dataset info
# Source: Oxford-IIIT Pet Dataset
# Format: JPG images with trimap-derived bounding boxes
# Classes: {len(CLASS_NAMES)} cat breeds
"""

    yaml_file = yolo_path / "data.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)

    logger.info(f"Configuration file created: {yaml_file}")


class AnnotationError(Exception):
    """Custom exception for annotation errors."""


def process_single_sample(
    image_id: str,
    class_id: int,
    dataset_path: Path,
    img_labeled: Path,
    img_unlabeled: Path,
    label_labeled: Path,
    label_unlabeled: Path,
) -> bool:
    """
    Process a single image sample and convert to YOLO format.

    Args:
        image_id: The image identifier (filename without extension)
        class_id: The original class ID from the dataset
        dataset_path: Path to the source dataset
        img_labeled: Destination directory for labeled images
        img_unlabeled: Destination directory for unlabeled images
        label_labeled: Destination directory for label files (non-empty)
        label_unlabeled: Destination directory for empty label files

    Returns:
        True if label was successfully extracted, False otherwise

    Raises:
        FileNotFoundError: If the image file is not found
    """
    image_path = dataset_path / "images" / f"{image_id}.jpg"
    trimap_path = dataset_path / "annotations" / "trimaps" / f"{image_id}.png"

    bounding_box = None
    has_label = False

    # Get image dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size

    # Try to extract bounding box from trimap
    try:
        with Image.open(trimap_path) as trimap_img:
            trimap_array = np.array(trimap_img)

        topmost = int(np.min(np.where(trimap_array == TRIMAP_FOREGROUND)[0]))
        bottommost = int(np.max(np.where(trimap_array == TRIMAP_FOREGROUND)[0]))
        leftmost = int(np.min(np.where(trimap_array == TRIMAP_FOREGROUND)[1]))
        rightmost = int(np.max(np.where(trimap_array == TRIMAP_FOREGROUND)[1]))

        bounding_box = DetectLabel(
            class_id=CLASS_MAPPING[class_id],
            left=leftmost,
            right=rightmost,
            top=topmost,
            bottom=bottommost,
            image_size=(img_width, img_height),
        )
        has_label = True
    except Exception as e:
        logger.debug(f"Could not extract trimap for {image_id}: {e}")
        has_label = False

    # Route to labeled or unlabeled directory
    if has_label:
        img_dest = img_labeled
        label_dest = label_labeled
    else:
        img_dest = img_unlabeled
        label_dest = label_unlabeled

    # Copy image to destination
    _ = shutil.copy2(image_path, img_dest / f"{image_id}.jpg")

    # Write YOLO annotation file
    label_file = label_dest / f"{image_id}.txt"
    with open(label_file, "w") as f:
        if bounding_box:
            content = bounding_box.to_yolo_format()
            _ = f.write(f"{content}\n")

    return has_label


def process_all_samples(
    df: pd.DataFrame,
    dataset_path: Path,
    img_labeled: Path,
    img_unlabeled: Path,
    label_labeled: Path,
    label_unlabeled: Path,
) -> tuple[int, int, int]:
    """Process all samples, routing to labeled or unlabeled directories."""
    labeled_count = 0
    unlabeled_count = 0
    errors = 0

    logger.info(f"Processing {len(df)} images")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
        image_id = row["id"]
        class_id = int(row["class"])

        try:
            has_label = process_single_sample(
                image_id=image_id,
                class_id=class_id,
                dataset_path=dataset_path,
                img_labeled=img_labeled,
                img_unlabeled=img_unlabeled,
                label_labeled=label_labeled,
                label_unlabeled=label_unlabeled,
            )
            if has_label:
                labeled_count += 1
            else:
                unlabeled_count += 1

        except FileNotFoundError as e:
            logger.warning(str(e))
            errors += 1
        except Exception as e:
            logger.error(f"Error processing {image_id}: {e}")
            errors += 1

    logger.info(f"Labeled: {labeled_count}, Unlabeled: {unlabeled_count}, Errors: {errors}")
    return labeled_count, unlabeled_count, errors


def convert_to_yolo(dataset_path: Path, output_path: Path) -> None:
    """Convert the entire dataset to YOLO format."""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    # Setup output directories
    (
        yolo_path,
        images_labeled,
        images_unlabeled,
        labels_labeled,
        labels_unlabeled,
    ) = setup_output_directories(output_path)

    # Setup logging
    setup_logging(yolo_path)

    # Load and prepare data (combine trainval + test)
    all_df = load_and_prepare_data(dataset_path)

    # Write class names file
    classes_file = yolo_path / "classes.txt"
    with open(classes_file, "w") as f:
        for name in CLASS_NAMES:
            _ = f.write(f"{name}\n")

    # Process all samples
    labeled, unlabeled, errors = process_all_samples(
        all_df,
        dataset_path,
        images_labeled,
        images_unlabeled,
        labels_labeled,
        labels_unlabeled,
    )

    # Create dataset configuration
    create_dataset_config(yolo_path)

    logger.info(
        f"Conversion completed. Labeled: {labeled}, Unlabeled: {unlabeled}, Errors: {errors}"
    )
    logger.info(f"YOLO dataset saved to: {yolo_path}")
    logger.info(f"Configuration file: {yolo_path / 'data.yaml'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Oxford-IIIT Pet Dataset to YOLO format"
    )
    _ = parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the Oxford-IIIT Pet dataset (default: {DEFAULT_DATASET_PATH})",
    )
    _ = parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output directory for YOLO dataset (default: {DEFAULT_OUTPUT_PATH})",
    )
    _ = parser.add_argument(
        "--force", action="store_true", help="Overwrite existing output directory"
    )

    args = parser.parse_args()

    logger.info("🚀 Converting Oxford-IIIT Pet Dataset to YOLO format")
    logger.info(f"Input dataset: {args.dataset_path}")
    logger.info(f"Output directory: {args.output_path}")

    # Check if dataset exists
    if not args.dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset_path}")
        print(
            "❌ Error: Dataset not found. Please run the dataset download script first."
        )
        exit(1)

    # Check if output exists
    yolo_path = args.output_path / "of3t-cats-yolo"
    if yolo_path.exists() and not args.force:
        logger.warning(f"Output directory already exists: {yolo_path}")
        print("❌ Error: Output directory already exists. Use --force to overwrite.")
        exit(1)

    # Convert dataset
    convert_to_yolo(args.dataset_path, args.output_path)
