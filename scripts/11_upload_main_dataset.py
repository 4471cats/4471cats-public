#!/usr/bin/env python3
"""
Script to upload to Kaggle.
"""

import json
import subprocess
from pathlib import Path
from textwrap import dedent

import typer

app = typer.Typer()

WORKSPACE_DIR = Path(__file__).parent.parent.resolve()
DATASET_PATH = WORKSPACE_DIR / "datasets" / "of3t-cats-yolo"

METADATA = {
    "title": "4471 Cats YOLO Dataset",
    "id": "inogai/of3t-cats-yolo",
    "licenses": [
        {"name": "CC BY-SA 4.0"},
    ],
    "description": dedent("""
      Derived from the Oxford-IIIT Pet Dataset.
      https://www.robots.ox.ac.uk/~vgg/data/pets/

      The original paper:
      ```
      O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
      Cats and Dogs  
      IEEE Conference on Computer Vision and Pattern Recognition, 2012
      ```

      Modifications:
      - Removed non-cat classes.
      - Converted annotations to YOLO format.
      - Split into training and validation sets with stratification by class.
      - Generated whole body bounding boxes from trimaps.
    """).strip(),
}


@app.command("init")
def create_metadata_file():
    """Create the metadata file required by Kaggle."""

    with open(DATASET_PATH / "dataset-metadata.json", "w") as f:
        _ = f.write(json.dumps(METADATA, indent=2))


@app.command()
def create():
    create_metadata_file()

    cmd = ["kaggle", "datasets", "create", "-p", str(DATASET_PATH), "-r", "zip"]
    _ = subprocess.run(cmd, cwd=WORKSPACE_DIR, check=True)


@app.command()
def upload(message: str):
    create_metadata_file()

    cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(DATASET_PATH),
        "-m",
        message,
        "-r",
        "zip",
    ]
    _ = subprocess.run(cmd, cwd=DATASET_PATH.parent)


if __name__ == "__main__":
    app()
