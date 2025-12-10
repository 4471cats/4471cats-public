#!/usr/bin/env python3
"""
Script to upload wiki images to Kaggle.
"""

import json
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

import typer

app = typer.Typer()

WORKSPACE_DIR = Path(__file__).parent.parent.resolve()
DATASET_PATH = WORKSPACE_DIR / "datasets" / "4471-cats-wiki-images"

METADATA = {
    "title": "4471 Cats Wiki Images",
    "id": "inogai/4471-cats-wiki-images",
    "licenses": [
        {"name": "CC BY-SA 4.0"},
    ],
    "description": dedent("""
        Collection of cat images from various Wikimedia Commons and other sources.
        Source information and licenses are documented in wiki_sources.csv.
        
        Images include various cat breeds and settings from around the world.
        All images are used with appropriate licenses as specified in the source file.

        See wiki_sources.csv for details.
    """).strip(),
}


@app.command("init")
def create_metadata_file():
    """Create the metadata file required by Kaggle."""

    _ = shutil.copy2(
        WORKSPACE_DIR / "scripts" / "wiki_sources.csv",
        DATASET_PATH / "wiki_sources.csv",
    )

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
