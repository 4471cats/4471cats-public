"""Script to download and prepare datasets."""

from pathlib import Path

from utils import extract, md5sum, prompt, wget

WORKSPACE_DIR = Path.cwd().resolve()

DATASET_SOURCE = {
    "oxford3tpet": [
        (
            "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz",
            "95a8c909bbe2e81eed6a22bccdf3f68f",
        ),
        (
            "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz",
            "5c4f3ee8e5d25df40f4fd59a7f44e54c",
        ),
    ]
}


def get_dataset(key: str):
    if key not in DATASET_SOURCE:
        raise ValueError(f"Dataset {key} is not supported.")

    print(f"🚀 Preparing dataset: {key}")
    dataset_dir = WORKSPACE_DIR / "datasets" / key
    dataset_dir.mkdir(parents=True, exist_ok=True)

    for url, hash in DATASET_SOURCE[key]:
        tar_path = dataset_dir / Path(url).name
        if not tar_path.exists():
            wget(url, tar_path)

        calculated_hash = md5sum(tar_path)
        if calculated_hash != hash:
            raise ValueError(
                f"❌ Checksum mismatch for {tar_path.name}: expected {hash}, got {calculated_hash}"
            )
        else:
            prompt(f"✅ Integrity verified for {tar_path.name}", f"{calculated_hash}")

        extract(tar_path, dataset_dir)

    dataset_dir_relative = dataset_dir.relative_to(WORKSPACE_DIR)
    prompt(f"🎉 Dataset {key} is ready at {dataset_dir_relative}", "")

    return dataset_dir


if __name__ == "__main__":
    _ = get_dataset("oxford3tpet")
