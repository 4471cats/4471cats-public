import requests
import tarfile
from pathlib import Path
from tqdm import tqdm


def prompt(left: str, right: str = ""):
    print(f"{left.ljust(50)}{right.rjust(30)}")


def wget(url: str, dest: Path):
    if dest.exists():
        file_size = dest.stat().st_size
        size_mb = file_size / (1024**2)
        prompt(f"✅ File {dest.name} already exists", f"({size_mb:.1f} MB)")
        return
    prompt(f"⬇️  Downloading {dest.name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(
            desc=f"Downloading {dest.name}",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                _ = f.write(chunk)
                _ = bar.update(len(chunk))
    file_size = dest.stat().st_size
    size_mb = file_size / (1024**2)
    prompt(f"✅ Downloaded {dest.name}", f"({size_mb:.1f} MB)")


def extract(tar_path: Path, dest_dir: Path):
    prompt(
        f"📦 Extracting {tar_path.name}...", f"to {dest_dir.relative_to(Path.cwd())}"
    )
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        with tqdm(
            total=len(members), desc=f"Extracting {tar_path.name}", unit="files"
        ) as bar:
            for member in members:
                tar.extract(member, dest_dir)
                _ = bar.update(1)
    prompt(f"✅ Extracted {tar_path.name}", f"{len(members)} files")


def md5sum(file_path: Path) -> str:
    import hashlib

    print(f"🔍 Verifying {file_path.name}...".ljust(50))
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    calculated_hash = hash_md5.hexdigest()
    return calculated_hash
