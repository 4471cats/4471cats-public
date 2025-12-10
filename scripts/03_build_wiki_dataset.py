"""Script to download images from wiki_sources.csv (Wikimedia Commons and Public Domain Pictures)."""

import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import polars as pl
import requests
from bs4 import BeautifulSoup
from loguru import logger
from tqdm import tqdm


class DownloadStatus(Enum):
    """Status of a download operation."""

    DOWNLOADED = auto()
    SKIPPED = auto()
    FAILED = auto()


@dataclass
class DownloadResult:
    """Result of a download operation."""

    filename: str | None
    status: DownloadStatus


USER_AGENT = "CatsDatasetDownloader/1.0 (Educational Research Project; Cat Image Classification; Contact: github.com/4471cats)"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}


def generate_filename(serial_id: int, original_filename: str) -> str:
    """Generate a filename in the format <serial_id>_<original_filename>."""
    return f"{serial_id}_{original_filename}"


def make_absolute_url(base_url: str, relative_url: str) -> str:
    """Convert a relative URL to absolute."""
    if relative_url.startswith("http"):
        return relative_url
    if relative_url.startswith("../"):
        return base_url + "/" + relative_url.lstrip("../")
    return base_url + relative_url


def save_response(response: requests.Response, dest: Path) -> None:
    """Write a streaming response to destination file."""
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def extract_wikimedia_title(url: str) -> str:
    """Extract the File:... title from a Wikimedia Commons URL."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    query = parse_qs(parsed.query)

    if "/wiki/" in path:
        return path.split("/wiki/")[-1]

    if "curid" in query:
        curid = query["curid"][0]
        api_url = f"https://commons.wikimedia.org/w/api.php?action=query&pageids={curid}&format=json"
        response = requests.get(api_url, headers={"User-Agent": USER_AGENT})
        response.raise_for_status()
        data = response.json()
        pages = data["query"]["pages"]
        if curid in pages:
            return pages[curid]["title"]
        raise ValueError(f"Could not find page with curid: {curid}")

    raise ValueError(f"Cannot extract title from URL: {url}")


def normalize_license(license_str: str) -> str:
    """Normalize license string for comparison."""
    return license_str.lower().replace("-", " ").replace("_", " ").strip()


def download_from_wikimedia(
    url: str,
    dest_dir: Path,
    serial_id: int,
    filename: str | None,
    expected_license: str,
) -> DownloadResult:
    """Download an image from Wikimedia Commons with license validation."""
    headers = {"User-Agent": USER_AGENT}

    # Early exit if file exists
    if filename and (dest_dir / filename).exists():
        return DownloadResult(filename, DownloadStatus.SKIPPED)

    file_title = extract_wikimedia_title(url)

    # Query API for image info and license
    api_url = (
        f"https://commons.wikimedia.org/w/api.php?"
        f"action=query&prop=imageinfo&iiprop=extmetadata|url&format=json&titles={file_title}"
    )
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    data = response.json()

    pages = data["query"]["pages"]
    page_id = list(pages.keys())[0]
    if page_id == "-1":
        raise ValueError(f"Image '{file_title}' not found on Wikimedia Commons.")

    imageinfo = pages[page_id]["imageinfo"][0]
    image_url = imageinfo["url"]

    # Validate license
    extmetadata = imageinfo.get("extmetadata", {})
    api_license = extmetadata.get("LicenseShortName", {}).get("value", "")
    if api_license and expected_license:
        if normalize_license(api_license) != normalize_license(expected_license):
            logger.warning(
                f"License mismatch: expected '{expected_license}', got '{api_license}'"
            )

    # Determine filename
    original_filename = Path(image_url).name
    dest_filename = filename or generate_filename(serial_id, original_filename)
    dest_path = dest_dir / dest_filename

    if dest_path.exists():
        return DownloadResult(dest_filename, DownloadStatus.SKIPPED)

    response = requests.get(image_url, stream=True, headers=headers)
    response.raise_for_status()
    save_response(response, dest_path)
    return DownloadResult(dest_filename, DownloadStatus.DOWNLOADED)


def download_from_publicdomain(
    url: str,
    dest_dir: Path,
    serial_id: int,
    filename: str | None,
    expected_license: str,
) -> DownloadResult:
    """Download an image from publicdomainpictures.net."""
    base_url = "https://www.publicdomainpictures.net"

    # Early exit if file exists
    if filename and (dest_dir / filename).exists():
        return DownloadResult(filename, DownloadStatus.SKIPPED)

    # Parse URL parameters
    parsed = urlparse(url)
    params = parse_qs(parsed.query)
    image_name = params.get("image", ["image"])[0]
    image_id = params.get("id", [""])[0]

    if not image_id:
        raise ValueError(f"Could not extract image ID from URL: {url}")

    # Determine filename
    original_filename = f"{image_name}.jpg"
    dest_filename = filename or generate_filename(serial_id, original_filename)
    dest_path = dest_dir / dest_filename

    if dest_path.exists():
        return DownloadResult(dest_filename, DownloadStatus.SKIPPED)

    # Validate license (publicdomainpictures.net is always CC0)
    if expected_license and "cc0" not in normalize_license(expected_license):
        logger.warning(
            f"License mismatch: expected '{expected_license}', publicdomainpictures.net uses CC0"
        )

    # Create session with browser headers
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    session.get(base_url)  # Get cookies

    # Fetch view page
    view_url = f"{base_url}/en/view-image.php?image={image_id}&picture={image_name}"
    response = session.get(view_url)

    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch page: {view_url}, status: {response.status_code}"
        )

    # Parse HTML and find image
    soup = BeautifulSoup(response.text, "html.parser")
    image_url = _find_publicdomain_image(soup, base_url)

    if not image_url:
        raise ValueError(f"Could not find image on page: {view_url}")

    # Download using session
    img_response = session.get(image_url, stream=True)
    img_response.raise_for_status()
    save_response(img_response, dest_path)
    return DownloadResult(dest_filename, DownloadStatus.DOWNLOADED)


def _find_publicdomain_image(soup: BeautifulSoup, base_url: str) -> str | None:
    """Find the main image URL from a publicdomainpictures.net page."""
    # Try main_image div first
    main_div = soup.find("div", {"id": "main_image"})
    if main_div:
        img = main_div.find("img")
        if img:
            src = img.get("src")
            if isinstance(src, str):
                return make_absolute_url(base_url, src)

    # Fallback: mainImage id or main-image class
    img = soup.find("img", {"id": "mainImage"}) or soup.find(
        "img", {"class": "main-image"}
    )
    if img:
        src = img.get("src")
        if isinstance(src, str):
            return make_absolute_url(base_url, src)

    # Fallback: any img with /pictures/ or velka in src
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if isinstance(src, str) and ("/pictures/" in src or "velka" in src):
            return make_absolute_url(base_url, src)

    return None


def download_image(
    url: str,
    dest_dir: Path,
    serial_id: int,
    filename: str | None,
    expected_license: str,
) -> DownloadResult:
    """Download an image from a supported source."""
    if "commons.wikimedia.org" in url:
        return download_from_wikimedia(
            url, dest_dir, serial_id, filename, expected_license
        )
    elif "publicdomainpictures.net" in url:
        return download_from_publicdomain(
            url, dest_dir, serial_id, filename, expected_license
        )
    else:
        logger.warning(f"Skipping unsupported URL: {url}")
        return DownloadResult(None, DownloadStatus.FAILED)


def download_from_csv(csv_file: Path, dest_dir: Path) -> None:
    """Download all images listed in a CSV file and update filename column."""
    if not csv_file.exists():
        logger.error(f"{csv_file} not found")
        sys.exit(1)

    dest_dir.mkdir(parents=True, exist_ok=True)
    df = pl.read_csv(csv_file)

    if df.is_empty():
        logger.error("No entries found in CSV file")
        sys.exit(1)

    # Configure loguru to work with tqdm
    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

    # Track filenames for updating CSV
    filenames: list[str | None] = (
        df["filename"].to_list() if "filename" in df.columns else [None] * len(df)
    )

    # Track stats
    downloaded = 0
    skipped = 0
    failed = 0

    progress_bar = tqdm(df.iter_rows(named=True), total=len(df))

    for i, row in enumerate(progress_bar, 1):
        url = str(row.get("url", "")).strip()
        if not url:
            continue

        license_str = str(row.get("license", "")).strip()
        filename_raw = row.get("filename")
        filename = (
            str(filename_raw).strip()
            if filename_raw and str(filename_raw).strip()
            else None
        )

        # Update progress bar description with current filename
        url_path = urlparse(url).path
        current_file = filename or url_path.split("/")[-1] or f"image_{i}"
        progress_bar.set_description(f"{current_file[:30]}")

        try:
            result = download_image(url, dest_dir, i, filename, license_str)
            if result.filename:
                filenames[i - 1] = result.filename
            if result.status == DownloadStatus.DOWNLOADED:
                downloaded += 1
            elif result.status == DownloadStatus.SKIPPED:
                skipped += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            failed += 1

    # Update CSV with filenames
    df = df.with_columns(pl.Series("filename", filenames))
    df.write_csv(csv_file)

    # Print summary
    print(f"\nDownloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")

    # Update CSV with filenames
    df = df.with_columns(pl.Series("filename", filenames))
    df.write_csv(csv_file)


def main() -> None:
    """Main entry point."""
    dest_dir = Path.cwd() / "datasets" / "4471-cats-wiki-images" / "images"

    if len(sys.argv) == 1:
        csv_file = Path(__file__).parent / "wiki_sources.csv"
    elif len(sys.argv) == 2:
        csv_file = Path(sys.argv[1])
    else:
        print("Usage:")
        print(
            "  python scripts/03_build_wiki_dataset.py                     # Read from wiki_sources.csv"
        )
        print(
            "  python scripts/03_build_wiki_dataset.py path/to/file.csv    # Read from specified CSV"
        )
        sys.exit(1)

    download_from_csv(csv_file, dest_dir)


if __name__ == "__main__":
    main()
