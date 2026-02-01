"""Download Community Notes public data from Twitter/X."""

import gzip
import shutil
import tempfile
from pathlib import Path

import requests

# Directory where this script lives (and where data will be downloaded)
SCRIPT_DIR = Path(__file__).parent

BASE_URL = "https://ton.twimg.com/birdwatch-public-data"


def download_file(url: str, destination: Path) -> bool:
    """Download a file from a URL to the specified destination."""
    response = requests.get(url, stream=True)
    if response.status_code != 200:
        print(f"Failed to download {url} (status {response.status_code})")
        return False

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def is_gzipped(file_path: Path) -> bool:
    """Check if a file is gzip-compressed by reading its magic bytes."""
    with open(file_path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def decompress_gzip_inplace(file_path: Path) -> None:
    """Decompress a gzip file in-place (file keeps the same name)."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with gzip.open(file_path, "rb") as gz_file:
            shutil.copyfileobj(gz_file, tmp)
    # Replace original with decompressed content
    shutil.move(tmp_path, file_path)


def build_ratings_urls(date: str, num_files: int) -> list[str]:
    """Build a list of ratings file URLs for the given date."""
    return [
        f"{BASE_URL}/{date}/noteRatings/ratings-{i:05d}.tsv"
        for i in range(num_files)
    ]


def download_data(date: str, num_ratings_files: int = 20) -> None:
    """
    Download Community Notes data for a given date.

    Args:
        date: Date string in 'YYYY/MM/DD' format.
        num_ratings_files: Number of ratings files to download (default 20).
    """
    output_dir = SCRIPT_DIR / "data"
    output_dir.mkdir(exist_ok=True)

    # Regular TSV files
    tsv_urls = [
        f"{BASE_URL}/{date}/notes/notes-00000.tsv",
        f"{BASE_URL}/{date}/noteStatusHistory/noteStatusHistory-00000.tsv",
        f"{BASE_URL}/{date}/userEnrollment/userEnrollment-00000.tsv",
    ]

    print(f"Downloading data for {date} to {output_dir}")

    # Download regular TSV files
    for url in tsv_urls:
        filename = url.split("/")[-1]
        destination = output_dir / filename
        print(f"  Downloading {filename}...")
        download_file(url, destination)

    # Create ratings subdirectory
    ratings_dir = output_dir / "ratings"
    ratings_dir.mkdir(exist_ok=True)

    # Download and process ratings files (potentially gzipped)
    ratings_urls = build_ratings_urls(date, num_ratings_files)
    for url in ratings_urls:
        filename = url.split("/")[-1]
        destination = ratings_dir / filename
        print(f"  Downloading {filename}...")

        if download_file(url, destination):
            if is_gzipped(destination):
                decompress_gzip_inplace(destination)
                print(f"    Decompressed {filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Community Notes public data")
    parser.add_argument("date", help="Date in YYYY/MM/DD format (e.g., 2024/03/07)")
    parser.add_argument(
        "--num-ratings-files",
        type=int,
        default=20,
        help="Number of ratings files to download (default: 20)",
    )

    args = parser.parse_args()
    download_data(args.date, args.num_ratings_files)