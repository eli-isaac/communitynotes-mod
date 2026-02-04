"""Download Community Notes public data from Twitter/X."""

import gzip
import shutil
import tempfile
import zipfile
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


def extract_zip_to_dir(zip_path: Path, dest_dir: Path) -> None:
    """Extract a zip file into dest_dir, then remove the zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    zip_path.unlink()


def build_ratings_urls(date: str, num_files: int) -> list[str]:
    """Build a list of ratings file URLs for the given date."""
    return [
        f"{BASE_URL}/{date}/noteRatings/ratings-{i:05d}.zip" for i in range(num_files)
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

    # Main data files (served as .zip; we extract to get .tsv)
    zip_urls = [
        f"{BASE_URL}/{date}/notes/notes-00000.zip",
        f"{BASE_URL}/{date}/noteStatusHistory/noteStatusHistory-00000.zip",
        f"{BASE_URL}/{date}/userEnrollment/userEnrollment-00000.zip",
    ]

    print(f"Downloading data for {date} to {output_dir}")

    for url in zip_urls:
        filename = url.split("/")[-1]
        destination = output_dir / filename
        print(f"  Downloading {filename}...")
        if download_file(url, destination):
            extract_zip_to_dir(destination, output_dir)
            print(f"    Extracted {filename}")

    # Create ratings subdirectory
    ratings_dir = output_dir / "ratings"
    ratings_dir.mkdir(exist_ok=True)

    # Download and process ratings files (.zip or legacy .tsv / .tsv.gz)
    ratings_urls = build_ratings_urls(date, num_ratings_files)
    for url in ratings_urls:
        filename = url.split("/")[-1]
        destination = ratings_dir / filename
        print(f"  Downloading {filename}...")

        if download_file(url, destination):
            if destination.suffix == ".zip":
                extract_zip_to_dir(destination, ratings_dir)
                print(f"    Extracted {filename}")
            elif is_gzipped(destination):
                decompress_gzip_inplace(destination)
                print(f"    Decompressed {filename}")


if __name__ == "__main__":
    import argparse
    from datetime import date

    parser = argparse.ArgumentParser(description="Download Community Notes public data")
    parser.add_argument(
        "date",
        nargs="?",
        default=date.today().strftime("%Y/%m/%d"),
        help="Date in YYYY/MM/DD format (default: today)",
    )
    parser.add_argument(
        "--num-ratings-files",
        type=int,
        default=20,
        help="Number of ratings files to download (default: 20)",
    )

    args = parser.parse_args()
    download_data(args.date, args.num_ratings_files)
