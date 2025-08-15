#!/usr/bin/env python3

import argparse
import hashlib
import logging
import os
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from tqdm import tqdm


# Configuration
DEFAULT_DATASETS_DIR = "training/data/datasets"  # Relative to project root
CHUNK_SIZE = 8192  # 8KB chunks for downloading
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # Initial retry delay in seconds

# Dataset metadata
DATASETS: Dict[str, Dict[str, str]] = {
    "test-clean": {
        "filename": "test-clean.tar.gz",
    },
    "dev-clean": {
        "filename": "dev-clean.tar.gz",
    },
    "train-clean-100": {
        "filename": "train-clean-100.tar.gz",
    },
    "train-clean-360": {
        "filename": "train-clean-360.tar.gz",
    },
    "train-other-500": {
        "filename": "train-other-500.tar.gz",
    },
}

OPENSLR_BASE_URL = "https://www.openslr.org/resources/12/"
MD5SUM_URL = "https://www.openslr.org/resources/12/md5sum.txt"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def resolve_output_dir(custom_dir: Optional[str]) -> Path:
    """Resolve the output directory path."""
    if custom_dir:
        return Path(custom_dir).expanduser().resolve()
    # Project root is two levels up from this script: training/scripts/
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / DEFAULT_DATASETS_DIR).resolve()


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        sys.exit(1)


def format_size(num_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f}{unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f}PB"


def check_disk_space(path: Path, required_bytes: int) -> bool:
    """Check if there's enough disk space for the download."""
    try:
        stat = shutil.disk_usage(path)
        available = stat.free
        if available < required_bytes * 1.1:  # 10% buffer
            logger.error(
                f"Insufficient disk space. Required: {format_size(required_bytes)}, "
                f"Available: {format_size(available)}"
            )
            return False
        return True
    except OSError as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Proceed anyway


def download_md5_checksums(downloads_dir: Path) -> Dict[str, str]:
    """Download and parse MD5 checksums from OpenSLR."""
    md5_file = downloads_dir / "md5sum.txt"
    checksums = {}
    
    try:
        logger.info("Downloading MD5 checksums from OpenSLR...")
        response = requests.get(MD5SUM_URL, timeout=30)
        response.raise_for_status()
        
        # Save the checksums file
        with open(md5_file, 'w') as f:
            f.write(response.text)
        
        # Parse checksums
        for line in response.text.strip().split('\n'):
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 2:
                    md5_hash = parts[0]
                    filename = parts[1]
                    checksums[filename] = md5_hash
        
        logger.info(f"Downloaded {len(checksums)} checksums from OpenSLR")
        return checksums
        
    except (requests.RequestException, OSError) as e:
        logger.warning(f"Could not download MD5 checksums: {e}")
        logger.warning("Checksum verification will be skipped")
        return {}


def calculate_md5(file_path: Path, chunk_size: int = CHUNK_SIZE) -> str:
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            with tqdm(
                total=file_path.stat().st_size,
                unit='B',
                unit_scale=True,
                desc="Verifying checksum"
            ) as pbar:
                for chunk in iter(lambda: f.read(chunk_size), b""):
                    md5_hash.update(chunk)
                    pbar.update(len(chunk))
        return md5_hash.hexdigest()
    except OSError as e:
        logger.error(f"Failed to calculate checksum for {file_path}: {e}")
        return ""


def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum against expected value."""
    if not expected_md5:
        logger.warning("No checksum available for verification")
        return True
    
    actual_md5 = calculate_md5(file_path)
    if not actual_md5:
        return False
        
    if actual_md5.lower() == expected_md5.lower():
        logger.info("✓ Checksum verification passed")
        return True
    else:
        logger.error(f"✗ Checksum mismatch! Expected: {expected_md5}, Got: {actual_md5}")
        return False


def get_remote_file_size(url: str) -> Optional[int]:
    """Get the size of a remote file without downloading it."""
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        response.raise_for_status()
        content_length = response.headers.get('content-length')
        if content_length:
            return int(content_length)
    except (requests.RequestException, ValueError) as e:
        logger.debug(f"Could not determine file size: {e}")
    return None


def download_with_resume(url: str, destination: Path, expected_size: Optional[int] = None) -> bool:
    """Download file with resume capability and progress bar."""
    headers = {}
    initial_pos = 0
    
    # Check if partial file exists
    if destination.exists():
        initial_pos = destination.stat().st_size
        headers['Range'] = f'bytes={initial_pos}-'
        logger.info(f"Resuming download from byte {initial_pos}")
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, stream=True, timeout=30)
            
            # Handle range request response
            if response.status_code == 206:  # Partial content
                total_size = int(response.headers.get('content-range', '').split('/')[-1])
                mode = 'ab'
            elif response.status_code == 200:  # Full content
                total_size = int(response.headers.get('content-length', 0))
                mode = 'wb'
                initial_pos = 0  # Reset if server doesn't support ranges
            else:
                response.raise_for_status()
            
            # Note: We don't validate expected size as it may change over time
            
            # Check disk space
            if not check_disk_space(destination.parent, total_size - initial_pos):
                return False
            
            # Download with progress bar
            with open(destination, mode) as f:
                with tqdm(
                    total=total_size,
                    initial=initial_pos,
                    unit='B',
                    unit_scale=True,
                    desc=f"Downloading {destination.name}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Download completed: {destination}")
            return True
            
        except (requests.RequestException, OSError) as e:
            logger.warning(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                logger.error(f"Download failed after {MAX_RETRIES} attempts")
                return False
    
    return False


def safe_extract_tar_gz(archive_path: Path, target_dir: Path) -> bool:
    """Safely extract tar.gz file with path traversal protection."""
    def is_safe_path(base_path: str, path: str) -> bool:
        """Check if the path is safe (no path traversal)."""
        abs_base = os.path.abspath(base_path)
        abs_path = os.path.abspath(os.path.join(base_path, path))
        return os.path.commonpath([abs_base, abs_path]) == abs_base
    
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            
            # Validate all paths before extraction
            for member in members:
                if not is_safe_path(str(target_dir), member.name):
                    logger.error(f"Unsafe path detected in archive: {member.name}")
                    return False
                
                # Additional security checks
                if member.name.startswith('/') or '..' in member.name:
                    logger.error(f"Potentially dangerous path in archive: {member.name}")
                    return False
            
            # Extract with progress bar
            with tqdm(total=len(members), desc="Extracting", unit="files") as pbar:
                for member in members:
                    tar.extract(member, path=target_dir)
                    pbar.update(1)
            
            logger.info(f"Extraction completed to {target_dir}")
            return True
            
    except (tarfile.TarError, OSError) as e:
        logger.error(f"Failed to extract {archive_path}: {e}")
        return False


def parse_size_string(size_str: str) -> int:
    """Parse size string (e.g., '6.3G') to bytes."""
    size_str = size_str.upper().strip()
    multipliers = {'B': 1, 'K': 1024, 'M': 1024**2, 'G': 1024**3, 'T': 1024**4}
    
    for suffix, multiplier in multipliers.items():
        if size_str.endswith(suffix):
            try:
                number = float(size_str[:-1])
                return int(number * multiplier)
            except ValueError:
                break
    
    logger.warning(f"Could not parse size string: {size_str}")
    return 0


def select_datasets_interactively() -> List[str]:
    """Interactive dataset selection."""
    print("\nSelect one or more LibriSpeech subsets to download:")
    options = list(DATASETS.keys())
    
    # Get sizes for display
    print("Checking file sizes...")
    for idx, key in enumerate(options, start=1):
        filename = DATASETS[key]["filename"]
        url = OPENSLR_BASE_URL + filename
        size = get_remote_file_size(url)
        size_str = f" [{format_size(size)}]" if size else " [Size unknown]"
        print(f"  {idx}) {key}{size_str}")
    print("  a) all")
    
    while True:
        choice = input("\nEnter selection (e.g., 1,3 or 'a'): ").strip().lower()
        
        if choice in {"a", "all"}:
            return options
        
        selected: List[str] = []
        valid_selection = True
        
        for token in choice.split(","):
            token = token.strip()
            if not token:
                continue
            
            if not token.isdigit():
                print(f"  Invalid entry: {token}")
                valid_selection = False
                break
            
            idx = int(token)
            if 1 <= idx <= len(options):
                dataset = options[idx - 1]
                if dataset not in selected:
                    selected.append(dataset)
            else:
                print(f"  Index out of range: {idx}")
                valid_selection = False
                break
        
        if valid_selection and selected:
            return selected
        elif valid_selection and not selected:
            print("  No datasets selected.")
        
        print("  Please try again.")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description=f"Download and prepare LibriSpeech training samples into {DEFAULT_DATASETS_DIR}/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --datasets test-clean
  %(prog)s --all --keep-archives
  %(prog)s --datasets train-clean-100 train-clean-360 --output-dir /data/speech
        """
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS.keys()),
        help="One or more subsets to download (multi-select)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available subsets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Directory to place downloads/extracted data (default: <repo>/{DEFAULT_DATASETS_DIR})",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep .tar.gz files after extraction (default: delete)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction; only download the .tar.gz archives",
    )
    parser.add_argument(
        "--skip-checksum",
        action="store_true",
        help="Skip checksum verification (not recommended)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Resolve output directory
    try:
        output_dir = resolve_output_dir(args.output_dir)
        ensure_directory(output_dir)
        downloads_dir = output_dir / "_downloads"
        ensure_directory(downloads_dir)
    except Exception as e:
        logger.error(f"Failed to setup directories: {e}")
        sys.exit(1)

    # Select datasets
    if args.all:
        selected = list(DATASETS.keys())
    elif args.datasets:
        selected = args.datasets
    else:
        selected = select_datasets_interactively()

    if not selected:
        logger.info("No datasets selected. Exiting.")
        sys.exit(0)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Selected datasets: {', '.join(selected)}")

    # Calculate estimated total download size
    logger.info("Checking file sizes...")
    total_estimated_size = 0
    for key in selected:
        filename = DATASETS[key]["filename"]
        url = OPENSLR_BASE_URL + filename
        size = get_remote_file_size(url)
        if size:
            total_estimated_size += size
    
    if total_estimated_size > 0:
        logger.info(f"Total estimated download size: {format_size(total_estimated_size)}")
    else:
        logger.info("Total download size: Unknown")

    # Download MD5 checksums if not skipping verification
    checksums = {}
    if not args.skip_checksum:
        checksums = download_md5_checksums(downloads_dir)
        if not checksums:
            logger.warning("⚠️  Proceeding without checksum verification")

    # Process each dataset
    success_count = 0
    for key in selected:
        meta = DATASETS[key]
        filename = meta["filename"]
        url = OPENSLR_BASE_URL + filename
        dest_archive = downloads_dir / filename
        expected_md5 = checksums.get(filename, "")
        # Size will be determined from HTTP headers during download

        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {key}")
        logger.info(f"URL: {url}")
        logger.info(f"Archive: {dest_archive}")

        # Get estimated file size
        estimated_size = get_remote_file_size(url)
        if estimated_size:
            logger.info(f"Estimated size: {format_size(estimated_size)}")
        else:
            logger.info("Size: Unknown")

        # Check if file already exists and is valid
        if dest_archive.exists():
            logger.info("Archive already exists.")
            if not args.skip_checksum and expected_md5:
                if verify_checksum(dest_archive, expected_md5):
                    logger.info("Existing archive is valid.")
                else:
                    logger.warning("Existing archive failed checksum. Re-downloading...")
                    dest_archive.unlink()
            else:
                if expected_md5:
                    logger.info("Skipping checksum verification for existing archive.")
                else:
                    logger.info("No checksum available for verification.")

        # Download if needed
        if not dest_archive.exists():
            logger.info("Starting download...")
            if not download_with_resume(url, dest_archive):
                logger.error(f"Failed to download {key}. Skipping.")
                continue

            # Verify checksum of downloaded file
            if not args.skip_checksum and expected_md5:
                if not verify_checksum(dest_archive, expected_md5):
                    logger.error(f"Downloaded file failed checksum verification. Removing.")
                    dest_archive.unlink()
                    continue

        # Extract if requested
        if not args.no_extract:
            logger.info("Extracting archive...")
            if not safe_extract_tar_gz(dest_archive, output_dir):
                logger.error(f"Failed to extract {key}.")
                continue

            # Remove archive if requested
            if not args.keep_archives:
                try:
                    dest_archive.unlink()
                    logger.info("Archive removed.")
                except OSError as e:
                    logger.warning(f"Could not remove archive: {e}")

        success_count += 1
        logger.info(f"Successfully processed {key}")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing complete: {success_count}/{len(selected)} datasets successful")
    
    if success_count == len(selected):
        logger.info("All datasets processed successfully!")
    elif success_count == 0:
        logger.error("No datasets were processed successfully.")
        sys.exit(1)
    else:
        logger.warning(f"{len(selected) - success_count} datasets failed to process.")


if __name__ == "__main__":
    main()