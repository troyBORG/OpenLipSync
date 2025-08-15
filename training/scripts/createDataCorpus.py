#!/usr/bin/env python3

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import tqdm


# Configuration
DEFAULT_LIBRISPEECH_DIR = "training/data/raw/LibriSpeech"  # Relative to project root
DEFAULT_PREPARED_DIR = "training/data/prepared"  # Relative to project root
SUPPORTED_SUBSETS = {"test-clean", "train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-other"}

# Audio conversion settings
TARGET_SAMPLE_RATE = 16000  # Hz - MFA standard
TARGET_CHANNELS = 1  # Mono - MFA standard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def resolve_path(custom_path: Optional[str], default_path: str) -> Path:
    """Resolve a path, using custom if provided, otherwise default relative to project root."""
    if custom_path:
        return Path(custom_path).expanduser().resolve()
    # Project root is two levels up from this script: training/scripts/
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / default_path).resolve()


def ensure_directory(path: Path) -> None:
    """Create directory if it doesn't exist."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {path}: {e}")
        sys.exit(1)


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_flac_to_wav(flac_path: Path, wav_path: Path) -> bool:
    """Convert FLAC file to WAV format using ffmpeg."""
    try:
        # Ensure output directory exists
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert with specific settings for MFA
        subprocess.run([
            'ffmpeg', '-i', str(flac_path),
            '-ar', str(TARGET_SAMPLE_RATE),  # Sample rate
            '-ac', str(TARGET_CHANNELS),     # Mono
            '-y',                           # Overwrite existing files
            str(wav_path)
        ], capture_output=True, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert {flac_path} to {wav_path}: {e}")
        return False


def parse_transcript_file(trans_path: Path) -> Dict[str, str]:
    """Parse LibriSpeech transcript file and return utterance ID to text mapping."""
    transcripts = {}
    try:
        with open(trans_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # LibriSpeech format: "utterance_id TRANSCRIPT TEXT"
                parts = line.split(' ', 1)
                if len(parts) >= 2:
                    utterance_id = parts[0]
                    transcript = parts[1]
                    transcripts[utterance_id] = transcript
                else:
                    logger.warning(f"Malformed transcript line in {trans_path}: {line}")
    
    except OSError as e:
        logger.error(f"Failed to read transcript file {trans_path}: {e}")
    
    return transcripts


def create_lab_file(text: str, lab_path: Path) -> bool:
    """Create a .lab file with the transcript text."""
    try:
        lab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lab_path, 'w', encoding='utf-8') as f:
            f.write(text.strip())
        return True
    except OSError as e:
        logger.error(f"Failed to create lab file {lab_path}: {e}")
        return False


def get_utterance_id_from_filename(filename: str) -> str:
    """Extract utterance ID from FLAC filename."""
    # Remove .flac extension
    return filename.replace('.flac', '')


def find_librispeech_subsets(librispeech_dir: Path) -> List[str]:
    """Find available LibriSpeech subsets in the directory."""
    available_subsets = []
    
    for subset in SUPPORTED_SUBSETS:
        subset_path = librispeech_dir / subset
        if subset_path.exists() and subset_path.is_dir():
            available_subsets.append(subset)
    
    return available_subsets


def count_files_in_subset(librispeech_dir: Path, subset: str) -> int:
    """Count FLAC files in a subset for progress tracking."""
    subset_path = librispeech_dir / subset
    count = 0
    
    for speaker_dir in subset_path.iterdir():
        if not speaker_dir.is_dir():
            continue
        for chapter_dir in speaker_dir.iterdir():
            if not chapter_dir.is_dir():
                continue
            for file_path in chapter_dir.iterdir():
                if file_path.suffix == '.flac':
                    count += 1
    
    return count


def process_subset(librispeech_dir: Path, prepared_dir: Path, subset: str) -> tuple[int, int]:
    """
    Process a LibriSpeech subset and convert to prepared flat format.
    
    Args:
        librispeech_dir: Path to raw LibriSpeech data
        prepared_dir: Path to prepared datasets directory 
        subset: Name of the subset to process
    
    Returns:
        tuple: (successful_files, total_files)
    """
    subset_path = librispeech_dir / subset
    if not subset_path.exists():
        logger.error(f"Subset directory not found: {subset_path}")
        return 0, 0
    
    logger.info(f"Processing subset: {subset}")
    
    # Create output directory for this dataset
    output_dir = prepared_dir / subset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Count files for progress bar
    total_files = count_files_in_subset(librispeech_dir, subset)
    if total_files == 0:
        logger.warning(f"No FLAC files found in subset {subset}")
        return 0, 0
    
    successful_files = 0
    
    with tqdm.tqdm(total=total_files, desc=f"Converting {subset}", unit="files") as pbar:
        # Walk through speaker directories
        for speaker_dir in subset_path.iterdir():
            if not speaker_dir.is_dir():
                continue
            
            speaker_id = speaker_dir.name
            
            # Walk through chapter directories
            for chapter_dir in speaker_dir.iterdir():
                if not chapter_dir.is_dir():
                    continue
                
                chapter_id = chapter_dir.name
                
                # Find transcript file
                trans_file = None
                for file_path in chapter_dir.iterdir():
                    if file_path.suffix == '.txt' and file_path.stem.endswith('.trans'):
                        trans_file = file_path
                        break
                
                if not trans_file:
                    logger.warning(f"No transcript file found in {chapter_dir}")
                    continue
                
                # Parse transcripts
                transcripts = parse_transcript_file(trans_file)
                if not transcripts:
                    logger.warning(f"No transcripts found in {trans_file}")
                    continue
                
                # Process FLAC files
                for file_path in chapter_dir.iterdir():
                    if file_path.suffix != '.flac':
                        continue
                    
                    utterance_id = get_utterance_id_from_filename(file_path.name)
                    
                    if utterance_id not in transcripts:
                        logger.warning(f"No transcript found for {utterance_id}")
                        pbar.update(1)
                        continue
                    
                    # Create flat file names (utterance_id is already unique)
                    wav_path = output_dir / f"{utterance_id}.wav"
                    lab_path = output_dir / f"{utterance_id}.lab"
                    
                    # Convert audio
                    if convert_flac_to_wav(file_path, wav_path):
                        # Create transcript file
                        if create_lab_file(transcripts[utterance_id], lab_path):
                            successful_files += 1
                        else:
                            # Remove WAV file if lab creation failed
                            if wav_path.exists():
                                wav_path.unlink()
                    
                    pbar.update(1)
    
    return successful_files, total_files


def select_subsets_interactively(available_subsets: List[str]) -> List[str]:
    """Interactive subset selection."""
    if not available_subsets:
        logger.error("No LibriSpeech subsets found in the directory")
        return []
    
    print("\nAvailable LibriSpeech subsets:")
    for idx, subset in enumerate(available_subsets, start=1):
        print(f"  {idx}) {subset}")
    print("  a) all")
    
    while True:
        choice = input("\nEnter selection (e.g., 1,3 or 'a'): ").strip().lower()
        
        if choice in {"a", "all"}:
            return available_subsets
        
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
            if 1 <= idx <= len(available_subsets):
                subset = available_subsets[idx - 1]
                if subset not in selected:
                    selected.append(subset)
            else:
                print(f"  Index out of range: {idx}")
                valid_selection = False
                break
        
        if valid_selection and selected:
            return selected
        elif valid_selection and not selected:
            print("  No subsets selected.")
        
        print("  Please try again.")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Convert LibriSpeech dataset to prepared format with flat structure for training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --subsets test-clean
  %(prog)s --all
  %(prog)s --subsets train-clean-100 --librispeech-dir /data/LibriSpeech --prepared-dir /data/prepared
        """
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        help="LibriSpeech subsets to process (interactive selection if not specified)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available subsets",
    )
    parser.add_argument(
        "--librispeech-dir",
        type=str,
        default=None,
        help=f"LibriSpeech dataset directory (default: <repo>/{DEFAULT_LIBRISPEECH_DIR})",
    )
    parser.add_argument(
        "--prepared-dir",
        type=str,
        default=None,
        help=f"Output prepared dataset directory (default: <repo>/{DEFAULT_PREPARED_DIR})",
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

    # Check dependencies
    if not check_ffmpeg():
        logger.error("ffmpeg is required but not found. Please install ffmpeg.")
        logger.error("On Ubuntu/Debian: sudo apt install ffmpeg")
        logger.error("On macOS: brew install ffmpeg")
        sys.exit(1)

    # Resolve paths
    try:
        librispeech_dir = resolve_path(args.librispeech_dir, DEFAULT_LIBRISPEECH_DIR)
        prepared_dir = resolve_path(args.prepared_dir, DEFAULT_PREPARED_DIR)
        
        if not librispeech_dir.exists():
            logger.error(f"LibriSpeech directory not found: {librispeech_dir}")
            logger.error("Please run the download script first or specify --librispeech-dir")
            sys.exit(1)
        
        ensure_directory(prepared_dir)
    except Exception as e:
        logger.error(f"Failed to setup directories: {e}")
        sys.exit(1)

    # Find available subsets
    available_subsets = find_librispeech_subsets(librispeech_dir)
    if not available_subsets:
        logger.error(f"No LibriSpeech subsets found in {librispeech_dir}")
        logger.error("Please run the download script first.")
        sys.exit(1)

    # Select subsets to process
    if args.all:
        selected_subsets = available_subsets
    elif args.subsets:
        selected_subsets = []
        for subset in args.subsets:
            if subset in available_subsets:
                selected_subsets.append(subset)
            else:
                logger.warning(f"Subset '{subset}' not found in {librispeech_dir}")
        
        if not selected_subsets:
            logger.error("No valid subsets selected.")
            sys.exit(1)
    else:
        selected_subsets = select_subsets_interactively(available_subsets)

    if not selected_subsets:
        logger.info("No subsets selected. Exiting.")
        sys.exit(0)

    logger.info(f"LibriSpeech directory: {librispeech_dir}")
    logger.info(f"Prepared dataset directory: {prepared_dir}")
    logger.info(f"Selected subsets: {', '.join(selected_subsets)}")
    logger.info(f"Organization: Flat structure per dataset")

    # Process each subset
    total_successful = 0
    total_files = 0
    
    for subset in selected_subsets:
        successful, total = process_subset(
            librispeech_dir, 
            prepared_dir, 
            subset
        )
        total_successful += successful
        total_files += total
        
        if successful == total:
            logger.info(f"Successfully processed {subset}: {successful}/{total} files")
        else:
            logger.warning(f"⚠ Partially processed {subset}: {successful}/{total} files")

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info(f"Conversion complete: {total_successful}/{total_files} files successful")
    
    if total_successful == total_files:
        logger.info("All files converted successfully.")
        logger.info(f"Prepared datasets ready at: {prepared_dir}")
        logger.info("\nNext steps:")
        logger.info("1. Run MFA alignment on the prepared datasets")
        logger.info("2. Start training with: python training/train.py --config training/recipes/tcn_config.toml")
    elif total_successful == 0:
        logger.error("No files were converted successfully.")
        sys.exit(1)
    else:
        logger.warning(f"⚠ {total_files - total_successful} files failed to convert.")
        logger.info(f"Prepared datasets partially ready at: {prepared_dir}")


if __name__ == "__main__":
    main()
