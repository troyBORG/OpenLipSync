#!/usr/bin/env python3

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages LibriSpeech dataset download, preparation, and organization.
    Provides automatic dataset preparation with user prompts for missing data.
    """
    
    # Supported LibriSpeech subsets
    SUPPORTED_SUBSETS = {
        "test-clean", "dev-clean", "train-clean-100", 
        "train-clean-360", "train-other-500", "dev-other", "test-other"
    }
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize DatasetManager
        
        Args:
            project_root: Root directory of the project. If None, auto-detected.
        """
        if project_root is None:
            # Auto-detect project root (assuming we're in training/modules/)
            self.project_root = Path(__file__).resolve().parents[2]
        else:
            self.project_root = Path(project_root).resolve()
            
        # Define directory structure
        self.data_dir = self.project_root / "training" / "data"
        self.prepared_dir = self.data_dir / "prepared"
        self.raw_dir = self.data_dir / "raw"
        self.cache_dir = self.data_dir / "cache"
        
        # Script paths
        self.scripts_dir = self.project_root / "training" / "scripts"
        self.download_script = self.scripts_dir / "downloadTrainingData.py"
        self.corpus_script = self.scripts_dir / "createDataCorpus.py"
        
        # Ensure directories exist
        self.prepared_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def check_dataset_availability(self, datasets: List[str]) -> Dict[str, Dict[str, bool]]:
        """
        Check availability of datasets at different preparation stages
        
        Args:
            datasets: List of dataset names to check
            
        Returns:
            Dict mapping dataset -> {raw: bool, corpus: bool, aligned: bool, prepared: bool}
        """
        availability = {}
        
        for dataset in datasets:
            if dataset not in self.SUPPORTED_SUBSETS:
                logger.warning(f"Unsupported dataset: {dataset}")
                continue
                
            status = {
                "raw": self._check_raw_data(dataset),
                "corpus": self._check_corpus_data(dataset), 
                "aligned": self._check_aligned_data(dataset),
                "prepared": self._check_prepared_data(dataset)
            }
            availability[dataset] = status
            
        return availability
    
    def _check_raw_data(self, dataset: str) -> bool:
        """Check if raw LibriSpeech data exists"""
        raw_dataset_dir = self.raw_dir / "LibriSpeech" / dataset
        return raw_dataset_dir.exists() and any(raw_dataset_dir.rglob("*.flac"))
    
    def _check_corpus_data(self, dataset: str) -> bool:
        """Check if MFA corpus data exists (WAV + LAB files)"""
        # Check old structure (organized by speaker)
        corpus_dirs = list((self.data_dir / "data_corpus").glob(f"{dataset}_*"))
        return len(corpus_dirs) > 0 and any(
            list(d.glob("*.wav")) and list(d.glob("*.lab"))
            for d in corpus_dirs if d.is_dir()
        )
    
    def _check_aligned_data(self, dataset: str) -> bool:
        """Check if MFA alignment data exists in cache (JSON files)."""
        cache_dir = self.data_dir / "cache" / f"out_align_{dataset}"
        if cache_dir.is_dir() and any(cache_dir.rglob("*.json")):
            return True
        return False
    
    def _check_prepared_data(self, dataset: str) -> bool:
        """Check if prepared data exists (WAV + LAB in flat structure).

        JSON alignment files are optional and may be added later.
        """
        prepared_dataset_dir = self.prepared_dir / dataset
        if not prepared_dataset_dir.exists():
            return False
            
        # Require matching WAV and LAB pairs; JSONs are not required
        wav_files = set(f.stem for f in prepared_dataset_dir.glob("*.wav"))
        lab_files = set(f.stem for f in prepared_dataset_dir.glob("*.lab"))
        return len(wav_files) > 0 and wav_files == lab_files
    
    def prepare_datasets(self, datasets: List[str], interactive: bool = True) -> bool:
        """
        Ensure all requested datasets are prepared and ready for training
        
        Args:
            datasets: List of dataset names to prepare
            interactive: Whether to prompt user for missing datasets
            
        Returns:
            True if all datasets are ready, False if preparation failed
        """
        logger.info(f"Checking dataset availability: {', '.join(datasets)}")
        availability = self.check_dataset_availability(datasets)
        
        missing_datasets = []
        needs_preparation = []
        
        for dataset in datasets:
            if dataset not in availability:
                continue
                
            status = availability[dataset]
            
            if status["prepared"]:
                logger.info(f"Dataset {dataset} is ready")
            elif status["aligned"]:
                logger.info(f"⚠ Dataset {dataset} needs final preparation (has alignment)")
                needs_preparation.append(dataset)
            elif status["corpus"]:
                logger.info(f"⚠ Dataset {dataset} needs alignment and preparation (has corpus)")
                needs_preparation.append(dataset)
            elif status["raw"]:
                logger.info(f"⚠ Dataset {dataset} needs corpus creation and alignment (has raw data)")
                needs_preparation.append(dataset)
            else:
                logger.info(f"Dataset {dataset} is missing completely")
                missing_datasets.append(dataset)
        
        # Handle completely missing datasets
        if missing_datasets:
            if interactive:
                response = self._prompt_download(missing_datasets)
                if response:
                    if not self._download_datasets(missing_datasets):
                        logger.error("Failed to download datasets")
                        return False
                    needs_preparation.extend(missing_datasets)
                else:
                    logger.error("Cannot proceed without required datasets")
                    return False
            else:
                logger.error(f"Missing datasets: {', '.join(missing_datasets)}")
                return False
        
        # Handle datasets that need preparation
        if needs_preparation:
            logger.info(f"Preparing datasets: {', '.join(needs_preparation)}")
            for dataset in needs_preparation:
                if not self._prepare_single_dataset(dataset):
                    logger.error(f"Failed to prepare dataset: {dataset}")
                    return False
        
        logger.info("All datasets are ready for training.")
        return True
    
    def _prompt_download(self, datasets: List[str]) -> bool:
        """Prompt user whether to download missing datasets"""
        print(f"\nMissing datasets: {', '.join(datasets)}")
        print("\nThese datasets need to be downloaded and prepared:")
        for dataset in datasets:
            print(f"  - {dataset}")
        
        print("\nThis will:")
        print("  1. Download raw LibriSpeech data")
        print("  2. Convert to MFA corpus format")  
        print("  3. Run MFA alignment")
        print("  4. Organize into training-ready format")
        
        while True:
            response = input("\nDownload and prepare missing datasets? [y/N]: ").strip().lower()
            if response in {"y", "yes"}:
                return True
            elif response in {"n", "no", ""}:
                return False
            else:
                print("Please enter 'y' or 'n'")
    
    def _download_datasets(self, datasets: List[str]) -> bool:
        """Download raw datasets using downloadTrainingData.py"""
        logger.info("Downloading raw datasets...")
        
        try:
            cmd = [
                sys.executable, str(self.download_script),
                "--datasets", *datasets,
                "--output-dir", str(self.raw_dir)
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info("Download completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Download failed: {e}")
            return False
        except FileNotFoundError:
            logger.error(f"Download script not found: {self.download_script}")
            return False
    
    def _prepare_single_dataset(self, dataset: str) -> bool:
        """Prepare a single dataset through the full pipeline"""
        logger.info(f"Preparing dataset: {dataset}")
        
        # Step 1: Create prepared dataset (WAV + LAB) directly
        if not self._check_prepared_data(dataset):
            logger.info("Creating prepared dataset...")
            if not self._create_corpus(dataset):
                return False
        
        # Step 2: Check if we need to run MFA alignment
        prepared_dataset_dir = self.prepared_dir / dataset
        has_real_alignments = False
        
        if prepared_dataset_dir.exists():
            # Check if we have real alignment data (not just placeholder JSON files)
            for json_file in prepared_dataset_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        if "error" not in data:  # Real alignment data
                            has_real_alignments = True
                            break
                except:
                    continue
        
        if not has_real_alignments:
            logger.info("Running MFA alignment...")
            if not self._run_alignment(dataset):
                logger.warning("MFA alignment failed, but continuing with placeholders")
                # Still organize the dataset even if alignment failed
        
        # Step 3: Copy any available alignment JSONs (do not create placeholders)
        logger.info("Finalizing dataset organization (copying alignments if available)...")
        if not self._organize_prepared_dataset(dataset):
            return False
            
        logger.info(f"Dataset {dataset} prepared successfully")
        return True
    
    def _create_corpus(self, dataset: str) -> bool:
        """Create prepared dataset using createDataCorpus.py"""
        try:
            cmd = [
                sys.executable, str(self.corpus_script),
                "--subsets", dataset,
                "--librispeech-dir", str(self.raw_dir / "LibriSpeech"),
                "--prepared-dir", str(self.prepared_dir)
            ]
            
            result = subprocess.run(cmd, check=True, capture_output=False)
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Prepared dataset creation failed: {e}")
            return False
    
    def _run_alignment(self, dataset: str) -> bool:
        """Run MFA alignment using the prepared dataset MFA script"""
        try:
            # Path to the MFA alignment script
            mfa_script = self.project_root / "run_mfa_alignment_prepared.sh"
            
            if not mfa_script.exists():
                logger.error(f"MFA alignment script not found: {mfa_script}")
                return False
            
            logger.info(f"Running MFA alignment for {dataset}...")
            cmd = [str(mfa_script), dataset]
            
            result = subprocess.run(cmd, check=True, capture_output=False)
            logger.info(f"MFA alignment completed for {dataset}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"MFA alignment failed for {dataset}: {e}")
            return False
        except FileNotFoundError:
            logger.error(f"MFA alignment script not found or not executable: {mfa_script}")
            return False
    
    def _organize_prepared_dataset(self, dataset: str) -> bool:
        """Copy MFA alignment JSONs from cache into prepared dataset if available.

        Does not create placeholder JSON files.
        """
        try:
            prepared_dataset_dir = self.prepared_dir / dataset
            
            if not prepared_dataset_dir.exists():
                logger.error(f"Prepared dataset directory not found: {prepared_dataset_dir}")
                return False
            
            # Alignment directory in cache
            cache_dir = self.data_dir / "cache" / f"out_align_{dataset}"
            
            if not cache_dir.is_dir():
                logger.warning(f"No alignment data found for {dataset} - leaving JSONs absent")
                return True
            
            # Copy alignment files to prepared directory
            file_count = 0
            for wav_file in prepared_dataset_dir.glob("*.wav"):
                base_name = wav_file.stem
                json_file = wav_file.with_suffix(".json")
                
                if json_file.exists():
                    continue  # Already have alignment
                
                # Find corresponding JSON file in alignment directories
                source_json: Optional[Path] = None
                for potential_json in cache_dir.rglob(f"{base_name}.json"):
                    source_json = potential_json
                    break
                
                if source_json:
                    import shutil
                    shutil.copy2(source_json, json_file)
                    file_count += 1
            
            logger.info(f"Added alignment data for {file_count} samples in {dataset}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add alignment data to dataset {dataset}: {e}")
            return False
    
    def get_prepared_dataset_info(self, dataset: str) -> Dict:
        """Get information about a prepared dataset"""
        prepared_dir = self.prepared_dir / dataset
        
        if not prepared_dir.exists():
            return {"exists": False}
        
        wav_files = list(prepared_dir.glob("*.wav"))
        lab_files = list(prepared_dir.glob("*.lab"))
        json_files = list(prepared_dir.glob("*.json"))
        
        return {
            "exists": True,
            "path": str(prepared_dir),
            "num_samples": len(wav_files),
            "has_audio": len(wav_files) > 0,
            "has_transcripts": len(lab_files) > 0, 
            "has_alignments": len(json_files) > 0,
            "complete": len(wav_files) == len(lab_files) == len(json_files) > 0
        }


def check_datasets_for_training(datasets: List[str], project_root: Optional[Path] = None) -> bool:
    """
    Convenience function to check and prepare datasets for training
    
    Args:
        datasets: List of dataset names needed for training
        project_root: Project root directory (auto-detected if None)
        
    Returns:
        True if all datasets are ready, False otherwise
    """
    manager = DatasetManager(project_root)
    return manager.prepare_datasets(datasets, interactive=True)
