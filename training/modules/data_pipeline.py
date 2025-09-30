"""
Data Pipeline for OpenLipSync TCN Training

Handles audio data loading, preprocessing, and phoneme-to-viseme conversion.
Supports LibriSpeech dataset with configurable audio processing and augmentation.
"""

import os
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

from .config import TrainingConfiguration
from .dataset_manager import DatasetManager


@dataclass
class AudioSample:
    """Container for a processed audio sample"""
    audio_features: torch.Tensor  # Shape: (time_frames, n_mels)
    viseme_targets: torch.Tensor  # Shape: (time_frames,) for single-label or (time_frames, C) for multi-label
    sample_rate: int
    duration_seconds: float
    speaker_id: str
    utterance_id: str
    transcript: str = ""  # Text transcript
    phoneme_targets: Optional[torch.Tensor] = None  # Shape: (time_frames,)


class AudioProcessor:
    """
    Handles audio preprocessing: loading, mel spectrogram computation,
    and normalization according to configuration parameters.
    """
    
    def __init__(self, config: TrainingConfiguration):
        """
        Initialize audio processor with configuration
        
        Args:
            config: Training configuration containing audio parameters
        """
        self.config = config
        self.audio_config = config.audio
        
        # Create mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.audio_config.sample_rate,
            n_fft=self.audio_config.n_fft,
            win_length=self.audio_config.window_length_samples,
            hop_length=self.audio_config.hop_length_samples,
            n_mels=self.audio_config.n_mels,
            f_min=self.audio_config.fmin,
            f_max=self.audio_config.fmax,
            power=2.0,  # Power spectrogram
            normalized=False,
        )
        
        # Create amplitude to dB transform
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)
        
        # Initialize resampler if needed
        self.resampler = None
        
    def load_and_process_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """
        Load audio file and convert to mel spectrogram features
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            torch.Tensor: Mel spectrogram features, shape (time_frames, n_mels)
        """
        # Load audio using the new torchcodec-based approach
        waveform, original_sample_rate = torchaudio.load_with_torchcodec(audio_path)
        
        # Ensure mono audio
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if original_sample_rate != self.audio_config.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != original_sample_rate:
                self.resampler = T.Resample(
                    orig_freq=original_sample_rate,
                    new_freq=self.audio_config.sample_rate
                )
            waveform = self.resampler(waveform)
        
        # Compute mel spectrogram
        mel_spectrogram = self.mel_transform(waveform)  # Shape: (1, n_mels, time)
        
        # Convert to dB scale
        mel_spectrogram_db = self.amplitude_to_db(mel_spectrogram)
        
        # Remove batch dimension and transpose to (time, n_mels)
        mel_features = mel_spectrogram_db.squeeze(0).transpose(0, 1)
        
        return mel_features

    def waveform_to_mel_features(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Convert an in-memory waveform to mel features with current configuration.
        """
        # Ensure mono
        if waveform.dim() == 2 and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        elif waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        # Resample if needed
        if sample_rate != self.audio_config.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sample_rate:
                self.resampler = T.Resample(orig_freq=sample_rate, new_freq=self.audio_config.sample_rate)
            waveform = self.resampler(waveform)
        # Mel and dB
        mel = self.mel_transform(waveform)
        mel_db = self.amplitude_to_db(mel)
        return mel_db.squeeze(0).transpose(0, 1)
    
    def normalize_features(self, mel_features: torch.Tensor, 
                          normalization_stats: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Normalize mel features according to configuration
        
        Args:
            mel_features: Mel spectrogram features, shape (time_frames, n_mels)
            normalization_stats: Pre-computed stats for global normalization
            
        Returns:
            torch.Tensor: Normalized features
        """
        if self.audio_config.normalization == "per_utterance":
            # Normalize each utterance independently
            mean = mel_features.mean()
            std = mel_features.std()
            # Avoid division by zero
            std = torch.clamp(std, min=1e-8)
            normalized_features = (mel_features - mean) / std
            
        elif self.audio_config.normalization == "global":
            if normalization_stats is None:
                raise RuntimeError("Audio normalization is set to 'global' but no normalization_stats were provided.")
            # Use pre-computed global statistics
            mean = normalization_stats["mean"]
            std = normalization_stats["std"]
            normalized_features = (mel_features - mean) / std
            
        else:  # "none"
            normalized_features = mel_features
            
        return normalized_features


class PhonemeAligner:
    """
    Handles phoneme-to-viseme alignment and conversion.
    Maps phoneme sequences to frame-level viseme targets.
    """
    
    def __init__(self, config: TrainingConfiguration):
        """
        Initialize phoneme aligner
        
        Args:
            config: Training configuration with phoneme-viseme mapping
        """
        self.config = config
        self.phoneme_to_viseme = config.phoneme_to_viseme_mapping
        
    
    
    def get_viseme_index(self, phoneme: str) -> int:
        """Get the viseme index for a phoneme"""
        return self.phoneme_to_viseme.get(phoneme, 0)  # Default to silence


class SpecAugment:
    """
    SpecAugment data augmentation for mel spectrograms.
    Randomly masks time segments to improve model robustness.
    """
    
    def __init__(self, config: TrainingConfiguration):
        """
        Initialize SpecAugment with configuration
        
        Args:
            config: Training configuration with augmentation parameters
        """
        self.config = config
        self.training_config = config.training
        self.audio_config = config.audio
        
        # Convert time mask from ms to frames
        self.max_time_mask_frames = int(
            self.training_config.specaugment_time_mask_max_ms * 
            self.audio_config.fps / 1000
        )
    
    def __call__(self, mel_features: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to mel spectrogram features
        
        Args:
            mel_features: Input features, shape (time_frames, n_mels)
            
        Returns:
            torch.Tensor: Augmented features
        """
        if not self.training_config.specaugment_enabled:
            return mel_features
        
        augmented_features = mel_features.clone()
        time_frames, n_mels = augmented_features.shape
        
        # Time masking: randomly mask a contiguous segment of time frames
        if self.max_time_mask_frames > 0 and time_frames > self.max_time_mask_frames:
            mask_length = random.randint(1, min(self.max_time_mask_frames, time_frames // 4))
            mask_start = random.randint(0, time_frames - mask_length)
            
            # Set masked region to mean value
            mask_value = augmented_features.mean()
            augmented_features[mask_start:mask_start + mask_length, :] = mask_value
        
        return augmented_features


class AudioAugmentation:
    """
    Audio-level augmentations: noise addition and gain variation.
    Applied to raw audio before mel spectrogram computation.
    """
    
    def __init__(self, config: TrainingConfiguration):
        """
        Initialize audio augmentation
        
        Args:
            config: Training configuration with augmentation parameters
        """
        self.config = config
        self.data_config = config.data
    
    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to audio waveform
        
        Args:
            waveform: Input audio, shape (1, samples)
            
        Returns:
            torch.Tensor: Noisy audio
        """
        if not self.data_config.augmentation_enabled:
            return waveform
        
        # Random SNR in specified range
        snr_db = random.uniform(*self.data_config.noise_snr_range)
        
        # Calculate noise power
        signal_power = waveform.pow(2).mean()
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate and add noise
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        noisy_waveform = waveform + noise
        
        return noisy_waveform
    
    def apply_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Apply random gain to audio waveform
        
        Args:
            waveform: Input audio, shape (1, samples)
            
        Returns:
            torch.Tensor: Gain-adjusted audio
        """
        if not self.data_config.augmentation_enabled:
            return waveform
        
        # Random gain in specified range
        gain = random.uniform(*self.data_config.gain_range)
        return waveform * gain

    def synthesize_silence_or_near_silence(self, 
                                           sample_rate: int, 
                                           min_length_s: float, 
                                           max_length_s: float) -> torch.Tensor:
        """
        Generate a random-length chunk of near-silence: low-level Gaussian noise.
        Returns mono waveform tensor with shape (1, samples).
        """
        length_s = random.uniform(min_length_s, max_length_s)
        num_samples = int(length_s * sample_rate)
        # Choose noise level in dBFS (negative). Convert to linear std assuming signal peak ~1.
        dbfs_min, dbfs_max = self.config.data.silence_noise_dbfs_range
        noise_dbfs = random.uniform(dbfs_min, dbfs_max)
        noise_rms = 10 ** (noise_dbfs / 20.0)
        noise = torch.randn(1, num_samples) * noise_rms
        return noise


class LibriSpeechDataset(Dataset):
    """
    PyTorch Dataset for LibriSpeech with phoneme alignment and viseme targets.
    Handles audio chunking, preprocessing, and augmentation.
    """
    
    def __init__(self, config: TrainingConfiguration, split: str, 
                 is_training: bool = True, data_root: Optional[str] = None):
        """
        Initialize LibriSpeech dataset
        
        Args:
            config: Training configuration
            split: Dataset split (e.g., "train-clean-100", "dev-clean")
            is_training: Whether this is for training (affects augmentation)
            data_root: Root directory for LibriSpeech data
        """
        self.config = config
        self.split = split
        self.is_training = is_training
        
        # Initialize processing components
        self.audio_processor = AudioProcessor(config)
        self.phoneme_aligner = PhonemeAligner(config)
        self.spec_augment = SpecAugment(config)
        self.audio_augment = AudioAugmentation(config)
        
        # Setup dataset manager and ensure data is prepared
        if data_root:
            self.dataset_manager = DatasetManager(Path(data_root))
        else:
            self.dataset_manager = DatasetManager()
        
        # Ensure dataset is prepared
        if not self.dataset_manager.prepare_datasets([split], interactive=True):
            raise RuntimeError(f"Failed to prepare dataset: {split}")
        
        # Load prepared data file list
        self.dataset_dir = self.dataset_manager.prepared_dir / split
        if not self.dataset_dir.exists():
            raise RuntimeError(f"Prepared dataset directory not found: {self.dataset_dir}")
        
        # Get list of audio files (other files will be found by extension)
        self.audio_files = sorted(list(self.dataset_dir.glob("*.wav")))
        if not self.audio_files:
            raise RuntimeError(f"No audio files found in {self.dataset_dir}")
        
        # Optional cache for processed samples (validation/test only by default)
        self.sample_cache: Dict[int, AudioSample] = {}
        self.enable_cache = (not self.is_training) and bool(self.config.data.cache_validation_items)
        self.cache_max_items = int(getattr(self.config.data, 'validation_cache_max_items', 0) or 0)
        
        # Chunk parameters
        self.min_chunk_frames = int(
            config.training.min_chunk_length_s * config.audio.fps
        )
        self.max_chunk_frames = int(
            config.training.max_chunk_length_s * config.audio.fps
        )
        
        print(f"Loaded prepared {split}: {len(self.audio_files)} utterances from {self.dataset_dir}")
    
    def __len__(self) -> int:
        """Number of utterances in the dataset"""
        return len(self.audio_files)
    
    def __getitem__(self, index: int) -> AudioSample:
        """
        Get a processed audio sample
        
        Args:
            index: Sample index
            
        Returns:
            AudioSample: Processed audio with features and targets
        """
        # Check cache first (optional)
        if self.enable_cache and index in self.sample_cache:
            audio_sample = self.sample_cache[index]
        else:
            # Load and process new sample
            audio_sample = self._process_sample(index)
            # Cache per policy
            if self.enable_cache:
                # Maintain simple size-bounded cache (FIFO eviction)
                if self.cache_max_items > 0 and len(self.sample_cache) >= self.cache_max_items:
                    # Pop an arbitrary (oldest-like) item deterministically
                    first_key = next(iter(self.sample_cache.keys()))
                    self.sample_cache.pop(first_key, None)
                self.sample_cache[index] = audio_sample
        
        # Apply chunking for training
        if self.is_training:
            audio_sample = self._apply_chunking(audio_sample)
        
        # Apply augmentation
        if self.is_training:
            audio_sample.audio_features = self.spec_augment(audio_sample.audio_features)
        
        return audio_sample
    
    def _process_sample(self, index: int) -> AudioSample:
        """
        Process a single prepared sample
        
        Args:
            index: Sample index
            
        Returns:
            AudioSample: Processed sample
        """
        # Get file paths
        audio_file = self.audio_files[index]
        base_name = audio_file.stem
        lab_file = audio_file.with_suffix(".lab")
        json_file = audio_file.with_suffix(".json")
        
        # Load transcript
        if not lab_file.exists():
            raise RuntimeError(f"Transcript file not found: {lab_file}")
        
        with open(lab_file, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
        
        # Optional: replace with synthetic near-silence sample with some probability
        use_silence_aug = False
        if self.is_training and self.config.data.augmentation_enabled:
            try:
                if random.random() < float(self.config.data.silence_augment_prob):
                    use_silence_aug = True
            except Exception:
                use_silence_aug = False

        if use_silence_aug:
            # Generate near-silence waveform and targets set to silence
            sr = self.config.audio.sample_rate
            min_s, max_s = self.config.data.silence_chunk_length_s
            waveform = self.audio_augment.synthesize_silence_or_near_silence(sr, min_s, max_s)
            audio_features = self.audio_processor.waveform_to_mel_features(waveform, sr)
        else:
            # Process audio features (this loads and processes the audio)
            audio_features = self.audio_processor.load_and_process_audio(audio_file)
        
        # Apply normalization according to configuration
        audio_features = self.audio_processor.normalize_features(audio_features)
        
        # Waveform-level augmentation would require waveform; here we only have features.
        # Keep feature-level augment below.
        
        # Load alignment data if available and use for more accurate targets
        phoneme_targets = None
        viseme_targets = None
        
        if json_file.exists() and not use_silence_aug:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    alignment_data = json.load(f)
                
                # Try to use alignment data for more accurate targets
                if "tiers" in alignment_data and "phones" in alignment_data["tiers"]:
                    phoneme_targets, viseme_targets = self._process_alignment(
                        alignment_data, audio_features.shape[0]
                    )
            except Exception as e:
                print(f"Warning: Failed to process alignment data for {base_name}: {e}")
        
        # If no valid alignment was found, raise an error instead of training on silence
        if viseme_targets is None and not use_silence_aug:
            raise RuntimeError(
                f"Missing MFA alignment for sample {base_name} in {self.dataset_dir}. "
                f"Run alignment (see run_mfa_alignment_prepared.sh) before training."
            )
        elif use_silence_aug:
            # Create pure-silence targets matching feature frames
            num_frames = audio_features.shape[0]
            num_classes = self.config.model.num_visemes
            multi = bool(getattr(self.config.training, "multi_label", False))
            if multi:
                viseme_targets = torch.zeros(num_frames, num_classes, dtype=torch.float32)
                viseme_targets[:, 0] = 1.0
            else:
                viseme_targets = torch.zeros(num_frames, dtype=torch.long)  # 0 == silence
        
        # Extract metadata from filename (format: speaker-chapter-utterance)
        parts = base_name.split('-')
        if len(parts) >= 3:
            speaker_id = parts[0]
            utterance_id = base_name
        else:
            speaker_id = "unknown"
            utterance_id = base_name
        
        # Calculate duration from audio features
        duration_seconds = audio_features.shape[0] / self.config.audio.fps
        
        return AudioSample(
            audio_features=audio_features,
            viseme_targets=viseme_targets,
            sample_rate=self.config.audio.sample_rate,
            duration_seconds=duration_seconds,
            speaker_id=speaker_id,
            utterance_id=utterance_id,
            transcript=transcript,
            phoneme_targets=phoneme_targets
        )
    
    def _process_alignment(self, alignment_data: Dict, target_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process MFA alignment data to create frame-level targets
        
        Args:
            alignment_data: MFA alignment JSON data
            target_frames: Number of target frames to generate
            
        Returns:
            Tuple of (phoneme_targets, viseme_targets)
        """
        try:
            phones_tier = alignment_data["tiers"]["phones"]["entries"]
            audio_duration = alignment_data["end"] - alignment_data["start"]
            
            # Convert frame index to time
            frame_rate = target_frames / audio_duration
            
            # Initialize targets
            num_classes = self.config.model.num_visemes
            multi = bool(getattr(self.config.training, "multi_label", False))
            target_crossfade_ms = int(getattr(self.config.training, "target_crossfade_ms", 0))
            crossfade_frames = int(round((target_crossfade_ms / 1000.0) * self.config.audio.fps)) if target_crossfade_ms > 0 else 0
            if multi:
                viseme_targets = torch.zeros(target_frames, num_classes, dtype=torch.float32)
            else:
                viseme_targets = torch.zeros(target_frames, dtype=torch.long)
            
            # Process each phoneme interval
            for start_time, end_time, phone in phones_tier:
                if isinstance(phone, str) and phone.strip():
                    # Convert time to frame indices
                    start_frame = int(start_time * frame_rate)
                    end_frame = int(end_time * frame_rate)
                    
                    # Ensure frames are within bounds
                    start_frame = max(0, min(start_frame, target_frames - 1))
                    end_frame = max(start_frame + 1, min(end_frame, target_frames))
                    
                    # Get viseme index
                    viseme_idx = self.phoneme_aligner.get_viseme_index(phone)
                    if viseme_targets.dim() == 2:
                        # Independent per-class assignment with symmetric crossfade
                        viseme_targets[start_frame:end_frame, viseme_idx] = 1.0
                        if crossfade_frames > 0:
                            lead_start = max(0, start_frame - crossfade_frames)
                            lead_len = start_frame - lead_start
                            if lead_len > 0:
                                alpha = torch.linspace(0.0, 1.0, steps=lead_len, dtype=torch.float32)
                                viseme_targets[lead_start:start_frame, viseme_idx] = torch.maximum(
                                    viseme_targets[lead_start:start_frame, viseme_idx], alpha
                                )
                            tail_end = min(target_frames, end_frame + crossfade_frames)
                            tail_len = tail_end - end_frame
                            if tail_len > 0:
                                alpha = torch.linspace(1.0, 0.0, steps=tail_len, dtype=torch.float32)
                                viseme_targets[end_frame:tail_end, viseme_idx] = torch.maximum(
                                    viseme_targets[end_frame:tail_end, viseme_idx], alpha
                                )
                    else:
                        viseme_targets[start_frame:end_frame] = viseme_idx
            
            # For multi-label, set silence where no other viseme active
            if viseme_targets.dim() == 2 and num_classes > 0:
                non_silence = viseme_targets[:, 1:].sum(dim=-1) if num_classes > 1 else torch.zeros(target_frames)
                silence_active = (non_silence <= 0.0).to(viseme_targets.dtype)
                viseme_targets[:, 0] = torch.maximum(viseme_targets[:, 0], silence_active)
            return None, viseme_targets  # phoneme_targets not used, return None
            
        except Exception as e:
            raise RuntimeError(f"Failed to process alignment data: {e}")
    
    
    
    def _apply_chunking(self, audio_sample: AudioSample) -> AudioSample:
        """
        Apply random chunking for training
        
        Args:
            audio_sample: Input audio sample
            
        Returns:
            AudioSample: Chunked audio sample
        """
        num_frames = audio_sample.audio_features.shape[0]
        
        # If audio is shorter than minimum chunk, return as is
        if num_frames <= self.min_chunk_frames:
            return audio_sample
        
        # If audio is longer than maximum chunk, randomly select a chunk
        if num_frames > self.max_chunk_frames:
            chunk_length = random.randint(self.min_chunk_frames, self.max_chunk_frames)
            start_frame = random.randint(0, num_frames - chunk_length)
            end_frame = start_frame + chunk_length
            
            # Extract chunk
            chunked_features = audio_sample.audio_features[start_frame:end_frame]
            chunked_targets = audio_sample.viseme_targets[start_frame:end_frame]
            
            # Update duration
            chunked_duration = chunk_length / self.config.audio.fps
            
            return AudioSample(
                audio_features=chunked_features,
                viseme_targets=chunked_targets,
                sample_rate=audio_sample.sample_rate,
                duration_seconds=chunked_duration,
                speaker_id=audio_sample.speaker_id,
                utterance_id=f"{audio_sample.utterance_id}_chunk_{start_frame}_{end_frame}"
            )
        
        return audio_sample


def collate_audio_samples(batch: List[AudioSample]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader. Pads sequences to equal length.
    
    Args:
        batch: List of AudioSample objects
        
    Returns:
        Dict with batched tensors
    """
    # Find maximum sequence length in batch
    max_length = max(sample.audio_features.shape[0] for sample in batch)
    
    # Pad sequences
    batch_features = []
    batch_targets = []
    batch_lengths = []
    
    for sample in batch:
        features = sample.audio_features
        targets = sample.viseme_targets
        seq_length = features.shape[0]
        
        # Pad features
        if seq_length < max_length:
            padding = torch.zeros(max_length - seq_length, features.shape[1])
            features = torch.cat([features, padding], dim=0)
            
            # Pad targets: handle (T,) long or (T,C) float
            if targets.dim() == 1:
                target_padding = torch.zeros(max_length - seq_length, dtype=torch.long)
            else:
                num_classes = targets.shape[1]
                target_padding = torch.zeros(max_length - seq_length, num_classes, dtype=targets.dtype)
                target_padding[:, 0] = 1.0  # pad with silence for multi-label
            targets = torch.cat([targets, target_padding], dim=0)
        
        batch_features.append(features)
        batch_targets.append(targets)
        batch_lengths.append(seq_length)
    
    # Stack targets handling variable dims
    if batch_targets[0].dim() == 1:
        targets_out = torch.stack(batch_targets)  # (B, T)
    else:
        targets_out = torch.stack(batch_targets)  # (B, T, C)
    return {
        'features': torch.stack(batch_features),  # (batch, time, n_mels)
        'targets': targets_out,                   # (batch, time) or (batch, time, C)
        'lengths': torch.tensor(batch_lengths),   # (batch,)
        'speaker_ids': [sample.speaker_id for sample in batch],
        'utterance_ids': [sample.utterance_id for sample in batch],
    }


def create_data_loaders(config: TrainingConfiguration, 
                       data_root: Optional[str] = None,
                       pin_memory: Optional[bool] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and test data loaders
    
    Args:
        config: Training configuration
        data_root: Root directory for data (optional)
        pin_memory: Override pin_memory setting (optional)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Use provided pin_memory override or fall back to config
    use_pin_memory = pin_memory if pin_memory is not None else config.hardware.pin_memory
    
    # Create datasets
    train_datasets = []
    for split in config.data.splits:
        dataset = LibriSpeechDataset(
            config=config,
            split=split,
            is_training=True,
            data_root=data_root
        )
        train_datasets.append(dataset)
    
    # Concatenate training datasets
    from torch.utils.data import ConcatDataset
    train_dataset = ConcatDataset(train_datasets)
    
    val_dataset = LibriSpeechDataset(
        config=config,
        split=config.data.val_split,
        is_training=False,
        data_root=data_root
    )
    
    test_dataset = LibriSpeechDataset(
        config=config,
        split=config.data.test_split,
        is_training=False,
        data_root=data_root
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_audio_samples,
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_audio_samples,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_audio_samples,
        drop_last=False
    )
    
    print(f"Created data loaders:")
    print(f"  Training: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Test: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage and testing
    from .config import load_config
    
    # Load configuration
    config = load_config("../recipes/tcn_config.toml")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(config)
    
    # Test loading a batch
    print("\nTesting data loading...")
    batch = next(iter(train_loader))
    
    print(f"Batch features shape: {batch['features'].shape}")
    print(f"Batch targets shape: {batch['targets'].shape}")
    print(f"Batch lengths: {batch['lengths'][:5]}...")  # Show first 5
    print(f"Sample speaker IDs: {batch['speaker_ids'][:3]}")
    
    print("Data pipeline test completed successfully.")
