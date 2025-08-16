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
        
        # Create reverse mapping for debugging
        self.viseme_to_phonemes = {}
        for phoneme, viseme in self.phoneme_to_viseme.items():
            if viseme not in self.viseme_to_phonemes:
                self.viseme_to_phonemes[viseme] = []
            self.viseme_to_phonemes[viseme].append(phoneme)
    
    def align_phonemes_to_frames(self, phoneme_sequence: List[str], 
                               phoneme_timestamps: List[Tuple[float, float]], 
                               audio_duration: float) -> torch.Tensor:
        """
        Convert phoneme sequence with timestamps to frame-level viseme targets
        
        Args:
            phoneme_sequence: List of phoneme symbols
            phoneme_timestamps: List of (start_time, end_time) for each phoneme
            audio_duration: Total duration of audio in seconds
            
        Returns:
            torch.Tensor: Frame-level viseme targets, shape (time_frames,) or (time_frames, C) if multi_label
        """
        # Calculate number of frames
        num_frames = int(audio_duration * self.config.audio.fps)
        frame_duration = 1.0 / self.config.audio.fps
        
        num_classes = self.config.model.num_visemes
        is_multi = bool(getattr(self.config.training, "multi_label", False))
        target_crossfade_ms = int(getattr(self.config.training, "target_crossfade_ms", 0))
        crossfade_frames = int(round((target_crossfade_ms / 1000.0) * self.config.audio.fps)) if target_crossfade_ms > 0 else 0

        if is_multi:
            viseme_targets = torch.zeros(num_frames, num_classes, dtype=torch.float32)
            silence_idx = 0
            viseme_targets[:, silence_idx] = 1.0
        else:
            # Initialize with silence (viseme 0)
            viseme_targets = torch.zeros(num_frames, dtype=torch.long)
        
        # Fill in viseme targets based on phoneme timestamps
        for phoneme, (start_time, end_time) in zip(phoneme_sequence, phoneme_timestamps):
            # Convert phoneme to viseme
            viseme_id = self.phoneme_to_viseme.get(phoneme, 0)  # Default to silence
            
            # Calculate frame indices
            start_frame = int(start_time * self.config.audio.fps)
            end_frame = int(end_time * self.config.audio.fps)
            
            # Ensure frames are within bounds
            start_frame = max(0, start_frame)
            end_frame = min(num_frames, end_frame)
            
            # Assign viseme to frames
            if start_frame < end_frame:
                if is_multi:
                    # Clear default silence
                    viseme_targets[start_frame:end_frame, :] = 0.0
                    viseme_targets[start_frame:end_frame, viseme_id] = 1.0
                else:
                    viseme_targets[start_frame:end_frame] = viseme_id
        
        # Apply boundary-aware soft targets for multi-label
        if is_multi and crossfade_frames > 0:
            # Build a binary hard label timeline first to locate transitions
            hard = torch.argmax(viseme_targets, dim=-1)  # (T,)
            transitions: List[int] = []
            prev = int(hard[0].item()) if num_frames > 0 else 0
            for t in range(1, num_frames):
                cur = int(hard[t].item())
                if cur != prev:
                    transitions.append(t)
                prev = cur
            # For each transition, create a symmetric linear ramp from prev->cur over ±crossfade_frames
            for t0 in transitions:
                a = int(hard[max(0, t0 - 1)].item())  # class before boundary
                b = int(hard[min(num_frames - 1, t0)].item())  # class after boundary
                start = max(0, t0 - crossfade_frames)
                end = min(num_frames, t0 + crossfade_frames)
                for t in range(start, end):
                    # r in [0,1] centered at boundary
                    if t < t0:
                        r = (t - start) / max(1, (t0 - start))
                    else:
                        r = 1.0 - (t - t0) / max(1, (end - t0))
                    # Blend a and b; keep others at 0
                    base = torch.zeros(num_classes, dtype=torch.float32)
                    base[a] = 1.0 - float(r)
                    base[b] = float(r)
                    viseme_targets[t, :] = base
        return viseme_targets
    
    def get_phoneme_index(self, phoneme: str) -> int:
        """Get the index for a phoneme (for now, just return 0 as placeholder)"""
        # TODO: Implement proper phoneme indexing if needed
        return 0
    
    def get_viseme_index(self, phoneme: str) -> int:
        """Get the viseme index for a phoneme"""
        return self.phoneme_to_viseme.get(phoneme, 0)  # Default to silence
    
    def align_text(self, transcript: str, num_frames: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create dummy alignment from text (fallback when no MFA alignment available)
        
        Args:
            transcript: Text transcript
            num_frames: Number of target frames
            
        Returns:
            Tuple of (phoneme_targets, viseme_targets)
        """
        # For now, just return silence for all frames
        # In a real implementation, you'd use a text-to-phoneme system
        phoneme_targets = torch.zeros(num_frames, dtype=torch.long)
        viseme_targets = torch.zeros(num_frames, dtype=torch.long)
        
        return phoneme_targets, viseme_targets


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
        
        # Cache for processed samples
        self.sample_cache = {}
        
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
        # Check cache first
        if index in self.sample_cache:
            audio_sample = self.sample_cache[index]
        else:
            # Load and process new sample
            audio_sample = self._process_sample(index)
            # Cache only during validation/testing to save memory
            if not self.is_training:
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
        
        # Process audio features (this loads and processes the audio)
        audio_features = self.audio_processor.load_and_process_audio(audio_file)
        
        # TODO: Apply audio augmentation if training
        # For now, we skip augmentation to keep the pipeline simple
        
        # Load alignment data if available and use for more accurate targets
        phoneme_targets = None
        viseme_targets = None
        
        if json_file.exists():
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
        if phoneme_targets is None or viseme_targets is None:
            raise RuntimeError(
                f"Missing MFA alignment for sample {base_name} in {self.dataset_dir}. "
                f"Run alignment (see run_mfa_alignment_prepared.sh) before training."
            )
        
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
            phoneme_targets = torch.zeros(target_frames, dtype=torch.long)
            num_classes = self.config.model.num_visemes
            if bool(getattr(self.config.training, "multi_label", False)):
                viseme_targets = torch.zeros(target_frames, num_classes, dtype=torch.float32)
                viseme_targets[:, 0] = 1.0  # default silence
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
                    
                    # Get phoneme and viseme indices
                    phoneme_idx = self.phoneme_aligner.get_phoneme_index(phone)
                    viseme_idx = self.phoneme_aligner.get_viseme_index(phone)
                    
                    # Set targets for this interval
                    phoneme_targets[start_frame:end_frame] = phoneme_idx
                    if viseme_targets.dim() == 2:
                        viseme_targets[start_frame:end_frame, :] = 0.0
                        viseme_targets[start_frame:end_frame, viseme_idx] = 1.0
                    else:
                        viseme_targets[start_frame:end_frame] = viseme_idx
            
            return phoneme_targets, viseme_targets
            
        except Exception as e:
            raise RuntimeError(f"Failed to process alignment data: {e}")
    
    def _create_dummy_alignment(self, transcript: str, duration: float) -> torch.Tensor:
        """
        Create dummy phoneme alignment for demonstration.
        In production, use Montreal Forced Alignment (MFA) or similar.
        
        Args:
            transcript: Text transcript
            duration: Audio duration in seconds
            
        Returns:
            torch.Tensor: Frame-level viseme targets
        """
        # This is a simplified placeholder - real alignment needs MFA
        num_frames = int(duration * self.config.audio.fps)
        num_classes = self.config.model.num_visemes
        is_multi = bool(getattr(self.config.training, "multi_label", False))
        
        # Map common letter patterns to visemes (very crude approximation)
        letter_to_viseme = {
            'p': 1, 'b': 1, 'm': 1,  # PP viseme (bilabial)
            'f': 2, 'v': 2,          # FF viseme (labiodental)
            'th': 3,                 # TH viseme (dental)
            't': 4, 'd': 4, 'n': 4,  # DD viseme (alveolar)
            'k': 5, 'g': 5,          # kk viseme (velar)
            'ch': 6, 'j': 6, 'sh': 6, # CH viseme (palatal)
            's': 7, 'z': 7,          # SS viseme (sibilant)
            'l': 8, 'r': 8,          # RR viseme (liquid)
            'a': 10, 'aa': 10,       # aa viseme (open)
            'e': 11, 'eh': 11,       # E viseme (mid)
            'i': 12, 'ih': 12,       # ih viseme (close)
            'o': 13, 'oh': 13,       # oh viseme (back)
            'u': 14, 'oo': 14,       # ou viseme (round)
        }
        
        # Convert transcript to viseme sequence (very simplified)
        transcript_lower = transcript.lower()
        viseme_sequence = []
        
        for char in transcript_lower:
            if char in letter_to_viseme:
                viseme_sequence.append(letter_to_viseme[char])
            elif char.isalpha():
                viseme_sequence.append(10)  # Default to 'aa' viseme
            else:
                viseme_sequence.append(0)   # Silence for non-letters
        
        # Spread visemes over time frames
        if len(viseme_sequence) == 0:
            return torch.zeros((num_frames, num_classes), dtype=torch.float32) if is_multi else torch.zeros(num_frames, dtype=torch.long)
        
        frames_per_viseme = max(1, num_frames // len(viseme_sequence))
        targets = []
        
        for viseme in viseme_sequence:
            targets.extend([viseme] * frames_per_viseme)
        
        # Pad or truncate to exact length
        targets = targets[:num_frames]
        while len(targets) < num_frames:
            targets.append(0)  # Pad with silence
        if is_multi:
            out = torch.zeros(num_frames, num_classes, dtype=torch.float32)
            out[torch.arange(num_frames), torch.tensor(targets, dtype=torch.long)] = 1.0
            return out
        return torch.tensor(targets, dtype=torch.long)
    
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
