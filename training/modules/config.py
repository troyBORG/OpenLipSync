"""
Configuration Management for OpenLipSync TCN Training

Handles loading, validation, and type checking of training configuration
from TOML files. Provides structured access to all training parameters.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Fallback for older Python versions


@dataclass
class ModelConfig:
    """TCN model architecture configuration"""
    name: str
    num_visemes: int
    layers: int
    channels: int
    kernel_size: int
    dropout: float
    normalization: str
    
    def __post_init__(self):
        """Validate model parameters"""
        if self.num_visemes < 1:
            raise ValueError(f"num_visemes must be positive, got {self.num_visemes}")
        if self.layers < 1:
            raise ValueError(f"layers must be positive, got {self.layers}")
        if self.channels < 1:
            raise ValueError(f"channels must be positive, got {self.channels}")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be positive and odd, got {self.kernel_size}")
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be in [0,1], got {self.dropout}")
        if self.normalization not in ["weight_norm", "layer_norm", "batch_norm"]:
            raise ValueError(f"Invalid normalization: {self.normalization}")


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int
    hop_length_ms: int
    window_length_ms: int
    n_mels: int
    fmin: int
    fmax: int
    n_fft: int
    normalization: str
    
    def __post_init__(self):
        """Validate audio parameters and compute derived values"""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.hop_length_ms <= 0:
            raise ValueError(f"hop_length_ms must be positive, got {self.hop_length_ms}")
        if self.window_length_ms <= 0:
            raise ValueError(f"window_length_ms must be positive, got {self.window_length_ms}")
        if self.window_length_ms < self.hop_length_ms:
            raise ValueError(f"window_length_ms ({self.window_length_ms}) should be >= hop_length_ms ({self.hop_length_ms})")
        if self.n_mels <= 0:
            raise ValueError(f"n_mels must be positive, got {self.n_mels}")
        if self.fmin < 0:
            raise ValueError(f"fmin must be non-negative, got {self.fmin}")
        if self.fmax <= self.fmin:
            raise ValueError(f"fmax ({self.fmax}) must be > fmin ({self.fmin})")
        if self.fmax > self.sample_rate // 2:
            raise ValueError(f"fmax ({self.fmax}) must be <= Nyquist frequency ({self.sample_rate // 2})")
        if self.n_fft <= 0 or (self.n_fft & (self.n_fft - 1)) != 0:
            raise ValueError(f"n_fft must be a positive power of 2, got {self.n_fft}")
        if self.normalization not in ["per_utterance", "global", "none"]:
            raise ValueError(f"Invalid audio normalization: {self.normalization}")
        
        # Compute derived values
        self.hop_length_samples = int(self.sample_rate * self.hop_length_ms / 1000)
        self.window_length_samples = int(self.sample_rate * self.window_length_ms / 1000)
        self.fps = 1000.0 / self.hop_length_ms  # Frames per second


@dataclass
class TrainingConfig:
    """Training loop configuration"""
    batch_size: int
    max_chunk_length_s: float
    min_chunk_length_s: float
    mixed_precision: bool
    loss_type: str
    class_weighting: bool
    focal_loss_alpha: float
    focal_loss_gamma: float
    optimizer: str
    learning_rate: float
    betas: List[float]
    weight_decay: float
    scheduler: str
    warmup_ratio: float
    max_epochs: int
    early_stopping_patience: int
    early_stopping_metric: str
    specaugment_enabled: bool
    specaugment_time_mask_max_ms: int
    
    def __post_init__(self):
        """Validate training parameters"""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.max_chunk_length_s <= 0:
            raise ValueError(f"max_chunk_length_s must be positive, got {self.max_chunk_length_s}")
        if self.min_chunk_length_s <= 0:
            raise ValueError(f"min_chunk_length_s must be positive, got {self.min_chunk_length_s}")
        if self.min_chunk_length_s >= self.max_chunk_length_s:
            raise ValueError(f"min_chunk_length_s ({self.min_chunk_length_s}) must be < max_chunk_length_s ({self.max_chunk_length_s})")
        if self.loss_type not in ["cross_entropy", "focal_loss"]:
            raise ValueError(f"Invalid loss_type: {self.loss_type}")
        if self.optimizer not in ["adamw", "adam", "sgd"]:
            raise ValueError(f"Invalid optimizer: {self.optimizer}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if len(self.betas) != 2 or not all(0 <= b < 1 for b in self.betas):
            raise ValueError(f"betas must be [beta1, beta2] with values in [0,1), got {self.betas}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.scheduler not in ["onecycle", "cosine", "constant"]:
            raise ValueError(f"Invalid scheduler: {self.scheduler}")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError(f"warmup_ratio must be in [0,1], got {self.warmup_ratio}")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be positive, got {self.max_epochs}")
        if self.early_stopping_patience <= 0:
            raise ValueError(f"early_stopping_patience must be positive, got {self.early_stopping_patience}")
        if self.early_stopping_metric not in ["val_loss", "val_f1"]:
            raise ValueError(f"Invalid early_stopping_metric: {self.early_stopping_metric}")


@dataclass
class DataConfig:
    """Data pipeline configuration"""
    dataset: str
    splits: List[str]
    val_split: str
    test_split: str
    augmentation_enabled: bool
    noise_snr_range: List[float]
    gain_range: List[float]
    phoneme_viseme_map: str
    
    def __post_init__(self):
        """Validate data parameters"""
        if self.dataset not in ["librispeech"]:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if not self.splits:
            raise ValueError("splits cannot be empty")
        if len(self.noise_snr_range) != 2 or self.noise_snr_range[0] >= self.noise_snr_range[1]:
            raise ValueError(f"noise_snr_range must be [min, max] with min < max, got {self.noise_snr_range}")
        if len(self.gain_range) != 2 or self.gain_range[0] >= self.gain_range[1]:
            raise ValueError(f"gain_range must be [min, max] with min < max, got {self.gain_range}")


@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration"""
    metrics: List[str]
    compute_latency: bool
    target_hardware: str
    
    def __post_init__(self):
        """Validate evaluation parameters"""
        valid_metrics = ["frame_accuracy", "macro_f1", "confusion_matrix"]
        for metric in self.metrics:
            if metric not in valid_metrics:
                raise ValueError(f"Invalid metric: {metric}. Valid options: {valid_metrics}")
        if self.target_hardware not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Invalid target_hardware: {self.target_hardware}")


@dataclass
class HardwareConfig:
    """Hardware configuration"""
    device: str
    num_workers: int
    pin_memory: bool
    
    def __post_init__(self):
        """Validate hardware parameters"""
        if self.device not in ["cpu", "cuda", "mps"]:
            raise ValueError(f"Invalid device: {self.device}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {self.num_workers}")


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_interval: int
    save_interval: int
    max_checkpoints: int
    log_level: str
    
    def __post_init__(self):
        """Validate logging parameters"""
        if self.log_interval <= 0:
            raise ValueError(f"log_interval must be positive, got {self.log_interval}")
        if self.save_interval <= 0:
            raise ValueError(f"save_interval must be positive, got {self.save_interval}")
        if self.max_checkpoints <= 0:
            raise ValueError(f"max_checkpoints must be positive, got {self.max_checkpoints}")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")


@dataclass
class TensorBoardConfig:
    """TensorBoard logging configuration"""
    enabled: bool
    runs_dir: str
    run_name_format: str
    run_name: Optional[str] = None
    log_scalars: bool = True
    log_histograms: bool = False
    log_images: bool = False
    log_audio: bool = False
    scalar_log_interval: int = 20
    histogram_log_interval: int = 100
    image_log_interval: int = 500
    
    def __post_init__(self):
        """Validate TensorBoard parameters"""
        if self.scalar_log_interval <= 0:
            raise ValueError(f"scalar_log_interval must be positive, got {self.scalar_log_interval}")
        if self.histogram_log_interval <= 0:
            raise ValueError(f"histogram_log_interval must be positive, got {self.histogram_log_interval}")
        if self.image_log_interval <= 0:
            raise ValueError(f"image_log_interval must be positive, got {self.image_log_interval}")


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration"""
    name: str
    tags: List[str]
    notes: str


@dataclass
class TrainingConfiguration:
    """Complete training configuration"""
    model: ModelConfig
    audio: AudioConfig
    training: TrainingConfig
    data: DataConfig
    evaluation: EvaluationConfig
    hardware: HardwareConfig
    logging: LoggingConfig
    tensorboard: TensorBoardConfig
    experiment: ExperimentConfig
    
    # Runtime properties
    config_path: str = ""
    phoneme_to_viseme_mapping: Dict[str, int] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Load phoneme-to-viseme mapping
        self._load_phoneme_viseme_mapping()
        
        # Validate that num_visemes matches the mapping
        actual_num_visemes = len(set(self.phoneme_to_viseme_mapping.values()))
        if self.model.num_visemes != actual_num_visemes:
            raise ValueError(
                f"model.num_visemes ({self.model.num_visemes}) does not match "
                f"the number of unique visemes in phoneme_viseme_map ({actual_num_visemes})"
            )
    
    def _load_phoneme_viseme_mapping(self):
        """Load and validate phoneme-to-viseme mapping"""
        # Try to resolve path relative to config file first, then relative to current working directory
        config_path = Path(self.config_path)
        
        # First try: relative to config file directory
        mapping_path = config_path.parent / self.data.phoneme_viseme_map
        
        # If that doesn't work, try relative to current working directory
        if not mapping_path.exists():
            mapping_path = Path(self.data.phoneme_viseme_map)
        
        # If still not found, try to find project root and resolve from there
        if not mapping_path.exists():
            current_dir = Path.cwd()
            # Look for project root indicators (like pyproject.toml)
            project_root = current_dir
            while project_root != project_root.parent:
                if (project_root / "pyproject.toml").exists() or (project_root / "README.md").exists():
                    break
                project_root = project_root.parent
            mapping_path = project_root / self.data.phoneme_viseme_map
        
        if not mapping_path.exists():
            raise FileNotFoundError(f"Phoneme-viseme mapping file not found: {mapping_path}")
        
        try:
            with open(mapping_path, 'r') as file:
                mapping_data = json.load(file)
            
            # Validate required sections
            if "phoneme_to_viseme" not in mapping_data:
                raise ValueError("phoneme_viseme_map must contain 'phoneme_to_viseme' key")
            
            if "viseme_set" not in mapping_data or "visemes" not in mapping_data["viseme_set"]:
                raise ValueError("phoneme_viseme_map must contain 'viseme_set.visemes' section")
            
            # Get viseme name to index mapping
            viseme_name_to_index = mapping_data["viseme_set"]["visemes"]
            phoneme_to_viseme_names = mapping_data["phoneme_to_viseme"]
            
            # Validate viseme indices are integers
            for viseme_name, viseme_index in viseme_name_to_index.items():
                if not isinstance(viseme_index, int) or viseme_index < 0:
                    raise ValueError(f"Invalid viseme index for '{viseme_name}': {viseme_index}")
            
            # Convert phoneme -> viseme_name mapping to phoneme -> viseme_index mapping
            self.phoneme_to_viseme_mapping = {}
            for phoneme, viseme_name in phoneme_to_viseme_names.items():
                if viseme_name not in viseme_name_to_index:
                    available_visemes = list(viseme_name_to_index.keys())
                    raise ValueError(
                        f"Unknown viseme '{viseme_name}' for phoneme '{phoneme}'. "
                        f"Available visemes: {available_visemes}"
                    )
                self.phoneme_to_viseme_mapping[phoneme] = viseme_name_to_index[viseme_name]
            
            # Validate that we have the expected number of unique visemes
            unique_viseme_indices = set(viseme_name_to_index.values())
            expected_indices = set(range(len(viseme_name_to_index)))
            
            if unique_viseme_indices != expected_indices:
                missing = expected_indices - unique_viseme_indices
                extra = unique_viseme_indices - expected_indices
                error_msg = "Viseme index validation failed:"
                if missing:
                    error_msg += f" Missing indices: {sorted(missing)}."
                if extra:
                    error_msg += f" Unexpected indices: {sorted(extra)}."
                raise ValueError(error_msg)
                    
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON in phoneme_viseme_map: {error}")
        except KeyError as error:
            raise ValueError(f"Missing required key in phoneme_viseme_map: {error}")
    
    def get_run_name(self) -> str:
        """Generate run name for TensorBoard logging"""
        if self.tensorboard.run_name:
            return self.tensorboard.run_name
        
        # Format the run name using available variables
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return self.tensorboard.run_name_format.format(
            experiment_name=self.experiment.name,
            timestamp=timestamp,
            lr=self.training.learning_rate,
            batch_size=self.training.batch_size,
            epoch=0  # Will be updated during training
        )
    
    def get_tensorboard_log_dir(self) -> Path:
        """Get TensorBoard log directory for this run"""
        # Resolve runs_dir intelligently
        # First try relative to current working directory
        runs_dir = Path(self.tensorboard.runs_dir)
        
        # If not absolute, try to find project root and resolve from there
        if not runs_dir.is_absolute():
            current_dir = Path.cwd()
            # Look for project root indicators
            project_root = current_dir
            while project_root != project_root.parent:
                if (project_root / "pyproject.toml").exists() or (project_root / "README.md").exists():
                    break
                project_root = project_root.parent
            runs_dir = project_root / self.tensorboard.runs_dir
        
        return runs_dir / self.get_run_name()


def load_config(config_path: Union[str, Path]) -> TrainingConfiguration:
    """
    Load and validate training configuration from TOML file
    
    Args:
        config_path: Path to the TOML configuration file
        
    Returns:
        TrainingConfiguration: Validated configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load TOML file
    try:
        with open(config_path, 'rb') as file:
            config_data = tomllib.load(file)
    except Exception as error:
        raise ValueError(f"Failed to parse TOML configuration: {error}")
    
    # Extract each section with defaults
    try:
        config = TrainingConfiguration(
            model=ModelConfig(**config_data.get("model", {})),
            audio=AudioConfig(**config_data.get("audio", {})),
            training=TrainingConfig(**config_data.get("training", {})),
            data=DataConfig(**config_data.get("data", {})),
            evaluation=EvaluationConfig(**config_data.get("evaluation", {})),
            hardware=HardwareConfig(**config_data.get("hardware", {})),
            logging=LoggingConfig(**config_data.get("logging", {})),
            tensorboard=TensorBoardConfig(**config_data.get("tensorboard", {})),
            experiment=ExperimentConfig(**config_data.get("experiment", {})),
            config_path=str(config_path)
        )
        
        return config
        
    except TypeError as error:
        raise ValueError(f"Missing required configuration parameter: {error}")
    except Exception as error:
        raise ValueError(f"Configuration validation failed: {error}")


def validate_environment(config: TrainingConfiguration) -> None:
    """
    Validate that the environment supports the requested configuration
    
    Args:
        config: Training configuration to validate
        
    Raises:
        RuntimeError: If environment doesn't support the configuration
    """
    # Check device availability
    if config.hardware.device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("CUDA requested but not available. Falling back to CPU.")
                config.hardware.device = "cpu"
        except ImportError:
            print("PyTorch not installed. Cannot use CUDA. Falling back to CPU.")
            config.hardware.device = "cpu"
    
    elif config.hardware.device == "mps":
        try:
            import torch
            if not torch.backends.mps.is_available():
                print("MPS requested but not available. Falling back to CPU.")
                config.hardware.device = "cpu"
        except ImportError:
            print("PyTorch not installed. Cannot use MPS. Falling back to CPU.")
            config.hardware.device = "cpu"
    
    # Check required directories exist
    runs_dir = Path(config.tensorboard.runs_dir)
    if config.tensorboard.enabled and not runs_dir.parent.exists():
        runs_dir.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python config.py <config_file.toml>")
        sys.exit(1)
    
    try:
        config = load_config(sys.argv[1])
        validate_environment(config)
        print("Configuration loaded and validated successfully.")
        print(f"Experiment: {config.experiment.name}")
        print(f"Model: {config.model.name} with {config.model.num_visemes} visemes")
        print(f"Audio: {config.audio.sample_rate}Hz, {config.audio.fps:.1f}fps")
        print(f"Device: {config.hardware.device}")
        
    except Exception as error:
        print(f"Configuration error: {error}")
        sys.exit(1)
