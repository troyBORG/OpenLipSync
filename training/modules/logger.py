"""
Logging and TensorBoard Integration for OpenLipSync TCN Training

Provides comprehensive logging functionality including TensorBoard integration,
experiment tracking, and progress monitoring.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .config import TrainingConfiguration


class TrainingLogger:
    """
    Comprehensive logging system for training progress, metrics, and debugging.
    Handles both console output and file logging with configurable verbosity.
    """
    
    def __init__(self, config: TrainingConfiguration, log_dir: Optional[Path] = None):
        """
        Initialize training logger
        
        Args:
            config: Training configuration
            log_dir: Directory for log files (optional, auto-generated if None)
        """
        self.config = config
        self.logging_config = config.logging
        
        # Create log directory - default to TensorBoard runs directory for this run
        if log_dir is None:
            log_dir = config.get_tensorboard_log_dir()
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Log experiment info
        self.log_experiment_info()
        
        # Training state tracking
        self.current_epoch = 0
        self.current_step = 0
        self.start_time = None
        
    def _setup_logging(self):
        """Setup Python logging with file and console handlers"""
        # Create logger
        self.logger = logging.getLogger('openlipsync_training')
        self.logger.setLevel(getattr(logging, self.logging_config.log_level))
        
        # Clear existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter('%(levelname)-8s | %(message)s')
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(self.log_dir / 'training.log')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.logging_config.log_level))
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(self.log_dir / 'errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(error_handler)
    
    def log_experiment_info(self):
        """Log experiment configuration and system information"""
        self.info("="*60)
        self.info("OPENLIPSYNC TCN TRAINING")
        self.info("="*60)
        
        # Experiment details
        self.info(f"Experiment: {self.config.experiment.name}")
        self.info(f"Tags: {', '.join(self.config.experiment.tags)}")
        self.info(f"Notes: {self.config.experiment.notes}")
        
        # Model configuration
        self.info(f"\nModel Configuration:")
        self.info(f"  Architecture: {self.config.model.name}")
        self.info(f"  Visemes: {self.config.model.num_visemes}")
        self.info(f"  Layers: {self.config.model.layers}")
        self.info(f"  Channels: {self.config.model.channels}")
        self.info(f"  Dropout: {self.config.model.dropout}")
        
        # Audio configuration
        self.info(f"\nAudio Configuration:")
        self.info(f"  Sample rate: {self.config.audio.sample_rate} Hz")
        self.info(f"  Frame rate: {self.config.audio.fps:.1f} fps")
        self.info(f"  Mel bands: {self.config.audio.n_mels}")
        self.info(f"  Frequency range: {self.config.audio.fmin}-{self.config.audio.fmax} Hz")
        
        # Training configuration
        self.info(f"\nTraining Configuration:")
        self.info(f"  Batch size: {self.config.training.batch_size}")
        self.info(f"  Learning rate: {self.config.training.learning_rate}")
        self.info(f"  Optimizer: {self.config.training.optimizer}")
        self.info(f"  Scheduler: {self.config.training.scheduler}")
        self.info(f"  Max epochs: {self.config.training.max_epochs}")
        self.info(f"  Loss type: {self.config.training.loss_type}")
        
        # Hardware configuration
        self.info(f"\nHardware Configuration:")
        self.info(f"  Device: {self.config.hardware.device}")
        self.info(f"  Workers: {self.config.hardware.num_workers}")
        self.info(f"  Mixed precision: {self.config.training.mixed_precision}")
        
        # System information
        self.info(f"\nSystem Information:")
        self.info(f"  Python: {sys.version.split()[0]}")
        self.info(f"  PyTorch: {torch.__version__}")
        
        if torch.cuda.is_available():
            self.info(f"  CUDA: {torch.version.cuda}")
            self.info(f"  GPU: {torch.cuda.get_device_name()}")
            self.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.info("="*60)
        
        # Save configuration to file
        config_file = self.log_dir / 'config.json'
        with open(config_file, 'w') as f:
            # Convert config to serializable format
            config_dict = self._config_to_dict(self.config)
            json.dump(config_dict, f, indent=2)
    
    def _config_to_dict(self, config: TrainingConfiguration) -> Dict[str, Any]:
        """Convert configuration to JSON-serializable dictionary"""
        config_dict = {}
        
        for section_name in ['model', 'audio', 'training', 'data', 'evaluation', 
                            'hardware', 'logging', 'tensorboard', 'experiment']:
            section = getattr(config, section_name)
            config_dict[section_name] = section.__dict__
        
        return config_dict
    
    def start_training(self):
        """Mark the start of training"""
        self.start_time = datetime.now()
        self.info(f"Training started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def start_epoch(self, epoch: int, total_epochs: int):
        """Log the start of a new epoch"""
        self.current_epoch = epoch
        self.info(f"\nEpoch {epoch + 1}/{total_epochs}")
        self.info("-" * 40)
    
    def log_training_step(self, step: int, loss: float, learning_rate: float, 
                         batch_time: float, data_time: float):
        """
        Log training step information
        
        Args:
            step: Current training step
            loss: Training loss value
            learning_rate: Current learning rate
            batch_time: Time taken for this batch
            data_time: Time taken for data loading
        """
        self.current_step = step
        
        if step % self.logging_config.log_interval == 0:
            # Calculate timing statistics
            samples_per_sec = self.config.training.batch_size / batch_time
            
            self.info(
                f"Step {step:6d} | "
                f"Loss: {loss:.4f} | "
                f"LR: {learning_rate:.2e} | "
                f"Batch: {batch_time:.3f}s | "
                f"Data: {data_time:.3f}s | "
                f"Rate: {samples_per_sec:.1f} samples/s"
            )
    
    def log_epoch_summary(self, epoch: int, train_metrics: Dict[str, float], 
                         val_metrics: Dict[str, float], epoch_time: float):
        """
        Log end-of-epoch summary
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            epoch_time: Total epoch time
        """
        self.info(f"\nEpoch {epoch + 1} Summary:")
        self.info(f"  Time: {epoch_time:.1f}s")
        
        self.info("  Training:")
        for metric_name, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.info(f"    {metric_name}: {value:.4f}")
        
        self.info("  Validation:")
        for metric_name, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.info(f"    {metric_name}: {value:.4f}")
    
    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """Log when a new best model is found"""
        self.info(f"🏆 New best model at epoch {epoch + 1}! {metric_name}: {metric_value:.4f}")
    
    def log_early_stopping(self, epoch: int, patience_counter: int, patience: int):
        """Log early stopping information"""
        self.warning(
            f"Early stopping: {patience_counter}/{patience} epochs without improvement "
            f"(epoch {epoch + 1})"
        )
    
    def log_training_complete(self, total_epochs: int, best_epoch: int, 
                            best_metric: float, total_time: float):
        """Log training completion summary"""
        self.info("\n" + "="*60)
        self.info("TRAINING COMPLETED")
        self.info("="*60)
        
        self.info(f"Total epochs: {total_epochs}")
        self.info(f"Best epoch: {best_epoch + 1}")
        self.info(f"Best validation metric: {best_metric:.4f}")
        self.info(f"Total training time: {total_time:.1f}s ({total_time/3600:.2f}h)")
        
        if self.start_time:
            end_time = datetime.now()
            self.info(f"Training period: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} "
                     f"to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)


class TensorBoardLogger:
    """
    TensorBoard integration for visualizing training progress, metrics, and model behavior.
    """
    
    def __init__(self, config: TrainingConfiguration):
        """
        Initialize TensorBoard logger
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.tensorboard_config = config.tensorboard
        
        if not self.tensorboard_config.enabled:
            self.writer = None
            return
        
        # Create TensorBoard log directory
        log_dir = config.get_tensorboard_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create SummaryWriter
        self.writer = SummaryWriter(log_dir=str(log_dir))
        
        # Log configuration as text
        self._log_configuration()
        
        print(f"TensorBoard logging enabled: {log_dir}")
        print(f"View with: tensorboard --logdir {log_dir.parent}")
    
    def _log_configuration(self):
        """Log configuration as text to TensorBoard"""
        if not self.writer:
            return
        
        config_text = f"""
# OpenLipSync TCN Training Configuration

## Experiment
- **Name**: {self.config.experiment.name}
- **Tags**: {', '.join(self.config.experiment.tags)}
- **Notes**: {self.config.experiment.notes}

## Model
- **Architecture**: {self.config.model.name}
- **Visemes**: {self.config.model.num_visemes}
- **Layers**: {self.config.model.layers}
- **Channels**: {self.config.model.channels}
- **Kernel Size**: {self.config.model.kernel_size}
- **Dropout**: {self.config.model.dropout}

## Audio Processing
- **Sample Rate**: {self.config.audio.sample_rate} Hz
- **Frame Rate**: {self.config.audio.fps:.1f} fps
- **Mel Bands**: {self.config.audio.n_mels}
- **Frequency Range**: {self.config.audio.fmin}-{self.config.audio.fmax} Hz

## Training
- **Batch Size**: {self.config.training.batch_size}
- **Learning Rate**: {self.config.training.learning_rate}
- **Optimizer**: {self.config.training.optimizer}
- **Scheduler**: {self.config.training.scheduler}
- **Loss**: {self.config.training.loss_type}
- **Max Epochs**: {self.config.training.max_epochs}
        """
        
        self.writer.add_text("Configuration", config_text, 0)
    
    def log_scalars(self, step: int, **scalars):
        """
        Log scalar values to TensorBoard (with interval filtering for step-level logging)
        
        Args:
            step: Global step number
            **scalars: Named scalar values to log
        """
        if not self.writer or not self.tensorboard_config.log_scalars:
            return
        
        if step % self.tensorboard_config.scalar_log_interval == 0:
            for name, value in scalars.items():
                if isinstance(value, (int, float, np.number)):
                    self.writer.add_scalar(name, value, step)
    
    def log_scalars_immediate(self, step: int, **scalars):
        """
        Log scalar values to TensorBoard immediately (no interval filtering for epoch-level logging)
        
        Args:
            step: Global step number
            **scalars: Named scalar values to log
        """
        if not self.writer or not self.tensorboard_config.log_scalars:
            return
        
        for name, value in scalars.items():
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(name, value, step)
    
    def log_histograms(self, step: int, model: torch.nn.Module):
        """
        Log model parameter histograms to TensorBoard (with interval filtering for step-level logging)
        
        Args:
            step: Global step number
            model: PyTorch model
        """
        if (not self.writer or not self.tensorboard_config.log_histograms or
            step % self.tensorboard_config.histogram_log_interval != 0):
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f"model/parameters/{name}", param.data, step)
                self.writer.add_histogram(f"model/gradients/{name}", param.grad.data, step)
    
    def log_histograms_immediate(self, step: int, model: torch.nn.Module):
        """
        Log model parameter histograms to TensorBoard immediately (no interval filtering for epoch-level logging)
        
        Args:
            step: Global step number
            model: PyTorch model
        """
        if not self.writer or not self.tensorboard_config.log_histograms:
            return
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f"model/parameters/{name}", param.data, step)
                self.writer.add_histogram(f"model/gradients/{name}", param.grad.data, step)
    
    def log_confusion_matrix(self, step: int, confusion_matrix: np.ndarray, 
                           class_names: Optional[list] = None):
        """
        Log confusion matrix as image to TensorBoard (with interval filtering for step-level logging)
        
        Args:
            step: Global step number
            confusion_matrix: Confusion matrix array
            class_names: Names of classes for labeling
        """
        if (not self.writer or not self.tensorboard_config.log_images or
            step % self.tensorboard_config.image_log_interval != 0):
            return
        
        self._log_confusion_matrix_impl(step, confusion_matrix, class_names)
    
    def log_confusion_matrix_immediate(self, step: int, confusion_matrix: np.ndarray, 
                                     class_names: Optional[list] = None):
        """
        Log confusion matrix as image to TensorBoard immediately (no interval filtering for epoch-level logging)
        
        Args:
            step: Global step number
            confusion_matrix: Confusion matrix array
            class_names: Names of classes for labeling
        """
        if not self.writer or not self.tensorboard_config.log_images:
            return
        
        self._log_confusion_matrix_impl(step, confusion_matrix, class_names)
    
    def _log_confusion_matrix_impl(self, step: int, confusion_matrix: np.ndarray, 
                                 class_names: Optional[list] = None):
        """
        Internal implementation for logging confusion matrix
        
        Args:
            step: Global step number
            confusion_matrix: Confusion matrix array
            class_names: Names of classes for labeling
        """
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot confusion matrix
            im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Set labels
            if class_names:
                tick_marks = np.arange(len(class_names))
                ax.set_xticks(tick_marks)
                ax.set_yticks(tick_marks)
                ax.set_xticklabels(class_names, rotation=45, ha="right")
                ax.set_yticklabels(class_names)
            
            # Add text annotations
            thresh = confusion_matrix.max() / 2.
            for i in range(confusion_matrix.shape[0]):
                for j in range(confusion_matrix.shape[1]):
                    ax.text(j, i, format(confusion_matrix[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if confusion_matrix[i, j] > thresh else "black")
            
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title('Confusion Matrix')
            
            # Convert to tensor and log
            fig.canvas.draw()
            image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Convert to CHW format for TensorBoard
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            self.writer.add_image('metrics/confusion_matrix', image_tensor, step)
            
            plt.close(fig)
            
        except ImportError:
            # matplotlib not available, skip confusion matrix logging
            pass
        except Exception as e:
            print(f"Warning: Could not log confusion matrix: {e}")
    
    def log_audio_sample(self, step: int, audio: torch.Tensor, sample_rate: int, 
                        tag: str = "audio_sample"):
        """
        Log audio sample to TensorBoard
        
        Args:
            step: Global step number
            audio: Audio tensor, shape (samples,) or (1, samples)
            sample_rate: Sample rate of audio
            tag: Tag for the audio sample
        """
        if (not self.writer or not self.tensorboard_config.log_audio or
            step % self.tensorboard_config.image_log_interval != 0):
            return
        
        # Ensure audio is 1D
        if audio.dim() > 1:
            audio = audio.squeeze()
        
        # Normalize audio to [-1, 1] range
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max()
        
        self.writer.add_audio(f"audio/{tag}", audio.unsqueeze(0), step, sample_rate=sample_rate)
    
    def log_learning_rate(self, step: int, optimizer: torch.optim.Optimizer):
        """
        Log current learning rate to TensorBoard
        
        Args:
            step: Global step number
            optimizer: PyTorch optimizer
        """
        if not self.writer:
            return
        
        # Log learning rate under training group to reduce top-level groups
        for param_group_idx, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            if param_group_idx == 0:  # Most common case - single param group
                self.writer.add_scalar('training/learning_rate', lr, step)
            else:  # Multiple param groups (rare)
                self.writer.add_scalar(f'training/learning_rate_group_{param_group_idx}', lr, step)
    
    def flush(self):
        """Flush TensorBoard writer"""
        if self.writer:
            self.writer.flush()
    
    def close(self):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()


class CombinedLogger:
    """
    Combines TrainingLogger and TensorBoardLogger for comprehensive logging.
    """
    
    def __init__(self, config: TrainingConfiguration, log_dir: Optional[Path] = None):
        """
        Initialize combined logger
        
        Args:
            config: Training configuration
            log_dir: Directory for log files
        """
        # Ensure a single consistent run directory/name across file logs and TensorBoard
        if log_dir is None:
            # Fix the run_name now (once) using the configured format to avoid drift
            if not config.tensorboard.run_name:
                config.tensorboard.run_name = config.get_run_name()
            log_dir = config.get_tensorboard_log_dir()
        
        self.training_logger = TrainingLogger(config, log_dir)
        self.tensorboard_logger = TensorBoardLogger(config)
        
        # Store log directory
        self.log_dir = self.training_logger.log_dir
    
    def _is_scalar_value(self, value: Any) -> bool:
        """Check if a value can be logged as a scalar to TensorBoard"""
        if isinstance(value, (int, float, np.number)):
            return True
        if hasattr(value, 'item') and callable(getattr(value, 'item')):  # torch.Tensor with single value
            try:
                value.item()
                return True
            except (ValueError, RuntimeError):
                return False
        return False
    
    def _extract_scalar_value(self, value: Any) -> Union[int, float]:
        """Extract scalar value from various numeric types"""
        if isinstance(value, (int, float, np.number)):
            return float(value)
        if hasattr(value, 'item') and callable(getattr(value, 'item')):
            return float(value.item())
        return float(value)
    
    def start_training(self):
        """Start training logging"""
        self.training_logger.start_training()
    
    def start_epoch(self, epoch: int, total_epochs: int):
        """Start epoch logging"""
        self.training_logger.start_epoch(epoch, total_epochs)
    
    def log_training_step(self, step: int, loss: float, learning_rate: float,
                         batch_time: float, data_time: float, 
                         optimizer: Optional[torch.optim.Optimizer] = None):
        """Log training step to both loggers"""
        # Console/file logging
        self.training_logger.log_training_step(step, loss, learning_rate, batch_time, data_time)
        
        # TensorBoard logging - group related metrics together
        self.tensorboard_logger.log_scalars(
            step,
            **{
                'metrics/train_loss': loss,
                'training/learning_rate': learning_rate,
                'training/batch_time': batch_time,
                'training/data_time': data_time
            }
        )
    
    def log_epoch_metrics(self, epoch: int, step: int, train_metrics: Dict[str, Any],
                         val_metrics: Dict[str, Any], epoch_time: float,
                         model: Optional[torch.nn.Module] = None):
        """Log epoch metrics to both loggers"""
        # Console/file logging
        self.training_logger.log_epoch_summary(epoch, train_metrics, val_metrics, epoch_time)
        
        # TensorBoard logging - group metrics by type rather than train/val
        all_scalars = {}
        
        # Group training metrics under metrics/train/
        for k, v in train_metrics.items():
            if self._is_scalar_value(v):
                all_scalars[f"metrics/train/{k}"] = self._extract_scalar_value(v)
        
        # Group validation metrics under metrics/val/  
        for k, v in val_metrics.items():
            if self._is_scalar_value(v):
                all_scalars[f"metrics/val/{k}"] = self._extract_scalar_value(v)
        
        # Add epoch timing to training group
        all_scalars["training/epoch_time"] = epoch_time
        
        self.tensorboard_logger.log_scalars_immediate(step, **all_scalars)
        
        # Log confusion matrix if available
        if 'confusion_matrix' in val_metrics:
            class_names = None
            # Try to get class names from config
            if hasattr(self.training_logger.config, 'phoneme_to_viseme_mapping'):
                # Create class names from viseme mapping
                class_names = [f"V{i}" for i in range(self.training_logger.config.model.num_visemes)]
            
            self.tensorboard_logger.log_confusion_matrix_immediate(
                step, val_metrics['confusion_matrix'], class_names
            )
        
        # Log model histograms
        if model:
            self.tensorboard_logger.log_histograms_immediate(step, model)
    
    def log_best_model(self, epoch: int, metric_name: str, metric_value: float):
        """Log best model found"""
        self.training_logger.log_best_model(epoch, metric_name, metric_value)
    
    def log_early_stopping(self, epoch: int, patience_counter: int, patience: int):
        """Log early stopping"""
        self.training_logger.log_early_stopping(epoch, patience_counter, patience)
    
    def log_training_complete(self, total_epochs: int, best_epoch: int,
                            best_metric: float, total_time: float):
        """Log training completion"""
        self.training_logger.log_training_complete(total_epochs, best_epoch, best_metric, total_time)
    
    def info(self, message: str):
        """Log info message"""
        self.training_logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.training_logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.training_logger.error(message)
    
    def flush(self):
        """Flush all loggers"""
        self.tensorboard_logger.flush()
    
    def close(self):
        """Close all loggers"""
        self.tensorboard_logger.close()


if __name__ == "__main__":
    # Example usage and testing
    from .config import load_config
    
    # Load configuration
    config = load_config("../recipes/tcn_config.toml")
    
    # Test logger
    logger = CombinedLogger(config)
    
    # Test logging functions
    logger.start_training()
    logger.start_epoch(0, 10)
    
    # Simulate training steps
    for step in range(5):
        logger.log_training_step(
            step=step,
            loss=1.0 - step * 0.1,
            learning_rate=3e-4,
            batch_time=0.5,
            data_time=0.1
        )
    
    # Test epoch metrics
    train_metrics = {'loss': 0.5, 'frame_accuracy': 85.2}
    val_metrics = {'loss': 0.6, 'frame_accuracy': 82.1}
    
    logger.log_epoch_metrics(0, 5, train_metrics, val_metrics, 30.0)
    logger.log_best_model(0, 'val_loss', 0.6)
    
    logger.info("✅ Logger test completed successfully!")
    
    # Close loggers
    logger.close()
