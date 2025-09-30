#!/usr/bin/env python3
"""
Main Training Script for OpenLipSync TCN Model

This script orchestrates the entire training process for the Temporal Convolutional
Network (TCN) used for audio-to-viseme mapping in lip synchronization.

Usage:
    python train.py --config recipes/tcn_config.toml [options]
    
Features:
    - Configuration-driven training
    - Logging and monitoring
    - Automatic checkpointing and resuming
    - Real-time metrics and TensorBoard integration
    - Early stopping and best model saving
    - Multi-GPU support (if available)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Import our modules
# …existing imports…
from modules.mfa_runner import ensure_aligned
from modules.config import load_config, validate_environment, TrainingConfiguration
from modules.data_pipeline import create_data_loaders
from modules.tcn_model import create_model, print_model_summary
from modules.training_utils import (
    MetricsTracker, LossFunction, EarlyStopping, ModelCheckpoint,
    create_optimizer, create_scheduler, compute_class_counts
)
from modules.logger import CombinedLogger


class TCNTrainer:
    """
    Main trainer class that orchestrates the TCN training process.
    
    Handles training loop, validation, checkpointing, and logging with
    clear error handling and progress monitoring.
    """
    
    def __init__(self, config: TrainingConfiguration, resume_from: Optional[str] = None, data_root: Optional[str] = None):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration
            resume_from: Path to checkpoint to resume from (optional)
        """
        self.config = config
        
        # Initialize logging first
        self.logger = CombinedLogger(config)
        self.logger.info("Initializing TCN Trainer...")
        
        # Smart device selection with automatic fallback
        requested_device = config.hardware.device
        if requested_device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available, falling back to CPU")
            self.device = torch.device("cpu")
            self.device_type = "cpu"
        else:
            self.device = torch.device(requested_device)
            self.device_type = "cuda" if self.device.type == "cuda" else "cpu"
        
        # Adjust config based on actual device
        self.use_mixed_precision = (config.training.mixed_precision and 
                                   self.device_type == "cuda")
        self.pin_memory = (config.hardware.pin_memory and 
                          self.device_type == "cuda")
        
        self.logger.info(f"Using device: {self.device} (type: {self.device_type})")
        if not self.use_mixed_precision and config.training.mixed_precision:
            self.logger.info("Mixed precision disabled (requires CUDA)")
        if not self.pin_memory and config.hardware.pin_memory:
            self.logger.info("Pin memory disabled (requires CUDA)")
        
        # Create data loaders
        self.logger.info("Setting up data pipeline...")
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            config, data_root=data_root, pin_memory=self.pin_memory)
        
        # Create model
        self.logger.info("Creating TCN model...")
        self.model = create_model(config)
        print_model_summary(self.model)
        
        # Initialize training components
        self._initialize_training_components()
        
        # Setup checkpointing - use TensorBoard runs directory for consistency
        checkpoint_dir = config.get_tensorboard_log_dir() / "checkpoints"
        self.checkpoint_manager = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            config=config
        )
        
        # Initialize training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = float('inf')
        self.best_epoch = 0
        
        # Resume from checkpoint if specified
        if resume_from:
            self._resume_from_checkpoint(resume_from)
        
        # If we're maximizing F1 for early stopping/best model, initialize best to -inf
        try:
            if self.config.training.early_stopping_metric == "val_f1":
                self.best_val_metric = float('-inf')
        except Exception:
            pass

        self.logger.info("Trainer initialization completed!")
    
    def _initialize_training_components(self):
        """Initialize optimizer, scheduler, loss function, and other training components"""
        # Create optimizer
        self.optimizer = create_optimizer(self.model, self.config)
        self.logger.info(f"Optimizer: {self.config.training.optimizer}")
        
        # Create learning rate scheduler
        steps_per_epoch = len(self.train_loader)
        self.scheduler = create_scheduler(self.optimizer, self.config, steps_per_epoch)
        if self.scheduler:
            self.logger.info(f"Scheduler: {self.config.training.scheduler}")
        
        # Create loss function (optionally with class weighting)
        class_counts = None
        if getattr(self.config.training, "class_weighting", False):
            from modules.training_utils import load_class_counts_cache, save_class_counts_cache
            cached = load_class_counts_cache(self.config)
            if cached is not None:
                class_counts = cached
                self.logger.info("Loaded cached class counts for class weighting.")
            else:
                self.logger.info("Computing class counts for class weighting... (one pass over training data)")
                class_counts = compute_class_counts(
                    train_loader=self.train_loader,
                    num_classes=self.config.model.num_visemes,
                    multi_label=bool(getattr(self.config.training, "multi_label", False)),
                    mask_padded=bool(getattr(self.config.training, "mask_padded_frames", False))
                )
                # Basic validation before caching
                if not isinstance(class_counts, dict) or len(class_counts) != self.config.model.num_visemes:
                    raise RuntimeError("Class weighting requested but class count computation returned invalid result.")
                if sum(int(v) for v in class_counts.values()) <= 0:
                    raise RuntimeError("Class weighting requested but computed zero total frames across classes.")
                save_class_counts_cache(self.config, class_counts)
            self.logger.info(f"Class counts: {class_counts}")
        self.loss_function = LossFunction(self.config, class_counts=class_counts)
        
        # Initialize mixed precision training
        self.scaler = GradScaler() if self.use_mixed_precision else None
        if self.scaler:
            self.logger.info("Mixed precision training enabled")
        
        # Initialize metrics tracking
        self.train_metrics = MetricsTracker(self.config)
        self.val_metrics = MetricsTracker(self.config)
        
        # Initialize early stopping
        metric_name = self.config.training.early_stopping_metric
        minimize = (metric_name == "val_loss")
        self.early_stopping = EarlyStopping(
            patience=self.config.training.early_stopping_patience,
            metric_name=metric_name,
            minimize=minimize
        )

    def _apply_viseme_crossfade(self, predictions: torch.Tensor) -> torch.Tensor:
        """Optionally apply temporal cross-fade smoothing on per-frame class probabilities for metrics.

        Uses a SINGLE unified configuration value: training.target_crossfade_ms.
        If set to 0, no cross-fade is applied. This mirrors the boundary softening used for targets.
        """
        try:
            ms = int(getattr(self.config.training, "target_crossfade_ms", 0))
            if ms <= 0:
                return predictions
            # Convert half-width milliseconds to frames (round to nearest)
            half_width_frames = int(round((ms / 1000.0) * float(self.config.audio.fps)))
            if half_width_frames <= 0:
                return predictions
            with torch.no_grad():
                # Convert logits -> probabilities suitable for current training mode
                if self.config.training.multi_label:
                    probs = torch.sigmoid(predictions.detach())  # (B, T, C)
                else:
                    probs = torch.softmax(predictions.detach(), dim=-1)  # (B, T, C)
                x = probs.transpose(1, 2)  # (B, C, T)
                pad = (half_width_frames, half_width_frames)
                x = F.pad(x, pad=pad, mode="replicate")  # pad time dimension
                kernel = 2 * half_width_frames + 1
                smoothed = F.avg_pool1d(x, kernel_size=kernel, stride=1)  # (B, C, T)
                smoothed = smoothed.transpose(1, 2)  # (B, T, C)
                return smoothed
        except Exception:
            # In case of any unexpected issue, fall back to original predictions
            return predictions
    
    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from a saved checkpoint"""
        self.logger.info(f"Resuming training from: {checkpoint_path}")
        
        try:
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch'] + 1
            self.global_step = checkpoint['step']
            
            # Load best model tracking
            if 'metrics' in checkpoint:
                val_loss = checkpoint['metrics'].get('val_loss', float('inf'))
                if val_loss < self.best_val_metric:
                    self.best_val_metric = val_loss
                    self.best_epoch = checkpoint['epoch']
            
            self.logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
            
        except Exception as error:
            self.logger.error(f"Failed to resume from checkpoint: {error}")
            raise
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dict[str, float]: Training metrics for the epoch
        """
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start_time = time.time()
        batch_times = []
        data_times = []
        total_train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Move data to device
            audio_features = batch['features'].to(self.device)  # (batch, time, n_mels)
            viseme_targets = batch['targets'].to(self.device)   # (batch, time)
            sequence_lengths = batch['lengths'].to(self.device) # (batch,)
            
            data_time = time.time() - batch_start_time
            data_times.append(data_time)
            
            # Forward pass with optional mixed precision
            if self.scaler:
                with autocast(device_type=self.device_type):
                    predictions = self.model(audio_features, sequence_lengths)
                    loss = self.loss_function(predictions, viseme_targets, sequence_lengths)
            else:
                predictions = self.model(audio_features, sequence_lengths)
                loss = self.loss_function(predictions, viseme_targets, sequence_lengths)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Update scheduler (if step-based)
            if self.scheduler and self.config.training.scheduler == "onecycle":
                self.scheduler.step()
            
            # Update metrics (optionally apply cross-fade smoothing for metrics only)
            preds_for_metrics = self._apply_viseme_crossfade(predictions).detach()
            self.train_metrics.update(preds_for_metrics, viseme_targets, sequence_lengths)
            total_train_loss += loss.item()
            num_batches += 1
            
            # Calculate timing
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Log training step
            current_lr = self.optimizer.param_groups[0]['lr']
            # Compute done/total samples for the current epoch
            samples_done = None
            samples_total = None
            try:
                steps_per_epoch = len(self.train_loader)
                batch_idx_in_epoch = (self.global_step % steps_per_epoch)
                samples_done = (batch_idx_in_epoch + 1) * self.config.training.batch_size
                samples_total = steps_per_epoch * self.config.training.batch_size
            except Exception:
                pass
            self.logger.log_training_step(
                step=self.global_step,
                loss=loss.item(),
                learning_rate=current_lr,
                batch_time=batch_time,
                data_time=data_time,
                optimizer=self.optimizer,
                samples_done=samples_done,
                samples_total=samples_total
            )
            
            # Save checkpoint periodically
            if (self.global_step > 0 and 
                self.global_step % self.config.logging.save_interval == 0):
                self._save_checkpoint(is_best=False)
            
            self.global_step += 1
        
        # Update scheduler (if epoch-based)
        if self.scheduler and self.config.training.scheduler in ["cosine", "constant"]:
            self.scheduler.step()
        
        # Compute epoch metrics
        epoch_metrics = self.train_metrics.compute()
        # Also track average training loss for the epoch for checkpoint metadata
        epoch_metrics['loss'] = (total_train_loss / num_batches) if num_batches > 0 else float('nan')
        epoch_time = time.time() - epoch_start_time
        
        # Add timing metrics
        epoch_metrics['epoch_time'] = epoch_time
        epoch_metrics['avg_batch_time'] = sum(batch_times) / len(batch_times)
        epoch_metrics['avg_data_time'] = sum(data_times) / len(data_times)
        
        return epoch_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch
        
        Returns:
            Dict[str, float]: Validation metrics for the epoch
        """
        self.model.eval()
        self.val_metrics.reset()
        
        total_val_loss = 0.0
        num_batches = 0
        
        amp_ctx = autocast(device_type=self.device_type) if self.device_type == "cuda" else nullcontext()
        with torch.no_grad():
            for batch in self.val_loader:
                # Move data to device
                audio_features = batch['features'].to(self.device)
                viseme_targets = batch['targets'].to(self.device)
                sequence_lengths = batch['lengths'].to(self.device)
                
                # Forward pass
                with (autocast(device_type=self.device_type) if self.device_type == "cuda" else nullcontext()):
                    predictions = self.model(audio_features, sequence_lengths)
                    loss = self.loss_function(predictions, viseme_targets, sequence_lengths)
                
                # Update metrics (optionally apply cross-fade smoothing for metrics only)
                preds_for_metrics = self._apply_viseme_crossfade(predictions).detach()
                self.val_metrics.update(preds_for_metrics, viseme_targets, sequence_lengths)
                total_val_loss += loss.item()
                num_batches += 1
        
        # Compute validation metrics
        val_metrics = self.val_metrics.compute()
        val_metrics['loss'] = (total_val_loss / num_batches) if num_batches > 0 else float('nan')
        
        return val_metrics
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        try:
            # Get current metrics for saving
            current_metrics = {
                'train_loss': getattr(self, '_last_train_loss', 0.0),
                'val_loss': getattr(self, '_last_val_loss', 0.0)
            }
            
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=self.current_epoch,
                step=self.global_step,
                metrics=current_metrics,
                is_best=is_best
            )
            
            if is_best:
                self.logger.info(f"💾 Best model saved: {checkpoint_path}")
            
        except Exception as error:
            self.logger.error(f"Failed to save checkpoint: {error}")
    
    def train(self):
        """Main training loop"""
        self.logger.start_training()
        training_start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.training.max_epochs):
                self.current_epoch = epoch
                
                # Start epoch logging
                self.logger.start_epoch(epoch, self.config.training.max_epochs)
                epoch_start_time = time.time()
                
                # Training phase
                train_metrics = self.train_epoch()
                self._last_train_loss = train_metrics.get('loss', 0.0)
                
                # Validation phase
                val_metrics = self.validate_epoch()
                self._last_val_loss = val_metrics.get('loss', 0.0)
                
                # Calculate epoch time
                epoch_time = time.time() - epoch_start_time
                
                # Log epoch results
                self.logger.log_epoch_metrics(
                    epoch=epoch,
                    step=self.global_step,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    epoch_time=epoch_time,
                    model=self.model
                )
                
                # Check for best model
                metric_name = self.config.training.early_stopping_metric
                if metric_name == "val_loss":
                    current_val_metric = val_metrics["loss"]
                    is_best = current_val_metric < self.best_val_metric
                elif metric_name == "val_f1":
                    current_val_metric = val_metrics["macro_f1"]
                    is_best = current_val_metric > self.best_val_metric
                else:
                    raise ValueError(f"Unsupported early_stopping_metric: {metric_name}")
                
                if is_best:
                    self.best_val_metric = current_val_metric
                    self.best_epoch = epoch
                    self.logger.log_best_model(epoch, self.config.training.early_stopping_metric, current_val_metric)
                
                # Save checkpoint
                self._save_checkpoint(is_best=is_best)
                
                # Check early stopping
                should_stop = self.early_stopping.update(current_val_metric, epoch)
                
                if self.early_stopping.patience_counter > 0:
                    self.logger.log_early_stopping(
                        epoch, 
                        self.early_stopping.patience_counter,
                        self.early_stopping.patience
                    )
                
                if should_stop:
                    self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
                # Flush logs
                self.logger.flush()
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        except Exception as error:
            self.logger.error(f"Training failed with error: {error}")
            raise
        
        finally:
            # Training completion summary
            total_training_time = time.time() - training_start_time
            self.logger.log_training_complete(
                total_epochs=self.current_epoch + 1,
                best_epoch=self.best_epoch,
                best_metric=self.best_val_metric,
                total_time=total_training_time
            )
            
            # Close loggers
            self.logger.close()
    
    def test(self) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Returns:
            Dict[str, float]: Test metrics
        """
        self.logger.info("Starting test evaluation...")
        
        # Load best model if available
        best_model_path = self.checkpoint_manager.best_checkpoint_path
        if best_model_path and best_model_path.exists():
            self.logger.info(f"Loading best model from: {best_model_path}")
            checkpoint = self.checkpoint_manager.load_checkpoint(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate on test set
        self.model.eval()
        test_metrics = MetricsTracker(self.config)
        test_metrics.reset()
        
        total_test_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                audio_features = batch['features'].to(self.device)
                viseme_targets = batch['targets'].to(self.device)
                sequence_lengths = batch['lengths'].to(self.device)
                
                with (autocast(device_type=self.device_type) if self.device_type == "cuda" else nullcontext()):
                    predictions = self.model(audio_features, sequence_lengths)
                    loss = self.loss_function(predictions, viseme_targets, sequence_lengths)
                
                preds_for_metrics = self._apply_viseme_crossfade(predictions)
                test_metrics.update(preds_for_metrics, viseme_targets, sequence_lengths)
                total_test_loss += loss.item()
                num_batches += 1
        
        # Compute final test metrics
        final_test_metrics = test_metrics.compute()
        final_test_metrics['loss'] = (total_test_loss / num_batches) if num_batches > 0 else float('nan')
        
        # Log test results
        self.logger.info("\nTest Results:")
        for metric_name, value in final_test_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {metric_name}: {value:.4f}")
        
        # Test model latency if requested
        if self.config.evaluation.compute_latency:
            self.logger.info("\nTesting model latency...")
            latency_ms = self.model.get_latency_ms(self.config.evaluation.target_hardware)
            rtf = self.model.get_real_time_factor(self.config.evaluation.target_hardware)
            
            self.logger.info(f"Latency ({self.config.evaluation.target_hardware}): {latency_ms:.2f} ms/second")
            self.logger.info(f"Real-time factor: {rtf:.3f}")
            
            if rtf < 1.0:
                self.logger.info("Model can run faster than real-time.")
            else:
                self.logger.warning("Model may struggle with real-time processing.")
        
        return final_test_metrics


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train OpenLipSync TCN model for audio-to-viseme mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic training
    python train.py --config recipes/tcn_config.toml
    
    # Resume from checkpoint
    python train.py --config recipes/tcn_config.toml --resume checkpoints/best_model.pt
    
    # Test only (no training)
    python train.py --config recipes/tcn_config.toml --test-only --resume checkpoints/best_model.pt
    
    # Override data directory
    python train.py --config recipes/tcn_config.toml --data-root /path/to/librispeech
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to training configuration TOML file'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to checkpoint file to resume training from'
    )
    
    parser.add_argument(
        '--test-only',
        action='store_true',
        help='Skip training and only run test evaluation'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        help='Root directory for dataset (overrides default)'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for training script"""
    # Parse arguments
    args = parse_arguments()
    
    # Load and validate configuration
    try:
        config = load_config(args.config)
        validate_environment(config)
        subsets = config.data.splits
        ensure_aligned(subsets)
    except Exception as error:
        print(f"Configuration error: {error}")
        sys.exit(1)
    
    # Set data root if specified
    if args.data_root:
        # This would need to be handled in data pipeline
        print(f"Using data root: {args.data_root}")
    
    try:
        # Create trainer
        trainer = TCNTrainer(config, resume_from=args.resume, data_root=args.data_root)
        
        if args.test_only:
            # Test only mode
            if not args.resume:
                print("--test-only requires --resume to specify model checkpoint")
                sys.exit(1)
            
            test_metrics = trainer.test()
            print("Test evaluation completed.")
        
        else:
            # Full training mode
            trainer.train()
            
            # Run test evaluation after training
            test_metrics = trainer.test()
            print("Training and testing completed.")
    
    except Exception as error:
        print(f"Training failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
