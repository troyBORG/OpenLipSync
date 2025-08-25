"""
Training Utilities for OpenLipSync TCN Training

Provides metrics computation, checkpointing, early stopping, loss functions,
and other utilities for model training.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict, deque
import time
import hashlib
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR, ConstantLR

from .config import TrainingConfiguration


def ensure_probabilities(predictions: torch.Tensor, multi_label: bool = False) -> torch.Tensor:
    """Convert model outputs to per-frame probabilities if needed.

    If `predictions` already look like probabilities (values in [0,1] and rows sum ~ 1),
    return as-is. Otherwise apply softmax along the class dimension.
    """
    if multi_label:
        return torch.sigmoid(predictions)
    with torch.no_grad():
        x = predictions.detach()
        try:
            if torch.all(x >= 0) and torch.all(x <= 1):
                row_sums = x.sum(dim=-1, keepdim=True)
                if torch.all(torch.isfinite(row_sums)) and torch.max(torch.abs(row_sums - 1.0)) < 1e-3:
                    return x
        except Exception:
            pass
    return torch.softmax(predictions, dim=-1)


class FrameAccuracy:
    """
    Computes frame-level classification accuracy.
    Simple metric: percentage of frames classified correctly.
    """
    
    def __init__(self, overlap_enabled: bool = False, overlap_threshold: float = 0.0, multi_label: bool = False):
        self.overlap_enabled = overlap_enabled
        self.overlap_threshold = float(overlap_threshold)
        self.multi_label = multi_label
        self.reset()
    
    def reset(self):
        """Reset accumulated statistics"""
        self.correct_frames = 0
        self.total_frames = 0
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, 
               sequence_lengths: Optional[torch.Tensor] = None):
        """
        Update accuracy with new predictions
        
        Args:
            predictions: Model predictions, shape (batch, time, num_classes)
            targets: Ground truth targets, shape (batch, time)
            sequence_lengths: Actual sequence lengths, shape (batch,)
        """
        # Use multilabel/overlap semantics when either overlap is enabled
        # or we are training in multi-label mode. This avoids mismatches
        # when targets are (B,T,C) but overlap is disabled.
        if self.overlap_enabled or self.multi_label:
            # Guard: if predictions don't look like (B, T, C) with C > 1, fall back to single-label
            if predictions.ndim != 3 or predictions.shape[-1] <= 1:
                predicted_classes = torch.argmax(predictions, dim=-1)
                if sequence_lengths is not None:
                    batch_size = predictions.shape[0]
                    for batch_idx in range(batch_size):
                        seq_len = int(sequence_lengths[batch_idx].item())
                        batch_predictions = predicted_classes[batch_idx, :seq_len]
                        batch_targets = targets[batch_idx, :seq_len]
                        self.correct_frames += (batch_predictions == batch_targets).sum().item()
                        self.total_frames += seq_len
                else:
                    self.correct_frames += (predicted_classes == targets).sum().item()
                    self.total_frames += targets.numel()
                return
            # Use multilabel correctness: a frame counts as correct if any true label is among active predictions
            probs = ensure_probabilities(predictions, multi_label=self.multi_label)  # (B, T, C)
            active = probs >= self.overlap_threshold  # bool (B, T, C)
            # Ensure at least one active per frame by forcing argmax to active where needed
            argmax_idx = torch.argmax(probs, dim=-1, keepdim=True)  # (B, T, 1)
            none_active_flat = ~active.any(dim=-1)  # (B, T)
            one_hot = F.one_hot(argmax_idx.squeeze(-1), num_classes=active.shape[-1]).to(dtype=torch.bool)  # (B, T, C)
            force_active = (none_active_flat.unsqueeze(-1) & one_hot)
            active = active | force_active
            # Build true-active mask from targets
            if targets.dim() == 3:
                true_active = (targets > 0.5)  # (B, T, C)
            else:
                true_active = F.one_hot(targets, num_classes=active.shape[-1]).to(dtype=torch.bool)
            # Ensure at least one true label per frame by forcing argmax of targets if none (rare with soft labels)
            if targets.dim() == 3:
                none_true = ~true_active.any(dim=-1)  # (B, T)
                targ_argmax = torch.argmax(targets, dim=-1, keepdim=True)
                one_hot_true = F.one_hot(targ_argmax.squeeze(-1), num_classes=active.shape[-1]).to(dtype=torch.bool)
                force_true = (none_true.unsqueeze(-1) & one_hot_true)
                true_active = true_active | force_true
            # Per-frame correctness
            if sequence_lengths is not None:
                batch_size = predictions.shape[0]
                for batch_idx in range(batch_size):
                    seq_len = int(sequence_lengths[batch_idx].item())
                    act = active[batch_idx, :seq_len, :]
                    tru = true_active[batch_idx, :seq_len, :]
                    correct_flags = (act & tru).any(dim=-1)
                    self.correct_frames += correct_flags.sum().item()
                    self.total_frames += seq_len
            else:
                correct_flags = (active & true_active).any(dim=-1)
                self.correct_frames += correct_flags.sum().item()
                self.total_frames += int(correct_flags.numel())
            return
        # Get predicted classes (single-label)
        predicted_classes = torch.argmax(predictions, dim=-1)  # (batch, time)
        
        if sequence_lengths is not None:
            # Only count frames within actual sequence length
            batch_size = predictions.shape[0]
            for batch_idx in range(batch_size):
                seq_len = sequence_lengths[batch_idx].item()
                batch_predictions = predicted_classes[batch_idx, :seq_len]
                batch_targets = targets[batch_idx, :seq_len]
                
                self.correct_frames += (batch_predictions == batch_targets).sum().item()
                self.total_frames += seq_len
        else:
            # Count all frames
            self.correct_frames += (predicted_classes == targets).sum().item()
            self.total_frames += targets.numel()
    
    def compute(self) -> float:
        """
        Compute final accuracy
        
        Returns:
            float: Frame accuracy as percentage
        """
        if self.total_frames == 0:
            return 0.0
        return (self.correct_frames / self.total_frames) * 100.0


class MacroF1Score:
    """
    Computes macro-averaged F1 score across all viseme classes.
    Better than accuracy for imbalanced datasets.
    """
    
    def __init__(self, num_classes: int, overlap_enabled: bool = False, overlap_threshold: float = 0.0, multi_label: bool = False):
        self.num_classes = num_classes
        self.overlap_enabled = overlap_enabled
        self.overlap_threshold = float(overlap_threshold)
        self.multi_label = multi_label
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets"""
        self.all_predictions = []  # list of 1D arrays (labels) or 2D rows (multilabel)
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               sequence_lengths: Optional[torch.Tensor] = None):
        """
        Accumulate predictions and targets for F1 computation
        
        Args:
            predictions: Model predictions, shape (batch, time, num_classes)
            targets: Ground truth targets, shape (batch, time)
            sequence_lengths: Actual sequence lengths, shape (batch,)
        """
        if self.overlap_enabled or self.multi_label:
            # Guard: if predictions don't look like (B, T, C) with C > 1, convert single-label to one-hot
            if predictions.ndim != 3 or predictions.shape[-1] <= 1:
                predicted_classes = torch.argmax(predictions, dim=-1)  # (B,T)
                num_classes = self.num_classes
                if sequence_lengths is not None:
                    batch_size = predictions.shape[0]
                    for batch_idx in range(batch_size):
                        seq_len = int(sequence_lengths[batch_idx].item())
                        batch_predictions = predicted_classes[batch_idx, :seq_len]
                        batch_targets = targets[batch_idx, :seq_len]
                        y_pred = F.one_hot(batch_predictions, num_classes=num_classes).to(torch.int)
                        y_true = F.one_hot(batch_targets, num_classes=num_classes).to(torch.int)
                        self.all_predictions.append(y_pred.cpu().numpy())
                        self.all_targets.append(y_true.cpu().numpy())
                else:
                    y_pred = F.one_hot(predicted_classes, num_classes=num_classes).to(torch.int)
                    y_true = F.one_hot(targets, num_classes=num_classes).to(torch.int)
                    self.all_predictions.append(y_pred.cpu().numpy())
                    self.all_targets.append(y_true.cpu().numpy())
                return
            probs = ensure_probabilities(predictions, multi_label=self.multi_label)  # (B, T, C)
            active = (probs >= self.overlap_threshold).to(torch.int)
            # Ensure at least one active per frame by forcing argmax to active where needed
            argmax_idx = torch.argmax(probs, dim=-1, keepdim=True)
            none_active_flat = (active.sum(dim=-1) == 0)  # (B, T)
            one_hot = F.one_hot(argmax_idx.squeeze(-1), num_classes=active.shape[-1]).to(dtype=active.dtype)  # (B, T, C)
            force_active = (none_active_flat.unsqueeze(-1).to(dtype=active.dtype) * one_hot)
            active = (active + force_active).clamp_max(1)
            # y_true from targets: one-hot or binarized multi-hot
            if targets.dim() == 3 and targets.size(-1) == self.num_classes:
                y_true_full = (targets > 0.5).to(torch.int)
            else:
                y_true_full = F.one_hot(targets, num_classes=self.num_classes).to(torch.int)
            if sequence_lengths is not None:
                batch_size = predictions.shape[0]
                for batch_idx in range(batch_size):
                    seq_len = int(sequence_lengths[batch_idx].item())
                    act = active[batch_idx, :seq_len, :]
                    y_true = y_true_full[batch_idx, :seq_len, :]
                    self.all_predictions.append(act.cpu().numpy())
                    self.all_targets.append(y_true.cpu().numpy())
            else:
                self.all_predictions.append(active.cpu().numpy())
                self.all_targets.append(y_true_full.cpu().numpy())
            return
        # Single-label path
        predicted_classes = torch.argmax(predictions, dim=-1)  # (batch, time)
        if sequence_lengths is not None:
            batch_size = predictions.shape[0]
            for batch_idx in range(batch_size):
                seq_len = sequence_lengths[batch_idx].item()
                batch_predictions = predicted_classes[batch_idx, :seq_len]
                batch_targets = targets[batch_idx, :seq_len]
                self.all_predictions.extend(batch_predictions.cpu().numpy())
                self.all_targets.extend(batch_targets.cpu().numpy())
        else:
            self.all_predictions.extend(predicted_classes.cpu().numpy().flatten())
            self.all_targets.extend(targets.cpu().numpy().flatten())
    
    def compute(self) -> float:
        """
        Compute macro F1 score
        
        Returns:
            float: Macro F1 score (0-100)
        """
        if len(self.all_predictions) == 0:
            return 0.0
        if self.overlap_enabled or self.multi_label:
            # Stack to (N, C) multilabel indicator for both preds and targets
            y_pred = np.concatenate([arr.reshape(-1, self.num_classes) for arr in self.all_predictions], axis=0)
            y_true = np.concatenate([arr.reshape(-1, self.num_classes) for arr in self.all_targets], axis=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            return float(f1) * 100.0
        # Single-label F1
        f1 = f1_score(
            self.all_targets,
            self.all_predictions,
            labels=list(range(self.num_classes)),
            average='macro',
            zero_division=0,
        )
        return float(f1) * 100.0


class ConfusionMatrixMetric:
    """
    Computes and tracks confusion matrix for viseme classification.
    Useful for understanding which visemes get confused with each other.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"viseme_{i}" for i in range(num_classes)]
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and targets"""
        self.all_predictions = []
        self.all_targets = []
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               sequence_lengths: Optional[torch.Tensor] = None):
        """
        Accumulate predictions and targets
        
        Args:
            predictions: Model predictions, shape (batch, time, num_classes)
            targets: Ground truth targets, shape (batch, time)
            sequence_lengths: Actual sequence lengths, shape (batch,)
        """
        # Get predicted classes
        predicted_classes = torch.argmax(predictions, dim=-1)  # (batch, time)
        # Ensure targets are class indices
        if targets.dim() == 3 and targets.size(-1) == self.num_classes:
            targets = torch.argmax(targets, dim=-1)
        
        if sequence_lengths is not None:
            # Only include frames within actual sequence length
            batch_size = predictions.shape[0]
            for batch_idx in range(batch_size):
                seq_len = sequence_lengths[batch_idx].item()
                batch_predictions = predicted_classes[batch_idx, :seq_len]
                batch_targets = targets[batch_idx, :seq_len]
                
                self.all_predictions.extend(batch_predictions.cpu().numpy())
                self.all_targets.extend(batch_targets.cpu().numpy())
        else:
            # Include all frames
            self.all_predictions.extend(predicted_classes.cpu().numpy().flatten())
            self.all_targets.extend(targets.cpu().numpy().flatten())
    
    def compute(self) -> np.ndarray:
        """
        Compute confusion matrix
        
        Returns:
            np.ndarray: Confusion matrix, shape (num_classes, num_classes)
        """
        if len(self.all_predictions) == 0:
            return np.zeros((self.num_classes, self.num_classes))
        
        return confusion_matrix(
            self.all_targets,
            self.all_predictions,
            labels=list(range(self.num_classes))
        )
    
    def get_per_class_accuracy(self) -> Dict[str, float]:
        """
        Get per-class accuracy from confusion matrix
        
        Returns:
            Dict[str, float]: Per-class accuracies
        """
        conf_matrix = self.compute()
        per_class_acc = {}
        
        for class_idx in range(self.num_classes):
            class_total = conf_matrix[class_idx, :].sum()
            if class_total > 0:
                class_correct = conf_matrix[class_idx, class_idx]
                accuracy = (class_correct / class_total) * 100.0
            else:
                accuracy = 0.0
            
            per_class_acc[self.class_names[class_idx]] = accuracy
        
        return per_class_acc


class MetricsTracker:
    """
    Aggregates and tracks multiple metrics during training/validation.
    """
    
    def __init__(self, config: TrainingConfiguration):
        self.config = config
        self.num_classes = config.model.num_visemes
        
        # Initialize metrics based on configuration
        self.metrics = {}
        
        if "frame_accuracy" in config.evaluation.metrics:
            self.metrics["frame_accuracy"] = FrameAccuracy(
                overlap_enabled=config.training.viseme_overlap_enabled,
                overlap_threshold=config.training.viseme_overlap_threshold,
                multi_label=config.training.multi_label,
            )
        
        if "macro_f1" in config.evaluation.metrics:
            self.metrics["macro_f1"] = MacroF1Score(
                self.num_classes,
                overlap_enabled=config.training.viseme_overlap_enabled,
                overlap_threshold=config.training.viseme_overlap_threshold,
                multi_label=config.training.multi_label,
            )
        
        if "confusion_matrix" in config.evaluation.metrics:
            # Get viseme names from phoneme mapping if available
            viseme_names = None
            if config.phoneme_to_viseme_mapping:
                # Extract unique visemes and their names
                viseme_to_phonemes = defaultdict(list)
                for phoneme, viseme in config.phoneme_to_viseme_mapping.items():
                    viseme_to_phonemes[viseme].append(phoneme)
                
                viseme_names = []
                for viseme_id in range(self.num_classes):
                    if viseme_id in viseme_to_phonemes:
                        # Use first phoneme as representative name
                        representative_phoneme = viseme_to_phonemes[viseme_id][0]
                        viseme_names.append(f"V{viseme_id}_{representative_phoneme}")
                    else:
                        viseme_names.append(f"V{viseme_id}")
            
            self.metrics["confusion_matrix"] = ConfusionMatrixMetric(
                self.num_classes, viseme_names
            )
    
    def reset(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor,
               sequence_lengths: Optional[torch.Tensor] = None):
        """
        Update all metrics with new predictions
        
        Args:
            predictions: Model predictions, shape (batch, time, num_classes)
            targets: Ground truth targets, shape (batch, time)
            sequence_lengths: Actual sequence lengths, shape (batch,)
        """
        for metric in self.metrics.values():
            metric.update(predictions, targets, sequence_lengths)
    
    def compute(self) -> Dict[str, Any]:
        """
        Compute all metrics
        
        Returns:
            Dict[str, Any]: Dictionary of computed metrics
        """
        results = {}
        
        for metric_name, metric in self.metrics.items():
            if metric_name == "confusion_matrix":
                # Store both matrix and per-class accuracies
                results[metric_name] = metric.compute()
                results["per_class_accuracy"] = metric.get_per_class_accuracy()
            else:
                results[metric_name] = metric.compute()
        
        return results


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Focuses learning on hard examples and down-weights easy examples.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, 
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class (default: 1.0)
            gamma: Focusing parameter (default: 2.0)
            class_weights: Per-class weights for imbalance handling
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            predictions: Model predictions, shape (batch, time, num_classes)
            targets: Ground truth targets, shape (batch, time)
            
        Returns:
            torch.Tensor: Focal loss value
        """
        # Reshape for cross entropy computation
        batch_size, time_steps, num_classes = predictions.shape
        predictions_flat = predictions.view(-1, num_classes)  # (batch*time, num_classes)
        targets_flat = targets.view(-1)  # (batch*time,)
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(
            predictions_flat, targets_flat, 
            weight=self.class_weights, 
            reduction='none'
        )  # (batch*time,)
        
        # Compute probabilities for focal term
        probabilities = F.softmax(predictions_flat, dim=1)
        target_probabilities = probabilities.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        
        # Compute focal loss
        focal_term = (1 - target_probabilities) ** self.gamma
        focal_loss = self.alpha * focal_term * ce_loss
        
        return focal_loss.mean()


class LossFunction:
    """
    Configurable loss function wrapper that handles different loss types
    and class weighting based on configuration.
    """
    
    def __init__(self, config: TrainingConfiguration, class_counts: Optional[Dict[int, int]] = None):
        """
        Initialize loss function from configuration
        
        Args:
            config: Training configuration
            class_counts: Per-class sample counts for weight computation
        """
        self.config = config
        self.loss_type = config.training.loss_type
        self.class_weighting = config.training.class_weighting
        if self.class_weighting and class_counts is None:
            raise RuntimeError("Class weighting enabled but class_counts not provided. Compute or load counts before constructing LossFunction.")
        
        # Compute class weights if requested
        class_weights = None
        if self.class_weighting and class_counts:
            class_weights = self._compute_class_weights(class_counts)
        
        # Initialize loss function
        if self.loss_type == "cross_entropy":
            if config.training.multi_label:
                # BCE for multi-label setup
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        elif self.loss_type == "focal_loss":
            self.criterion = FocalLoss(
                alpha=config.training.focal_loss_alpha,
                gamma=config.training.focal_loss_gamma,
                class_weights=class_weights
            )
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _compute_class_weights(self, class_counts: Dict[int, int]) -> torch.Tensor:
        """
        Compute inverse frequency weights for class balancing
        
        Args:
            class_counts: Dictionary mapping class indices to sample counts
            
        Returns:
            torch.Tensor: Class weights
        """
        num_classes = self.config.model.num_visemes
        total_samples = sum(class_counts.values())
        
        weights = torch.ones(num_classes)
        
        for class_idx in range(num_classes):
            class_count = class_counts.get(class_idx, 1)  # Avoid division by zero
            weight = total_samples / (num_classes * class_count)
            weights[class_idx] = weight
        
        # Normalize weights
        weights = weights / weights.mean()
        
        return weights
    
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor, sequence_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute loss
        
        Args:
            predictions: Model predictions, shape (batch, time, num_classes)
            targets: Ground truth targets, shape (batch, time)
            
        Returns:
            torch.Tensor: Loss value
        """
        if self.loss_type == "cross_entropy":
            batch_size, time_steps, num_classes = predictions.shape
            mask_padded = bool(getattr(self.config.training, "mask_padded_frames", False))
            if self.config.training.multi_label:
                # Targets expected as multi-hot per frame: (B, T, C)
                # If provided as (B, T) indices, convert to one-hot
                if targets.dim() == 2:
                    targets = F.one_hot(targets, num_classes=num_classes).float()
                elif targets.dim() == 3 and targets.size(-1) == num_classes:
                    targets = targets.float()
                else:
                    raise ValueError("Multi-label training expects targets of shape (B,T) or (B,T,C)")
                if mask_padded and sequence_lengths is not None:
                    device = predictions.device
                    valid_mask_bt = (torch.arange(time_steps, device=device).unsqueeze(0).expand(batch_size, -1)
                                     < sequence_lengths.to(device).unsqueeze(1))  # (B,T)
                    preds_flat = predictions.reshape(batch_size * time_steps, num_classes)
                    targets_flat = targets.reshape(batch_size * time_steps, num_classes)
                    valid_flat = valid_mask_bt.reshape(batch_size * time_steps)
                    # Guard in case mask is empty
                    if valid_flat.any():
                        preds_flat = preds_flat[valid_flat]
                        targets_flat = targets_flat[valid_flat]
                    return self.criterion(preds_flat, targets_flat)
                else:
                    preds_flat = predictions.reshape(-1, num_classes)
                    targets_flat = targets.reshape(-1, num_classes)
                    return self.criterion(preds_flat, targets_flat)
            # Single-label cross entropy path
            if mask_padded and sequence_lengths is not None:
                device = predictions.device
                valid_mask_bt = (torch.arange(time_steps, device=device).unsqueeze(0).expand(batch_size, -1)
                                 < sequence_lengths.to(device).unsqueeze(1))  # (B,T)
                predictions_flat = predictions.view(batch_size * time_steps, num_classes)
                targets_flat = targets.view(batch_size * time_steps)
                valid_flat = valid_mask_bt.reshape(batch_size * time_steps)
                if valid_flat.any():
                    predictions_flat = predictions_flat[valid_flat]
                    targets_flat = targets_flat[valid_flat]
                return self.criterion(predictions_flat, targets_flat)
            else:
                predictions_flat = predictions.view(-1, num_classes)
                targets_flat = targets.view(-1)
                return self.criterion(predictions_flat, targets_flat)
        
        else:  # focal_loss
            if self.config.training.multi_label:
                # Convert to BCE-style focal loss for multi-label if needed
                num_classes = predictions.size(-1)
                if targets.dim() == 2:
                    targets = F.one_hot(targets, num_classes=num_classes).float()
                elif targets.dim() == 3 and targets.size(-1) == num_classes:
                    targets = targets.float()
                else:
                    raise ValueError("Multi-label focal_loss expects targets of shape (B,T) or (B,T,C)")
                preds_flat = predictions.reshape(-1, num_classes)
                targets_flat = targets.reshape(-1, num_classes)
                probs = torch.sigmoid(preds_flat)
                pt = probs * targets_flat + (1 - probs) * (1 - targets_flat)
                alpha = float(self.config.training.focal_loss_alpha)
                gamma = float(self.config.training.focal_loss_gamma)
                focal = (alpha * (1 - pt) ** gamma) * F.binary_cross_entropy_with_logits(
                    preds_flat, targets_flat, reduction='none'
                )
                return focal.mean()
            return self.criterion(predictions, targets)


def compute_class_counts(train_loader, num_classes: int, multi_label: bool, mask_padded: bool = True) -> Dict[int, int]:
    """One-pass computation of class counts over the training data.

    Counts valid frames only (excludes padding when lengths are provided).
    Prints lightweight progress as 'done/total'.
    """
    counts = {i: 0 for i in range(num_classes)}
    total = len(train_loader)
    with torch.no_grad():
        for batch in tqdm(train_loader, total=total, desc="Class count", unit="batch", leave=False):
            targets = batch['targets']  # (B,T) or (B,T,C)
            lengths = batch.get('lengths', None)
            if multi_label:
                # Expect (B,T,C). If (B,T), convert to one-hot.
                if targets.dim() == 2:
                    targets = F.one_hot(targets, num_classes=num_classes).to(torch.float32)
                if mask_padded and lengths is not None:
                    B, T, C = targets.shape
                    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
                    mask = mask.unsqueeze(-1).expand(B, T, C)
                    targets = targets * mask
                # Sum positives per class
                cls_sum = targets.sum(dim=(0, 1))  # (C,)
                for i in range(num_classes):
                    counts[i] += int(cls_sum[i].item())
            else:
                # Single-label (B,T)
                if mask_padded and lengths is not None:
                    B, T = targets.shape
                    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
                    targets = torch.where(mask, targets, torch.zeros_like(targets))
                flat = targets.view(-1)
                hist = torch.bincount(flat, minlength=num_classes)
                for i in range(num_classes):
                    counts[i] += int(hist[i].item())
    return counts


def _class_count_cache_key(config: TrainingConfiguration) -> str:
    """Build a deterministic key for class-count caching based on data and labeling config."""
    # Include a hash of the resolved phoneme->viseme mapping so cache invalidates when mapping changes
    try:
        # Deterministic mapping serialization
        mapping_items = sorted((str(k), int(v)) for k, v in (config.phoneme_to_viseme_mapping or {}).items())
        mapping_blob = json.dumps(mapping_items, separators=(",", ":")).encode("utf-8")
        mapping_hash = hashlib.sha256(mapping_blob).hexdigest()[:16]
    except Exception:
        mapping_hash = "unknown"

    payload = {
        'dataset': config.data.dataset,
        'splits': config.data.splits,
        'val': config.data.val_split,
        'test': config.data.test_split,
        'num_visemes': config.model.num_visemes,
        'multi_label': bool(getattr(config.training, 'multi_label', False)),
        'target_crossfade_ms': int(getattr(config.training, 'target_crossfade_ms', 0)),
        'phoneme_viseme_map': str(config.data.phoneme_viseme_map),
        'mapping_hash': mapping_hash,
        'fps': float(config.audio.fps),
    }
    blob = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:16]


def get_class_count_cache_path(config: TrainingConfiguration) -> Path:
    """Return a stable cache file path for class counts under dataset cache.

    Location: <project_root>/training/data/cache/class_counts/counts_<key>.json
    """
    project_root = Path(__file__).resolve().parents[2]
    cache_dir = project_root / 'training' / 'data' / 'cache' / 'class_counts'
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _class_count_cache_key(config)
    return cache_dir / f"counts_{key}.json"


def load_class_counts_cache(config: TrainingConfiguration) -> Optional[Dict[int, int]]:
    """Load cached class counts if present."""
    path = get_class_count_cache_path(config)
    if path.exists():
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            # Ensure keys are int
            return {int(k): int(v) for k, v in data.items()}
        except Exception:
            return None
    return None


def save_class_counts_cache(config: TrainingConfiguration, counts: Dict[int, int]) -> None:
    """Save class counts to cache for reuse in subsequent runs."""
    path = get_class_count_cache_path(config)
    try:
        with open(path, 'w') as f:
            json.dump({int(k): int(v) for k, v in counts.items()}, f)
    except Exception:
        pass


class EarlyStopping:
    """
    Early stopping utility to prevent overfitting.
    Monitors a metric and stops training when it stops improving.
    """
    
    def __init__(self, patience: int, metric_name: str, minimize: bool = True):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs to wait without improvement
            metric_name: Name of metric to monitor
            minimize: Whether lower values are better (True for loss, False for accuracy)
        """
        self.patience = patience
        self.metric_name = metric_name
        self.minimize = minimize
        
        self.best_value = float('inf') if minimize else float('-inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.should_stop = False
    
    def update(self, current_value: float, current_epoch: int) -> bool:
        """
        Update early stopping state with new metric value
        
        Args:
            current_value: Current metric value
            current_epoch: Current epoch number
            
        Returns:
            bool: Whether training should stop
        """
        improved = False
        
        if self.minimize:
            if current_value < self.best_value:
                improved = True
        else:
            if current_value > self.best_value:
                improved = True
        
        if improved:
            self.best_value = current_value
            self.best_epoch = current_epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current early stopping status
        
        Returns:
            Dict with early stopping information
        """
        return {
            'best_value': self.best_value,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'should_stop': self.should_stop
        }


class ModelCheckpoint:
    """
    Handles model checkpointing with automatic best model saving.
    """
    
    def __init__(self, checkpoint_dir: Union[str, Path], config: TrainingConfiguration):
        """
        Initialize model checkpointing
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            config: Training configuration
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config
        self.max_checkpoints = config.logging.max_checkpoints
        
        # Track saved checkpoints
        self.checkpoint_files = deque(maxlen=self.max_checkpoints)
        self.best_checkpoint_path = None
        self.best_metric_value = float('inf')  # Assume we minimize validation loss
    
    def _serialize_config(self, config: TrainingConfiguration) -> Dict[str, Any]:
        """Convert TrainingConfiguration dataclasses into plain nested dicts."""
        def to_plain(obj: Any) -> Any:
            if hasattr(obj, "__dict__"):
                return {k: to_plain(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
            if isinstance(obj, dict):
                return {k: to_plain(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [to_plain(v) for v in obj]
            return obj
        return to_plain(config)

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any], epoch: int, step: int,
                       metrics: Dict[str, float], is_best: bool = False) -> Path:
        """
        Save model checkpoint
        
        Args:
            model: PyTorch model
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            step: Current training step
            metrics: Current metrics
            is_best: Whether this is the best model so far
            
        Returns:
            Path: Path to saved checkpoint
        """
        checkpoint_data = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            # Save a plain, JSON-serializable config snapshot to avoid pickle issues
            'config_serialized': self._serialize_config(self.config),
        }
        
        # Create checkpoint filename
        checkpoint_filename = f"checkpoint_epoch_{epoch:04d}_step_{step:06d}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_filename
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Track checkpoint file
        self.checkpoint_files.append(checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_checkpoint_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_data, best_checkpoint_path)
            self.best_checkpoint_path = best_checkpoint_path
        
        # Clean up old checkpoints (keep only max_checkpoints recent ones)
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoint files beyond the maximum limit"""
        # Get all checkpoint files in directory
        all_checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        # Sort by modification time
        all_checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest files if we exceed the limit
        while len(all_checkpoints) > self.max_checkpoints:
            old_checkpoint = all_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load checkpoint from file
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dict containing checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Prefer safe tensor-only load; fall back to full load for older checkpoints
        try:
            return torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        except Exception:
            # As a fallback for locally produced checkpoints, allow full unpickling
            return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get path to the most recent checkpoint
        
        Returns:
            Path to latest checkpoint or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if not checkpoints:
            return None
        
        # Sort by modification time and return the latest
        latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return latest_checkpoint


def create_optimizer(model: nn.Module, config: TrainingConfiguration) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration
    
    Args:
        model: PyTorch model
        config: Training configuration
        
    Returns:
        torch.optim.Optimizer: Configured optimizer
    """
    optimizer_name = config.training.optimizer.lower()
    learning_rate = config.training.learning_rate
    weight_decay = config.training.weight_decay
    
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=config.training.betas,
            weight_decay=weight_decay,
            foreach=False
        )
    
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=config.training.betas,
            weight_decay=weight_decay
        )
    
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=config.training.betas[0],  # Use first beta as momentum
            weight_decay=weight_decay
        )
    
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfiguration,
                    steps_per_epoch: int) -> Optional[Any]:
    """
    Create learning rate scheduler from configuration
    
    Args:
        optimizer: PyTorch optimizer
        config: Training configuration
        steps_per_epoch: Number of training steps per epoch
        
    Returns:
        Learning rate scheduler or None
    """
    scheduler_name = config.training.scheduler.lower()
    max_epochs = config.training.max_epochs
    warmup_ratio = config.training.warmup_ratio
    
    if scheduler_name == "onecycle":
        total_steps = max_epochs * steps_per_epoch
        warmup_steps = int(total_steps * warmup_ratio)
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.training.learning_rate,
            total_steps=total_steps,
            pct_start=warmup_ratio,
            anneal_strategy='cos'
        )
    
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=config.training.learning_rate * 0.01  # Final LR is 1% of initial
        )
    
    elif scheduler_name == "constant":
        # Truly constant learning rate: do not create a scheduler
        scheduler = None
    
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


if __name__ == "__main__":
    # Example usage and testing
    from .config import load_config
    
    # Load configuration
    config = load_config("../recipes/tcn_config.toml")
    
    # Test metrics
    print("Testing metrics...")
    metrics_tracker = MetricsTracker(config)
    
    # Create dummy predictions and targets
    batch_size, time_steps, num_classes = 4, 100, config.model.num_visemes
    dummy_predictions = torch.randn(batch_size, time_steps, num_classes)
    dummy_targets = torch.randint(0, num_classes, (batch_size, time_steps))
    dummy_lengths = torch.randint(50, time_steps, (batch_size,))
    
    # Update metrics
    metrics_tracker.update(dummy_predictions, dummy_targets, dummy_lengths)
    
    # Compute metrics
    results = metrics_tracker.compute()
    
    print("Computed metrics:")
    for metric_name, value in results.items():
        if metric_name == "confusion_matrix":
            print(f"  {metric_name}: {value.shape}")
        elif metric_name == "per_class_accuracy":
            print(f"  {metric_name}: {len(value)} classes")
        else:
            print(f"  {metric_name}: {value:.2f}")
    
    # Test loss function
    print("\nTesting loss function...")
    loss_fn = LossFunction(config)
    loss_value = loss_fn(dummy_predictions, dummy_targets)
    print(f"Loss value: {loss_value.item():.4f}")
    
    # Test early stopping
    print("\nTesting early stopping...")
    early_stopping = EarlyStopping(patience=5, metric_name="val_loss", minimize=True)
    
    # Simulate training with improving then degrading loss
    for epoch in range(10):
        if epoch < 5:
            val_loss = 1.0 - epoch * 0.1  # Improving loss
        else:
            val_loss = 0.5 + (epoch - 5) * 0.05  # Degrading loss
        
        should_stop = early_stopping.update(val_loss, epoch)
        print(f"Epoch {epoch}: val_loss={val_loss:.3f}, should_stop={should_stop}")
        
        if should_stop:
            break
    
    print("Training utilities test completed successfully.")
