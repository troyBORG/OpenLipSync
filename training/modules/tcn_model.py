"""
Temporal Convolutional Network (TCN) Model for OpenLipSync

Implements a TCN architecture for audio-to-viseme mapping with configurable
depth, channels, and normalization. Designed for real-time lip synchronization.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from .config import TrainingConfiguration


class TemporalConvolutionBlock(nn.Module):
    """
    A single temporal convolution block with residual connections.
    
    Features:
    - Dilated convolutions for large receptive fields
    - Weight normalization for training stability
    - Residual connections for gradient flow
    - Configurable dropout and normalization
    """
    
    def __init__(self, input_channels: int, output_channels: int, 
                 kernel_size: int, dilation: int, dropout: float = 0.1,
                 normalization: str = "weight_norm"):
        """
        Initialize temporal convolution block
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels
            kernel_size: Convolution kernel size (should be odd)
            dilation: Dilation factor for temporal modeling
            dropout: Dropout probability
            normalization: Normalization type ("weight_norm", "layer_norm", "batch_norm")
        """
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout
        
        # Calculate padding for causal convolution (no future information)
        # Padding = (kernel_size - 1) * dilation ensures output length = input length
        self.padding = (kernel_size - 1) * dilation
        
        # First convolution layer
        self.conv1 = nn.Conv1d(
            input_channels, output_channels, kernel_size,
            dilation=dilation, padding=self.padding, bias=False
        )
        
        # Second convolution layer
        self.conv2 = nn.Conv1d(
            output_channels, output_channels, kernel_size,
            dilation=dilation, padding=self.padding, bias=False
        )
        
        # Apply normalization
        if normalization == "weight_norm":
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
        
        # Normalization layers
        if normalization == "layer_norm":
            self.norm1 = nn.LayerNorm(output_channels)
            self.norm2 = nn.LayerNorm(output_channels)
        elif normalization == "batch_norm":
            self.norm1 = nn.BatchNorm1d(output_channels)
            self.norm2 = nn.BatchNorm1d(output_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection adjustment
        if input_channels != output_channels:
            self.residual_conv = nn.Conv1d(input_channels, output_channels, 1, bias=False)
            if normalization == "weight_norm":
                self.residual_conv = weight_norm(self.residual_conv)
        else:
            self.residual_conv = nn.Identity()
        
        # Activation function
        self.activation = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize convolution weights using He initialization"""
        for module in [self.conv1, self.conv2]:
            if hasattr(module, 'weight_v'):  # Weight-normalized layer
                nn.init.kaiming_normal_(module.weight_v, mode='fan_out', nonlinearity='relu')
            elif hasattr(module, 'weight'):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        
        if hasattr(self.residual_conv, 'weight'):
            nn.init.kaiming_normal_(self.residual_conv.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal convolution block
        
        Args:
            input_tensor: Input tensor, shape (batch, channels, time)
            
        Returns:
            torch.Tensor: Output tensor, shape (batch, output_channels, time)
        """
        # Store input for residual connection
        residual = input_tensor
        
        # First convolution path
        output = self.conv1(input_tensor)
        
        # Remove future information (causal convolution)
        if self.padding > 0:
            output = output[:, :, :-self.padding]
        
        # Apply normalization and activation
        if isinstance(self.norm1, nn.LayerNorm):
            # LayerNorm expects (batch, time, channels)
            output = output.transpose(1, 2)
            output = self.norm1(output)
            output = output.transpose(1, 2)
        else:
            output = self.norm1(output)
        
        output = self.activation(output)
        output = self.dropout1(output)
        
        # Second convolution path
        output = self.conv2(output)
        
        # Remove future information (causal convolution)
        if self.padding > 0:
            output = output[:, :, :-self.padding]
        
        # Apply normalization
        if isinstance(self.norm2, nn.LayerNorm):
            output = output.transpose(1, 2)
            output = self.norm2(output)
            output = output.transpose(1, 2)
        else:
            output = self.norm2(output)
        
        output = self.dropout2(output)
        
        # Residual connection
        residual = self.residual_conv(residual)
        
        # Ensure temporal dimensions match for residual connection
        min_time = min(output.size(-1), residual.size(-1))
        output = output[:, :, :min_time]
        residual = residual[:, :, :min_time]
        
        # Add residual and apply final activation
        output = self.activation(output + residual)
        
        return output


class TemporalConvolutionalNetwork(nn.Module):
    """
    Complete TCN model for audio-to-viseme mapping.
    
    Architecture:
    - Input projection from mel features to hidden channels
    - Stack of dilated temporal convolution blocks
    - Output projection to viseme classes
    - Optional temporal pooling for efficiency
    """
    
    def __init__(self, config: TrainingConfiguration):
        """
        Initialize TCN model from configuration
        
        Args:
            config: Training configuration with model parameters
        """
        super().__init__()
        
        self.config = config
        self.model_config = config.model
        self.audio_config = config.audio
        
        # Model dimensions
        self.input_features = self.audio_config.n_mels
        self.hidden_channels = self.model_config.channels
        self.num_layers = self.model_config.layers
        self.num_classes = self.model_config.num_visemes
        self.kernel_size = self.model_config.kernel_size
        self.dropout = self.model_config.dropout
        self.normalization = self.model_config.normalization
        
        # Calculate receptive field
        self.receptive_field = self._calculate_receptive_field()
        
        # Input projection layer
        self.input_projection = nn.Conv1d(
            self.input_features, self.hidden_channels, 1, bias=False
        )
        
        if self.normalization == "weight_norm":
            self.input_projection = weight_norm(self.input_projection)
        
        # TCN layers with exponentially increasing dilation
        self.tcn_layers = nn.ModuleList()
        
        for layer_idx in range(self.num_layers):
            dilation = 2 ** layer_idx  # Exponential dilation: 1, 2, 4, 8, 16, ...
            
            # All hidden layers have same channel count for residual connections
            tcn_block = TemporalConvolutionBlock(
                input_channels=self.hidden_channels,
                output_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                dilation=dilation,
                dropout=self.dropout,
                normalization=self.normalization
            )
            
            self.tcn_layers.append(tcn_block)
        
        # Output projection layer
        self.output_projection = nn.Linear(self.hidden_channels, self.num_classes)
        
        # Initialize output layer with smaller weights for stable training
        nn.init.normal_(self.output_projection.weight, mean=0, std=0.01)
        nn.init.zeros_(self.output_projection.bias)
        
        # Model info
        self.num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Initialized TCN model:")
        print(f"  - Layers: {self.num_layers}")
        print(f"  - Hidden channels: {self.hidden_channels}")
        print(f"  - Receptive field: {self.receptive_field} frames ({self.receptive_field / self.audio_config.fps:.2f}s)")
        print(f"  - Parameters: {self.num_parameters:,}")
    
    def _calculate_receptive_field(self) -> int:
        """
        Calculate the receptive field of the TCN in number of frames
        
        Returns:
            int: Receptive field in frames
        """
        # Start with 1 frame (current frame)
        receptive_field = 1
        # Each TemporalConvolutionBlock has two convs with same kernel and dilation
        for layer_idx in range(self.num_layers):
            dilation = 2 ** layer_idx
            receptive_field += 2 * (self.kernel_size - 1) * dilation
        return receptive_field
    
    def forward(self, mel_features: torch.Tensor, 
                sequence_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through TCN model
        
        Args:
            mel_features: Input mel spectrogram features, shape (batch, time, n_mels)
            sequence_lengths: Actual lengths of sequences (for padding), shape (batch,)
            
        Returns:
            torch.Tensor: Viseme logits, shape (batch, time, num_visemes)
        """
        batch_size, time_steps, feature_dim = mel_features.shape
        
        # Transpose to (batch, channels, time) for conv1d
        features = mel_features.transpose(1, 2)  # (batch, n_mels, time)
        
        # Input projection
        hidden = self.input_projection(features)  # (batch, hidden_channels, time)
        
        # Pass through TCN layers
        for tcn_layer in self.tcn_layers:
            hidden = tcn_layer(hidden)
        
        # Transpose back to (batch, time, channels) for output layer
        hidden = hidden.transpose(1, 2)  # (batch, time, hidden_channels)
        
        # Output projection to viseme classes
        viseme_logits = self.output_projection(hidden)  # (batch, time, num_visemes)
        
        # Apply silence bias if configured
        if hasattr(self.config.training, 'silence_bias') and self.config.training.silence_bias > 0:
            viseme_logits[:, :, 0] += self.config.training.silence_bias
        
        return viseme_logits
    
    def get_latency_ms(self, device: str = "cpu") -> float:
        """
        Measure model inference latency for real-time performance evaluation
        
        Args:
            device: Device to test on ("cpu", "cuda")
            
        Returns:
            float: Average inference time per second of audio (in milliseconds)
        """
        import time
        
        # Move model to specified device
        self.to(device)
        self.eval()
        
        # Create dummy input for 1 second of audio
        frames_per_second = int(self.audio_config.fps)
        dummy_input = torch.randn(
            1, frames_per_second, self.input_features, 
            device=device, dtype=torch.float32
        )
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(10):
                _ = self.forward(dummy_input)
        
        # Timing runs
        num_runs = 100
        torch.cuda.synchronize() if device == "cuda" else None
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.forward(dummy_input)
        
        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()
        
        # Calculate average latency per second of audio
        total_time_seconds = end_time - start_time
        average_latency_per_run = total_time_seconds / num_runs
        latency_ms = average_latency_per_run * 1000  # Convert to milliseconds
        
        return latency_ms
    
    def get_real_time_factor(self, device: str = "cpu") -> float:
        """
        Calculate Real-Time Factor (RTF) for the model
        
        RTF < 1.0 means the model can process audio faster than real-time
        RTF = 1.0 means the model processes at exactly real-time speed
        RTF > 1.0 means the model is slower than real-time
        
        Args:
            device: Device to test on
            
        Returns:
            float: Real-time factor
        """
        latency_ms = self.get_latency_ms(device)
        rtf = latency_ms / 1000.0  # 1 second of audio should take 1000ms to process in real-time
        return rtf


def create_model(config: TrainingConfiguration) -> TemporalConvolutionalNetwork:
    """
    Create TCN model from configuration
    
    Args:
        config: Training configuration
        
    Returns:
        TemporalConvolutionalNetwork: Initialized model
    """
    model = TemporalConvolutionalNetwork(config)
    
    # Move to specified device
    device = torch.device(config.hardware.device)
    model = model.to(device)
    
    return model


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model
    
    Args:
        model: PyTorch model
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        int: Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: TemporalConvolutionalNetwork):
    """
    Print a detailed summary of the TCN model
    
    Args:
        model: TCN model to summarize
    """
    print("\n" + "="*60)
    print("TCN MODEL SUMMARY")
    print("="*60)
    
    print(f"Architecture: {model.model_config.name.upper()}")
    print(f"Input features: {model.input_features} (mel bands)")
    print(f"Output classes: {model.num_classes} (visemes)")
    print(f"Hidden channels: {model.hidden_channels}")
    print(f"Number of layers: {model.num_layers}")
    print(f"Kernel size: {model.kernel_size}")
    print(f"Dropout rate: {model.dropout}")
    print(f"Normalization: {model.normalization}")
    
    print(f"\nReceptive field:")
    print(f"  - Frames: {model.receptive_field}")
    print(f"  - Time: {model.receptive_field / model.audio_config.fps:.2f} seconds")
    
    print(f"\nModel size:")
    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Total parameters: {total_params:,}")
    
    # Estimate memory usage (rough approximation)
    param_size_mb = (trainable_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"  - Estimated size: {param_size_mb:.1f} MB")
    
    print(f"\nAudio processing:")
    print(f"  - Sample rate: {model.audio_config.sample_rate} Hz")
    print(f"  - Frame rate: {model.audio_config.fps:.1f} fps")
    print(f"  - Hop length: {model.audio_config.hop_length_ms} ms")
    
    print("="*60)


if __name__ == "__main__":
    # Example usage and testing
    from .config import load_config
    
    # Load configuration (resolve relative to project root if not provided)
    from pathlib import Path
    here = Path(__file__).resolve()
    project_root = here.parents[3] if (here.parents[3] / 'pyproject.toml').exists() else here.parents[2]
    default_cfg = project_root / 'training' / 'recipes' / 'tcn_config.toml'
    config = load_config(default_cfg)
    
    # Create model
    model = create_model(config)
    
    # Print model summary
    print_model_summary(model)
    
    # Test forward pass
    batch_size = 4
    seq_length = 100  # 1 second at 100fps
    
    dummy_input = torch.randn(batch_size, seq_length, config.audio.n_mels)
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test latency
    print(f"\nTesting inference latency...")
    latency_ms = model.get_latency_ms(config.hardware.device)
    rtf = model.get_real_time_factor(config.hardware.device)
    
    print(f"Latency: {latency_ms:.2f} ms per second of audio")
    print(f"Real-time factor: {rtf:.3f}")
    
    if rtf < 1.0:
        print("Model can run faster than real-time.")
    else:
        print("Model may struggle with real-time processing.")
    
    print("TCN model test completed successfully.")
