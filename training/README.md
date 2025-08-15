# OpenLipSync TCN Training

A comprehensive training system for the Temporal Convolutional Network (TCN) used in OpenLipSync for audio-to-viseme mapping.

## 🚀 Quick Start

### Prerequisites

```bash
# Install dependencies
pip install torch torchaudio numpy scikit-learn tensorboard
pip install tomli  # for Python < 3.11, or use tomllib for Python 3.11+
```

### Basic Usage

```bash
# Train with default configuration
python train.py --config recipes/tcn_config.toml

# Resume from checkpoint
python train.py --config recipes/tcn_config.toml --resume checkpoints/best_model.pt

# Test only (skip training)
python train.py --config recipes/tcn_config.toml --test-only --resume checkpoints/best_model.pt
```

## 📁 Project Structure

```
training/
├── train.py                    # Main training script
├── modules/                    # Core training modules
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── data_pipeline.py        # Data loading & preprocessing
│   ├── tcn_model.py           # TCN model architecture
│   ├── training_utils.py      # Training utilities & metrics
│   └── logger.py              # Logging & TensorBoard
├── recipes/
│   └── tcn_config.toml        # Training configuration
└── README.md                  # This file
```

## ⚙️ Configuration

The training system is fully configuration-driven through `tcn_config.toml`. Key sections:

### Model Architecture
```toml
[model]
num_visemes = 15      # Number of viseme classes
layers = 5            # TCN depth
channels = 128        # Hidden channels
kernel_size = 3       # Convolution kernel size
dropout = 0.1         # Dropout rate
normalization = "weight_norm"
```

### Audio Processing
```toml
[audio]
sample_rate = 16000   # Audio sample rate
hop_length_ms = 10    # Frame rate (100fps)
n_mels = 80          # Mel spectrogram bands
fmin = 50            # Min frequency
fmax = 8000          # Max frequency
```

### Training Parameters
```toml
[training]
batch_size = 32
learning_rate = 3e-4
optimizer = "adamw"
scheduler = "onecycle"
max_epochs = 100
loss_type = "cross_entropy"
mixed_precision = true
```

## 🔧 Key Features

### 📊 Comprehensive Metrics
- **Frame Accuracy**: Percentage of correctly classified frames
- **Macro F1 Score**: Balanced accuracy across all viseme classes
- **Confusion Matrix**: Detailed class-wise performance analysis
- **Real-time Factor**: Performance benchmarking for real-time inference

### 🎯 Advanced Training Features
- **Mixed Precision**: Faster training with minimal accuracy loss
- **Early Stopping**: Automatic training termination to prevent overfitting
- **Learning Rate Scheduling**: OneCycle, Cosine, or Constant schedules
- **Data Augmentation**: SpecAugment and audio-level augmentations
- **Class Weighting**: Handle imbalanced viseme distributions

### 📈 Monitoring & Logging
- **TensorBoard Integration**: Real-time training visualization
- **Comprehensive Logging**: Console and file logging with configurable verbosity
- **Automatic Checkpointing**: Save best models and periodic checkpoints
- **Experiment Tracking**: Organized experiment metadata and reproducibility

### 🏗️ Modular Architecture
- **Configuration-Driven**: All parameters controlled via TOML files
- **Pluggable Components**: Easy to swap models, optimizers, and data loaders
- **Error Handling**: Graceful handling of common training issues
- **Resume Training**: Seamlessly continue interrupted training runs

## 📝 Usage Examples

### Training from Scratch
```bash
python train.py --config recipes/tcn_config.toml
```

### Custom Data Location
```bash
python train.py --config recipes/tcn_config.toml --data-root /path/to/librispeech
```

### Experiment with Different Parameters
```bash
# Edit tcn_config.toml to change:
# - Model architecture (layers, channels)
# - Training parameters (learning rate, batch size)
# - Audio processing (sample rate, mel bands)
# Then run training
python train.py --config recipes/tcn_config.toml
```

### Monitor Training Progress
```bash
# In another terminal, start TensorBoard
tensorboard --logdir training/runs

# View in browser at http://localhost:6006
```

## 🎛️ Model Architecture

The TCN (Temporal Convolutional Network) is designed for efficient real-time audio processing:

- **Dilated Convolutions**: Large receptive fields without heavy computation
- **Residual Connections**: Stable gradient flow for deep networks
- **Causal Convolutions**: No future information for real-time inference
- **Weight Normalization**: Training stability and faster convergence

**Architecture Flow**:
```
Audio → Mel Spectrogram → Input Projection → TCN Layers → Output Projection → Visemes
```

## 📊 Data Pipeline

### LibriSpeech Integration
- Automatic download and preprocessing of LibriSpeech datasets
- Support for multiple splits (train-clean-100, train-clean-360, etc.)
- Configurable audio chunking for memory efficiency

### Audio Processing
- Mel spectrogram computation with configurable parameters
- Per-utterance or global normalization
- Real-time compatible processing pipeline

### Phoneme-to-Viseme Mapping
- Configurable phoneme-to-viseme mapping via JSON files
- Frame-level alignment (currently simplified, MFA integration planned)
- Support for different languages and accent variations

## 🚨 Important Notes

### Current Limitations
1. **Phoneme Alignment**: Currently uses simplified letter-to-viseme mapping
   - **Production Ready**: Integrate Montreal Forced Alignment (MFA)
   - **Recommended**: Use pre-aligned datasets like GRID or TCD-TIMIT

2. **Dataset**: Built for LibriSpeech (speech-only)
   - **For Production**: Consider audiovisual datasets with ground-truth visemes

### Performance Expectations
- **Real-time Factor**: Typically < 0.1 on modern GPUs (10x faster than real-time)
- **Accuracy**: Expect 70-85% frame accuracy on clean speech
- **Training Time**: ~2-6 hours for 100 epochs on single GPU (depends on data size)

## 🔍 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Reduce batch size in config
batch_size = 16  # or 8
```

**Slow Training**:
```bash
# Enable mixed precision
mixed_precision = true

# Increase num_workers
num_workers = 8
```

**Poor Convergence**:
```bash
# Adjust learning rate
learning_rate = 1e-4  # lower

# Enable class weighting
class_weighting = true
```

### Debug Mode
```bash
# Enable detailed logging
log_level = "DEBUG"

# Log model histograms
log_histograms = true
```

## 🤝 Contributing

To extend the training system:

1. **New Models**: Implement in `modules/tcn_model.py`
2. **New Datasets**: Extend `modules/data_pipeline.py`
3. **New Metrics**: Add to `modules/training_utils.py`
4. **New Optimizers**: Update `create_optimizer()` function

## 📚 References

- [TCN Paper](https://arxiv.org/abs/1803.01271): An Empirical Evaluation of Generic Convolutional and Recurrent Networks
- [LibriSpeech](http://www.openslr.org/12/): Large-scale English speech corpus
- [Montreal Forced Alignment](https://github.com/MontrealCorpusTools/Montreal-Forced-Alignment): Phoneme alignment tool

## 📄 License

See the main OpenLipSync repository for license information.
