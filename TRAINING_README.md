# Training Pipeline for Self-Supervised 3D Medical Image Interpolation

This training pipeline implements a self-supervised approach for 3D medical image interpolation using multi-view consistency. The method uses RIFE for interpolation and U-Net for segmentation.

## Overview

The pipeline performs the following steps:
1. Load and preprocess DICOM medical images
2. Interpolate slices using RIFE to create denser volumes
3. Extract three orthogonal views (axial, sagittal, coronal)
4. Generate segmentations for each view using a frozen U-Net
5. Optimize interpolation network using multi-view consistency loss

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or with conda
conda create -n med-interp python=3.9
conda activate med-interp
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Organize your DICOM files in the following structure:

```
data/
├── train/
│   ├── patient001.dcm
│   ├── patient002.dcm
│   └── ...
├── val/
│   ├── patient101.dcm
│   └── ...
└── test/
    └── ...
```

Or create split files:

```
data/
├── volumes/
│   ├── patient001.dcm
│   └── ...
├── train.txt
├── val.txt
└── test.txt
```

### 2. Configure Training

Edit `config.py` or create a custom configuration:

```python
from config import Config

config = Config(
    data_dir="./data",
    batch_size=2,
    num_epochs=100,
    learning_rate=1e-4,
    num_slices=16,
    img_size=(256, 256)
)
```

### 3. Run Training

```bash
python train.py
```

Or with custom configuration:

```python
from train import TrainingPipeline
from config import Config

config = Config(
    data_dir="./your_data",
    use_wandb=True,
    project_name="my-experiment"
)

pipeline = TrainingPipeline(config)
pipeline.train()
```

## Configuration Options

### Data Settings

- `data_dir`: Path to data directory
- `num_slices`: Number of slices to extract from each volume (default: 16)
- `img_size`: Target size for each slice, e.g., (256, 256)
- `in_channels`: Number of input channels (default: 1 for grayscale)
- `num_classes`: Number of segmentation classes (default: 2)

### Training Settings

- `batch_size`: Batch size (default: 2, limited by GPU memory)
- `num_epochs`: Number of training epochs (default: 100)
- `learning_rate`: Initial learning rate (default: 1e-4)
- `grad_clip`: Gradient clipping value (default: 1.0)
- `num_workers`: Number of data loading workers (default: 4)

### Loss Weights

- `lambda_consistency`: Weight for multi-view consistency loss (default: 1.0)
- `lambda_smoothness`: Weight for temporal smoothness loss (default: 0.1)
- `lambda_reconstruction`: Weight for reconstruction loss (default: 1.0)
- `consistency_loss_type`: Type of consistency loss ("dice", "ce", "mse", "combined")

### Model Settings

- `interpolation_factor`: Interpolation scale (default: 2 for 2x)
- `segmentation_checkpoint`: Path to pre-trained segmentation model (optional)
- `freeze_segmentation`: Whether to freeze segmentation model (default: True)

## Pre-trained Configurations

Three pre-configured setups are available:

### 1. Fast Debug Configuration

For quick testing and debugging:

```python
from config import get_fast_debug_config

config = get_fast_debug_config()
pipeline = TrainingPipeline(config)
pipeline.train()
```

### 2. High Quality Configuration

For production-quality results:

```python
from config import get_high_quality_config

config = get_high_quality_config()
```

### 3. Low Memory Configuration

For limited GPU memory:

```python
from config import get_low_memory_config

config = get_low_memory_config()
```

## Model Architecture

### Interpolation: RIFE

The pipeline uses RIFE (Real-Time Intermediate Flow Estimation):
- Flow estimation network for optical flow between slices
- Refinement network for improving interpolated frames
- Located in `models/rife_interpolator.py`

Alternative interpolators are also available:
- `SimpleInterpolator`: Lightweight encoder-decoder
- `UNetInterpolator`: U-Net with skip connections

### Segmentation: U-Net

Standard U-Net for medical image segmentation:
- Encoder-decoder architecture with skip connections
- Optional attention gates (AttentionUNet)
- Frozen during training (not updated)
- Located in `models/unet_segmentation.py`

## Loss Functions

The total loss combines multiple components:

```
L_total = λ₁ * L_consistency + λ₂ * L_smoothness + λ₃ * L_reconstruction
```

### 1. Consistency Loss

Measures agreement between segmentations from different views:
- Dice loss: Overlap between predictions
- Cross-entropy: KL divergence between probability distributions
- MSE: Mean squared error between predictions

### 2. Smoothness Loss

Ensures smooth transitions between interpolated slices:
- First-order: Gradient magnitude along depth
- Second-order: Laplacian (curvature)

### 3. Reconstruction Loss

Preserves original slice information:
- L1 or L2 loss between interpolated and original slices

## Logging and Monitoring

### Weights & Biases

Enable W&B logging:

```python
config = Config(
    use_wandb=True,
    project_name="medical-interpolation",
    experiment_name="experiment_001"
)
```

Logged metrics:
- Training/validation losses
- Individual loss components
- Learning rate
- Multi-view consistency scores

### Checkpointing

Checkpoints are saved automatically:
- `latest.pth`: Most recent model
- `best.pth`: Best validation loss
- `epoch_N.pth`: Periodic checkpoints

Load from checkpoint:

```python
config = Config(resume_from="checkpoints/best.pth")
```

## Data Loading

### Single DICOM Files

For datasets with one volume per file:

```python
from data_loader import MedicalVolumeDataset

dataset = MedicalVolumeDataset(
    data_dir="./data/train",
    num_slices=16,
    img_size=(256, 256)
)
```

### DICOM Series

For datasets with multiple files per series:

```python
from data_loader import DICOMSeriesDataset

dataset = DICOMSeriesDataset(
    series_dir="./data/series",
    num_slices=16,
    img_size=(256, 256)
)
```

## Multi-View Processing

The pipeline extracts three orthogonal views:

```python
from utils.multi_view import MultiViewExtractor

extractor = MultiViewExtractor()
axial, sagittal, coronal = extractor.extract_views(volume)
```

Calculate consistency metrics:

```python
from utils.multi_view import calculate_consistency_metrics

metrics = calculate_consistency_metrics(
    axial_pred,
    sagittal_pred,
    coronal_pred
)

print(f"Average Dice: {metrics['dice_average']:.4f}")
```

## Examples

### Example 1: Basic Training

```python
from train import TrainingPipeline
from config import Config

config = Config(
    data_dir="./data",
    batch_size=2,
    num_epochs=50,
    num_slices=16
)

pipeline = TrainingPipeline(config)
pipeline.train()
```

### Example 2: Training with Pre-trained Segmentation

```python
config = Config(
    data_dir="./data",
    segmentation_checkpoint="path/to/unet.pth",
    freeze_segmentation=True
)

pipeline = TrainingPipeline(config)
pipeline.train()
```

### Example 3: Custom Loss Weights

```python
config = Config(
    data_dir="./data",
    lambda_consistency=2.0,
    lambda_smoothness=0.5,
    lambda_reconstruction=1.5,
    consistency_loss_type="combined"
)

pipeline = TrainingPipeline(config)
pipeline.train()
```

## File Structure

```
.
├── train.py                    # Main training script
├── config.py                   # Configuration classes
├── data_loader.py              # Data loading utilities
├── losses.py                   # Loss functions
├── requirements.txt            # Dependencies
├── models/
│   ├── __init__.py
│   ├── rife_interpolator.py   # RIFE interpolation model
│   └── unet_segmentation.py   # U-Net segmentation model
└── utils/
    ├── __init__.py
    └── multi_view.py           # Multi-view utilities
```

## Expected Results

After training, you should observe:
1. Decreasing consistency loss (better multi-view agreement)
2. Smooth interpolated slices
3. Preservation of anatomical structures
4. Improved Dice scores between views

## Troubleshooting

### Out of Memory

- Reduce `batch_size` to 1
- Reduce `num_slices` (e.g., 8 or 12)
- Reduce `img_size` (e.g., 128x128)
- Enable `mixed_precision=True`

### Slow Training

- Increase `num_workers` for data loading
- Enable `cache_data=True` if you have enough RAM
- Use `mixed_precision=True`
- Reduce `num_slices` if not needed

### Poor Convergence

- Increase `lambda_consistency`
- Adjust learning rate
- Add more training data
- Use data augmentation
- Try different consistency loss types

## Citation

If you use this code in your research, please cite the corresponding paper.

## License

See LICENSE file for details.
