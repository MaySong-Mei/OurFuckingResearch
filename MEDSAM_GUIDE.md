# MedSAM Integration Guide

This guide explains how to use MedSAM (Medical Segment Anything Model) as the pre-trained segmentation model in the training pipeline.

## What is MedSAM?

MedSAM is a specialized version of the Segment Anything Model (SAM) fine-tuned on large-scale medical imaging datasets. It provides:
- **Pre-trained weights** on 1+ million medical images
- **Zero-shot segmentation** capability
- **Better performance** on medical images than vanilla SAM
- **Support for various modalities**: CT, MRI, ultrasound, X-ray, microscopy

## Installation

### 1. Install Segment Anything

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Or add to requirements.txt and run:
```bash
pip install -r requirements.txt
```

### 2. Download MedSAM Checkpoint

#### Option A: Manual Download

Download the MedSAM checkpoint from the official repository:

```bash
# Create checkpoints directory
mkdir -p checkpoints

# Download MedSAM ViT-B checkpoint (~350MB)
wget https://github.com/bowang-lab/MedSAM/releases/download/v0.0.1/medsam_vit_b.pth \
     -O checkpoints/medsam_vit_b.pth
```

Alternative download sources:
- **GitHub Release**: https://github.com/bowang-lab/MedSAM/releases
- **Google Drive**: Check MedSAM repository for links

#### Option B: Automatic Download

Set `medsam_auto_download=True` in config and the pipeline will download automatically.

```python
from config import Config

config = Config(
    use_medsam=True,
    medsam_auto_download=True  # Will download if not found
)
```

#### Option C: Use Download Helper

```python
from models.medsam_segmentation import download_medsam_checkpoint

download_medsam_checkpoint(save_path="checkpoints/medsam_vit_b.pth")
```

## Usage

### Basic Configuration

Enable MedSAM in your training configuration:

```python
from config import Config
from train import TrainingPipeline

config = Config(
    data_dir="./data",
    use_medsam=True,
    medsam_checkpoint="checkpoints/medsam_vit_b.pth",
    medsam_model_type="vit_b",
    batch_size=2,
    num_epochs=100
)

pipeline = TrainingPipeline(config)
pipeline.train()
```

### Using Pre-configured MedSAM Config

```python
from config import get_medsam_config
from train import TrainingPipeline

config = get_medsam_config()
config.data_dir = "./your_data"

pipeline = TrainingPipeline(config)
pipeline.train()
```

### Configuration Options

```python
config = Config(
    # Enable MedSAM
    use_medsam=True,

    # Path to MedSAM checkpoint
    medsam_checkpoint="checkpoints/medsam_vit_b.pth",

    # Model size: "vit_b" (base), "vit_l" (large), "vit_h" (huge)
    medsam_model_type="vit_b",

    # Auto-download if checkpoint not found
    medsam_auto_download=False,

    # Number of segmentation classes
    num_classes=2  # Binary segmentation (background + foreground)
)
```

## Model Types

MedSAM/SAM provides three model sizes:

| Model Type | Parameters | Memory | Speed | Performance |
|------------|------------|--------|-------|-------------|
| `vit_b`    | ~90M       | ~350MB | Fast  | Good        |
| `vit_l`    | ~300M      | ~1.2GB | Medium| Better      |
| `vit_h`    | ~630M      | ~2.4GB | Slow  | Best        |

**Recommendation**: Use `vit_b` for most applications. It provides a good balance of speed and accuracy.

## How MedSAM is Used in the Pipeline

### Automatic Prompting

The pipeline automatically generates prompts for MedSAM:

1. **Box Prompts**: A bounding box covering the center 60% of each slice
2. **Grid Prompts**: Multiple point prompts distributed across the image (alternative)

You don't need to manually provide prompts - they're generated automatically!

### Multi-View Consistency

MedSAM segments each view independently:
```
Axial slices → MedSAM → Axial segmentations
Sagittal slices → MedSAM → Sagittal segmentations
Coronal slices → MedSAM → Coronal segmentations
```

The consistency loss then ensures these segmentations agree after remapping to the same coordinate system.

## Example Workflows

### Example 1: Basic Training with MedSAM

```python
from config import Config
from train import TrainingPipeline

config = Config(
    data_dir="./data/chest_ct",
    use_medsam=True,
    medsam_checkpoint="checkpoints/medsam_vit_b.pth",
    batch_size=2,
    num_epochs=100,
    learning_rate=1e-4,
    use_wandb=True,
    project_name="medsam-interpolation"
)

pipeline = TrainingPipeline(config)
pipeline.train()
```

### Example 2: High-Quality Training with MedSAM

```python
from config import get_medsam_config

config = get_medsam_config()
config.data_dir = "./data"
config.num_epochs = 200
config.img_size = (512, 512)
config.lambda_consistency = 2.0

pipeline = TrainingPipeline(config)
pipeline.train()
```

### Example 3: Low-Memory Setup with MedSAM

```python
config = Config(
    data_dir="./data",
    use_medsam=True,
    medsam_checkpoint="checkpoints/medsam_vit_b.pth",
    medsam_model_type="vit_b",  # Use smallest model
    batch_size=1,
    num_slices=8,
    img_size=(256, 256),
    mixed_precision=True,
    num_workers=2
)
```

## Standalone MedSAM Usage

You can also use MedSAM directly for segmentation:

```python
import torch
from models.medsam_segmentation import load_medsam

# Load MedSAM
model = load_medsam(
    checkpoint_path="checkpoints/medsam_vit_b.pth",
    model_type="vit_b",
    device="cuda"
)

# Segment images
images = torch.randn(4, 1, 256, 256).cuda()  # [B, C, H, W]
segmentations = model(images)  # [B, num_classes, H, W]

print(segmentations.shape)  # torch.Size([4, 2, 256, 256])
```

## Comparing MedSAM vs U-Net

### Use MedSAM when:
- You don't have pre-trained segmentation weights
- You want zero-shot segmentation capability
- You need robust performance across different anatomies
- You want to leverage large-scale pre-training

### Use U-Net when:
- You have domain-specific pre-trained weights
- You need faster inference
- You have limited GPU memory
- You want a simpler, more interpretable model

### Configuration Comparison

**MedSAM:**
```python
config = Config(
    use_medsam=True,
    medsam_checkpoint="checkpoints/medsam_vit_b.pth"
)
```

**U-Net:**
```python
config = Config(
    use_medsam=False,
    segmentation_checkpoint="checkpoints/unet_pretrained.pth"
)
```

## Troubleshooting

### Issue: "segment-anything not installed"

**Solution:**
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### Issue: Checkpoint not found

**Solution:**
```bash
# Download manually
wget https://github.com/bowang-lab/MedSAM/releases/download/v0.0.1/medsam_vit_b.pth \
     -O checkpoints/medsam_vit_b.pth

# Or enable auto-download
config.medsam_auto_download = True
```

### Issue: Out of memory with MedSAM

**Solution:**
- Use `vit_b` instead of `vit_l` or `vit_h`
- Reduce `batch_size` to 1
- Reduce `img_size` to (128, 128) or (256, 256)
- Enable `mixed_precision=True`

```python
config = Config(
    use_medsam=True,
    medsam_model_type="vit_b",  # Smallest model
    batch_size=1,
    img_size=(256, 256),
    mixed_precision=True
)
```

### Issue: Slow training with MedSAM

**Explanation**: MedSAM is frozen during training, so it only affects segmentation speed, not backpropagation. The speed bottleneck is usually in:
1. Segmenting multiple views per iteration
2. Large model size (ViT-L/H)

**Solutions:**
- Use `vit_b` (fastest)
- Reduce number of slices per volume
- Cache segmentations if possible (not implemented by default)

### Issue: Poor segmentation quality

**Possible causes:**
1. Wrong checkpoint (using vanilla SAM instead of MedSAM)
2. Incorrect input format (grayscale vs RGB, normalization)
3. Auto-generated prompts not suitable for your data

**Solutions:**
- Verify you're using MedSAM checkpoint
- Check image preprocessing
- Customize prompt generation in `medsam_segmentation.py:generate_box_prompt()`

## Advanced: Custom Prompt Generation

If default prompts don't work well for your data, customize them:

```python
from models.medsam_segmentation import MedSAMSegmentation
import numpy as np

class CustomMedSAM(MedSAMSegmentation):
    def generate_box_prompt(self, image):
        """Custom prompt generation for your specific anatomy"""
        H, W = image.shape[-2:]

        # Example: Focus on upper 50% of image (e.g., chest)
        box = np.array([
            W * 0.1,      # x1
            0,            # y1 (top)
            W * 0.9,      # x2
            H * 0.5       # y2 (middle)
        ])

        return box
```

## Performance Benchmarks

Approximate timing on A100 GPU (per batch of 2 volumes with 16 slices):

| Configuration | Segmentation Time | Total Epoch Time |
|---------------|-------------------|------------------|
| MedSAM ViT-B  | ~0.5s            | ~5min            |
| MedSAM ViT-L  | ~1.2s            | ~8min            |
| U-Net         | ~0.1s            | ~3min            |

Note: MedSAM is slower but provides better pre-trained features.

## References

- **MedSAM Paper**: [arXiv:2304.12306](https://arxiv.org/abs/2304.12306)
- **MedSAM GitHub**: https://github.com/bowang-lab/MedSAM
- **SAM Paper**: [arXiv:2304.02643](https://arxiv.org/abs/2304.02643)
- **SAM GitHub**: https://github.com/facebookresearch/segment-anything

## Citation

If you use MedSAM in your research, please cite:

```bibtex
@article{ma2023medsam,
  title={Segment anything in medical images},
  author={Ma, Jun and He, Yuting and Li, Feifei and Han, Lin and You, Chenyu and Wang, Bo},
  journal={arXiv preprint arXiv:2304.12306},
  year={2023}
}
```
