"""
为三个视角可视化一致性权重，并叠加到切片上
简化版：先在轴向视图上计算权重，然后在三个不同视角上展示

关键想法：
- 权重基于轴向视图的一致性计算 [Z, H, W]
- 然后在轴向、矢状、冠状三个视图上分别可视化这个权重
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import logging
import sys

# Setup paths
_current_dir = str(Path(__file__).parent)
if _current_dir in sys.path:
    sys.path.remove(_current_dir)
sys.path.insert(0, _current_dir)

from data_loader import MedicalVolumeDataset
from models.medsam_infer import MedSAM2Segmenter
from losses import SegmentationConsistencyWeighting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleThreeViewWeightVisualizer:
    """Visualize consistency weights on three views"""

    def __init__(self, data_dir: str, num_samples: int = 3, device: str = 'cuda',
                 w_min: float = 0.5, w_max: float = 3.0, tau: float = 0.02, kappa: float = 0.1):
        """Initialize"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_samples = num_samples
        self.w_min = w_min
        self.w_max = w_max
        self.tau = tau
        self.kappa = kappa

        # Initialize dataset
        self.dataset = MedicalVolumeDataset(
            data_dir=data_dir,
            split='train',
            max_slices=32
        )

        # Initialize MedSAM2
        medsam2_config = "//" + "/gpfs/radev/project/zhuoran_yang/sl3348/Med_Segmentation/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
        medsam2_ckpt = "/gpfs/radev/project/zhuoran_yang/sl3348/Med_Segmentation/MedSAM2/checkpoints/MedSAM2_latest.pt"

        try:
            self.segmentation = MedSAM2Segmenter(
                config_file=medsam2_config,
                ckpt_path=medsam2_ckpt,
                num_classes=2,
                device=str(self.device),
                use_bfloat16=True
            )
            logger.info(f"Loaded MedSAM2")
        except Exception as e:
            logger.error(f"Failed to load MedSAM2: {e}")
            raise

        self.segmentation.eval()
        for param in self.segmentation.parameters():
            param.requires_grad = False

        # Consistency weighting
        self.consistency_weighting = SegmentationConsistencyWeighting(
            w_min=w_min, w_max=w_max, tau=tau, kappa=kappa
        )

    def get_axial_segmentation(self, volume: torch.Tensor):
        """Get axial segmentation"""
        volume_3d = volume.unsqueeze(1)
        with torch.no_grad():
            seg = self.segmentation(volume_3d)
        return seg  # [B, Z, H, W, 2]

    def create_weight_overlay(self, slice_2d: np.ndarray, weight_2d: np.ndarray) -> Image.Image:
        """Create weight overlay on 2D slice with smooth gradient colormap"""
        # Normalize slice
        if slice_2d.max() > slice_2d.min():
            img_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min())
        else:
            img_norm = np.zeros_like(slice_2d)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        img_rgb = np.stack([img_uint8] * 3, axis=-1).astype(np.float32)

        # Normalize weight to [0, 1]
        if weight_2d.max() > weight_2d.min():
            w_norm = (weight_2d - weight_2d.min()) / (weight_2d.max() - weight_2d.min())
        else:
            w_norm = np.zeros_like(weight_2d)

        # Create single-color (yellow) gradient heatmap
        # Map value [0, 1] to yellow intensity gradient
        # 0.0: Very light yellow (almost transparent/white)
        # 0.5: Bright yellow (255, 255, 0)
        # 1.0: Deep orange (255, 165, 0)

        # Using vectorized operations for efficiency
        heatmap_rgb = np.zeros((*w_norm.shape, 3), dtype=np.float32)

        r = np.zeros_like(w_norm)
        g = np.zeros_like(w_norm)
        b = np.zeros_like(w_norm)

        # Enhanced yellow to orange-red gradient (single color with larger variation)
        # 0.0: Very light yellow (255, 255, 150)
        # 0.5: Golden yellow (255, 200, 0)
        # 1.0: Deep red-orange (255, 100, 0)

        # Light yellow to golden yellow (0.0-0.5)
        mask1 = w_norm <= 0.5
        t1 = w_norm[mask1] / 0.5
        r[mask1] = 255
        g[mask1] = 255 - 55 * t1  # 255 to 200
        b[mask1] = 150 * (1 - t1)  # 150 to 0

        # Golden yellow to deep orange-red (0.5-1.0)
        mask2 = w_norm > 0.5
        t2 = (w_norm[mask2] - 0.5) / 0.5
        r[mask2] = 255
        g[mask2] = 200 - 100 * t2  # 200 to 100
        b[mask2] = 0

        heatmap_rgb[:, :, 0] = r
        heatmap_rgb[:, :, 1] = g
        heatmap_rgb[:, :, 2] = b

        # Blend: 50% original image + 50% heatmap for better visibility of color variations
        # Note: Weights have range [1.625, 2.895] but normalize to [0,1], so small weight differences
        # are nearly invisible with low heatmap blend. 50% blend provides better visual clarity.
        overlay = (0.50 * img_rgb + 0.50 * heatmap_rgb).astype(np.uint8)

        return Image.fromarray(overlay)

    def visualize_sample(self, sample_idx: int, output_dir: Path):
        """Visualize one sample"""
        logger.info(f"\n[{sample_idx + 1}/{self.num_samples}] Processing sample {sample_idx}...")

        # Load data
        sample = self.dataset[sample_idx]
        volume = sample['ground_truth_slices'].unsqueeze(0).to(self.device)  # [1, Z, H, W]
        volume_np = volume[0].cpu().numpy()  # [Z, H, W]

        Z, H, W = volume_np.shape
        logger.info(f"Volume shape: {volume_np.shape}")

        # Create sample directory
        sample_dir = output_dir / f'sample_{sample_idx:03d}'
        sample_dir.mkdir(parents=True, exist_ok=True)

        # ============ Compute axial segmentation and weights ============
        with torch.no_grad():
            # Axial segmentation
            seg_axial = self.get_axial_segmentation(volume)  # [1, Z, H, W, 2]
            prob_axial = F.softmax(seg_axial, dim=-1)[0].cpu().numpy()  # [Z, H, W, 2]

            # Sagittal segmentation
            volume_sagittal = volume.permute(0, 3, 1, 2).unsqueeze(1)  # [1, 1, D, H, W]
            seg_sagittal = self.segmentation(volume_sagittal)  # [1, D, H, W, 2]
            prob_sagittal = F.softmax(seg_sagittal, dim=-1)[0].cpu().numpy()  # [D, H, W, 2]

            # Coronal segmentation
            volume_coronal = volume.permute(0, 2, 3, 1).unsqueeze(1)  # [1, 1, W, D, H]
            seg_coronal = self.segmentation(volume_coronal)  # [1, W, D, H, 2]
            prob_coronal = F.softmax(seg_coronal, dim=-1)[0].cpu().numpy()  # [W, D, H, 2]

        # 简化方法：只使用axial视图计算一致性权重
        # 然后在三个视图上分别可视化这个权重图

        # 原因：三个视图的深度维度可能不同，为了避免维度不匹配，
        # 我们需要确保所有概率图都有相同的空间维度后再计算权重

        with torch.no_grad():
            # Axial: [C=2, Z, H, W] -> [Z, H, W, C] -> [1, Z, H, W, C]
            prob_axial_transposed = prob_axial.transpose(1, 2, 3, 0)  # [Z, H, W, C]
            w_axial = torch.from_numpy(prob_axial_transposed).unsqueeze(0).to(self.device)  # [1, Z, H, W, C]

            # Sagittal: [C=2, W, Z, H] -> [Z, H, W, C]
            # From permute(0,3,1,2): original [1, Z, H, W] -> [1, W, Z, H]
            # Output: [C, W_orig, Z_orig, H_orig] = [2, 256, Z, 256]
            # Need: [Z, H, W, C] = [Z, 256, 256, 2]
            prob_sagittal_transposed = prob_sagittal.transpose(2, 3, 1, 0)  # [Z, H, W, C]

            # Ensure Z dimension matches axial's Z
            z_sag = prob_sagittal_transposed.shape[0]
            if z_sag < Z:
                prob_sagittal_transposed = np.pad(prob_sagittal_transposed, ((0, Z-z_sag), (0,0), (0,0), (0,0)), mode='edge')
            elif z_sag > Z:
                prob_sagittal_transposed = prob_sagittal_transposed[:Z]

            w_sag = torch.from_numpy(prob_sagittal_transposed).unsqueeze(0).to(self.device)  # [1, Z, H, W, C]

            # Coronal: [C=2, H, W, Z] -> [Z, H, W, C]
            # From permute(0,2,3,1): original [1, Z, H, W] -> [1, H, W, Z]
            # Output: [C, H_orig, W_orig, Z_orig] = [2, 256, 256, Z]
            # Need: [Z, H, W, C] = [Z, 256, 256, 2]
            prob_coronal_transposed = prob_coronal.transpose(3, 1, 2, 0)  # [Z, H, W, C]

            # Ensure Z dimension matches axial's Z
            z_cor = prob_coronal_transposed.shape[0]
            if z_cor < Z:
                prob_coronal_transposed = np.pad(prob_coronal_transposed, ((0, Z-z_cor), (0,0), (0,0), (0,0)), mode='edge')
            elif z_cor > Z:
                prob_coronal_transposed = prob_coronal_transposed[:Z]

            w_cor = torch.from_numpy(prob_coronal_transposed).unsqueeze(0).to(self.device)  # [1, Z, H, W, C]

            # All three inputs now have shape [1, Z, H, W, C]
            weights = self.consistency_weighting(w_axial, w_sag, w_cor)  # [1, Z, H, W]

        weights_np = weights[0].cpu().numpy()  # [Z, H, W]

        logger.info(f"Weights - mean={weights_np.mean():.4f}, min={weights_np.min():.4f}, max={weights_np.max():.4f}")

        # ============ 1. AXIAL VIEW: iterate Z axis ============
        axial_dir = sample_dir / 'axial_view'
        axial_dir.mkdir(exist_ok=True)
        axial_weight_dir = axial_dir / 'weights'
        axial_overlay_dir = axial_dir / 'weights_overlay'
        axial_weight_dir.mkdir(exist_ok=True)
        axial_overlay_dir.mkdir(exist_ok=True)

        for z in range(Z):
            # Heatmap
            w_heatmap = self.create_weight_overlay(np.zeros((H, W)), weights_np[z])
            w_heatmap.save(axial_weight_dir / f'axial_weight_{z:03d}.png')

            # Overlay
            w_overlay = self.create_weight_overlay(volume_np[z], weights_np[z])
            w_overlay.save(axial_overlay_dir / f'axial_weight_overlay_{z:03d}.png')

        logger.info(f"  ✓ Axial view: {Z} slices")

        # ============ 2. SAGITTAL VIEW: iterate W (depth) axis ============
        sagittal_dir = sample_dir / 'sagittal_view'
        sagittal_dir.mkdir(exist_ok=True)
        sagittal_weight_dir = sagittal_dir / 'weights'
        sagittal_overlay_dir = sagittal_dir / 'weights_overlay'
        sagittal_weight_dir.mkdir(exist_ok=True)
        sagittal_overlay_dir.mkdir(exist_ok=True)

        for w in range(W):
            # Sagittal slice at W: [Z, H]
            w_slice = weights_np[:, :, w]  # [Z, H]
            img_slice = volume_np[:, :, w]  # [Z, H]

            # Heatmap
            w_heatmap = self.create_weight_overlay(np.zeros_like(w_slice), w_slice)
            w_heatmap.save(sagittal_weight_dir / f'sagittal_weight_{w:03d}.png')

            # Overlay
            w_overlay = self.create_weight_overlay(img_slice, w_slice)
            w_overlay.save(sagittal_overlay_dir / f'sagittal_weight_overlay_{w:03d}.png')

        logger.info(f"  ✓ Sagittal view: {W} slices")

        # ============ 3. CORONAL VIEW: iterate H (height) axis ============
        coronal_dir = sample_dir / 'coronal_view'
        coronal_dir.mkdir(exist_ok=True)
        coronal_weight_dir = coronal_dir / 'weights'
        coronal_overlay_dir = coronal_dir / 'weights_overlay'
        coronal_weight_dir.mkdir(exist_ok=True)
        coronal_overlay_dir.mkdir(exist_ok=True)

        for h in range(H):
            # Coronal slice at H: [Z, W]
            w_slice = weights_np[:, h, :]  # [Z, W]
            img_slice = volume_np[:, h, :]  # [Z, W]

            # Heatmap
            w_heatmap = self.create_weight_overlay(np.zeros_like(w_slice), w_slice)
            w_heatmap.save(coronal_weight_dir / f'coronal_weight_{h:03d}.png')

            # Overlay
            w_overlay = self.create_weight_overlay(img_slice, w_slice)
            w_overlay.save(coronal_overlay_dir / f'coronal_weight_overlay_{h:03d}.png')

        logger.info(f"  ✓ Coronal view: {H} slices")
        logger.info(f"✓ Sample {sample_idx} complete: {sample_dir}")

    def visualize(self):
        """Visualize all samples"""
        output_dir = Path('/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/weight_checkpoints/3view_consistency_weights_overlay_v2')
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("THREE-VIEW CONSISTENCY WEIGHT VISUALIZATION (Simple)")
        logger.info("=" * 80)
        logger.info(f"Output: {output_dir}\n")

        for i in range(self.num_samples):
            try:
                self.visualize_sample(i, output_dir)
            except Exception as e:
                logger.error(f"Sample {i} error: {e}")
                import traceback
                traceback.print_exc()

        logger.info("\n" + "=" * 80)
        logger.info("✓ COMPLETE")
        logger.info("=" * 80)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/Colon_data')
    parser.add_argument('--num_samples', type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    visualizer = SimpleThreeViewWeightVisualizer(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        device=args.device
    )

    visualizer.visualize()
