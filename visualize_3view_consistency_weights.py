"""
可视化三个视角上的一致性权重
为轴向、矢状、冠状视图都生成权重热力图并叠加到原始切片上
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


class ThreeViewConsistencyWeightVisualizer:
    """Visualize consistency weights from all three views"""

    def __init__(self, data_dir: str, num_samples: int = 3, device: str = 'cuda',
                 w_min: float = 0.5, w_max: float = 2.0, tau: float = 0.15, kappa: float = 0.4):
        """
        Initialize visualizer for three-view consistency weights

        Args:
            data_dir: Path to data directory
            num_samples: Number of samples to visualize
            device: 'cuda' or 'cpu'
            w_min, w_max, tau, kappa: Consistency weight parameters
        """
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

        # Initialize MedSAM2 segmenter
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
            logger.info(f"Loaded MedSAM2 from {medsam2_ckpt}")
        except Exception as e:
            logger.error(f"Failed to load MedSAM2: {e}")
            raise

        self.segmentation.eval()
        for param in self.segmentation.parameters():
            param.requires_grad = False

        # Initialize consistency weighting
        self.consistency_weighting = SegmentationConsistencyWeighting(
            w_min=w_min, w_max=w_max, tau=tau, kappa=kappa
        )

    def compute_multi_view_segmentations(self, volume: torch.Tensor):
        """Compute segmentations for three orthogonal views"""
        B = volume.shape[0]
        volume_3d = volume.unsqueeze(1)

        with torch.no_grad():
            # Axial view (original XY slices)
            seg_axial = self.segmentation(volume_3d)  # [B, Z, H, W, 2]

            # Sagittal view (XZ slices)
            vol_sagittal = volume.permute(0, 3, 1, 2).unsqueeze(1)  # [B, 1, D, H, W]
            seg_sagittal = self.segmentation(vol_sagittal)  # [B, D, H, W, 2]

            # Coronal view (YZ slices)
            vol_coronal = volume.permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, W, D, H]
            seg_coronal = self.segmentation(vol_coronal)  # [B, W, D, H, 2]

        return seg_axial, seg_sagittal, seg_coronal

    def create_weight_heatmap(self, weight_values: np.ndarray) -> Image.Image:
        """Create heatmap for consistency weights using Red-Green gradient"""
        # Normalize to [0, 1]
        if weight_values.max() > weight_values.min():
            normalized = (weight_values - weight_values.min()) / (weight_values.max() - weight_values.min())
        else:
            normalized = np.zeros_like(weight_values)

        # Create heatmap using Red-Green gradient (matching segmentation heatmap style)
        heatmap = np.zeros((*normalized.shape, 3), dtype=np.uint8)
        heatmap[:, :, 0] = (normalized * 255).astype(np.uint8)  # Red
        heatmap[:, :, 1] = (np.sqrt(normalized) * 255).astype(np.uint8)  # Green
        heatmap[:, :, 2] = 0  # No blue

        return Image.fromarray(heatmap)

    def create_weight_overlay(self, slice_np: np.ndarray, weight_values: np.ndarray) -> Image.Image:
        """Create weight heatmap overlaid on original slice"""
        # Normalize slice to 0-255
        if slice_np.max() > slice_np.min():
            img_normalized = (slice_np - slice_np.min()) / (slice_np.max() - slice_np.min())
        else:
            img_normalized = np.zeros_like(slice_np)
        img_uint8 = (img_normalized * 255).astype(np.uint8)
        img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

        # Normalize weight values to [0, 1]
        if weight_values.max() > weight_values.min():
            weight_normalized = (weight_values - weight_values.min()) / (weight_values.max() - weight_values.min())
        else:
            weight_normalized = np.zeros_like(weight_values)

        # Create heatmap using same Red-Green gradient
        weight_uint8 = (weight_normalized * 255).astype(np.uint8)
        weight_sqrt = (np.sqrt(weight_normalized) * 255).astype(np.uint8)

        heatmap_rgb = np.zeros((*weight_normalized.shape, 3), dtype=np.uint8)
        heatmap_rgb[:, :, 0] = weight_uint8  # Red
        heatmap_rgb[:, :, 1] = weight_sqrt   # Green
        heatmap_rgb[:, :, 2] = 0             # No blue

        # Blend: 70% image + 30% heatmap
        overlay = (0.7 * img_rgb + 0.3 * heatmap_rgb).astype(np.uint8)

        return Image.fromarray(overlay)

    def visualize_sample(self, sample_idx: int, output_dir: Path):
        """Visualize one sample with three-view consistency weights"""
        logger.info(f"\n[{sample_idx + 1}/{self.num_samples}] Processing sample {sample_idx}...")

        # Load data
        sample = self.dataset[sample_idx]
        # Data loader returns 'slices' (4 sampled slices) and 'ground_truth_slices' (all slices)
        # Use ground_truth_slices for full volume
        volume = sample['ground_truth_slices'].unsqueeze(0).to(self.device)  # [1, 2D-1, H, W]
        volume_np = volume[0].cpu().numpy()  # [2D-1, H, W] (all interpolated+original slices)

        logger.info(f"Volume shape: {volume_np.shape}")

        # Create sample directory
        sample_dir = output_dir / f'sample_{sample_idx:03d}'
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Compute three-view segmentations
        seg_axial, seg_sagittal, seg_coronal = self.compute_multi_view_segmentations(volume)

        # Extract probabilities
        prob_axial = F.softmax(seg_axial, dim=-1).squeeze(0).cpu().numpy()  # [Z, H, W, 2]
        prob_sagittal = F.softmax(seg_sagittal, dim=-1).squeeze(0).cpu().numpy()  # [D, H, W, 2]
        prob_coronal = F.softmax(seg_coronal, dim=-1).squeeze(0).cpu().numpy()  # [W, D, H, 2]

        Z, H, W = volume_np.shape

        # ============ Remap to consistent spatial format [Z, H, W, 2] ============
        # Sagittal is [D, H, W, 2] where D=depth (Z), need to remap to [Z, H, W, 2]
        # No remapping needed, they're already aligned
        prob_sagittal_remapped = prob_sagittal  # Already [Z, H, W, 2]

        # Coronal is [W, D, H, 2] where D=depth (Z), W=width, H=height
        # Need to remap to [Z, H, W, 2]: [W, D, H, 2] -> [D, H, W, 2]
        prob_coronal_remapped = prob_coronal.transpose(1, 2, 0, 3)  # [W, D, H, 2] -> [D, H, W, 2]

        # ============ Compute consistency weights (same for all views) ============
        prob_axial_torch = torch.from_numpy(prob_axial).unsqueeze(0).to(self.device)  # [1, Z, H, W, 2]
        prob_sagittal_torch = torch.from_numpy(prob_sagittal_remapped).unsqueeze(0).to(self.device)
        prob_coronal_torch = torch.from_numpy(prob_coronal_remapped).unsqueeze(0).to(self.device)

        with torch.no_grad():
            consistency_weights = self.consistency_weighting(
                prob_axial_torch, prob_sagittal_torch, prob_coronal_torch
            )  # [1, Z, H, W]

        weights_np = consistency_weights[0].cpu().numpy()  # [Z, H, W]

        logger.info(f"Consistency weights - mean={weights_np.mean():.4f}, min={weights_np.min():.4f}, max={weights_np.max():.4f}")

        # ============ Visualize for three views ============

        # 1. AXIAL VIEW (Z, H, W)
        axial_dir = sample_dir / 'axial_view'
        axial_dir.mkdir(exist_ok=True)

        axial_weight_dir = axial_dir / 'weights'
        axial_overlay_dir = axial_dir / 'weights_overlay'
        axial_weight_dir.mkdir(exist_ok=True)
        axial_overlay_dir.mkdir(exist_ok=True)

        for z_idx in range(Z):
            weight_slice = weights_np[z_idx]  # [H, W]
            slice_data = volume_np[z_idx]  # [H, W]

            # Save standalone heatmap
            weight_img = self.create_weight_heatmap(weight_slice)
            weight_path = axial_weight_dir / f'axial_weight_{z_idx:03d}.png'
            weight_img.save(weight_path)

            # Save overlay
            overlay_img = self.create_weight_overlay(slice_data, weight_slice)
            overlay_path = axial_overlay_dir / f'axial_weight_overlay_{z_idx:03d}.png'
            overlay_img.save(overlay_path)

        logger.info(f"  ✓ Axial view: {Z} slices saved")

        # 2. SAGITTAL VIEW (Z, W, H)
        sagittal_dir = sample_dir / 'sagittal_view'
        sagittal_dir.mkdir(exist_ok=True)

        sagittal_weight_dir = sagittal_dir / 'weights'
        sagittal_overlay_dir = sagittal_dir / 'weights_overlay'
        sagittal_weight_dir.mkdir(exist_ok=True)
        sagittal_overlay_dir.mkdir(exist_ok=True)

        # Sagittal view uses different slicing: volume.permute(0, 3, 1, 2) means [Z, W, H]
        volume_sagittal = volume_np.transpose(2, 0, 1)  # [H, Z, W] -> need [W, Z, H]
        volume_sagittal = volume_np.transpose(2, 1, 0)  # [W, H, Z] -> [W, Z, H] (depth slices perpendicular)

        # For sagittal: iterate through W (depth in original volume)
        for w_idx in range(W):
            weight_slice = weights_np[:, :, w_idx]  # [Z, H] - sagittal slice at position w
            slice_data = volume_np[:, :, w_idx]  # [Z, H] - original sagittal slice

            # Save standalone heatmap
            weight_img = self.create_weight_heatmap(weight_slice)
            weight_path = sagittal_weight_dir / f'sagittal_weight_{w_idx:03d}.png'
            weight_img.save(weight_path)

            # Save overlay
            overlay_img = self.create_weight_overlay(slice_data, weight_slice)
            overlay_path = sagittal_overlay_dir / f'sagittal_weight_overlay_{w_idx:03d}.png'
            overlay_img.save(overlay_path)

        logger.info(f"  ✓ Sagittal view: {W} slices saved")

        # 3. CORONAL VIEW (Z, H, W)
        coronal_dir = sample_dir / 'coronal_view'
        coronal_dir.mkdir(exist_ok=True)

        coronal_weight_dir = coronal_dir / 'weights'
        coronal_overlay_dir = coronal_dir / 'weights_overlay'
        coronal_weight_dir.mkdir(exist_ok=True)
        coronal_overlay_dir.mkdir(exist_ok=True)

        # For coronal: iterate through H (height in original volume)
        for h_idx in range(H):
            weight_slice = weights_np[:, h_idx, :]  # [Z, W] - coronal slice at position h
            slice_data = volume_np[:, h_idx, :]  # [Z, W] - original coronal slice

            # Save standalone heatmap
            weight_img = self.create_weight_heatmap(weight_slice)
            weight_path = coronal_weight_dir / f'coronal_weight_{h_idx:03d}.png'
            weight_img.save(weight_path)

            # Save overlay
            overlay_img = self.create_weight_overlay(slice_data, weight_slice)
            overlay_path = coronal_overlay_dir / f'coronal_weight_overlay_{h_idx:03d}.png'
            overlay_img.save(overlay_path)

        logger.info(f"  ✓ Coronal view: {H} slices saved")

        logger.info(f"✓ Saved all three-view visualizations to {sample_dir}")

    def visualize(self):
        """Visualize all samples"""
        output_dir = Path('/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/weight_checkpoints/3view_consistency_weights_overlay')
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("═" * 80)
        logger.info("THREE-VIEW CONSISTENCY WEIGHT VISUALIZATION")
        logger.info("═" * 80)
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Weight parameters: w_min={self.w_min}, w_max={self.w_max}, tau={self.tau}, kappa={self.kappa}")
        logger.info(f"Visualizing {self.num_samples} samples...\n")

        for sample_idx in range(self.num_samples):
            try:
                self.visualize_sample(sample_idx, output_dir)
            except Exception as e:
                logger.error(f"Error processing sample {sample_idx}: {e}")
                import traceback
                traceback.print_exc()

        logger.info("\n" + "═" * 80)
        logger.info("✓ VISUALIZATION COMPLETE")
        logger.info("═" * 80)
        logger.info(f"\nOutput location: {output_dir}")
        logger.info("\nFor each sample, you'll find:")
        logger.info("  • axial_view/weights/ - Standalone weight heatmaps")
        logger.info("  • axial_view/weights_overlay/ - Weights overlaid on slices")
        logger.info("  • sagittal_view/weights/ - Standalone weight heatmaps")
        logger.info("  • sagittal_view/weights_overlay/ - Weights overlaid on slices")
        logger.info("  • coronal_view/weights/ - Standalone weight heatmaps")
        logger.info("  • coronal_view/weights_overlay/ - Weights overlaid on slices")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize three-view consistency weights')
    parser.add_argument('--data_dir', type=str, default='/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/Colon_data',
                        help='Path to data directory')
    parser.add_argument('--num_samples', type=int, default=3,
                        help='Number of samples to visualize')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--weight_w_min', type=float, default=0.5,
                        help='Minimum weight for consistent regions')
    parser.add_argument('--weight_w_max', type=float, default=2.0,
                        help='Maximum weight for inconsistent regions')
    parser.add_argument('--weight_tau', type=float, default=0.15,
                        help='Variance tolerance threshold')
    parser.add_argument('--weight_kappa', type=float, default=0.4,
                        help='Sigmoid steepness coefficient')

    args = parser.parse_args()

    visualizer = ThreeViewConsistencyWeightVisualizer(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        device=args.device,
        w_min=args.weight_w_min,
        w_max=args.weight_w_max,
        tau=args.weight_tau,
        kappa=args.weight_kappa
    )

    visualizer.visualize()
