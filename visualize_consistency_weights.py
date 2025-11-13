"""
Visualize consistency weights from three-view segmentation results
Shows pixel-wise weight distribution based on multi-view segmentation variance
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
from models.i3net_adapter import I3NetInterpolator
from losses import SegmentationConsistencyWeighting

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsistencyWeightVisualizer:
    """Visualize consistency weights from three-view segmentations"""

    def __init__(self, data_dir: str, num_samples: int = 10, device: str = 'cuda',
                 w_min: float = 0.5, w_max: float = 2.0, tau: float = 0.15, kappa: float = 0.4):
        """
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

        logger.info("Using ground truth slices directly (7 consecutive slices)")

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
            seg_axial = self.segmentation(volume_3d)

            vol_sagittal = volume.permute(0, 3, 1, 2).unsqueeze(1)
            seg_sagittal = self.segmentation(vol_sagittal)

            vol_coronal = volume.permute(0, 2, 3, 1).unsqueeze(1)
            seg_coronal = self.segmentation(vol_coronal)

        return seg_axial, seg_sagittal, seg_coronal

    def compute_consistency_weights(self, seg_axial, seg_sagittal, seg_coronal):
        """
        Compute consistency weights from three-view segmentations

        Returns:
            weights: [B, Z, H, W] consistency weights
            variance: [B, Z, H, W] inconsistency metric
            probs: tuple of (prob_axial, prob_sagittal, prob_coronal)
        """
        # Remap to axial coordinates
        seg_sagittal_remapped = seg_sagittal.permute(0, 1, 3, 4, 2)
        seg_coronal_remapped = seg_coronal.permute(0, 1, 4, 2, 3)

        # Convert to probability format [B, Z, H, W, C]
        seg_axial_fmt = seg_axial.permute(0, 2, 3, 4, 1)
        seg_sag_fmt = seg_sagittal_remapped.permute(0, 2, 3, 4, 1)
        seg_cor_fmt = seg_coronal_remapped.permute(0, 2, 3, 4, 1)

        # Apply softmax to get probabilities
        prob_axial = F.softmax(seg_axial_fmt, dim=-1)
        prob_sagittal = F.softmax(seg_sag_fmt, dim=-1)
        prob_coronal = F.softmax(seg_cor_fmt, dim=-1)

        # Compute consistency weights
        weights = self.consistency_weighting(prob_axial, prob_sagittal, prob_coronal)

        # Compute variance for detailed analysis
        probs_stacked = torch.stack([prob_axial, prob_sagittal, prob_coronal], dim=-1)
        class_variances = torch.var(probs_stacked, dim=-1)
        variance = class_variances.sum(dim=-1)

        return weights, variance, (prob_axial, prob_sagittal, prob_coronal)

    def create_weight_heatmap(self, weights_np: np.ndarray) -> Image.Image:
        """
        Create heatmap visualization for consistency weights

        Args:
            weights_np: [H, W] weight values in range [w_min, w_max]

        Returns:
            PIL Image with weight heatmap
        """
        # Normalize weights to [0, 1] for visualization
        w_normalized = (weights_np - self.w_min) / (self.w_max - self.w_min)
        w_normalized = np.clip(w_normalized, 0, 1)

        # Create RGB heatmap: Red (low weight) → Yellow → White (high weight)
        H, W = weights_np.shape
        heatmap_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # Red channel: always high
        heatmap_rgb[:, :, 0] = (w_normalized * 255).astype(np.uint8)

        # Green channel: increases with weight
        heatmap_rgb[:, :, 1] = (w_normalized ** 0.5 * 255).astype(np.uint8)

        # Blue channel: zero for red-yellow-white gradient
        heatmap_rgb[:, :, 2] = 0

        return Image.fromarray(heatmap_rgb)

    def create_heatmap_overlay(self, slice_np: np.ndarray, heatmap_values: np.ndarray,
                               heatmap_type: str, z_idx: int, output_dir: Path):
        """
        Create and save heatmap overlay on slice, reusing same heatmap style as standalone heatmap

        Args:
            slice_np: [H, W] original medical image slice
            heatmap_values: [H, W] heatmap values (normalized to [0, 1])
            heatmap_type: 'weight' or 'variance'
            z_idx: slice index
            output_dir: output directory
        """
        # Normalize slice to 0-255
        if slice_np.max() > slice_np.min():
            img_normalized = (slice_np - slice_np.min()) / (slice_np.max() - slice_np.min())
        else:
            img_normalized = np.zeros_like(slice_np)
        img_uint8 = (img_normalized * 255).astype(np.uint8)

        # Create RGB image
        img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

        # Create heatmap using same method as standalone heatmap
        # Red-Green gradient: Black → Red → Yellow (matching the standalone heatmap style)
        heatmap_uint8 = (heatmap_values * 255).astype(np.uint8)
        heatmap_sqrt = (heatmap_values ** 0.5 * 255).astype(np.uint8)

        heatmap_rgb = np.zeros((*heatmap_values.shape, 3), dtype=np.uint8)
        heatmap_rgb[:, :, 0] = heatmap_uint8  # Red channel
        heatmap_rgb[:, :, 1] = heatmap_sqrt   # Green channel (sqrt for hotter look)
        heatmap_rgb[:, :, 2] = 0              # No blue

        # Blend: 70% image + 30% heatmap (matching overlay style)
        overlay = (0.7 * img_rgb + 0.3 * heatmap_rgb).astype(np.uint8)

        # Save overlay image
        overlay_img = Image.fromarray(overlay)
        overlay_path = output_dir / f'{heatmap_type}_overlay_{z_idx:03d}.png'
        overlay_img.save(overlay_path)

    def create_variance_heatmap(self, variance_np: np.ndarray) -> Image.Image:
        """
        Create heatmap for variance (inconsistency metric)

        Args:
            variance_np: [H, W] variance values

        Returns:
            PIL Image with variance heatmap
        """
        # Normalize variance for visualization
        v_max = np.percentile(variance_np, 99)
        v_normalized = np.clip(variance_np / v_max, 0, 1)

        # Same colormap as weights
        H, W = variance_np.shape
        heatmap_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        heatmap_rgb[:, :, 0] = (v_normalized * 255).astype(np.uint8)
        heatmap_rgb[:, :, 1] = (v_normalized ** 0.5 * 255).astype(np.uint8)
        heatmap_rgb[:, :, 2] = 0

        return Image.fromarray(heatmap_rgb)

    def visualize_sample(self, sample_idx: int, output_dir: Path):
        """
        Visualize consistency weights for a single sample

        Args:
            sample_idx: Index of sample to visualize
            output_dir: Directory to save visualizations
        """
        # Load data
        batch = self.dataset[sample_idx]
        # Use ground truth slices directly (all slices from the volume)
        volume = batch['ground_truth_slices'].unsqueeze(0).to(self.device)
        file_path = batch['file_path']

        logger.info(f"\n{'='*80}")
        logger.info(f"Sample {sample_idx}: {Path(file_path).name}")
        logger.info(f"Original shape: {batch['volume_shape']}")
        logger.info(f"Using ground truth slices (all {volume.shape[1]} slices)")
        logger.info(f"{'='*80}")

        # Get segmentations from ground truth volume
        seg_axial, seg_sagittal, seg_coronal = self.compute_multi_view_segmentations(
            volume
        )

        # Compute consistency weights
        weights, variance, (prob_axial, prob_sagittal, prob_coronal) = \
            self.compute_consistency_weights(seg_axial, seg_sagittal, seg_coronal)

        # Convert to numpy
        volume_np = volume[0].cpu().numpy()  # [Z, H, W]
        weights_np = weights[0].cpu().numpy()  # [Z, H, W]
        variance_np = variance[0].cpu().numpy()  # [Z, H, W]
        prob_axial_np = prob_axial[0].cpu().numpy()  # [Z, H, W, C]
        prob_sagittal_np = prob_sagittal[0].cpu().numpy()  # [Z, H, W, C]
        prob_coronal_np = prob_coronal[0].cpu().numpy()  # [Z, H, W, C]

        # Create output directory for this sample
        sample_dir = output_dir / f'sample_{sample_idx:03d}'
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Visualize consistency weights and variance (with overlay on original volume)
        self._visualize_weight_slices(weights_np, variance_np, sample_dir, volume_np)

        # Visualize individual view segmentations
        self._visualize_view_segmentations(prob_axial_np, prob_sagittal_np, prob_coronal_np, sample_dir)

        # Create summary statistics
        self._save_statistics(weights_np, variance_np, sample_dir, sample_idx)

        logger.info(f"✓ Saved weight visualizations to {sample_dir}")

    def _visualize_weight_slices(self, weights_np: np.ndarray, variance_np: np.ndarray,
                                 output_dir: Path, volume_np: np.ndarray = None):
        """
        Visualize weights and variance for all slices

        Args:
            weights_np: [Z, H, W]
            variance_np: [Z, H, W]
            output_dir: Output directory
            volume_np: [Z, H, W] optional original volume for overlay visualization
        """
        Z, H, W = weights_np.shape

        # Create subdirectories
        weights_dir = output_dir / 'weights'
        variance_dir = output_dir / 'variance'
        weights_overlay_dir = output_dir / 'weights_overlay'
        variance_overlay_dir = output_dir / 'variance_overlay'

        weights_dir.mkdir(exist_ok=True)
        variance_dir.mkdir(exist_ok=True)
        if volume_np is not None:
            weights_overlay_dir.mkdir(exist_ok=True)
            variance_overlay_dir.mkdir(exist_ok=True)

        # Visualize each slice
        for z_idx in range(Z):
            weight_slice = weights_np[z_idx]  # [H, W]
            variance_slice = variance_np[z_idx]  # [H, W]

            # Create weight heatmap
            weight_img = self.create_weight_heatmap(weight_slice)
            weight_path = weights_dir / f'weight_slice_{z_idx:03d}.png'
            weight_img.save(weight_path)

            # Create variance heatmap
            variance_img = self.create_variance_heatmap(variance_slice)
            variance_path = variance_dir / f'variance_slice_{z_idx:03d}.png'
            variance_img.save(variance_path)

            # Create overlaid heatmaps if original volume provided
            if volume_np is not None:
                slice_data = volume_np[z_idx]  # [H, W]

                # Normalize weights and variance to [0, 1] before overlaying
                weight_normalized = (weight_slice - self.w_min) / (self.w_max - self.w_min)
                weight_normalized = np.clip(weight_normalized, 0, 1)
                variance_normalized = np.clip(variance_slice / np.percentile(variance_slice, 99), 0, 1)

                # Overlay weights on original slice (reuse same heatmap style)
                self.create_heatmap_overlay(slice_data, weight_normalized, 'weight', z_idx, weights_overlay_dir)

                # Overlay variance on original slice (reuse same heatmap style)
                self.create_heatmap_overlay(slice_data, variance_normalized, 'variance', z_idx, variance_overlay_dir)

        logger.info(f"  ✓ Weights: {Z} slices saved to {weights_dir}")
        logger.info(f"  ✓ Variance: {Z} slices saved to {variance_dir}")
        if volume_np is not None:
            logger.info(f"  ✓ Weights overlay: {Z} slices saved to {weights_overlay_dir}")
            logger.info(f"  ✓ Variance overlay: {Z} slices saved to {variance_overlay_dir}")

    def _visualize_view_segmentations(self, prob_axial: np.ndarray, prob_sagittal: np.ndarray,
                                     prob_coronal: np.ndarray, output_dir: Path):
        """
        Visualize individual view segmentations

        Args:
            prob_axial: [Z, H, W, C] probabilities for axial view
            prob_sagittal: [Z, H, W, C] probabilities for sagittal view (already remapped to axial)
            prob_coronal: [Z, H, W, C] probabilities for coronal view (already remapped to axial)
            output_dir: Output directory
        """
        Z, H, W = prob_axial.shape[:3]

        # Create subdirectories for each view
        axial_dir = output_dir / 'segmentation_axial'
        sag_dir = output_dir / 'segmentation_sagittal'
        cor_dir = output_dir / 'segmentation_coronal'

        axial_dir.mkdir(exist_ok=True)
        sag_dir.mkdir(exist_ok=True)
        cor_dir.mkdir(exist_ok=True)

        # Visualize each slice for each view
        for z_idx in range(Z):
            # Get foreground probabilities for each view
            fg_axial = prob_axial[z_idx, :, :, 1]  # [H, W]
            fg_sagittal = prob_sagittal[z_idx, :, :, 1]  # [H, W]
            fg_coronal = prob_coronal[z_idx, :, :, 1]  # [H, W]

            # Create heatmaps for each view
            for view_name, fg_prob, save_dir in [
                ('axial', fg_axial, axial_dir),
                ('sagittal', fg_sagittal, sag_dir),
                ('coronal', fg_coronal, cor_dir),
            ]:
                # Normalize to [0, 255]
                fg_prob_uint8 = (fg_prob * 255).astype(np.uint8)

                # Create heatmap (hot colormap for foreground probability)
                heatmap = np.zeros((fg_prob.shape[0], fg_prob.shape[1], 3), dtype=np.uint8)
                heatmap[:, :, 0] = fg_prob_uint8  # Red channel
                heatmap[:, :, 1] = (fg_prob ** 0.5 * 255).astype(np.uint8)  # Green

                heatmap_img = Image.fromarray(heatmap)
                heatmap_path = save_dir / f'seg_{view_name}_slice_{z_idx:03d}.png'
                heatmap_img.save(heatmap_path)

        logger.info(f"  ✓ Axial segmentations: {Z} slices saved to {axial_dir}")
        logger.info(f"  ✓ Sagittal segmentations: {Z} slices saved to {sag_dir}")
        logger.info(f"  ✓ Coronal segmentations: {Z} slices saved to {cor_dir}")

    def _save_statistics(self, weights_np: np.ndarray, variance_np: np.ndarray,
                        output_dir: Path, sample_idx: int):
        """
        Save statistical analysis of weights and variance

        Args:
            weights_np: [Z, H, W]
            variance_np: [Z, H, W]
            output_dir: Output directory
            sample_idx: Sample index
        """
        stats_file = output_dir / 'weight_statistics.txt'

        with open(stats_file, 'w') as f:
            f.write(f"CONSISTENCY WEIGHT STATISTICS\n")
            f.write(f"{'='*80}\n\n")

            f.write(f"Sample: {sample_idx}\n")
            f.write(f"Shape: {weights_np.shape} (Z, H, W)\n")
            f.write(f"Weight parameters:\n")
            f.write(f"  w_min = {self.w_min}\n")
            f.write(f"  w_max = {self.w_max}\n")
            f.write(f"  tau = {self.tau}\n")
            f.write(f"  kappa = {self.kappa}\n\n")

            # Weight statistics
            f.write(f"WEIGHT STATISTICS\n")
            f.write(f"{'-'*80}\n")
            f.write(f"  Min:        {weights_np.min():.4f}\n")
            f.write(f"  Max:        {weights_np.max():.4f}\n")
            f.write(f"  Mean:       {weights_np.mean():.4f}\n")
            f.write(f"  Median:     {np.median(weights_np):.4f}\n")
            f.write(f"  Std:        {weights_np.std():.4f}\n")
            f.write(f"  Q25:        {np.percentile(weights_np, 25):.4f}\n")
            f.write(f"  Q75:        {np.percentile(weights_np, 75):.4f}\n\n")

            # Weight distribution
            f.write(f"WEIGHT DISTRIBUTION\n")
            f.write(f"{'-'*80}\n")
            low_weight = (weights_np < 0.8).sum()
            mid_weight = ((weights_np >= 0.8) & (weights_np < 1.5)).sum()
            high_weight = (weights_np >= 1.5).sum()
            total = weights_np.size

            f.write(f"  Low weight (< 0.8):     {low_weight} ({low_weight/total*100:.1f}%)\n")
            f.write(f"  Mid weight (0.8-1.5):   {mid_weight} ({mid_weight/total*100:.1f}%)\n")
            f.write(f"  High weight (>= 1.5):   {high_weight} ({high_weight/total*100:.1f}%)\n\n")

            # Variance statistics
            f.write(f"VARIANCE (INCONSISTENCY) STATISTICS\n")
            f.write(f"{'-'*80}\n")
            f.write(f"  Min:        {variance_np.min():.6f}\n")
            f.write(f"  Max:        {variance_np.max():.6f}\n")
            f.write(f"  Mean:       {variance_np.mean():.6f}\n")
            f.write(f"  Median:     {np.median(variance_np):.6f}\n")
            f.write(f"  Std:        {variance_np.std():.6f}\n\n")

            # Variance distribution
            f.write(f"VARIANCE DISTRIBUTION\n")
            f.write(f"{'-'*80}\n")
            f.write(f"  Below tau ({self.tau}):    {(variance_np < self.tau).sum()} ({(variance_np < self.tau).sum()/variance_np.size*100:.1f}%)\n")
            f.write(f"  Above tau ({self.tau}):    {(variance_np >= self.tau).sum()} ({(variance_np >= self.tau).sum()/variance_np.size*100:.1f}%)\n\n")

            # Correlation analysis
            f.write(f"CORRELATION ANALYSIS\n")
            f.write(f"{'-'*80}\n")
            correlation = np.corrcoef(weights_np.flatten(), variance_np.flatten())[0, 1]
            f.write(f"  Correlation(weight, variance): {correlation:.4f}\n")
            f.write(f"  (Should be close to 1.0, showing weights follow variance)\n\n")

            # Slice-by-slice summary
            f.write(f"SLICE-BY-SLICE SUMMARY\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{'Slice':>6} {'Mean W':>10} {'Std W':>10} {'Mean V':>12} {'Std V':>12}\n")
            f.write(f"{'-'*80}\n")

            for z in range(weights_np.shape[0]):
                w_slice = weights_np[z]
                v_slice = variance_np[z]
                f.write(f"{z:6d} {w_slice.mean():10.4f} {w_slice.std():10.4f} "
                       f"{v_slice.mean():12.6f} {v_slice.std():12.6f}\n")

        logger.info(f"  ✓ Statistics saved to {stats_file}")

    def run(self, output_base_dir: str = './consistency_weights_vis'):
        """
        Visualize multiple samples

        Args:
            output_base_dir: Base directory for all visualizations
        """
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nVisualizing {self.num_samples} samples...")
        logger.info(f"Output directory: {output_base_dir}")
        logger.info(f"Weight parameters: w_min={self.w_min}, w_max={self.w_max}, tau={self.tau}, kappa={self.kappa}")

        # Limit to available samples
        num_samples_to_vis = min(self.num_samples, len(self.dataset))

        for idx in range(num_samples_to_vis):
            logger.info(f"[{idx+1}/{num_samples_to_vis}] Processing sample {idx}...")
            try:
                self.visualize_sample(idx, output_base_dir)
            except Exception as e:
                logger.error(f"Error processing sample {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Visualization complete! Results saved to {output_base_dir}")
        logger.info(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize consistency weights')
    parser.add_argument('--data_dir', type=str,
                       default='/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/Colon_data',
                       help='Data directory')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str,
                       default='/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/weight_checkpoints/consistency_weights_vis',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    # Consistency weight parameters
    parser.add_argument('--weight_w_min', type=float, default=0.5,
                       help='Minimum weight for consistent regions')
    parser.add_argument('--weight_w_max', type=float, default=2.0,
                       help='Maximum weight for inconsistent regions')
    parser.add_argument('--weight_tau', type=float, default=0.15,
                       help='Tolerance threshold for variance')
    parser.add_argument('--weight_kappa', type=float, default=0.4,
                       help='Smoothness coefficient')

    args = parser.parse_args()

    visualizer = ConsistencyWeightVisualizer(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        device=args.device,
        w_min=args.weight_w_min,
        w_max=args.weight_w_max,
        tau=args.weight_tau,
        kappa=args.weight_kappa
    )

    visualizer.run(output_base_dir=args.output_dir)


if __name__ == '__main__':
    main()
