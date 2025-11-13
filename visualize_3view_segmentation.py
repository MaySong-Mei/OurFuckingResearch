"""
Visualize 3-view segmentation results with masks overlaid on original slices
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThreeViewSegmentationVisualizer:
    """Visualize segmentation from three orthogonal views"""

    def __init__(self, data_dir: str, num_samples: int = 10, device: str = 'cuda'):
        """
        Args:
            data_dir: Path to data directory
            num_samples: Number of samples to visualize
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_samples = num_samples

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


    def compute_multi_view_segmentations(self, volume: torch.Tensor):
        """
        Compute segmentations for three orthogonal views

        Args:
            volume: [B, Z, H, W]

        Returns:
            seg_axial: [B, C, Z, H, W]
            seg_sagittal: [B, C, W, Z, H]
            seg_coronal: [B, C, H, W, Z]
        """
        B = volume.shape[0]
        volume_3d = volume.unsqueeze(1)  # [B, 1, Z, H, W]

        with torch.no_grad():
            # Axial view
            seg_axial = self.segmentation(volume_3d)  # [B, C, Z, H, W]

            # Sagittal view
            vol_sagittal = volume.permute(0, 3, 1, 2).unsqueeze(1)  # [B, 1, W, Z, H]
            seg_sagittal = self.segmentation(vol_sagittal)  # [B, C, W, Z, H]

            # Coronal view
            vol_coronal = volume.permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, H, W, Z]
            seg_coronal = self.segmentation(vol_coronal)  # [B, C, H, W, Z]

        return seg_axial, seg_sagittal, seg_coronal

    def visualize_sample(self, sample_idx: int, output_dir: Path):
        """
        Visualize a single sample with 3 views using ground truth slices

        Args:
            sample_idx: Index of sample to visualize
            output_dir: Directory to save visualizations
        """
        # Load data
        batch = self.dataset[sample_idx]
        # Use ground truth slices directly (all slices from the volume)
        volume = batch['ground_truth_slices'].unsqueeze(0).to(self.device)  # [1, Z, H, W]
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

        # Convert to numpy
        volume_np = volume[0].cpu().numpy()  # [7, H, W] = [Z, H, W]
        seg_axial_np = seg_axial[0].cpu().numpy()  # [2, 7, H, W] = [C, Z, H, W]
        seg_sag_np = seg_sagittal[0].cpu().numpy()  # [2, W, Z, H] = [C, W, Z, H]
        seg_cor_np = seg_coronal[0].cpu().numpy()  # [2, H, W, Z] = [C, H, W, Z]

        # Create permuted versions for sagittal and coronal volumes
        # Original volume: [Z, H, W]
        # Sagittal view needs: [W, Z, H]
        # Coronal view needs: [H, Z, W]
        volume_sag = volume_np.transpose(2, 0, 1)  # [W, Z, H]
        volume_cor = volume_np.transpose(1, 0, 2)  # [H, Z, W]

        # Remap segmentations to match the permuted volumes
        # Sagittal: [C, W, Z, H] -> [C, W, Z, H] (already correct)
        # Coronal: [C, H, W, Z] -> [C, H, Z, W]
        seg_cor_np = seg_cor_np.transpose(0, 1, 3, 2)  # [C, H, Z, W]

        # Create output directory for this sample
        sample_dir = output_dir / f'sample_{sample_idx:03d}'
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Visualize each slice with masks
        self._visualize_view_slices(volume_np, seg_axial_np, 'axial', sample_dir)
        self._visualize_view_slices(volume_sag, seg_sag_np, 'sagittal', sample_dir)
        self._visualize_view_slices(volume_cor, seg_cor_np, 'coronal', sample_dir)

        logger.info(f"✓ Saved visualizations to {sample_dir}")

    def create_heatmap_overlay(self, slice_np: np.ndarray, prob_values: np.ndarray,
                               view_name: str, z_idx: int, output_dir: Path):
        """
        Create and save heatmap overlay on slice, reusing same heatmap style as standalone heatmap

        Args:
            slice_np: [H, W] original medical image slice
            prob_values: [H, W] probability values to visualize
            view_name: 'axial', 'sagittal', or 'coronal'
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
        prob_uint8 = (prob_values * 255).astype(np.uint8)
        prob_sqrt = (prob_values ** 0.5 * 255).astype(np.uint8)

        heatmap_rgb = np.zeros((*prob_values.shape, 3), dtype=np.uint8)
        heatmap_rgb[:, :, 0] = prob_uint8  # Red channel
        heatmap_rgb[:, :, 1] = prob_sqrt   # Green channel (sqrt for hotter look)
        heatmap_rgb[:, :, 2] = 0           # No blue

        # Blend: 70% image + 30% heatmap (matching mask overlay style)
        overlay = (0.7 * img_rgb + 0.3 * heatmap_rgb).astype(np.uint8)

        # Save overlay image
        overlay_img = Image.fromarray(overlay)
        overlay_path = output_dir / f'{view_name}_slice_{z_idx:03d}_prob_overlay.png'
        overlay_img.save(overlay_path)

    def _visualize_view_slices(self, volume_np: np.ndarray, seg_logits: np.ndarray,
                               view_name: str, output_dir: Path):
        """
        Visualize slices with segmentation masks overlaid

        Args:
            volume_np: [Z, H, W]
            seg_logits: [C, Z, H, W]
            view_name: 'axial', 'sagittal', or 'coronal'
            output_dir: Output directory
        """
        view_dir = output_dir / view_name
        view_dir.mkdir(exist_ok=True)

        Z, H, W = volume_np.shape

        # Get class probabilities and predictions
        seg_probs = torch.softmax(torch.from_numpy(seg_logits).float(), dim=0)
        seg_preds = torch.argmax(seg_probs, dim=0).numpy()  # [Z, H, W]
        seg_probs_np = seg_probs.numpy()  # [2, Z, H, W]

        # Foreground probability
        fg_prob = seg_probs_np[1, :, :, :]  # [Z, H, W]

        # Visualize each slice
        for z_idx in range(Z):
            img_slice = volume_np[z_idx]  # [H, W]
            mask_slice = seg_preds[z_idx]  # [H, W]
            fg_prob_slice = fg_prob[z_idx]  # [H, W]

            # Normalize image to 0-255
            if img_slice.max() > img_slice.min():
                img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            else:
                img_normalized = np.zeros_like(img_slice)
            img_uint8 = (img_normalized * 255).astype(np.uint8)

            # Create RGB image
            img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

            # Create mask overlay (green for foreground)
            mask_overlay = np.zeros_like(img_rgb)
            mask_overlay[mask_slice == 1, 1] = 200  # Green channel for foreground

            # Blend images
            overlay = (0.7 * img_rgb + 0.3 * mask_overlay).astype(np.uint8)

            # Save overlay image
            overlay_img = Image.fromarray(overlay)
            save_path = view_dir / f'{view_name}_slice_{z_idx:03d}_overlay.png'
            overlay_img.save(save_path)

            # Also save probability heatmap (standalone)
            fg_prob_uint8 = (fg_prob_slice * 255).astype(np.uint8)
            # Create hot colormap manually (red-yellow-white for high prob)
            heatmap = np.zeros((fg_prob_slice.shape[0], fg_prob_slice.shape[1], 3), dtype=np.uint8)
            heatmap[:, :, 0] = fg_prob_uint8  # Red channel
            heatmap[:, :, 1] = (fg_prob_slice ** 0.5 * 255).astype(np.uint8)  # Green (sqrt for hotter look)
            heatmap_img = Image.fromarray(heatmap)
            heatmap_path = view_dir / f'{view_name}_slice_{z_idx:03d}_heatmap.png'
            heatmap_img.save(heatmap_path)

            # Save probability heatmap overlaid on original slice (reuse same heatmap style)
            self.create_heatmap_overlay(img_slice, fg_prob_slice, view_name, z_idx, view_dir)

        logger.info(f"  ✓ {view_name}: {Z} slices saved to {view_dir}")

    def _create_summary_figure(self, volume_np: np.ndarray, seg_axial_np: np.ndarray,
                               seg_sag_np: np.ndarray, seg_cor_np: np.ndarray,
                               output_dir: Path):
        """
        Create a summary figure showing middle slice from all three views

        Args:
            volume_np: [Z, H, W]
            seg_axial_np: [2, Z, H, W]
            seg_sag_np: [2, Z, H, W]
            seg_cor_np: [2, Z, H, W]
            output_dir: Output directory
        """
        Z, H, W = volume_np.shape
        mid_z = Z // 2

        # Get probabilities
        seg_axial_probs = torch.softmax(torch.from_numpy(seg_axial_np).float(), dim=0).numpy()
        seg_sag_probs = torch.softmax(torch.from_numpy(seg_sag_np).float(), dim=0).numpy()
        seg_cor_probs = torch.softmax(torch.from_numpy(seg_cor_np).float(), dim=0).numpy()

        # Get predictions
        seg_axial_preds = np.argmax(seg_axial_np, axis=0)[mid_z]
        seg_sag_preds = np.argmax(seg_sag_np, axis=0)[mid_z]
        seg_cor_preds = np.argmax(seg_cor_np, axis=0)[mid_z]

        # Foreground probabilities
        fg_axial = seg_axial_probs[1, mid_z]
        fg_sag = seg_sag_probs[1, mid_z]
        fg_cor = seg_cor_probs[1, mid_z]

        # Create summary images for three views
        views = [
            ('axial', volume_np[mid_z], seg_axial_preds, fg_axial),
            ('sagittal', volume_np[mid_z], seg_sag_preds, fg_sag),
            ('coronal', volume_np[mid_z], seg_cor_preds, fg_cor),
        ]

        for view_name, img_slice, mask, fg_prob in views:
            # Normalize image to 0-255
            if img_slice.max() > img_slice.min():
                img_normalized = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            else:
                img_normalized = np.zeros_like(img_slice)
            img_uint8 = (img_normalized * 255).astype(np.uint8)

            # Create RGB image
            img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)

            # Create mask overlay (green for foreground)
            mask_overlay = np.zeros_like(img_rgb)
            mask_overlay[mask == 1, 1] = 200

            # Blend images
            overlay = (0.7 * img_rgb + 0.3 * mask_overlay).astype(np.uint8)

            # Save overlay
            overlay_img = Image.fromarray(overlay)
            save_path = output_dir / f'summary_{view_name}_overlay.png'
            overlay_img.save(save_path)

            # Save heatmap
            fg_prob_uint8 = (fg_prob * 255).astype(np.uint8)
            heatmap = np.zeros((fg_prob.shape[0], fg_prob.shape[1], 3), dtype=np.uint8)
            heatmap[:, :, 0] = fg_prob_uint8
            heatmap[:, :, 1] = (fg_prob ** 0.5 * 255).astype(np.uint8)
            heatmap_img = Image.fromarray(heatmap)
            heatmap_path = output_dir / f'summary_{view_name}_heatmap.png'
            heatmap_img.save(heatmap_path)

        logger.info(f"  ✓ Summary figures saved to {output_dir}")

    def run(self, output_base_dir: str = './3view_segmentation_vis'):
        """
        Visualize multiple samples

        Args:
            output_base_dir: Base directory for all visualizations
        """
        output_base_dir = Path(output_base_dir)
        output_base_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"\nVisualizing {self.num_samples} samples...")
        logger.info(f"Output directory: {output_base_dir}")

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

    parser = argparse.ArgumentParser(description='Visualize 3-view segmentation results')
    parser.add_argument('--data_dir', type=str,
                       default='/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/Colon_data',
                       help='Data directory')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str,
                       default='/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/weight_checkpoints/3view_segmentation_vis',
                       help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')

    args = parser.parse_args()

    visualizer = ThreeViewSegmentationVisualizer(
        data_dir=args.data_dir,
        num_samples=args.num_samples,
        device=args.device
    )

    visualizer.run(output_base_dir=args.output_dir)


if __name__ == '__main__':
    main()
