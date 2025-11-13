"""Analyze weight distribution to understand spatial variation patterns"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys
import logging

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


class WeightDistributionAnalyzer:
    """Analyze weight distribution patterns"""

    def __init__(self, data_dir: str, device: str = 'cuda',
                 w_min: float = 0.5, w_max: float = 3.0, tau: float = 0.02, kappa: float = 0.1):
        """Initialize"""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
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
        medsam2_config = "///" + "/gpfs/radev/project/zhuoran_yang/sl3348/Med_Segmentation/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
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

    def analyze_sample(self, sample_idx: int):
        """Analyze weight distribution for one sample"""
        logger.info(f"\n{'='*80}")
        logger.info(f"ANALYZING SAMPLE {sample_idx}")
        logger.info(f"{'='*80}")

        # Load data
        sample = self.dataset[sample_idx]
        volume = sample['ground_truth_slices'].unsqueeze(0).to(self.device)  # [1, Z, H, W]
        volume_np = volume[0].cpu().numpy()  # [Z, H, W]

        Z, H, W = volume_np.shape
        logger.info(f"Volume shape: {volume_np.shape}")

        # Compute segmentations
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

        # Prepare inputs with proper dimension handling
        with torch.no_grad():
            # Axial: [Z, H, W, C]
            prob_axial_transposed = prob_axial.transpose(1, 2, 3, 0)
            w_axial = torch.from_numpy(prob_axial_transposed).unsqueeze(0).to(self.device)

            # Sagittal: [Z, H, W, C]
            prob_sagittal_transposed = prob_sagittal.transpose(2, 3, 1, 0)
            z_sag = prob_sagittal_transposed.shape[0]
            if z_sag < Z:
                prob_sagittal_transposed = np.pad(prob_sagittal_transposed, ((0, Z-z_sag), (0,0), (0,0), (0,0)), mode='edge')
            elif z_sag > Z:
                prob_sagittal_transposed = prob_sagittal_transposed[:Z]
            w_sag = torch.from_numpy(prob_sagittal_transposed).unsqueeze(0).to(self.device)

            # Coronal: [Z, H, W, C]
            prob_coronal_transposed = prob_coronal.transpose(3, 1, 2, 0)
            z_cor = prob_coronal_transposed.shape[0]
            if z_cor < Z:
                prob_coronal_transposed = np.pad(prob_coronal_transposed, ((0, Z-z_cor), (0,0), (0,0), (0,0)), mode='edge')
            elif z_cor > Z:
                prob_coronal_transposed = prob_coronal_transposed[:Z]
            w_cor = torch.from_numpy(prob_coronal_transposed).unsqueeze(0).to(self.device)

            # Compute weights
            weights = self.consistency_weighting(w_axial, w_sag, w_cor)  # [1, Z, H, W]

        weights_np = weights[0].cpu().numpy()  # [Z, H, W]

        logger.info(f"\nWeight statistics:")
        logger.info(f"  Global - min={weights_np.min():.4f}, max={weights_np.max():.4f}, mean={weights_np.mean():.4f}")
        logger.info(f"  Global - std={weights_np.std():.4f}, range={weights_np.max()-weights_np.min():.4f}")

        # Analyze Z-axis variation
        logger.info(f"\nZ-axis variation (mean weight per Z slice):")
        z_means = []
        for z in range(Z):
            z_mean = weights_np[z].mean()
            z_std = weights_np[z].std()
            z_min = weights_np[z].min()
            z_max = weights_np[z].max()
            z_means.append(z_mean)
            if z % 5 == 0 or z < 3 or z >= Z-3:
                logger.info(f"  Z={z:3d}: mean={z_mean:.4f}, std={z_std:.4f}, min={z_min:.4f}, max={z_max:.4f}")

        # Check if weights vary within Z-planes
        logger.info(f"\nWithin-plane spatial variation (H,W dimensions):")
        within_plane_stds = []
        for z in range(Z):
            plane_std = weights_np[z].std()
            within_plane_stds.append(plane_std)

        mean_within_std = np.mean(within_plane_stds)
        logger.info(f"  Average within-plane std={mean_within_std:.6f}")
        logger.info(f"  Max within-plane std={np.max(within_plane_stds):.6f}")
        logger.info(f"  Min within-plane std={np.min(within_plane_stds):.6f}")

        if mean_within_std < 0.0001:
            logger.warning(f"  ⚠ Within-plane variation is EXTREMELY SMALL (nearly uniform)")
        elif mean_within_std < 0.001:
            logger.warning(f"  ⚠ Within-plane variation is VERY SMALL (mostly uniform)")
        else:
            logger.info(f"  ✓ Within-plane variation exists")

        # Check if all variations come from Z-axis
        z_range = np.max(z_means) - np.min(z_means)
        logger.info(f"\nZ-axis variation:")
        logger.info(f"  Z-mean range: {z_range:.4f}")
        logger.info(f"  Total global range: {weights_np.max() - weights_np.min():.4f}")
        logger.info(f"  Percentage from Z-variation: {100*z_range/(weights_np.max()-weights_np.min()):.1f}%")

        # Sample specific H,W positions to see if they vary with Z
        logger.info(f"\nSampling specific (H,W) positions across Z:")
        test_positions = [(0, 0), (H//4, W//4), (H//2, W//2), (3*H//4, 3*W//4), (H-1, W-1)]

        for h, w in test_positions:
            z_profile = weights_np[:, h, w]
            logger.info(f"  (H={h:3d}, W={w:3d}): min={z_profile.min():.4f}, max={z_profile.max():.4f}, std={z_profile.std():.4f}")

        # Analyze if weights are truly constant within planes
        logger.info(f"\nChecking if weights are constant within Z-planes:")
        is_constant = True
        for z in range(Z):
            plane = weights_np[z]
            if plane.max() - plane.min() > 0.0001:
                is_constant = False
                logger.info(f"  Z={z}: NOT constant (range={plane.max()-plane.min():.6f})")
                break

        if is_constant:
            logger.warning(f"  ⚠ CRITICAL: Weights appear to be CONSTANT within each Z-plane!")
            logger.warning(f"  This means weights[z, h, w] ≈ weights[z, 0, 0] for all (h,w)")
        else:
            logger.info(f"  ✓ Weights vary within Z-planes (as expected)")

        # Detailed statistics per Z-plane
        logger.info(f"\nDetailed within-plane statistics:")
        logger.info(f"  Z | Plane Mean | Plane Std | Plane Min | Plane Max | Plane Range")
        logger.info(f"  {'-'*70}")
        for z in range(0, Z, max(1, Z//10)):
            plane = weights_np[z]
            logger.info(f"  {z:3d} | {plane.mean():.6f} | {plane.std():.6f} | {plane.min():.6f} | {plane.max():.6f} | {plane.max()-plane.min():.6f}")

        return weights_np

    def analyze(self):
        """Analyze first sample"""
        logger.info("WEIGHT DISTRIBUTION ANALYSIS")
        logger.info(f"Parameters: w_min={self.w_min}, w_max={self.w_max}, tau={self.tau}, kappa={self.kappa}\n")

        try:
            self.analyze_sample(0)
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/Colon_data')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    analyzer = WeightDistributionAnalyzer(
        data_dir=args.data_dir,
        device=args.device
    )

    analyzer.analyze()
