"""
3D Medical Image Interpolation Training Pipeline
Self-Supervised Multi-View Consistency Approach

Uses IFNet for interpolation and MedSam for segmentation.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
from datetime import datetime
import numpy as np
from PIL import Image
import pydicom
import random

from data_loader import MedicalVolumeDataset
from models.IFNet import IFNet
from models.medsam_infer import MedSAM2Segmenter
from losses import ConsistencyLoss, SmoothnessLoss, InterpolationGroundTruthLoss


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value (default: 42)
    """
    # Set seeds for different libraries
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Enable deterministic mode for CUDA operations
    # Note: This may impact performance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"✓ Random seed set to {seed} for reproducibility")
    logger.info(f"  - torch.backends.cudnn.deterministic = {torch.backends.cudnn.deterministic}")
    logger.info(f"  - torch.backends.cudnn.benchmark = {torch.backends.cudnn.benchmark}")


class TrainingPipeline:
    """Main training pipeline for self-supervised 3D interpolation"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize models
        self.interpolator = IFNet().to(self.device)

        # Initialize MedSAM2 for 3D segmentation
        logger.info("Using MedSAM2 for 3D segmentation")
        # Use absolute path with // prefix for Hydra
        medsam2_config = "//" + "/gpfs/radev/project/zhuoran_yang/sl3348/Med_Segmentation/MedSAM2/sam2/configs/sam2.1_hiera_t512.yaml"
        medsam2_ckpt = "/gpfs/radev/project/zhuoran_yang/sl3348/Med_Segmentation/MedSAM2/checkpoints/MedSAM2_latest.pt"

        try:
            self.segmentation = MedSAM2Segmenter(
                config_file=medsam2_config,
                ckpt_path=medsam2_ckpt,
                num_classes=self.args.num_classes,
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

        # Loss functions
        self.consistency_loss = ConsistencyLoss(loss_type='dice')
        self.smoothness_loss = SmoothnessLoss()
        self.interpolation_gt_loss = InterpolationGroundTruthLoss(loss_type='l1', use_ssim=True)

        # Optimizer
        self.optimizer = optim.Adam(
            self.interpolator.parameters(),
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2)
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.args.num_epochs,
            eta_min=self.args.min_lr
        )

        # Data loaders
        self.train_loader = self._create_data_loader('train')
        self.val_loader = self._create_data_loader('val')

        # Metrics tracking
        self.best_val_loss = float('inf')
        self.global_step = 0
        self.save_interval = 10
        self.log_interval = 10

    def _create_data_loader(self, split: str) -> DataLoader:
        """Create data loader for train/val/test split"""
        dataset = MedicalVolumeDataset(
            data_dir=self.args.data_dir,
            split=split,
            num_slices=self.args.num_slices,
            img_size=self.args.img_size
        )

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(split == 'train'),
            num_workers=self.args.num_workers,
            pin_memory=True
        )

    def interpolate_volume(self, slices: torch.Tensor) -> torch.Tensor:
        """
        Interpolate between slices to create denser volume using IFNet
        Takes 129 slices and interpolates to 256 slices

        Args:
            slices: [B, 129, H, W] - batch of 129 original slices

        Returns:
            interpolated: [B, 256, H, W] - batch of 256 interpolated slices
        """
        batch_size, num_slices, H, W = slices.shape

        # First use IFNet to do 2x interpolation: 129 -> 257 slices
        # For 2x interpolation: N' = 2*N - 1
        interpolated_slices = [slices[:, 0:1]]  # Keep dimension [B, 1, H, W]

        for i in range(num_slices - 1):
            # Get consecutive slice pairs
            frame0 = slices[:, i:i+1]  # [B, 1, H, W]
            frame1 = slices[:, i+1:i+2]  # [B, 1, H, W]

            # IFNet expects [B, 6, H, W] input (concatenated RGB images)
            # Convert grayscale to 3-channel by repeating
            frame0_rgb = frame0.repeat(1, 3, 1, 1)  # [B, 3, H, W]
            frame1_rgb = frame1.repeat(1, 3, 1, 1)  # [B, 3, H, W]

            # Concatenate along channel dimension
            ifnet_input = torch.cat([frame0_rgb, frame1_rgb], dim=1)  # [B, 6, H, W]

            # Interpolate middle frame(s)
            with torch.set_grad_enabled(self.interpolator.training):
                # IFNet returns: (flow_list, mask, merged, flow_teacher, merged_teacher, loss_distill)
                # merged is a list of 3 frames, merged[2] is the final refined output
                _, _, merged = self.interpolator(ifnet_input)
                middle_frame_rgb = merged[2]  # [B, 3, H, W]

                # Convert back to grayscale by averaging channels
                middle_frame = middle_frame_rgb.mean(dim=1, keepdim=True)  # [B, 1, H, W]

            interpolated_slices.append(middle_frame)
            interpolated_slices.append(frame1)

        # Stack all slices: 129 -> 257 slices
        interpolated = torch.cat(interpolated_slices, dim=1)  # [B, 257, H, W]

        # Drop the last frame to get exactly 256 slices
        interpolated = interpolated[:, :256, :, :]  # [B, 256, H, W]

        return interpolated

    def compute_multi_view_segmentations(self, volume: torch.Tensor):
        """
        Compute segmentations for three orthogonal views separately
        Segment axial view, sagittal view, and coronal view independently

        Args:
            volume: [B, 256, 256, 256] - 3D interpolated volume

        Returns:
            seg_axial: [B, C, 256, 256, 256] - axial view segmentation (3D)
            seg_sagittal: [B, C, 256, 256, 256] - sagittal view segmentation (3D)
            seg_coronal: [B, C, 256, 256, 256] - coronal view segmentation (3D)
        """
        B = volume.shape[0]

        # Add channel dimension: [B, 256, 256, 256] -> [B, 1, 256, 256, 256]
        volume_3d = volume.unsqueeze(1)

        with torch.no_grad():
            # Axial view: segment original volume (looking along Z-axis)
            seg_axial = self.segmentation(volume_3d)  # [B, C, 256, 256, 256]

            # Sagittal view: permute and segment (looking along X-axis)
            # [B, 1, 256, 256, 256] -> [B, 1, 256, 256, 256] (rearrange to x-axis view)
            vol_sagittal = volume.permute(0, 3, 1, 2).unsqueeze(1)  # [B, 1, 256, 256, 256]
            seg_sagittal = self.segmentation(vol_sagittal)  # [B, C, 256, 256, 256]

            # Coronal view: permute and segment (looking along Y-axis)
            # [B, 256, 256, 256] -> [B, 1, 256, 256, 256] (rearrange to y-axis view)
            vol_coronal = volume.permute(0, 2, 3, 1).unsqueeze(1)  # [B, 1, 256, 256, 256]
            seg_coronal = self.segmentation(vol_coronal)  # [B, C, 256, 256, 256]

        return seg_axial, seg_sagittal, seg_coronal

    def compute_loss(self, seg_axial, seg_sagittal, seg_coronal,
                     interpolated_volume, ground_truth_slices=None):
        """
        Compute total training loss from multi-view consistency and ground truth

        Args:
            seg_axial, seg_sagittal, seg_coronal: Segmentations from different views
            interpolated_volume: Interpolated volume [B, 256, H, W]
            ground_truth_slices: Ground truth slices [B, 257, H, W] (optional)

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual loss components
        """
        # Remap sagittal and coronal to axial space for comparison
        # seg_axial: [B, C, Z, Y, X]
        # seg_sagittal: [B, C, X, Z, Y] (from permuted volume) -> need [B, C, Z, Y, X]
        # seg_coronal: [B, C, Y, X, Z] (from permuted volume) -> need [B, C, Z, Y, X]
        seg_sagittal_remapped = seg_sagittal.permute(0, 1, 3, 4, 2)  # [B, C, Z, Y, X]
        seg_coronal_remapped = seg_coronal.permute(0, 1, 4, 2, 3)   # [B, C, Z, Y, X]

        # Convert to [B, Z, Y, X, C] format for consistency loss (expects last dim as classes)
        seg_axial_fmt = seg_axial.permute(0, 2, 3, 4, 1)  # [B, Z, Y, X, C]
        seg_sag_fmt = seg_sagittal_remapped.permute(0, 2, 3, 4, 1)  # [B, Z, Y, X, C]
        seg_cor_fmt = seg_coronal_remapped.permute(0, 2, 3, 4, 1)  # [B, Z, Y, X, C]

        is_first_batch = getattr(self, '_is_first_batch', False)

        prob_axial = F.softmax(seg_axial_fmt, dim=-1)
        class_pred_axial = torch.argmax(seg_axial_fmt, dim=-1)
        logit_diff = seg_axial_fmt[..., 1] - seg_axial_fmt[..., 0]  # class1_logit - class0_logit

        # Debug: Check segmentation output shape and values
        logger.info(f"DEBUG Axial - seg_axial_fmt shape: {seg_axial_fmt.shape}, dtype: {seg_axial_fmt.dtype}")
        logger.info(f"DEBUG Axial - class_pred_axial shape: {class_pred_axial.shape}, min: {class_pred_axial.min()}, max: {class_pred_axial.max()}")
        logger.info(f"Axial - Logits: max={seg_axial_fmt.max():.4f}, min={seg_axial_fmt.min():.4f}, mean={seg_axial_fmt.mean():.4f}")
        logger.info(f"Axial - Logit diff (class1-class0): mean={logit_diff.mean():.4f}, std={logit_diff.std():.4f}")
        logger.info(f"Axial - Prob class1: mean={prob_axial[..., 1].mean():.6f}, max={prob_axial[..., 1].max():.6f}")

        # Fixed bincount with proper handling
        class_pred_flat = class_pred_axial.flatten().int()
        class_dist = torch.bincount(class_pred_flat)
        logger.info(f"Axial - Class distribution: {class_dist.tolist()} (ratio: {(class_dist.float() / class_pred_axial.numel()).tolist()})")
        prob_sagittal = F.softmax(seg_sag_fmt, dim=-1)
        class_pred_sagittal = torch.argmax(seg_sag_fmt, dim=-1)
        logit_diff_sag = seg_sag_fmt[..., 1] - seg_sag_fmt[..., 0]
        logger.info(f"DEBUG Sagittal - seg_sag_fmt shape: {seg_sag_fmt.shape}, dtype: {seg_sag_fmt.dtype}")
        logger.info(f"DEBUG Sagittal - class_pred_sagittal shape: {class_pred_sagittal.shape}, min: {class_pred_sagittal.min()}, max: {class_pred_sagittal.max()}")
        logger.info(f"Sagittal - Logit diff (class1-class0): mean={logit_diff_sag.mean():.4f}, std={logit_diff_sag.std():.4f}")
        logger.info(f"Sagittal - Prob class1: mean={prob_sagittal[..., 1].mean():.6f}, max={prob_sagittal[..., 1].max():.6f}")
        class_pred_sag_flat = class_pred_sagittal.flatten().int()
        class_dist_sag = torch.bincount(class_pred_sag_flat)
        logger.info(f"Sagittal - Class distribution: {class_dist_sag.tolist()} (ratio: {(class_dist_sag.float() / class_pred_sagittal.numel()).tolist()})")

        prob_coronal = F.softmax(seg_cor_fmt, dim=-1)
        class_pred_coronal = torch.argmax(seg_cor_fmt, dim=-1)
        logit_diff_cor = seg_cor_fmt[..., 1] - seg_cor_fmt[..., 0]
        logger.info(f"DEBUG Coronal - seg_cor_fmt shape: {seg_cor_fmt.shape}, dtype: {seg_cor_fmt.dtype}")
        logger.info(f"DEBUG Coronal - class_pred_coronal shape: {class_pred_coronal.shape}, min: {class_pred_coronal.min()}, max: {class_pred_coronal.max()}")
        logger.info(f"Coronal - Logit diff (class1-class0): mean={logit_diff_cor.mean():.4f}, std={logit_diff_cor.std():.4f}")
        logger.info(f"Coronal - Prob class1: mean={prob_coronal[..., 1].mean():.6f}, max={prob_coronal[..., 1].max():.6f}")
        class_pred_cor_flat = class_pred_coronal.flatten().int()
        class_dist_cor = torch.bincount(class_pred_cor_flat)
        logger.info(f"Coronal - Class distribution: {class_dist_cor.tolist()} (ratio: {(class_dist_cor.float() / class_pred_coronal.numel()).tolist()})")

        # Multi-view consistency losses (self-supervised signal)
        consistency_sag = self.consistency_loss(seg_axial_fmt, seg_sag_fmt, debug=is_first_batch)
        consistency_cor = self.consistency_loss(seg_axial_fmt, seg_cor_fmt, debug=is_first_batch)
        consistency_total = (consistency_sag + consistency_cor) / 2

        # Smoothness regularization
        smoothness = self.smoothness_loss(interpolated_volume)

        # Ground truth interpolation loss and metrics
        interpolation_gt = torch.tensor(0.0, device=interpolated_volume.device)
        metrics = {'psnr': 0.0, 'ssim': 0.0}
        if ground_truth_slices is not None:
            # Align ground truth to match interpolated volume size
            # ground_truth_slices: [B, 257, H, W] -> take first 256 slices
            ground_truth_aligned = ground_truth_slices[:, :256, :, :]

            # Set debug=True only for first batch of each epoch
            is_first_batch = getattr(self, '_is_first_batch', False)
            try:
                interpolation_gt = self.interpolation_gt_loss(
                    interpolated_volume, ground_truth_aligned, debug=is_first_batch
                )
                # Compute PSNR and SSIM metrics
                metrics = self.interpolation_gt_loss.compute_metrics(
                    interpolated_volume, ground_truth_aligned
                )
            except Exception as e:
                logger.warning(f"Failed to compute ground truth loss and metrics: {e}")
                interpolation_gt = torch.tensor(0.0, device=interpolated_volume.device)
                metrics = {'psnr': 0.0, 'ssim': 0.0}

        # Total weighted loss
        total_loss = (
            self.args.lambda_consistency * consistency_total +
            self.args.lambda_smoothness * smoothness +
            self.args.lambda_interpolation_gt * interpolation_gt
        )

        loss_dict = {
            'total': total_loss.item(),
            'consistency': consistency_total.item(),
            'consistency_sagittal': consistency_sag.item(),
            'consistency_coronal': consistency_cor.item(),
            'smoothness': smoothness.item(),
            'interpolation_gt': interpolation_gt.item() if isinstance(interpolation_gt, torch.Tensor) else interpolation_gt,
            'psnr': metrics['psnr'],
            'ssim': metrics['ssim']
        }

        return total_loss, loss_dict

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.interpolator.train()

        epoch_losses = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.args.num_epochs}')
        for batch_idx, batch in enumerate(pbar):
            slices = batch['slices'].to(self.device)  # [B, N, H, W]
            ground_truth_slices = batch.get('ground_truth_slices', None)
            if ground_truth_slices is not None:
                ground_truth_slices = ground_truth_slices.to(self.device)  # [B, 257, H, W]

            # Set debug flag for first batch
            self._is_first_batch = (batch_idx == 0)

            # Forward pass
            self.optimizer.zero_grad()

            # 1. Interpolate volume
            interpolated_volume = self.interpolate_volume(slices)

            # 2. Generate multi-view segmentations
            seg_axial, seg_sagittal, seg_coronal = \
                self.compute_multi_view_segmentations(interpolated_volume)

            # 3. Compute losses
            loss, loss_dict = self.compute_loss(
                seg_axial, seg_sagittal, seg_coronal,
                interpolated_volume, ground_truth_slices
            )

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.interpolator.parameters(),
                    self.args.grad_clip
                )

            self.optimizer.step()

            # Logging
            epoch_losses.append(loss_dict)

            # Print detailed loss breakdown with metrics
            loss_str = f"L:{loss_dict['total']:.4f} | C:{loss_dict['consistency']:.4f} | S:{loss_dict['smoothness']:.4f} | GT:{loss_dict['interpolation_gt']:.4f}"

            # Add PSNR/SSIM if available
            if loss_dict['psnr'] > 0:
                loss_str += f" | PSNR:{loss_dict['psnr']:.2f}dB | SSIM:{loss_dict['ssim']:.4f}"

            pbar.set_postfix({'status': loss_str}, refresh=True)

            # Visualize first batch of training
            if batch_idx == 0:
                output_dir = Path(self.args.checkpoint_dir) / f'epoch_{epoch:03d}_train'
                self.visualize_segmentation(
                    slices[0],  # 原始切片 [N, H, W]
                    interpolated_volume[0],  # 插值体积 [D, H, W]
                    seg_axial[0],  # 轴向分割 [C, D, H, W]
                    seg_sagittal[0],  # 矢状分割 [C, D, H, W]
                    seg_coronal[0],  # 冠状分割 [C, D, H, W]
                    output_dir=output_dir,
                    epoch=epoch
                )
                logger.info(f"Saved training visualization at epoch {epoch}")

            self.global_step += 1

        # Average epoch metrics
        avg_loss = {
            k: sum(d[k] for d in epoch_losses) / len(epoch_losses)
            for k in epoch_losses[0].keys()
        }

        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validation loop with segmentation visualization"""
        self.interpolator.eval()

        val_losses = []

        for batch_idx, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
            slices = batch['slices'].to(self.device)
            ground_truth_slices = batch.get('ground_truth_slices', None)
            if ground_truth_slices is not None:
                ground_truth_slices = ground_truth_slices.to(self.device)

            # Forward pass
            interpolated_volume = self.interpolate_volume(slices)
            seg_axial, seg_sagittal, seg_coronal = \
                self.compute_multi_view_segmentations(interpolated_volume)

            loss, loss_dict = self.compute_loss(
                seg_axial, seg_sagittal, seg_coronal,
                interpolated_volume, ground_truth_slices
            )

            val_losses.append(loss_dict)

        # Average validation metrics
        avg_val_loss = {
            k: sum(d[k] for d in val_losses) / len(val_losses)
            for k in val_losses[0].keys()
        }

        return avg_val_loss

    def visualize_segmentation(self, original_slices, interpolated_volume, seg_axial,
                              seg_sagittal, seg_coronal, output_dir: Path, epoch: int):
        """
        Visualize segmentation results from multiple views

        Args:
            original_slices: [N, H, W] - 原始输入切片
            interpolated_volume: [D, H, W] - 插值后的体积
            seg_axial: [C, D, H, W] - 轴向分割
            seg_sagittal: [C, D, H, W] - 矢状分割
            seg_coronal: [C, D, H, W] - 冠状分割
            output_dir: 保存目录
            epoch: 训练轮次
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 转换为 numpy
        interp_np = interpolated_volume.cpu().detach().numpy()  # [D, H, W]
        seg_axial_np = seg_axial.cpu().detach().numpy()  # [C, D, H, W]
        seg_sag_np = seg_sagittal.cpu().detach().numpy()  # [C, D, H, W]
        seg_cor_np = seg_coronal.cpu().detach().numpy()  # [C, D, H, W]
        orig_slices_np = original_slices.cpu().detach().numpy()  # [N, H, W]

        D, H, W = interp_np.shape

        # ===== 1. 保存原始切片 =====
        orig_dir = output_dir / '1_original_slices'
        orig_dir.mkdir(exist_ok=True)
        for i, slice_data in enumerate(orig_slices_np):
            slice_norm = np.clip(slice_data, 0, 1)
            img = Image.fromarray((slice_norm * 255).astype(np.uint8))
            img.save(orig_dir / f'slice_{i:03d}.png')

        # ===== 2. 保存插值体积 =====
        interp_dir = output_dir / '2_interpolated_volume'
        interp_dir.mkdir(exist_ok=True)
        v_min, v_max = interp_np.min(), interp_np.max()
        if v_max > v_min:
            interp_norm = (interp_np - v_min) / (v_max - v_min) * 255
        else:
            interp_norm = interp_np * 255
        for i in range(D):
            img = Image.fromarray(interp_norm[i].astype(np.uint8))
            img.save(interp_dir / f'frame_{i:03d}.png')

        # ===== 3. 保存分割掩码 =====
        def save_segmentation_view(seg_logits, view_name):
            """保存某个视图的分割结果"""
            view_dir = output_dir / f'3_masks_{view_name}'
            view_dir.mkdir(exist_ok=True)

            # argmax 得到类别预测
            mask = np.argmax(seg_logits, axis=0)  # [D, H, W]

            # 保存每一帧
            for frame_idx in range(mask.shape[0]):
                mask_frame = mask[frame_idx].astype(np.uint8) * 127
                img = Image.fromarray(mask_frame)
                img.save(view_dir / f'mask_{frame_idx:03d}.png')

            # 保存 3D numpy
            np.save(view_dir / f'mask_3d.npy', mask)
            logger.info(f"Saved {view_name} masks to {view_dir}")

        save_segmentation_view(seg_axial_np, 'axial')
        save_segmentation_view(seg_sag_np, 'sagittal')
        save_segmentation_view(seg_cor_np, 'coronal')

        # ===== 4. 保存彩色叠加图 =====
        def save_overlay(seg_logits, interp_vol, view_name):
            """保存分割掩码与原始图像的叠加"""
            overlay_dir = output_dir / f'4_overlay_{view_name}'
            overlay_dir.mkdir(exist_ok=True)

            # 获取前景概率
            foreground_prob = np.exp(seg_logits[1]) / (np.exp(seg_logits[0]) + np.exp(seg_logits[1]))  # [D, H, W]

            # 逐帧保存叠加
            for frame_idx in range(interp_vol.shape[0]):
                img_frame = (interp_vol[frame_idx] * 255).astype(np.uint8)
                prob_frame = (foreground_prob[frame_idx] * 255).astype(np.uint8)

                # 转换为 RGB
                img_rgb = np.stack([img_frame, img_frame, img_frame], axis=-1)

                # 创建热力图
                heatmap = np.stack([
                    np.zeros_like(prob_frame),  # R
                    (prob_frame * 0.7).astype(np.uint8),  # G
                    prob_frame  # B
                ], axis=-1)

                # 叠加
                overlay = (0.5 * img_rgb + 0.5 * heatmap).astype(np.uint8)
                img = Image.fromarray(overlay)
                img.save(overlay_dir / f'overlay_{frame_idx:03d}.png')

            logger.info(f"Saved {view_name} overlays to {overlay_dir}")

        save_overlay(seg_axial_np, interp_np, 'axial')
        save_overlay(seg_sag_np, interp_np, 'sagittal')
        save_overlay(seg_cor_np, interp_np, 'coronal')

        # ===== 5. 保存统计信息 =====
        stats_file = output_dir / 'visualization_stats.txt'
        with open(stats_file, 'w') as f:
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Interpolated Volume Shape: {interp_np.shape}\n")
            f.write(f"Interpolated Volume Range: [{v_min:.4f}, {v_max:.4f}]\n\n")

            for view_name, seg_logits in [('Axial', seg_axial_np),
                                         ('Sagittal', seg_sag_np),
                                         ('Coronal', seg_cor_np)]:
                mask = np.argmax(seg_logits, axis=0)
                bg_count = (mask == 0).sum()
                fg_count = (mask == 1).sum()
                total = bg_count + fg_count
                f.write(f"{view_name}:\n")
                f.write(f"  Background: {bg_count} ({bg_count/total*100:.1f}%)\n")
                f.write(f"  Foreground: {fg_count} ({fg_count/total*100:.1f}%)\n\n")

        logger.info(f"Saved visualization stats to {stats_file}")

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'interpolator_state_dict': self.interpolator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': vars(self.args)
        }

        # Save latest
        save_path = Path(self.args.checkpoint_dir) / 'latest.pth'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)

        # Save best
        if is_best:
            best_path = Path(self.args.checkpoint_dir) / 'best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {val_loss:.4f}")

        # Save periodic
        if epoch % self.save_interval == 0:
            epoch_path = Path(self.args.checkpoint_dir) / f'epoch_{epoch}.pth'
            torch.save(checkpoint, epoch_path)

    def _log_losses(self, losses: dict, prefix: str, epoch: int):
        """Log loss dictionary with appropriate precision for each metric"""
        # Separate metrics and performance metrics
        loss_metrics = {}
        perf_metrics = {}

        for k, v in losses.items():
            if k in ['psnr', 'ssim']:
                perf_metrics[k] = v
            else:
                loss_metrics[k] = v

        # Log loss metrics
        loss_strs = []
        for k in ['total', 'consistency', 'consistency_sagittal', 'consistency_coronal', 'smoothness', 'interpolation_gt']:
            if k in loss_metrics:
                loss_strs.append(f"{k}={loss_metrics[k]:.4f}")

        msg = f"\n{'='*80}\n"
        msg += f"{prefix} Epoch {epoch:03d}\n"
        msg += f"Losses: {' | '.join(loss_strs)}\n"

        # Log performance metrics
        if perf_metrics:
            perf_strs = []
            if 'psnr' in perf_metrics:
                perf_strs.append(f"PSNR={perf_metrics['psnr']:.2f} dB")
            if 'ssim' in perf_metrics:
                perf_strs.append(f"SSIM={perf_metrics['ssim']:.4f}")

            if perf_strs:
                msg += f"Metrics: {' | '.join(perf_strs)}\n"

        msg += f"{'='*80}"
        logger.info(msg)

    def train(self):
        """Main training loop"""
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Dataset: {len(self.train_loader.dataset)} train, {len(self.val_loader.dataset)} val")

        for epoch in range(1, self.args.num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            self._log_losses(train_loss, "TRAIN", epoch)

            val_loss = self.validate(epoch)
            self._log_losses(val_loss, "VAL", epoch)

            self.scheduler.step()

            # Save checkpoint
            is_best = val_loss['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss['total']
            self.save_checkpoint(epoch, val_loss['total'], is_best)

        logger.info("Training complete!")


    def _load_full_volume_slices(self, file_path: str, num_output_slices: int = 257) -> np.ndarray:
        """
        Load all slices from a DICOM/npy file and return exactly num_output_slices
        by linearly sampling across the full volume.

        Args:
            file_path: Path to DICOM or npy file
            num_output_slices: Number of slices to extract (default 257 for 0-256)

        Returns:
            slices: [num_output_slices, H, W] normalized slices
        """
        file_path = Path(file_path)

        # Load volume
        if file_path.suffix == '.dcm':
            dicom_data = pydicom.dcmread(str(file_path))
            volume = dicom_data.pixel_array.astype(np.float32)
        elif file_path.suffix == '.npy':
            volume = np.load(file_path).astype(np.float32)
        elif file_path.suffix == '.npz':
            data = np.load(file_path)
            volume = data['volume'].astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Handle different dimensions
        if len(volume.shape) == 2:
            # Single slice - replicate
            volume = np.stack([volume] * num_output_slices, axis=0)
        elif len(volume.shape) == 4:
            # Multi-frame - take first timepoint
            volume = volume[0]

        # Normalize
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        vol_min = volume.min()
        vol_max = volume.max()
        if vol_max > vol_min:
            volume = (volume - vol_min) / (vol_max - vol_min)
        else:
            volume = volume - vol_min

        # Sample num_output_slices uniformly across the volume
        num_available = volume.shape[0]
        indices = np.linspace(0, num_available - 1, num_output_slices, dtype=int)
        slices = volume[indices]

        return slices

    def test(self):
        """Testing: load checkpoint, interpolate volume, visualize multi-view segmentations"""
        logger.info("Starting testing...")

        test_loader = self._create_data_loader('test')
        if len(test_loader.dataset) == 0:
            logger.warning("No test data found, skipping test")
            return

        output_dir = Path(self.args.checkpoint_dir) / 'test_results'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load best checkpoint
        checkpoint_path = Path(self.args.checkpoint_dir) / 'best.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.interpolator.load_state_dict(checkpoint['interpolator_state_dict'])
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning("Checkpoint not found, using current model state")

        self.interpolator.eval()

        # Track metrics across all test samples
        test_metrics = {
            'sample_id': [],
            'file_path': [],
            'psnr': [],
            'ssim': []
        }

        # Process all test samples
        for batch_idx, batch in enumerate(test_loader):
            file_path = batch['file_path'][0]
            logger.info(f"Processing test sample {batch_idx + 1}: {file_path}")

            # Load full 257 slices
            full_slices = self._load_full_volume_slices(file_path, num_output_slices=257)

            # Save original slices
            original_dir = output_dir / f'sample_{batch_idx:03d}/original_257'
            original_dir.mkdir(parents=True, exist_ok=True)
            for i, slice_data in enumerate(full_slices):
                img = Image.fromarray((slice_data * 255).astype(np.uint8))
                img.save(original_dir / f'slice_{i:03d}.png')
            logger.info(f"  Saved {len(full_slices)} original slices")

            # Extract every other slice (129 slices)
            sampled_slices = full_slices[::2]
            sampled_tensor = torch.from_numpy(sampled_slices[np.newaxis, :, :, :]).float().to(self.device)

            # Use 256 slices from ground truth (matching interpolated output)
            ground_truth_slices_256 = full_slices[:256]
            ground_truth_tensor = torch.from_numpy(ground_truth_slices_256[np.newaxis, :, :, :]).float().to(self.device)

            # Interpolate
            with torch.no_grad():
                interpolated_volume = self.interpolate_volume(sampled_tensor)
            interpolated_np = interpolated_volume[0].cpu().numpy()

            # Save interpolated slices
            interpolated_dir = output_dir / f'sample_{batch_idx:03d}/interpolated_256'
            interpolated_dir.mkdir(parents=True, exist_ok=True)
            for i, slice_data in enumerate(interpolated_np):
                slice_norm = np.clip(slice_data, 0, 1)
                img = Image.fromarray((slice_norm * 255).astype(np.uint8))
                img.save(interpolated_dir / f'slice_{i:03d}.png')
            logger.info(f"  Saved {len(interpolated_np)} interpolated slices")

            # Compute metrics for this sample
            if ground_truth_tensor.shape[1] == interpolated_volume.shape[1]:
                try:
                    metrics = self.interpolation_gt_loss.compute_metrics(
                        interpolated_volume, ground_truth_tensor
                    )
                    psnr_val = metrics['psnr']
                    ssim_val = metrics['ssim']

                    # Record metrics
                    test_metrics['sample_id'].append(batch_idx)
                    test_metrics['file_path'].append(Path(file_path).name)
                    test_metrics['psnr'].append(psnr_val)
                    test_metrics['ssim'].append(ssim_val)

                    logger.info(f"  Sample {batch_idx + 1}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")
                except Exception as e:
                    logger.error(f"  Failed to compute metrics: {e}")
            else:
                logger.warning(f"  Shape mismatch: GT {ground_truth_tensor.shape[1]} vs Interp {interpolated_volume.shape[1]}")

        # Save metrics to CSV
        if test_metrics['psnr']:
            import csv

            metrics_file = output_dir / 'test_metrics.csv'
            with open(metrics_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Sample ID', 'File Path', 'PSNR (dB)', 'SSIM'])
                for i, sample_id in enumerate(test_metrics['sample_id']):
                    writer.writerow([
                        sample_id,
                        test_metrics['file_path'][i],
                        f"{test_metrics['psnr'][i]:.2f}",
                        f"{test_metrics['ssim'][i]:.4f}"
                    ])
            logger.info(f"Saved per-sample metrics to {metrics_file}")

            # Compute statistics
            psnr_values = np.array(test_metrics['psnr'])
            ssim_values = np.array(test_metrics['ssim'])

            avg_psnr = np.mean(psnr_values)
            std_psnr = np.std(psnr_values)
            min_psnr = np.min(psnr_values)
            max_psnr = np.max(psnr_values)

            avg_ssim = np.mean(ssim_values)
            std_ssim = np.std(ssim_values)
            min_ssim = np.min(ssim_values)
            max_ssim = np.max(ssim_values)

            # Log detailed test results
            logger.info("\n" + "="*80)
            logger.info("TEST RESULTS - PER-SAMPLE METRICS")
            logger.info("="*80)
            for i, sample_id in enumerate(test_metrics['sample_id']):
                logger.info(f"Sample {sample_id}: {test_metrics['file_path'][i]}")
                logger.info(f"  PSNR: {test_metrics['psnr'][i]:.2f} dB")
                logger.info(f"  SSIM: {test_metrics['ssim'][i]:.4f}")

            logger.info("\n" + "="*80)
            logger.info("TEST RESULTS - STATISTICS")
            logger.info("="*80)
            logger.info(f"PSNR Statistics (dB):")
            logger.info(f"  Mean:   {avg_psnr:.2f}")
            logger.info(f"  Std:    {std_psnr:.2f}")
            logger.info(f"  Min:    {min_psnr:.2f}")
            logger.info(f"  Max:    {max_psnr:.2f}")
            logger.info(f"\nSSIM Statistics:")
            logger.info(f"  Mean:   {avg_ssim:.4f}")
            logger.info(f"  Std:    {std_ssim:.4f}")
            logger.info(f"  Min:    {min_ssim:.4f}")
            logger.info(f"  Max:    {max_ssim:.4f}")
            logger.info(f"\nTotal samples: {len(test_metrics['psnr'])}")
            logger.info("="*80)

            # Save summary to text file
            summary_file = output_dir / 'test_summary.txt'
            with open(summary_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("TEST RESULTS - PER-SAMPLE METRICS\n")
                f.write("="*80 + "\n")
                for i, sample_id in enumerate(test_metrics['sample_id']):
                    f.write(f"Sample {sample_id}: {test_metrics['file_path'][i]}\n")
                    f.write(f"  PSNR: {test_metrics['psnr'][i]:.2f} dB\n")
                    f.write(f"  SSIM: {test_metrics['ssim'][i]:.4f}\n")

                f.write("\n" + "="*80 + "\n")
                f.write("TEST RESULTS - STATISTICS\n")
                f.write("="*80 + "\n")
                f.write(f"PSNR Statistics (dB):\n")
                f.write(f"  Mean:   {avg_psnr:.2f}\n")
                f.write(f"  Std:    {std_psnr:.2f}\n")
                f.write(f"  Min:    {min_psnr:.2f}\n")
                f.write(f"  Max:    {max_psnr:.2f}\n")
                f.write(f"\nSSIM Statistics:\n")
                f.write(f"  Mean:   {avg_ssim:.4f}\n")
                f.write(f"  Std:    {std_ssim:.4f}\n")
                f.write(f"  Min:    {min_ssim:.4f}\n")
                f.write(f"  Max:    {max_ssim:.4f}\n")
                f.write(f"\nTotal samples: {len(test_metrics['psnr'])}\n")
                f.write("="*80 + "\n")
            logger.info(f"Saved test summary to {summary_file}")

        logger.info(f"Test complete. Results saved to {output_dir}")
        logger.info("Testing finished!")


def main():
    """Entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Train 3D Medical Image Interpolation')

    # Data
    parser.add_argument('--data_dir', type=str, default='/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/data',
                       help='Data directory')
    parser.add_argument('--num_slices', type=int, default=129, help='Number of slices')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')

    # Training
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')

    # Optimizer & Loss
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--lambda_consistency', type=float, default=0, help='Consistency loss weight')
    parser.add_argument('--lambda_smoothness', type=float, default=0.1, help='Smoothness loss weight')
    parser.add_argument('--lambda_interpolation_gt', type=float, default=1.0, help='Interpolation ground truth loss weight')

    # Checkpoint & Device
    parser.add_argument('--checkpoint_dir', type=str, default='/gpfs/radev/scratch/zhuoran_yang/sl3348/med_data/original_checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=0, help='Data loading workers (0=main process)')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # ✓ Set random seed BEFORE any model/data initialization
    set_random_seed(args.seed)

    # Convert img_size from int to tuple
    args.img_size = (args.img_size, args.img_size)

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"

    # Print config summary
    logger.info("-" * 80)
    logger.info("TRAINING CONFIGURATION")
    logger.info(f"  Data: {args.data_dir} | {args.num_slices} slices | Size: {args.img_size}")
    logger.info(f"  Training: batch={args.batch_size} epochs={args.num_epochs} lr={args.learning_rate}")
    logger.info(f"  Models: Interpolator=IFNet | Segmentation=MedSam (pretrained)")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Reproducibility: seed={args.seed}")
    logger.info("-" * 80)

    # Create and run pipeline
    pipeline = TrainingPipeline(args)
    pipeline.train()
    pipeline.test()


if __name__ == '__main__':
    main()
