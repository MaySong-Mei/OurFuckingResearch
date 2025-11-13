"""
MedSAM2 segmentation model wrapper for 3D medical images.
Provides a unified interface compatible with the pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging
import os
import sys

# Use MedSAM2's build_sam function
medsam2_path = "/gpfs/radev/project/zhuoran_yang/sl3348/Med_Segmentation/MedSAM2"
if medsam2_path not in sys.path:
    sys.path.insert(0, medsam2_path)

try:
    from sam2.build_sam import build_sam2_video_predictor_npz
except ImportError:
    # Fallback if MedSAM2 path not available
    from sam2 import build_sam
    build_sam2_video_predictor_npz = None

logger = logging.getLogger(__name__)


class MedSAM2Segmenter(nn.Module):
    """
    Wrapper for MedSAM2 video predictor to segment 3D medical volumes.

    Converts 3D volumes into 2D frames for MedSAM2, handles prompting,
    and outputs segmentation logits in MONAI UNet format [B, C, D, H, W].
    """

    def __init__(
        self,
        config_file: str,
        ckpt_path: str,
        num_classes: int = 2,
        device: str = "cuda",
        use_bfloat16: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.use_bfloat16 = use_bfloat16

        # Build MedSAM2 predictor using MedSAM2's build function
        if build_sam2_video_predictor_npz is None:
            raise RuntimeError("Could not import build_sam2_video_predictor_npz from MedSAM2")

        self.predictor = build_sam2_video_predictor_npz(config_file, ckpt_path)
        self.predictor = self.predictor.to(device)

        logger.info(f"Loaded MedSAM2 from {ckpt_path}")

    def _get_full_image_box(self, H: int, W: int) -> np.ndarray:
        """Get full image bounding box as automatic prompt.

        For initial mask prompt on middle slice, use the entire image
        as the bounding box since we don't have prior segmentation information.
        """
        # [x_min, y_min, x_max, y_max]
        return np.array([0, 0, W - 1, H - 1], dtype=np.float32)

    @torch.inference_mode()
    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Segment a 3D volume using MedSAM2.

        Args:
            volume: [B, 1, D, H, W] - 3D medical volume (grayscale)

        Returns:
            logits: [B, C, D, H, W] - segmentation logits for C classes
                   logits[b, 0] = background probability
                   logits[b, 1] = foreground probability
        """
        B, C, D, H, W = volume.shape
        assert C == 1, f"Expected single channel input, got {C}"

        # Convert to numpy and prepare as RGB frames for MedSAM2
        volume_np = volume[0, 0].cpu().numpy()  # [D, H, W]

        # Normalize to [0, 255] for visualization
        v_min, v_max = volume_np.min(), volume_np.max()
        if v_max > v_min:
            volume_norm = ((volume_np - v_min) / (v_max - v_min) * 255).astype(np.uint8)
        else:
            volume_norm = volume_np.astype(np.uint8)

        # Resize to 512x512 if needed (MedSAM2 expects 512x512)
        target_size = 512
        if H != target_size or W != target_size:
            from PIL import Image
            volume_resized = np.zeros((D, target_size, target_size), dtype=np.uint8)
            for i in range(D):
                img_pil = Image.fromarray(volume_norm[i])
                img_resized = img_pil.resize((target_size, target_size), Image.BILINEAR)
                volume_resized[i] = np.array(img_resized)
            volume_norm = volume_resized
            orig_H, orig_W = H, W
        else:
            orig_H, orig_W = H, W

        # Convert grayscale to RGB by repeating channels
        volume_rgb = np.stack([volume_norm, volume_norm, volume_norm], axis=-1)  # [D, 512, 512, 3]
        volume_rgb = volume_rgb.transpose(0, 3, 1, 2)  # [D, 3, 512, 512]

        # Convert to tensor and apply normalization
        img_tensor = torch.from_numpy(volume_rgb).float().to(self.device)  # [D, 3, 512, 512]
        img_tensor = img_tensor / 255.0

        # ImageNet normalization
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device)[:, None, None]
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device)[:, None, None]
        img_tensor = (img_tensor - img_mean) / img_std

        # Initialize MedSAM2 predictor state
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_bfloat16 else torch.no_grad():
            # Use resized dimensions for init_state and box
            inference_state = self.predictor.init_state(img_tensor, 512, 512)

            # Step 1: Initial segmentation with full image bbox to locate object
            logger.info("Step 1: Initial segmentation with full image bbox")
            full_box = self._get_full_image_box(512, 512)

            _, _, middle_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=D // 2,  # Start from middle frame
                obj_id=1,
                box=full_box,
            )

            # Step 2: Extract refined bbox from the middle frame's segmentation
            logger.info("Step 2: Extracting refined bbox from middle frame segmentation")
            middle_mask_binary = (middle_mask_logits[0, 0] > 0).cpu().numpy().astype(np.uint8)  # [512, 512]

            # Find foreground pixels
            foreground_indices = np.where(middle_mask_binary > 0)

            if len(foreground_indices[0]) > 0:
                # Get bounds of foreground region
                y_min, y_max = foreground_indices[0].min(), foreground_indices[0].max()
                x_min, x_max = foreground_indices[1].min(), foreground_indices[1].max()

                # Add some padding (e.g., 5% of size) to avoid cutting off edges
                h_pad = int(0.05 * (y_max - y_min))
                w_pad = int(0.05 * (x_max - x_min))
                y_min = max(0, y_min - h_pad)
                y_max = min(511, y_max + h_pad)
                x_min = max(0, x_min - w_pad)
                x_max = min(511, x_max + w_pad)

                box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
                logger.info(f"Refined bbox: {box}")
            else:
                logger.warning("No foreground found in initial segmentation, using full image bbox")
                box = full_box

            # Reuse the middle_mask_logits from step 1 (no need to re-segment)

            # Propagate segmentation through entire volume
            # Store at original size (resize back from 512x512)
            middle_mask_resized = F.interpolate(middle_mask_logits, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
            mask_logits_all = torch.zeros(D, 1, orig_H, orig_W, device=self.device)
            mask_logits_all[D // 2] = middle_mask_resized[0]  # Store initial mask

            # Step 2a: Propagate forward (D//2 → D-1) using refined box
            logger.info("Step 2a: Propagating forward using refined bbox")
            # Propagate forward
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                inference_state, start_frame_idx=D // 2, reverse=False
            ):
                out_mask_resized = F.interpolate(out_mask_logits, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
                mask_logits_all[out_frame_idx] = out_mask_resized[0]

            # Step 3: Reset and propagate backward using refined box
            logger.info("Step 3: Propagating backward using refined bbox")
            self.predictor.reset_state(inference_state)
            inference_state = self.predictor.init_state(img_tensor, 512, 512)

            # Re-add initial mask with refined box
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=D // 2,
                obj_id=1,
                box=box,
            )

            # Propagate backward
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                inference_state, start_frame_idx=D // 2, reverse=True
            ):
                # Resize to original size
                out_mask_resized = F.interpolate(out_mask_logits, size=(orig_H, orig_W), mode='bilinear', align_corners=False)
                # Average with forward pass if frame was already computed
                if mask_logits_all[out_frame_idx].abs().sum() > 0:
                    mask_logits_all[out_frame_idx] = (mask_logits_all[out_frame_idx] + out_mask_resized[0]) / 2
                else:
                    mask_logits_all[out_frame_idx] = out_mask_resized[0]

            self.predictor.reset_state(inference_state)

        # Convert mask logits to class logits: [D, 1, H, W] -> [D, 2, H, W]
        # MedSAM2 outputs logits where > 0 means foreground
        foreground_logits = mask_logits_all.squeeze(1)  # [D, H, W]
        background_logits = -foreground_logits  # Inverse for background

        # Stack into [D, 2, H, W]
        logits = torch.stack([background_logits, foreground_logits], dim=1)  # [D, 2, H, W]

        # Add batch dimension: [D, 2, H, W] -> [1, 2, D, H, W]
        # Permute to get [B, C, D, H, W] format
        logits = logits.unsqueeze(0)  # [1, D, 2, H, W]
        logits = logits.permute(0, 2, 1, 3, 4)  # [1, 2, D, H, W]
        logits = logits.to(volume.device)

        return logits
