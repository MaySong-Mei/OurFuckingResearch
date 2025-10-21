"""
Multi-view extraction utilities for 3D volumes
"""

import torch
from typing import Tuple


class MultiViewExtractor:
    """Extract orthogonal views from 3D volume"""

    def __init__(self):
        pass

    def extract_views(self, volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract three orthogonal views from a 3D volume

        Args:
            volume: [B, N, H, W] - batch of volumes

        Returns:
            axial: [B, N, H, W] - axial slices (original)
            sagittal: [B, H, N, W] - sagittal slices
            coronal: [B, W, N, H] - coronal slices
        """
        axial = volume  # [B, N, H, W]
        sagittal = volume.permute(0, 2, 1, 3)  # [B, H, N, W]
        coronal = volume.permute(0, 3, 1, 2)   # [B, W, N, H]

        return axial, sagittal, coronal

    def remap_to_axial(
        self,
        sagittal_pred: torch.Tensor,
        coronal_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Remap sagittal and coronal predictions back to axial coordinate system

        Args:
            sagittal_pred: [B, H, N, W, C] - predictions in sagittal view
            coronal_pred: [B, W, N, H, C] - predictions in coronal view

        Returns:
            sagittal_axial: [B, N, H, W, C] - sagittal remapped to axial
            coronal_axial: [B, N, H, W, C] - coronal remapped to axial
        """
        # Remap sagittal: [B, H, N, W, C] -> [B, N, H, W, C]
        sagittal_axial = sagittal_pred.permute(0, 2, 1, 3, 4)

        # Remap coronal: [B, W, N, H, C] -> [B, N, H, W, C]
        coronal_axial = coronal_pred.permute(0, 2, 3, 1, 4)

        return sagittal_axial, coronal_axial
