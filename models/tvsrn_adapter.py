"""
TVSRN Interpolation Adapter
Wraps TVSRN model to provide 4-input -> 7-output medical image interpolation
"""

import torch
import torch.nn as nn
from pathlib import Path
from .model_TransSR import TVSRN
from .config import opt


class TVSRNInterpolator(nn.Module):
    """
    Adapter for TVSRN
    Takes 4 grayscale medical images and produces 7 interpolated slices

    Args:
        device: device to use (cuda/cpu)
    """

    _opt_loaded = False  # Class variable to load config only once

    def __init__(self, device='cuda'):
        super(TVSRNInterpolator, self).__init__()
        self.device = device

        # Load config only once
        if not TVSRNInterpolator._opt_loaded:
            config_path = Path(__file__).parent / 'default.txt'
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            opt.load_config(str(config_path))
            TVSRNInterpolator._opt_loaded = True

        # Initialize TVSRN model
        self.model = TVSRN().to(device)

    def forward(self, x):
        """
        Interpolate 4 sampled slices to output slices

        Args:
            x: [B, 4, 256, 256] - four consecutive grayscale frames (256x256, pre-resized by DataLoader)

        Returns:
            output: [B, Z, 256, 256] - interpolated slices
        """
        B, C, H, W = x.shape

        # Process each sample in batch (TVSRN is optimized for single samples)
        outputs = []
        for i in range(B):
            sample = x[i]  # [4, 256, 256]
            output_sample = self.model(sample)  # TVSRN returns [1, C, Z, 256, 256]

            # Remove batch dimension and keep only first channel: [1, C, Z, 256, 256] -> [Z, 256, 256]
            output_sample = output_sample[0, 0]  # [Z, 256, 256]
            outputs.append(output_sample)

        # Stack outputs
        output = torch.stack(outputs, dim=0)  # [B, Z, 256, 256]

        # Ensure output is in valid range [0, 1]
        output = torch.clamp(output, 0, 1)

        return output

    def parameters(self):
        """Return model parameters for optimizer"""
        return self.model.parameters()

    def train(self, mode=True):
        """Set training mode"""
        self.model.train(mode)
        return self

    def eval(self):
        """Set evaluation mode"""
        self.model.eval()
        return self

    def state_dict(self):
        """Get state dict"""
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict"""
        return self.model.load_state_dict(state_dict, strict=strict)
