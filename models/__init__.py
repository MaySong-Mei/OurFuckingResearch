"""
Model architectures
"""

from .IFNet import IFNet
from .vit_seg_modeling import VisionTransformer, CONFIGS

__all__ = [
    'IFNet',
    'VisionTransformer',
    'CONFIGS',
]
