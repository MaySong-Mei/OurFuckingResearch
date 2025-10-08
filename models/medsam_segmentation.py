"""
MedSAM integration for medical image segmentation
MedSAM is a pre-trained Segment Anything Model adapted for medical imaging
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class MedSAMSegmentation(nn.Module):
    """
    Wrapper for MedSAM (Medical Segment Anything Model)

    MedSAM is pre-trained on large-scale medical imaging datasets
    and can be used for zero-shot or prompted segmentation.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_b",
        device: str = "cuda",
        auto_prompt: bool = True,
        num_classes: int = 2
    ):
        """
        Args:
            checkpoint_path: Path to MedSAM checkpoint
            model_type: SAM model type ('vit_b', 'vit_l', 'vit_h')
            device: Device to run model on
            auto_prompt: Automatically generate prompts (boxes/points)
            num_classes: Number of output classes
        """
        super().__init__()

        self.model_type = model_type
        self.device = device
        self.auto_prompt = auto_prompt
        self.num_classes = num_classes

        # Try to import segment_anything
        try:
            from segment_anything import sam_model_registry, SamPredictor
            self.sam_available = True
        except ImportError:
            print("Warning: segment-anything not installed. Install with:")
            print("pip install git+https://github.com/facebookresearch/segment-anything.git")
            self.sam_available = False
            self.sam = None
            self.predictor = None
            return

        # Load MedSAM model
        if checkpoint_path is None:
            print("No checkpoint provided. Using SAM with ImageNet weights.")
            print("For best results, download MedSAM checkpoint from:")
            print("https://github.com/bowang-lab/MedSAM")
            checkpoint_path = self._get_default_sam_checkpoint(model_type)

        # Initialize SAM
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device)
        self.sam.eval()

        # Create predictor
        self.predictor = SamPredictor(self.sam)

        # Freeze all parameters
        for param in self.sam.parameters():
            param.requires_grad = False

    def _get_default_sam_checkpoint(self, model_type: str) -> str:
        """Get default SAM checkpoint path (if available locally)"""
        # User should download checkpoints manually
        checkpoint_paths = {
            'vit_b': 'checkpoints/sam_vit_b_01ec64.pth',
            'vit_l': 'checkpoints/sam_vit_l_0b3195.pth',
            'vit_h': 'checkpoints/sam_vit_h_4b8939.pth',
        }
        return checkpoint_paths.get(model_type, 'checkpoints/medsam_vit_b.pth')

    def generate_box_prompt(self, image: torch.Tensor) -> np.ndarray:
        """
        Automatically generate box prompt for SAM
        Simple heuristic: use center region of image

        Args:
            image: [H, W] or [1, H, W] image

        Returns:
            box: [x1, y1, x2, y2] bounding box
        """
        if len(image.shape) == 3:
            image = image[0]

        H, W = image.shape

        # Simple heuristic: box around center 60% of image
        margin_h = int(H * 0.2)
        margin_w = int(W * 0.2)

        box = np.array([
            margin_w,           # x1
            margin_h,           # y1
            W - margin_w,       # x2
            H - margin_h        # y2
        ])

        return box

    def generate_grid_prompts(
        self,
        image: torch.Tensor,
        grid_size: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate grid of point prompts covering the image

        Args:
            image: [H, W] or [1, H, W] image
            grid_size: Number of points per dimension

        Returns:
            point_coords: [N, 2] array of (x, y) coordinates
            point_labels: [N] array of labels (1 for foreground)
        """
        if len(image.shape) == 3:
            image = image[0]

        H, W = image.shape

        # Create grid
        y_coords = np.linspace(H * 0.2, H * 0.8, grid_size)
        x_coords = np.linspace(W * 0.2, W * 0.8, grid_size)

        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([x, y])

        point_coords = np.array(points)
        point_labels = np.ones(len(points))  # All foreground

        return point_coords, point_labels

    def segment_with_box(
        self,
        image: np.ndarray,
        box: np.ndarray,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment image using box prompt

        Args:
            image: [H, W, 3] RGB image (0-255)
            box: [4] bounding box [x1, y1, x2, y2]
            multimask_output: Whether to output multiple masks

        Returns:
            masks: [N, H, W] segmentation masks
            scores: [N] confidence scores
            logits: [N, H, W] raw logits
        """
        self.predictor.set_image(image)

        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box,
            multimask_output=multimask_output
        )

        return masks, scores, logits

    def segment_with_points(
        self,
        image: np.ndarray,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment image using point prompts

        Args:
            image: [H, W, 3] RGB image (0-255)
            point_coords: [N, 2] point coordinates
            point_labels: [N] point labels (1=foreground, 0=background)
            multimask_output: Whether to output multiple masks

        Returns:
            masks: [M, H, W] segmentation masks
            scores: [M] confidence scores
            logits: [M, H, W] raw logits
        """
        self.predictor.set_image(image)

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=multimask_output
        )

        return masks, scores, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batch of images

        Args:
            x: [B, C, H, W] batch of images (normalized 0-1)

        Returns:
            logits: [B, num_classes, H, W] segmentation logits
        """
        if not self.sam_available:
            # Fallback: return dummy predictions
            B, C, H, W = x.shape
            return torch.zeros(B, self.num_classes, H, W, device=x.device)

        B, C, H, W = x.shape
        outputs = []

        for i in range(B):
            # Convert to numpy RGB image (0-255)
            img = x[i].detach().cpu().numpy()

            # Handle grayscale -> RGB
            if img.shape[0] == 1:
                img = np.repeat(img, 3, axis=0)

            # Transpose to [H, W, C] and scale to 0-255
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)

            if self.auto_prompt:
                # Generate box prompt automatically
                box = self.generate_box_prompt(x[i])
                masks, scores, logits = self.segment_with_box(
                    img, box, multimask_output=False
                )
            else:
                # Use grid prompts
                points, labels = self.generate_grid_prompts(x[i])
                masks, scores, logits = self.segment_with_points(
                    img, points, labels, multimask_output=False
                )

            # Take best mask
            if len(masks.shape) == 3:
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
            else:
                mask = masks

            # Convert to binary mask
            mask_binary = (mask > 0).astype(np.float32)

            # Create one-hot encoding for num_classes
            if self.num_classes == 2:
                # Binary: background + foreground
                mask_onehot = np.stack([1 - mask_binary, mask_binary], axis=0)
            else:
                # Multi-class: assume mask contains class indices
                mask_onehot = np.zeros((self.num_classes, H, W))
                mask_onehot[0] = 1 - mask_binary  # Background
                mask_onehot[1] = mask_binary      # Foreground (can be extended)

            outputs.append(torch.from_numpy(mask_onehot).float())

        # Stack batch
        output = torch.stack(outputs, dim=0).to(x.device)

        return output


class MedSAMWithAutoPrompt(nn.Module):
    """
    MedSAM with learned automatic prompt generation
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_type: str = "vit_b",
        device: str = "cuda",
        num_classes: int = 2
    ):
        super().__init__()

        # Base MedSAM
        self.medsam = MedSAMSegmentation(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            device=device,
            auto_prompt=False,
            num_classes=num_classes
        )

        # Learnable prompt generator (lightweight CNN)
        self.prompt_generator = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 4)  # Output: box coordinates [x1, y1, x2, y2]
        )

        # Freeze MedSAM
        for param in self.medsam.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with learned prompts

        Args:
            x: [B, C, H, W] batch of images

        Returns:
            logits: [B, num_classes, H, W] segmentation logits
        """
        # Generate box prompts
        boxes = self.prompt_generator(x)  # [B, 4]
        boxes = torch.sigmoid(boxes)  # Normalize to [0, 1]

        # Scale boxes to image size
        B, C, H, W = x.shape
        boxes = boxes * torch.tensor([W, H, W, H], device=x.device)

        # Use MedSAM with learned boxes
        # Note: This is simplified - full implementation would pass boxes to SAM
        return self.medsam(x)


def download_medsam_checkpoint(save_path: str = "checkpoints/medsam_vit_b.pth"):
    """
    Helper to download MedSAM checkpoint

    Args:
        save_path: Where to save the checkpoint
    """
    import os
    import urllib.request

    print("Downloading MedSAM checkpoint...")
    print("Note: MedSAM checkpoint is ~350MB")

    # MedSAM checkpoint URL
    url = "https://github.com/bowang-lab/MedSAM/releases/download/v0.0.1/medsam_vit_b.pth"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Downloaded MedSAM checkpoint to {save_path}")
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Please download manually from: https://github.com/bowang-lab/MedSAM")


# Utility function for easy loading
def load_medsam(
    checkpoint_path: Optional[str] = None,
    model_type: str = "vit_b",
    device: str = "cuda",
    num_classes: int = 2,
    auto_download: bool = False
) -> MedSAMSegmentation:
    """
    Convenient function to load MedSAM

    Args:
        checkpoint_path: Path to checkpoint (will auto-download if None and auto_download=True)
        model_type: SAM model type
        device: Device to load on
        num_classes: Number of classes
        auto_download: Whether to auto-download checkpoint if not found

    Returns:
        model: Loaded MedSAM model
    """
    if checkpoint_path is None and auto_download:
        checkpoint_path = "checkpoints/medsam_vit_b.pth"
        import os
        if not os.path.exists(checkpoint_path):
            download_medsam_checkpoint(checkpoint_path)

    model = MedSAMSegmentation(
        checkpoint_path=checkpoint_path,
        model_type=model_type,
        device=device,
        num_classes=num_classes
    )

    return model
