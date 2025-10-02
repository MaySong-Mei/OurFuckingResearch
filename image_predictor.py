import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.savefig(f"mask_{i+1}.png", bbox_inches='tight', pad_inches=0)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

def patches_to_points(image_size, positive_patches, negative_patches, patch_size=64):
    """
    Convert patch indices to point coordinates and labels for SAM2.

    Args:
        image_size: Tuple of (width, height) of the image
        positive_patches: List of positive patch indices (1-based)
        negative_patches: List of negative patch indices (1-based)
        patch_size: Size of each patch (default: 64)

    Returns:
        input_points: Array of shape (N, 2) with point coordinates (center of each patch)
        input_labels: Array of shape (N,) with labels (1 for positive, 0 for negative)
    """
    width, height = image_size
    patches_per_row = width // patch_size
    patches_per_col = height // patch_size

    input_points = []
    input_labels = []

    # Process positive patches
    for patch_idx in positive_patches:
        # Convert 1-based index to 0-based
        patch_idx_0 = patch_idx - 1

        # Calculate row and column
        row = patch_idx_0 // patches_per_row
        col = patch_idx_0 % patches_per_row

        # Calculate center point
        center_x = col * patch_size + patch_size // 2
        center_y = row * patch_size + patch_size // 2

        input_points.append([center_x, center_y])
        input_labels.append(1)

    # Process negative patches
    for patch_idx in negative_patches:
        # Convert 1-based index to 0-based
        patch_idx_0 = patch_idx - 1

        # Calculate row and column
        row = patch_idx_0 // patches_per_row
        col = patch_idx_0 % patches_per_row

        # Calculate center point
        center_x = col * patch_size + patch_size // 2
        center_y = row * patch_size + patch_size // 2

        input_points.append([center_x, center_y])
        input_labels.append(0)

    return np.array(input_points), np.array(input_labels)

def regions_to_points(regions_dict, positive_indices, negative_indices):
    """
    Convert region indices to point coordinates and labels for SAM2 using region centroids.

    Args:
        regions_dict: Dictionary from create_patch_image_v3 mapping region index to properties
                     (must include 'centroid' key with (x, y) coordinates)
        positive_indices: List of positive region indices
        negative_indices: List of negative region indices

    Returns:
        input_points: Array of shape (N, 2) with point coordinates (centroid of each region)
        input_labels: Array of shape (N,) with labels (1 for positive, 0 for negative)
    """
    input_points = []
    input_labels = []

    # Process positive regions
    for region_idx in positive_indices:
        if region_idx in regions_dict:
            centroid = regions_dict[region_idx]['centroid']
            input_points.append([centroid[0], centroid[1]])
            input_labels.append(1)
        else:
            print(f"Warning: Positive region index {region_idx} not found in regions_dict")

    # Process negative regions
    for region_idx in negative_indices:
        if region_idx in regions_dict:
            centroid = regions_dict[region_idx]['centroid']
            input_points.append([centroid[0], centroid[1]])
            input_labels.append(0)
        else:
            print(f"Warning: Negative region index {region_idx} not found in regions_dict")

    return np.array(input_points), np.array(input_labels)

def predict_mask(image_path, input_points, input_labels, sam2_checkpoint="./sam2/checkpoints/sam2.1_hiera_large.pt", model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"):
    """
    Predict segmentation masks for an image based on input points and labels.

    Args:
        image_path: Path to the input image
        input_points: Array of shape (N, 2) with point coordinates
        input_labels: Array of shape (N,) with point labels (1 for positive, 0 for negative)
        sam2_checkpoint: Path to SAM2 checkpoint file
        model_cfg: Path to SAM2 model config file

    Returns:
        masks: Predicted segmentation masks
        scores: Confidence scores for each mask
        logits: Raw logits from the model
    """

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    np.random.seed(3)

    # Load image
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    # Build model
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    # Set image and predict
    predictor.set_image(image)

    masks, scores, logits = predictor.predict(
        point_coords=np.array(input_points),
        point_labels=np.array(input_labels),
        multimask_output=False,
    )

    # Sort by scores (highest first)
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    print(f"Generated {len(masks)} masks with scores: {scores}")

    # Visualize and save - only show the best mask
    show_masks(image, masks[:1], scores[:1], point_coords=np.array(input_points), input_labels=np.array(input_labels), borders=True)

    return masks, scores, logits

# # Example usage
# if __name__ == "__main__":
#     image_path = 'image.jpg'
#     positive_patches = [2, 3, 10, 11, 18]  # Example positive patch indices
#     negative_patches = [1, 4, 5, 12, 19, 27]       # Example negative patch indices
#     image = Image.open(image_path)
#     image_size = image.size  # (width, height)
#     input_points, input_labels = patches_to_points(image_size, positive_patches, negative_patches)

#     masks, scores, logits = predict_mask(image_path, input_points, input_labels)