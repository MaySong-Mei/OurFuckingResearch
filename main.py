#!/usr/bin/env python3
"""
Main pipeline for medical image segmentation using ChatGPT patch selection and SAM2.
"""

import os
from PIL import Image
from patch_image import create_patch_image_v1, create_patch_image_v2
from chatgpt_controller import call_chatgpt_for_patches, chatgpt_supervise
from png_to_jpg import convert_png_to_jpg
from image_predictor import patches_to_points, predict_mask


def run_segmentation_pipeline(
    image_path,
    target_structure,
    api_key=None,
    patch_size=32,
    sam2_checkpoint="./sam2/checkpoints/sam2.1_hiera_large.pt",
    model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml"
):
    """
    Run the complete segmentation pipeline.

    Args:
        image_path: Path to the input image (PNG)
        target_structure: Anatomical structure to segment (e.g., "liver", "kidney")
        api_key: OpenAI API key (optional, uses OPENAI_API_KEY env var if not provided)
        patch_size: Size of each patch (default: 64)
        sam2_checkpoint: Path to SAM2 checkpoint
        model_cfg: Path to SAM2 model config

    Returns:
        masks: Segmentation masks
        scores: Confidence scores
        logits: Raw logits
    """
    print(f"Starting segmentation pipeline for: {image_path}")
    print(f"Target structure: {target_structure}\n")

    # Step 1: Create patch image with grid
    print("Step 1: Creating patch image with grid...")
    valid_points, patch_image_path = create_patch_image_v2(image_path, 'image_with_patches.png', patch_size)
    print()

    # Step 2: Call ChatGPT to select patches
    print("Step 2: Calling ChatGPT to select patches...")
    model = "gpt-5"
    positive_patches, negative_patches = call_chatgpt_for_patches(
        image_path,
        patch_image_path,
        target_structure,
        api_key=api_key,
        model=model
    )
    print(f"Positive patches: {positive_patches}")
    print(f"Negative patches: {negative_patches}\n")

    # Step 3: Convert original image to JPG
    print("Step 3: Converting image to JPG...")
    jpg_path = image_path.replace('.png', '.jpg')
    convert_png_to_jpg(image_path, jpg_path)
    print(f"Converted to: {jpg_path}\n")

    # Step 4: Convert patches to points
    print("Step 4: Converting patches to point coordinates...")
    img = Image.open(image_path)
    image_size = img.size  # (width, height)
    input_points, input_labels = patches_to_points(
        image_size,
        positive_patches,
        negative_patches,
        patch_size
    )
    print(f"Generated {len(input_points)} points\n")

    # Step 5: Run SAM2 prediction
    print("Step 5: Running SAM2 segmentation...")
    masks, scores, logits = predict_mask(
        jpg_path,
        input_points,
        input_labels,
        sam2_checkpoint,
        model_cfg
    )
    print(f"\nSegmentation complete! Generated {len(masks)} mask(s)")

    return masks, scores, logits

    # # Step 6: Save masked image for supervision
    # print("\nStep 6: Creating masked image for supervision...")
    # import numpy as np
    # masked_img = Image.open(jpg_path).convert('RGB')
    # masked_array = np.array(masked_img)

    # # Overlay the best mask (highest score)
    # best_mask = masks[0]  # SAM2 returns masks sorted by score
    # mask_overlay = np.zeros_like(masked_array)
    # mask_overlay[best_mask > 0] = [255, 0, 0]  # Red overlay

    # # Blend the mask with the original image
    # alpha = 0.5
    # masked_array = (masked_array * (1 - alpha) + mask_overlay * alpha).astype(np.uint8)
    # masked_img_result = Image.fromarray(masked_array)

    # masked_image_path = jpg_path.replace('.jpg', '_masked.jpg')
    # masked_img_result.save(masked_image_path)
    # print(f"Saved masked image to: {masked_image_path}\n")

    # # Step 7: Supervise results with ChatGPT
    # print("Step 7: Supervising segmentation results with ChatGPT...")
    # refined_positive_patches, refined_negative_patches = chatgpt_supervise(
    #     masked_image_path,
    #     jpg_path,
    #     patch_image_path,
    #     api_key=api_key,
    #     model=model
    # )
    # print(f"Refined positive patches: {refined_positive_patches}")
    # print(f"Refined negative patches: {refined_negative_patches}\n")

    # # Step 8: Convert refined patches to points
    # print("Step 8: Converting refined patches to point coordinates...")
    # refined_input_points, refined_input_labels = patches_to_points(
    #     image_size,
    #     refined_positive_patches,
    #     refined_negative_patches,
    #     patch_size
    # )
    # print(f"Generated {len(refined_input_points)} refined points\n")

    # # Step 9: Run SAM2 prediction again with refined points
    # print("Step 9: Running SAM2 segmentation with refined points...")
    # refined_masks, refined_scores, refined_logits = predict_mask(
    #     jpg_path,
    #     refined_input_points,
    #     refined_input_labels,
    #     sam2_checkpoint,
    #     model_cfg
    # )
    # print(f"\nRefined segmentation complete! Generated {len(refined_masks)} mask(s)")

    # return refined_masks, refined_scores, refined_logits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Medical image segmentation pipeline")
    parser.add_argument("image_path", help="Path to input image (PNG)")
    parser.add_argument("target", help="Target structure to segment (e.g., 'liver', 'kidney')")
    parser.add_argument("--api-key", help="OpenAI API key (optional)")
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size (default: 64)")
    parser.add_argument("--checkpoint", default="./sam2/checkpoints/sam2.1_hiera_large.pt", help="SAM2 checkpoint path")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_l.yaml", help="SAM2 config path")

    args = parser.parse_args()

    masks, scores, logits = run_segmentation_pipeline(
        args.image_path,
        args.target,
        api_key=args.api_key,
        patch_size=args.patch_size,
        sam2_checkpoint=args.checkpoint,
        model_cfg=args.config
    )
