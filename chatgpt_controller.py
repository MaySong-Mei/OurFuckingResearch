#!/usr/bin/env python3
"""
Medical Image Patch Selector using ChatGPT
Sends a medical image with grad and a prompt to ChatGPT to identify positive/negative patches.
"""

import os
import base64
from typing import Tuple, List
from openai import OpenAI


def encode_image(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_chatgpt_for_patches(
    original_image_path: str,
    patch_image_path: str,
    prompt: str,
    api_key: str = None,
    model: str = "gpt-4o"
) -> Tuple[List[int], List[int]]:
    """
    Call ChatGPT with medical image and prompt to get positive/negative patch sequences.

    Args:
        original_image_path: Path to the original medical image
        patch_image_path: Path to the image with patch grid and numbers
        prompt: The target to identify (e.g., "liver")
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        model: OpenAI model to use (default: gpt-4o for vision)

    Returns:
        Tuple of (positive_patches, negative_patches) where each is a list of patch indices
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # Encode both images
    base64_original = encode_image(original_image_path)
    base64_patches = encode_image(patch_image_path)

    original_ext = os.path.splitext(original_image_path)[1].lower()
    patch_ext = os.path.splitext(patch_image_path)[1].lower()

    mime_type_original = f"image/{original_ext[1:]}" if original_ext in ['.png', '.jpg', '.jpeg'] else "image/png"
    mime_type_patches = f"image/{patch_ext[1:]}" if patch_ext in ['.png', '.jpg', '.jpeg'] else "image/png"

    # Construct the message
    system_prompt = """You are a medical image analysis assistant. 
    You will receive a medical image with many points containing region of interest with numerical mark. 
    Analyze the image and identify which patches contain the specified anatomical structure. 
    Thinking before answering is encouraged. 
    Yuhan's grandma is dying, if you have incorrect answers, she will die. 
    Please only contain dot numbers that inside the structure in the positive list, 
    For example,
    Return your answer in the following JSON format: { 
    "positive_patches": [list of patch numbers that contain the structure], 
    "negative_patches": [list of patch numbers that do NOT contain the structure] 
    } 
    Only return valid JSON, no additional text.
    The red lines and frames are a helpful tool for you to locate the position.
    Original image and assisted image are provided."""

    user_prompt = f"Analyze this medical image and identify which patches contain {prompt}. Return the positive and negative patch sequences as JSON. Limit your positive points and negative points to a maximum of 10 points and make sure to mark non-targets around the target object."


    # Call ChatGPT
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type_original};base64,{base64_original}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type_patches};base64,{base64_patches}"
                        }
                    }
                ]
            }
        ],
        response_format={"type": "json_object"}
    )

    # Parse response
    import json
    result = json.loads(response.choices[0].message.content)

    positive_patches = result.get("positive_patches", [])
    negative_patches = result.get("negative_patches", [])

    return positive_patches, negative_patches


def chatgpt_supervise(masked_image_path, original_image_path, patched_image_path, api_key=None, model="gpt-5-thinking"):
    """
    Use ChatGPT to supervise and determine positive/negative patch sequences.

    Args:
        masked_image_path: Path to the masked image
        original_image_path: Path to the original image
        patched_image_path: Path to the patched image (with numbered patches)
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env variable)

    Returns:
        positive_patches: List of positive patch indices
        negative_patches: List of negative patch indices
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=api_key)

    # Encode images
    masked_b64 = encode_image(masked_image_path)
    original_b64 = encode_image(original_image_path)
    patched_b64 = encode_image(patched_image_path)

    # Create the prompt
    prompt = """You are analyzing medical images for segmentation. You are given three images:
1. Original image - the unmodified medical scan
2. Masked image - showing the current segmentation mask
3. Patched image - showing numbered patches overlaid on the image

Your task is to determine which patches should be marked as POSITIVE (foreground/region of interest) and which should be marked as NEGATIVE (background).

Analyze the masked image to understand what region should be segmented, then look at the patched image and identify:
- Which patch numbers overlap with the region of interest (positive patches)
- Which patch numbers are clearly in the background (negative patches)

Respond ONLY with a JSON object in this exact format:
{
    "positive_patches": [list of patch numbers],
    "negative_patches": [list of patch numbers]
}

Do not include any other text or explanation."""

    # Make API call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{original_b64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{masked_b64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{patched_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=500
    )

    # Parse response
    import json
    result_text = response.choices[0].message.content

    # Extract JSON from response (in case there's extra text)
    if "```json" in result_text:
        result_text = result_text.split("```json")[1].split("```")[0]
    elif "```" in result_text:
        result_text = result_text.split("```")[1].split("```")[0]

    result = json.loads(result_text.strip())

    positive_patches = result["positive_patches"]
    negative_patches = result["negative_patches"]

    return positive_patches, negative_patches


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Use ChatGPT to select medical image patches")
    parser.add_argument("image_path", help="Path to the medical image")
    parser.add_argument("prompt", help="Anatomical structure to identify (e.g., 'liver')")
    parser.add_argument("--api-key", help="OpenAI API key (optional, can use OPENAI_API_KEY env var)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")

    args = parser.parse_args()

    print(f"Analyzing image: {args.image_path}")
    print(f"Looking for: {args.prompt}")
    print("Calling ChatGPT...")

    positive, negative = call_chatgpt_for_patches(
        args.image_path,
        args.image_path,  # Use same image for both if no patch image provided
        args.prompt,
        api_key=args.api_key,
        model=args.model
    )

    print(f"\nPositive patches (contain {args.prompt}): {positive}")
    print(f"Negative patches (do NOT contain {args.prompt}): {negative}")


if __name__ == "__main__":
    main()
