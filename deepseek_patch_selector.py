#!/usr/bin/env python3
"""
Medical Image Patch Selector using DeepSeek
Sends a medical image with grad and a prompt to DeepSeek to identify positive/negative patches.
"""

import os
import base64
from typing import Tuple, List
from openai import OpenAI


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_deepseek_for_patches(
    original_image_path: str,
    patch_image_path: str,
    prompt: str,
    api_key: str = None,
    model: str = "deepseek-chat"
) -> Tuple[List[int], List[int]]:
    """
    Call DeepSeek with medical image and prompt to get positive/negative patch sequences.

    Args:
        original_image_path: Path to the original medical image
        patch_image_path: Path to the image with patch grid and numbers
        prompt: The target to identify (e.g., "liver")
        api_key: DeepSeek API key (if None, reads from DEEPSEEK_API_KEY env var)
        model: DeepSeek model to use (default: deepseek-chat)

    Returns:
        Tuple of (positive_patches, negative_patches) where each is a list of patch indices
    """
    # Initialize DeepSeek client with custom base_url
    client = OpenAI(
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    # Encode both images
    base64_original = encode_image_to_base64(original_image_path)
    base64_patches = encode_image_to_base64(patch_image_path)

    original_ext = os.path.splitext(original_image_path)[1].lower()
    patch_ext = os.path.splitext(patch_image_path)[1].lower()

    mime_type_original = f"image/{original_ext[1:]}" if original_ext in ['.png', '.jpg', '.jpeg'] else "image/png"
    mime_type_patches = f"image/{patch_ext[1:]}" if patch_ext in ['.png', '.jpg', '.jpeg'] else "image/png"

    # Construct the message
    system_prompt = """You are a medical image analysis assistant.
    You will receive a medical image divided into patches with gradients.
    Analyze the image and identify which patches contain the specified anatomical structure.
    Thinking before answering is encouraged.
    Someone's grandma is dying, if you have incorrect answers, she will die.
    Please only contain patch numbers that contain the structure in the positive list only when the whole patch contains most of the structure,
    and make sure the center pixel of the patch is the target.
    For example, if a patch contains only a small part of the target organ, do not include it in the positive list.
    Return your answer in the following JSON format: {
    "positive_patches": [list of patch numbers that contain the structure],
    "negative_patches": [list of patch numbers that do NOT contain the structure]
    }
    Only return valid JSON, no additional text.
    The red lines and frames are a helpful tool for you to locate the position.
    Original image and assisted image are provided."""

    user_prompt = f"Analyze this medical image and identify which patches contain {prompt}. Return the positive and negative patch sequences as JSON. Limit your negative points to a maximum of 6 points and make sure to mark non-targets around the target object."

    # Call DeepSeek
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
        temperature=0.0
    )

    # Parse response
    import json

    # Extract JSON from response content
    response_content = response.choices[0].message.content

    # Try to find JSON in the response
    try:
        # First try direct parsing
        result = json.loads(response_content)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from markdown code blocks
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(1))
        else:
            # Last attempt: find any JSON object in the text
            json_match = re.search(r'\{.*?\}', response_content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
            else:
                raise ValueError(f"Could not parse JSON from response: {response_content}")

    positive_patches = result.get("positive_patches", [])
    negative_patches = result.get("negative_patches", [])

    return positive_patches, negative_patches


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Use DeepSeek to select medical image patches")
    parser.add_argument("original_image", help="Path to the original medical image")
    parser.add_argument("patch_image", help="Path to the image with patch grid")
    parser.add_argument("prompt", help="Anatomical structure to identify (e.g., 'liver')")
    parser.add_argument("--api-key", help="DeepSeek API key (optional, can use DEEPSEEK_API_KEY env var)")
    parser.add_argument("--model", default="deepseek-reasoner", help="DeepSeek model to use")

    args = parser.parse_args()

    print(f"Analyzing images: {args.original_image} and {args.patch_image}")
    print(f"Looking for: {args.prompt}")
    print("Calling DeepSeek...")

    positive, negative = call_deepseek_for_patches(
        args.original_image,
        args.patch_image,
        args.prompt,
        api_key=args.api_key,
        model=args.model
    )

    print(f"\nPositive patches (contain {args.prompt}): {positive}")
    print(f"Negative patches (do NOT contain {args.prompt}): {negative}")


if __name__ == "__main__":
    main()
