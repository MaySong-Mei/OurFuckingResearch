#!/usr/bin/env python3
"""
Medical Image Patch Selector using ChatGPT
Sends a medical image with grad and a prompt to ChatGPT to identify positive/negative patches.
"""

import os
import base64
from typing import Tuple, List
from openai import OpenAI


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def call_chatgpt_for_patches(
    image_path: str,
    prompt: str,
    api_key: str = None,
    model: str = "gpt-4o"
) -> Tuple[List[int], List[int]]:
    """
    Call ChatGPT with medical image and prompt to get positive/negative patch sequences.

    Args:
        image_path: Path to the medical image
        prompt: The target to identify (e.g., "liver")
        api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
        model: OpenAI model to use (default: gpt-4o for vision)

    Returns:
        Tuple of (positive_patches, negative_patches) where each is a list of patch indices
    """
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    # Encode image
    base64_image = encode_image_to_base64(image_path)
    image_extension = os.path.splitext(image_path)[1].lower()
    mime_type = f"image/{image_extension[1:]}" if image_extension in ['.png', '.jpg', '.jpeg'] else "image/png"

    # Construct the message
    system_prompt = """You are a medical image analysis assistant. You will receive a medical image divided into patches with gradients.
    Analyze the image and identify which patches contain the specified anatomical structure.
    Return your answer in the following JSON format:
    {
        "positive_patches": [list of patch numbers that contain the structure],
        "negative_patches": [list of patch numbers that do NOT contain the structure]
    }
    Only return valid JSON, no additional text."""

    user_prompt = f"Analyze this medical image and identify which patches contain {prompt}. Return the positive and negative patch sequences as JSON."

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
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    # Parse response
    import json
    result = json.loads(response.choices[0].message.content)

    positive_patches = result.get("positive_patches", [])
    negative_patches = result.get("negative_patches", [])

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
        args.prompt,
        api_key=args.api_key,
        model=args.model
    )

    print(f"\nPositive patches (contain {args.prompt}): {positive}")
    print(f"Negative patches (do NOT contain {args.prompt}): {negative}")


if __name__ == "__main__":
    main()
