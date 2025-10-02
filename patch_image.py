import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_patch_image_v1(input_image_path, output_image_path='image_with_patches.png', patch_size=64):
    """
    Divide an image into patches and draw grid lines with patch numbers.

    Args:
        input_image_path: Path to the input image
        output_image_path: Path to save the output image with patches
        patch_size: Size of each patch (default: 64)

    Returns:
        output_image_path: Path to the saved image
    """
    # Load the image
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    # Calculate patch dimensions
    patches_per_row = width // patch_size
    patches_per_col = height // patch_size

    # Convert to PIL for drawing text and grid
    img_pil = Image.fromarray(img).convert('RGB')
    draw = ImageDraw.Draw(img_pil)

    # Draw grid lines
    for row in range(patches_per_col + 1):
        y = row * patch_size
        draw.line([(0, y), (width, y)], fill='red', width=2)

    for col in range(patches_per_row + 1):
        x = col * patch_size
        draw.line([(x, 0), (x, height)], fill='red', width=2)

    # Try to use a font, fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    # Add patch numbers
    patch_num = 1
    for row in range(patches_per_col):
        for col in range(patches_per_row):
            # Calculate center of patch
            center_x = col * patch_size + patch_size // 2
            center_y = row * patch_size + patch_size // 2

            # Draw the number
            text = str(patch_num)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center the text
            text_x = center_x - text_width // 2
            text_y = center_y - text_height // 2

            # Draw text with outline for visibility
            draw.text((text_x-1, text_y-1), text, fill='black', font=font)
            draw.text((text_x+1, text_y-1), text, fill='black', font=font)
            draw.text((text_x-1, text_y+1), text, fill='black', font=font)
            draw.text((text_x+1, text_y+1), text, fill='black', font=font)
            draw.text((text_x, text_y), text, fill='yellow', font=font)

            patch_num += 1

    # Save the result
    img_pil.save(output_image_path)
    print(f"Created image with {patch_num-1} patches")
    print(f"Patches per row: {patches_per_row}")
    print(f"Patches per column: {patches_per_col}")
    print(f"Output saved to: {output_image_path}")

    return output_image_path

def create_patch_image_v2(input_image_path, output_image_path='image_with_points.png', patch_size=32, gradient_threshold=45):
    """
    Place points at patch centers and filter out points in high-gradient (non-flat) areas.

    Args:
        input_image_path: Path to the input image
        output_image_path: Path to save the output image with points
        patch_size: Size of each patch (default: 64)
        gradient_threshold: Threshold for gradient magnitude. Points with gradient above this are dropped (default: 100)

    Returns:
        valid_points: List of (x, y, point_number) tuples for points in flat areas
        output_image_path: Path to the saved image
    """
    # Load the image
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape

    # Calculate Sobel gradients
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Calculate patch dimensions
    patches_per_row = width // patch_size
    patches_per_col = height // patch_size

    # Place points at patch centers and filter based on gradient
    valid_points = []
    point_num = 1

    for row in range(patches_per_col):
        for col in range(patches_per_row):
            # Calculate center of patch
            x = col * patch_size + patch_size // 2
            y = row * patch_size + patch_size // 2

            # Check gradient in a small neighborhood around the point
            neighborhood_size = 15
            y_start = max(0, y - neighborhood_size)
            y_end = min(height, y + neighborhood_size + 1)
            x_start = max(0, x - neighborhood_size)
            x_end = min(width, x + neighborhood_size + 1)

            local_gradient = gradient_magnitude[y_start:y_end, x_start:x_end]
            avg_gradient = np.mean(local_gradient)

            # Keep point if gradient is below threshold (flat area)
            if avg_gradient < gradient_threshold:
                valid_points.append((x, y, point_num))

            point_num += 1

    # Convert to PIL for drawing
    img_pil = Image.fromarray(img).convert('RGB')
    draw = ImageDraw.Draw(img_pil)

    # Try to use a font, fall back to default if unavailable
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/google-droid/DroidSans-Bold.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Draw valid points
    for x, y, num in valid_points:
        # Draw a circle at the point
        radius = 8
        draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill='red', outline='yellow', width=2)

        # Draw the number
        text = str(num)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text
        text_x = x - text_width // 2
        text_y = y - text_height // 2

        # Draw text with outline for visibility
        draw.text((text_x-1, text_y-1), text, fill='black', font=font)
        draw.text((text_x+1, text_y-1), text, fill='black', font=font)
        draw.text((text_x-1, text_y+1), text, fill='black', font=font)
        draw.text((text_x+1, text_y+1), text, fill='black', font=font)
        draw.text((text_x, text_y), text, fill='white', font=font)

    # Save the result
    img_pil.save(output_image_path)
    total_points = patches_per_row * patches_per_col
    print(f"Placed {len(valid_points)} valid points out of {total_points} total points")
    print(f"Gradient threshold: {gradient_threshold}")
    print(f"Output saved to: {output_image_path}")

    return valid_points, output_image_path


if __name__ == "__main__":
    create_patch_image_v2('image.png')
