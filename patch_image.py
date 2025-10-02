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


def create_patch_image_v3(input_image_path, output_image_path='image_with_indexed_regions.png', min_area=400,
                          use_adaptive=True, blur_kernel=5, canny_low=50, canny_high=100, morph_iterations=1):
    """
    Detect color blocks/regions in an image, find edges, draw edges, and index each color region.

    Args:
        input_image_path: Path to the input image
        output_image_path: Path to save the output image with indexed regions
        min_area: Minimum area threshold to filter out small regions (default: 400)
        use_adaptive: Use adaptive thresholding + Canny combination (default: True)
        blur_kernel: Gaussian blur kernel size for noise reduction (default: 5)
        canny_low: Lower threshold for Canny (default: 30)
        canny_high: Upper threshold for Canny (default: 100)
        morph_iterations: Morphological closing iterations (default: 1)

    Returns:
        regions: Dictionary mapping region index to region properties (centroid, area, color)
        output_image_path: Path to the saved image
    """
    # Load the image
    img = cv2.imread(input_image_path)
    if img is None:
        raise ValueError(f"Could not read image from {input_image_path}")

    # Convert to RGB for PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    if use_adaptive:
        # Combine adaptive thresholding with Canny for better region separation
        adaptive_thresh = cv2.adaptiveThreshold(gray_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2)
        # Also use Canny with lower thresholds
        edges_canny = cv2.Canny(gray_blurred, canny_low, canny_high)
        # Combine both edge detection methods
        edges = cv2.bitwise_or(edges_canny, cv2.Canny(adaptive_thresh, 50, 150))
    else:
        # Standard Canny edge detection with adjustable parameters
        edges = cv2.Canny(gray_blurred, canny_low, canny_high)

    # Apply morphological operations to close gaps in edges
    kernel = np.ones((3, 3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)

    # Dilate edges to make them thicker, helping to separate regions better
    edges_dilated = cv2.dilate(edges_closed, kernel, iterations=1)

    # Invert to get regions (not edges)
    regions_mask = cv2.bitwise_not(edges_dilated)

    # Use connected components instead of contours for cleaner region detection
    num_labels, labels = cv2.connectedComponents(regions_mask)

    # Convert back to contours for compatibility with existing code
    contours = []
    for label_id in range(1, num_labels):  # Skip 0 (background)
        # Create mask for this label
        label_mask = (labels == label_id).astype(np.uint8) * 255
        # Find contour for this connected component
        label_contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if label_contours:
            contours.append(label_contours[0])

    # Collect region properties
    regions = {}
    region_idx = 1

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > min_area:  # Filter out small regions
            # Calculate centroid using moments
            moments = cv2.moments(contour)

            # Use contour moments for centroid, with fallback to bounding box center
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                # Fallback: use center of bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2

            # Create mask for this contour to get color
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)

            # Get average color in this region
            region_pixels = img_rgb[mask > 0]
            if len(region_pixels) > 0:
                avg_color = tuple(np.mean(region_pixels, axis=0).astype(int))
            else:
                avg_color = (0, 0, 0)

            regions[region_idx] = {
                'centroid': (cx, cy),
                'area': int(area),
                'color': avg_color,
                'coordinates': contour.squeeze().tolist() if contour.ndim > 2 else contour.tolist()
            }
            region_idx += 1

    # Create visualization
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    # # Draw edges in red with wider line
    # edge_coords = np.column_stack(np.where(edges > 10))
    # for y, x in edge_coords:
    #     # Draw a 3x3 square to make edges wider
    #     draw.rectangle([(x-1, y-1), (x, y)], fill='red')

    # Try to use a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/google-droid/DroidSans-Bold.ttf", 20)
        except:
            font = ImageFont.load_default()

    # Draw region indices at centroids
    for idx, props in regions.items():
        cx, cy = props['centroid']

        # Draw the index number
        text = str(idx)
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        text_x = cx - text_width // 2
        text_y = cy - text_height // 2

        # Draw text with outline
        draw.text((text_x-1, text_y-1), text, fill='white', font=font)
        draw.text((text_x+1, text_y-1), text, fill='white', font=font)
        draw.text((text_x-1, text_y+1), text, fill='white', font=font)
        draw.text((text_x+1, text_y+1), text, fill='white', font=font)
        draw.text((text_x, text_y), text, fill='black', font=font)

    # Save result
    img_pil.save(output_image_path)

    print(f"Detected {len(regions)} color regions")
    print(f"Output saved to: {output_image_path}")

    return regions, output_image_path


if __name__ == "__main__":
    create_patch_image_v3('image.png')
