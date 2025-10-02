import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load the image
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
height, width = img.shape

# Calculate patch dimensions
patch_size = 64
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
img_pil.save('image_with_patches.png')
print(f"Created image with {patch_num-1} patches")
print(f"Patches per row: {patches_per_row}")
print(f"Patches per column: {patches_per_col}")
print(f"Output saved to: image_with_patches.png")
