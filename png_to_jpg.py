from PIL import Image

# Convert PNG to JPG
def convert_png_to_jpg(png_path, jpg_path):
    img = Image.open(png_path).convert('RGB')  # Convert to RGB to avoid alpha channel issues
    img.save(jpg_path, 'JPEG', quality=95)  # Save as JPG with quality setting
    