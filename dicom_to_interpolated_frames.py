import os
import sys
import numpy as np
import pydicom
from PIL import Image
import subprocess
import shutil
import argparse

def dicom_to_images(dicom_path, output_dir):
    """
    Convert DICOM file to a list of images.
    Handles both single-frame and multi-frame DICOM files.
    """
    print(f"Reading DICOM file: {dicom_path}")
    dcm = pydicom.dcmread(dicom_path)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if multi-frame DICOM
    if hasattr(dcm, 'NumberOfFrames') and dcm.NumberOfFrames > 1:
        print(f"Multi-frame DICOM detected: {dcm.NumberOfFrames} frames")
        pixel_array = dcm.pixel_array  # Shape: (num_frames, height, width)
        num_frames = pixel_array.shape[0]
    else:
        print("Single-frame DICOM detected")
        pixel_array = dcm.pixel_array
        if len(pixel_array.shape) == 2:
            pixel_array = pixel_array[np.newaxis, ...]  # Add frame dimension
        num_frames = 1

    # Normalize to 0-255 range
    pixel_array = pixel_array.astype(np.float32)
    pixel_min = pixel_array.min()
    pixel_max = pixel_array.max()
    pixel_array = ((pixel_array - pixel_min) / (pixel_max - pixel_min) * 255).astype(np.uint8)

    # Save each frame as PNG
    image_paths = []
    for i in range(num_frames):
        frame = pixel_array[i]
        # Convert grayscale to RGB
        if len(frame.shape) == 2:
            frame = np.stack([frame, frame, frame], axis=-1)

        img = Image.fromarray(frame)
        img_path = os.path.join(output_dir, f"{i}.png")
        img.save(img_path)
        image_paths.append(img_path)

    print(f"Saved {num_frames} frames to {output_dir}")
    return image_paths, num_frames

def interpolate_frames(input_dir, output_dir, exp=3, model_dir=None):
    """
    Use RIFE to interpolate frames.
    exp=3 means 2^3-1=7 intermediate frames, but we want 8, so we'll use exp=3
    and adjust accordingly.
    """
    print(f"Running RIFE interpolation with exp={exp} (2^{exp} = {2**exp}x frames)")

    # Path to RIFE directory and script
    rife_dir = os.path.join(os.path.dirname(__file__), "ECCV2022-RIFE")
    rife_script = "inference_video.py"

    # Convert input_dir to absolute path
    input_dir = os.path.abspath(input_dir)

    # Build command
    cmd = [
        sys.executable,
        rife_script,
        "--img", input_dir,
        "--exp", str(exp),
        "--png"
    ]

    # Add model directory if specified
    if model_dir:
        cmd.extend(["--model", model_dir])

    print(f"Running RIFE from directory: {rife_dir}")
    print(f"Running command: {' '.join(cmd)}")

    # Run RIFE from its own directory (important for imports and relative paths)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=rife_dir)

    if result.returncode != 0:
        print("RIFE stderr:", result.stderr)
        print("RIFE stdout:", result.stdout)
        raise RuntimeError(f"RIFE interpolation failed. See above for details.")

    print("RIFE interpolation completed")
    print(result.stdout)

    # RIFE outputs to 'vid_out' directory by default (in RIFE directory)
    rife_output_dir = os.path.join(rife_dir, "vid_out")

    return rife_output_dir

def convert_to_jpeg_format(input_dir, output_dir, quality=95):
    """
    Convert PNG frames to JPEG with the naming format: 00000.jpg, 00001.jpg, etc.
    """
    print(f"Converting frames to JPEG format in {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Get all PNG files from RIFE output
    png_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.png')])

    for idx, png_file in enumerate(png_files):
        # Read PNG
        img_path = os.path.join(input_dir, png_file)
        img = Image.open(img_path)

        # Convert to RGB if necessary (JPEG doesn't support transparency)
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (0, 0, 0))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # Save as JPEG with 5-digit filename
        jpeg_path = os.path.join(output_dir, f"{idx:05d}.jpg")
        img.save(jpeg_path, 'JPEG', quality=quality)

    print(f"Saved {len(png_files)} JPEG frames to {output_dir}")
    return len(png_files)

def main():
    parser = argparse.ArgumentParser(description='Convert DICOM to interpolated JPEG frames')
    parser.add_argument('--dicom', type=str, default='example.dcm', help='Path to DICOM file')
    parser.add_argument('--output', type=str, default='output_frames', help='Output directory for final JPEG frames')
    parser.add_argument('--exp', type=int, default=3, help='Interpolation exponent (exp=3 gives 8x frames)')
    parser.add_argument('--model', type=str, default=None, help='Path to RIFE model directory (default: train_log)')
    parser.add_argument('--keep-temp', action='store_true', help='Keep temporary directories')
    parser.add_argument('--quality', type=int, default=95, help='JPEG quality (1-100)')

    args = parser.parse_args()

    # Step 1: Convert DICOM to images
    temp_dicom_dir = "temp_dicom_frames"
    print("\n=== Step 1: Converting DICOM to images ===")
    image_paths, num_frames = dicom_to_images(args.dicom, temp_dicom_dir)

    # Step 2: Run RIFE interpolation
    print("\n=== Step 2: Running RIFE interpolation ===")
    rife_output_dir = interpolate_frames(temp_dicom_dir, "temp_interpolated", exp=args.exp, model_dir=args.model)

    # Step 3: Convert to JPEG format
    print("\n=== Step 3: Converting to JPEG format ===")
    num_output_frames = convert_to_jpeg_format(rife_output_dir, args.output, quality=args.quality)

    # Cleanup temporary directories
    if not args.keep_temp:
        print("\n=== Cleaning up temporary files ===")
        if os.path.exists(temp_dicom_dir):
            shutil.rmtree(temp_dicom_dir)
        if os.path.exists(rife_output_dir):
            shutil.rmtree(rife_output_dir)

    print("\n=== Done! ===")
    print(f"Original frames: {num_frames}")
    print(f"Interpolated frames: {num_output_frames}")
    print(f"Output directory: {args.output}")
    print(f"Frames are named as: 00000.jpg, 00001.jpg, ...")

if __name__ == "__main__":
    main()
