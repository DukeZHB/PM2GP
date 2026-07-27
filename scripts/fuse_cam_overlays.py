"""Fuse several CAM overlay images into a single visualization.

Averages the input images pixel-wise. Utility for visually inspecting
the three Stage-1 CAM branches (ic1 / ic2 / fc8) side by side; not part
of the training or evaluation pipeline.

Example:
    python scripts/fuse_cam_overlays.py \
        cam1_overlay.png cam2_overlay.png cam3_overlay.png \
        -o fused_overlay.png
"""

import argparse

import numpy as np
from PIL import Image


def pixel_wise_add_images(image_paths, output_path="added_image.png"):
    """Average multiple PNG images pixel-wise and save the result."""
    images = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        images.append(img)

    width, height = images[0].size
    for img in images[1:]:
        if img.size != (width, height):
            raise ValueError("All images must have identical width and height.")

    img_arrays = [np.array(img, dtype=np.float32) for img in images]
    added_array = sum(img_arrays) / len(img_arrays)

    added_array = np.clip(added_array, 0, 255).astype(np.uint8)
    result_image = Image.fromarray(added_array)
    result_image.save(output_path)
    print(f"Fused image saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pixel-wise average of CAM overlay images.")
    parser.add_argument("inputs", nargs="+", help="Input overlay image paths (same size).")
    parser.add_argument("-o", "--output", default="fused_overlay.png", help="Output image path.")
    args = parser.parse_args()
    pixel_wise_add_images(args.inputs, args.output)
