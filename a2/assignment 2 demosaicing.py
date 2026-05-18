import sys

import numpy as np
from PIL import Image


# rawpy was installed locally in this folder.
sys.path.insert(0, ".codex_deps")
import rawpy


# Assignment 2: Simple demosaicing
#
# A raw camera image has only one value per pixel.
# Because of the Bayer filter, each pixel measures only red, green, or blue.
#
# For IMG_4782.CR3, rawpy tells us the Bayer pattern is:
#
#   R G
#   G B
#
# We create one mask for each color. Then, for every channel, we fill missing
# values by averaging nearby known values from the same color.


# Load the visible raw sensor image. It is still a 2D Bayer mosaic, not RGB.
raw = rawpy.imread("exercise_2_data/02/IMG_4782.CR3")
raw_image = np.array(raw.raw_image_visible, dtype=np.float32)

print("Raw image shape:", raw_image.shape)
print("Raw Bayer pattern from rawpy:")
print(raw.raw_pattern)
print("Color description:", raw.color_desc.decode("ascii"))


# The masks mark where the real red, green and blue samples are located.
# Missing values stay zero for now and are filled during demosaicing.
red_mask = np.zeros(raw_image.shape, dtype=np.float32)
green_mask = np.zeros(raw_image.shape, dtype=np.float32)
blue_mask = np.zeros(raw_image.shape, dtype=np.float32)

red_mask[0::2, 0::2] = 1
green_mask[0::2, 1::2] = 1
green_mask[1::2, 0::2] = 1
blue_mask[1::2, 1::2] = 1


def convolution_3x3(image):
    # This is the simple 3x3 all-ones convolution from the lecture.
    padded = np.pad(image, 1, mode="edge")

    return (
        padded[:-2, :-2]
        + padded[:-2, 1:-1]
        + padded[:-2, 2:]
        + padded[1:-1, :-2]
        + padded[1:-1, 1:-1]
        + padded[1:-1, 2:]
        + padded[2:, :-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )


def demosaic_channel(mask):
    # Lecture formula:
    # channel = convolution(raw * mask) / convolution(mask)
    weighted_sum = convolution_3x3(raw_image * mask)
    weights = convolution_3x3(mask)
    return weighted_sum / weights


# Reconstruct the three color channels and combine them into one RGB image.
red = demosaic_channel(red_mask)
green = demosaic_channel(green_mask)
blue = demosaic_channel(blue_mask)

rgb = np.dstack([red, green, blue])

# Keep the processing in float. Convert to uint8 only for saving as JPG.
jpg = rgb / raw.white_level
jpg = np.clip(jpg, 0, 1)
jpg = (jpg * 255).astype(np.uint8)

Image.fromarray(jpg).save("assignment 2 demosaiced.jpg", quality=98)

print("Saved result as assignment 2 demosaiced.jpg")
