import sys

import numpy as np
from PIL import Image


sys.path.insert(0, ".codex_deps")
import rawpy


# Assignment 4: White balance
#
# After demosaicing and gamma correction, the image still has a color cast.
# The gray-world method assumes that the average color of the whole image
# should be gray.
#
# So we compute:
#   image mean = mean of all RGB values
#   channel mean = mean of one channel
#   corrected channel = channel * image mean / channel mean


raw = rawpy.imread("exercise_2_data/02/IMG_4782.CR3")
raw_image = np.array(raw.raw_image_visible, dtype=np.float32)


# Start with the same Bayer masks as in the demosaicing exercise.
red_mask = np.zeros(raw_image.shape, dtype=np.float32)
green_mask = np.zeros(raw_image.shape, dtype=np.float32)
blue_mask = np.zeros(raw_image.shape, dtype=np.float32)

# Bayer pattern for IMG_4782.CR3:
# R G
# G B
red_mask[0::2, 0::2] = 1
green_mask[0::2, 1::2] = 1
green_mask[1::2, 0::2] = 1
blue_mask[1::2, 1::2] = 1


def convolution_3x3(image):
    # Simple 3x3 all-ones convolution from the lecture.
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
    # Estimate one RGB channel from the known Bayer samples.
    weighted_sum = convolution_3x3(raw_image * mask)
    weights = convolution_3x3(mask)
    return weighted_sum / weights


def normalize_with_percentiles(image):
    # Normalize to [0, 1] before gamma correction.
    a = np.percentile(image, 0.01)
    b = np.percentile(image, 99.99)

    image = (image - a) / (b - a)
    image[image < 0] = 0
    image[image > 1] = 1

    return image


def gray_world(image):
    # Gray-world tries to make the average image color neutral gray.
    image_mean = image.mean()

    red_mean = image[:, :, 0].mean()
    green_mean = image[:, :, 1].mean()
    blue_mean = image[:, :, 2].mean()

    image[:, :, 0] = image[:, :, 0] * image_mean / red_mean
    image[:, :, 1] = image[:, :, 1] * image_mean / green_mean
    image[:, :, 2] = image[:, :, 2] * image_mean / blue_mean

    image[image < 0] = 0
    image[image > 1] = 1

    return image


def save_image(image, path):
    # Save only after all float processing is finished.
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path, quality=98)


# Demosaic the raw image, then test white balance after two gamma values.
red = demosaic_channel(red_mask)
green = demosaic_channel(green_mask)
blue = demosaic_channel(blue_mask)

rgb = np.dstack([red, green, blue])

rgb_normalized = normalize_with_percentiles(rgb)

for gamma in [0.3, 0.5]:
    # Apply gamma first, then correct the color cast with gray-world.
    rgb = rgb_normalized**gamma
    rgb = gray_world(rgb)

    output_name = "assignment 4 white balance gamma " + str(gamma) + ".jpg"
    save_image(rgb, output_name)

    print("Saved", output_name)
