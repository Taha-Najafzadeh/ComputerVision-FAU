import sys

import numpy as np
from PIL import Image


sys.path.insert(0, ".codex_deps")
import rawpy


# Assignment 3: Improve the luminosity
#
# The demosaiced image from Assignment 2 is dark.
# Here we brighten it with:
#   1. gamma correction: y = x^gamma, using gamma = 0.3
#   2. another curve: logarithmic correction
#
# We normalize with percentiles, as written in the exercise sheet.


raw = rawpy.imread("exercise_2_data/02/IMG_4782.CR3")
raw_image = np.array(raw.raw_image_visible, dtype=np.float32)


# Same Bayer masks and demosaicing as Assignment 2.
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
    # Simple 3x3 all-ones convolution.
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
    # Fill missing values of one color channel with the lecture formula.
    weighted_sum = convolution_3x3(raw_image * mask)
    weights = convolution_3x3(mask)
    return weighted_sum / weights


def normalize_with_percentiles(image):
    # Percentiles avoid one very dark or very bright pixel controlling the result.
    a = np.percentile(image, 0.01)
    b = np.percentile(image, 99.99)

    image = (image - a) / (b - a)
    image[image < 0] = 0
    image[image > 1] = 1

    return image


def save_image(image, path):
    # JPG needs 8-bit values, so conversion is done only at the end.
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path, quality=98)


# First demosaic the raw image into RGB.
red = demosaic_channel(red_mask)
green = demosaic_channel(green_mask)
blue = demosaic_channel(blue_mask)

rgb = np.dstack([red, green, blue])
rgb_normalized = normalize_with_percentiles(rgb)

# Gamma below 1 brightens the dark demosaiced image.
gamma = 0.3
gamma_corrected = rgb_normalized**gamma
save_image(gamma_corrected, "assignment 3 gamma 0.3.jpg")

gamma = 0.5
gamma_corrected = rgb_normalized**gamma
save_image(gamma_corrected, "assignment 3 gamma 0.5.jpg")

# The logarithmic curve is the second brightness curve for comparison.
log_corrected = np.log1p(9 * rgb_normalized) / np.log(10)
save_image(log_corrected, "assignment 3 log curve.jpg")

print("Saved assignment 3 gamma 0.3.jpg")
print("Saved assignment 3 gamma 0.5.jpg")
print("Saved assignment 3 log curve.jpg")
