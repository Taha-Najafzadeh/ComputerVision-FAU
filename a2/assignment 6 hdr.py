import sys

import numpy as np
from PIL import Image


sys.path.insert(0, ".codex_deps")
import rawpy


# Assignment 6: Initial HDR implementation
#
# Lecture method:
# 1. Load the brightest raw image, 00.CR3, and call it hdr.
# 2. For each next image:
#    - load it
#    - multiply it by the exposure difference to the first image
#      01.CR3 is multiplied by 2, 02.CR3 by 4, etc.
#    - replace values in hdr that are above a threshold.
# 3. Demosaic the HDR raw data.
# 4. Apply gray-world white balance.
# 5. Apply logarithm.
# 6. Normalize to [0, 255] and save.


files = [
    "exercise_2_data/06/00.CR3",
    "exercise_2_data/06/01.CR3",
    "exercise_2_data/06/02.CR3",
    "exercise_2_data/06/03.CR3",
    "exercise_2_data/06/04.CR3",
    "exercise_2_data/06/05.CR3",
    "exercise_2_data/06/06.CR3",
    "exercise_2_data/06/07.CR3",
    "exercise_2_data/06/08.CR3",
    "exercise_2_data/06/09.CR3",
    "exercise_2_data/06/10.CR3",
]


def load_raw_sensor_data(raw):
    # Raw files contain a black offset, so remove it before exposure scaling.
    image = np.array(raw.raw_image_visible, dtype=np.float32)
    black_levels = np.array(raw.black_level_per_channel, dtype=np.float32)
    black_image = black_levels[raw.raw_colors_visible]

    image = image - black_image
    image = np.clip(image, 0, None)

    return image


raw = rawpy.imread(files[0])
# Read the Bayer pattern from the CR3 metadata instead of guessing it.
color_description = raw.color_desc.decode("ascii")
pattern = "".join(color_description[value] for row in raw.raw_pattern for value in row)

# Start the HDR image with the brightest exposure, as required in the lecture.
hdr = load_raw_sensor_data(raw)
threshold = 0.8 * hdr.max()

print("Initial HDR image:", files[0])
print("Raw Bayer pattern:")
print(raw.raw_pattern)
print("Color description:", color_description)
print("Bayer pattern:", pattern)
print("Black levels:", raw.black_level_per_channel)
print("Threshold:", threshold)

# Each following image is darker, so scale it back by 2**index.
# If a pixel in the current HDR image is above the threshold, replace it.
for index, filename in enumerate(files[1:], start=1):
    raw = rawpy.imread(filename)
    image = load_raw_sensor_data(raw)

    exposure_factor = 2**index
    image = image * exposure_factor

    saturated_pixels = hdr > threshold
    hdr[saturated_pixels] = image[saturated_pixels]

    print(filename, "factor =", exposure_factor)


def convolution_3x3(image):
    # Simple 3x3 all-ones convolution used in the lecture formula.
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


def make_bayer_masks(pattern):
    # Build one mask per color from the four-letter Bayer pattern.
    red_mask = np.zeros(hdr.shape, dtype=np.float32)
    green_mask = np.zeros(hdr.shape, dtype=np.float32)
    blue_mask = np.zeros(hdr.shape, dtype=np.float32)

    positions = [
        (0, 0, pattern[0]),
        (0, 1, pattern[1]),
        (1, 0, pattern[2]),
        (1, 1, pattern[3]),
    ]

    for row_start, col_start, color in positions:
        if color == "R":
            red_mask[row_start::2, col_start::2] = 1
        elif color == "G":
            green_mask[row_start::2, col_start::2] = 1
        elif color == "B":
            blue_mask[row_start::2, col_start::2] = 1

    return red_mask, green_mask, blue_mask


def demosaic_channel(mask):
    # Lecture demosaicing formula for one color channel.
    weighted_sum = convolution_3x3(hdr * mask)
    weights = convolution_3x3(mask)
    return weighted_sum / weights


def gray_world(image):
    # Balance channels so their averages move toward the same gray value.
    image_mean = image.mean()

    red_mean = image[:, :, 0].mean()
    green_mean = image[:, :, 1].mean()
    blue_mean = image[:, :, 2].mean()

    image[:, :, 0] = image[:, :, 0] * image_mean / red_mean
    image[:, :, 1] = image[:, :, 1] * image_mean / green_mean
    image[:, :, 2] = image[:, :, 2] * image_mean / blue_mean

    return image


# Demosaic the HDR raw data before doing color and display corrections.
red_mask, green_mask, blue_mask = make_bayer_masks(pattern)

red = demosaic_channel(red_mask)
green = demosaic_channel(green_mask)
blue = demosaic_channel(blue_mask)

rgb = np.dstack([red, green, blue])
rgb = gray_world(rgb)
rgb = np.clip(rgb, 0, None)

# Log compression reduces the HDR range so it can be displayed.
rgb = np.log1p(rgb)

# Percentile normalization maps the log image to the JPG range.
low = np.percentile(rgb, 0.01)
high = np.percentile(rgb, 99.9)

rgb = (rgb - low) / (high - low)
rgb = np.clip(rgb, 0, 1)
rgb = rgb * 255
rgb = np.clip(rgb, 0, 255)
rgb = rgb.astype(np.uint8)

Image.fromarray(rgb).save("assignment 6 hdr.jpg", quality=98)

print("Saved assignment 6 hdr.jpg")
