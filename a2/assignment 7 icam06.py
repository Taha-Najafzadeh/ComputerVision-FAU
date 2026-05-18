import sys

import numpy as np
from PIL import Image


sys.path.insert(0, ".codex_deps")
import rawpy


# Assignment 7: iCAM06 HDR compression
#
# This script uses the same HDR raw data from Assignment 6.
# After HDR combination, demosaicing and gray-world white balance, it applies
# the iCAM06 pseudocode from the lecture slides.


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
    # Use the same black-level correction as in Assignment 6.
    image = np.array(raw.raw_image_visible, dtype=np.float32)
    black_levels = np.array(raw.black_level_per_channel, dtype=np.float32)
    black_image = black_levels[raw.raw_colors_visible]

    image = image - black_image
    image = np.clip(image, 0, None)

    return image


raw = rawpy.imread(files[0])
# Read the Bayer pattern from the first CR3 file.
color_description = raw.color_desc.decode("ascii")
pattern = "".join(color_description[value] for row in raw.raw_pattern for value in row)

# First create the HDR raw image exactly like Exercise 6.
hdr = load_raw_sensor_data(raw)
threshold = 0.8 * hdr.max()

print("Initial HDR image:", files[0])
print("Raw Bayer pattern:")
print(raw.raw_pattern)
print("Color description:", color_description)
print("Bayer pattern:", pattern)
print("Black levels:", raw.black_level_per_channel)
print("Threshold:", threshold)

# Scale each darker exposure and replace saturated values in hdr.
for index, filename in enumerate(files[1:], start=1):
    raw = rawpy.imread(filename)
    image = load_raw_sensor_data(raw)

    exposure_factor = 2**index
    image = image * exposure_factor

    saturated_pixels = hdr > threshold
    hdr[saturated_pixels] = image[saturated_pixels]

    print(filename, "factor =", exposure_factor)


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


def make_bayer_masks(pattern):
    # Create masks for the visible raw image using the Bayer pattern.
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
    # Same demosaicing formula as the earlier exercises.
    weighted_sum = convolution_3x3(hdr * mask)
    weights = convolution_3x3(mask)
    return weighted_sum / weights


def gray_world(image):
    # Apply gray-world before iCAM06, as in the exercise pipeline.
    image_mean = image.mean()

    red_mean = image[:, :, 0].mean()
    green_mean = image[:, :, 1].mean()
    blue_mean = image[:, :, 2].mean()

    image[:, :, 0] = image[:, :, 0] * image_mean / red_mean
    image[:, :, 1] = image[:, :, 1] * image_mean / green_mean
    image[:, :, 2] = image[:, :, 2] * image_mean / blue_mean

    return image


def bilateral_filter(image, kernel_radius, sigma_space, sigma_range):
    # Bilateral filter makes the smooth base while keeping stronger edges.
    padded = np.pad(image, kernel_radius, mode="edge")
    filtered_sum = np.zeros(image.shape, dtype=np.float32)
    weight_sum = np.zeros(image.shape, dtype=np.float32)

    for row_offset in range(-kernel_radius, kernel_radius + 1):
        for col_offset in range(-kernel_radius, kernel_radius + 1):
            shifted = padded[
                kernel_radius + row_offset : kernel_radius + row_offset + image.shape[0],
                kernel_radius + col_offset : kernel_radius + col_offset + image.shape[1],
            ]

            space_distance = row_offset**2 + col_offset**2
            space_weight = np.exp(-space_distance / (2 * sigma_space**2))

            range_distance = (shifted - image) ** 2
            range_weight = np.exp(-range_distance / (2 * sigma_range**2))

            weight = space_weight * range_weight
            filtered_sum += shifted * weight
            weight_sum += weight

    return filtered_sum / weight_sum


def icam06(rgb, output_range, kernel_radius, sigma_space, sigma_range):
    # This follows the iCAM06 pseudocode from the lecture slide.
    epsilon = 1e-6

    # Intensity is the weighted brightness value from the slide.
    input_intensity = (20 * rgb[:, :, 0] + 40 * rgb[:, :, 1] + rgb[:, :, 2]) / 61
    input_intensity = np.clip(input_intensity, epsilon, None)

    # Keep color ratios separate while compressing only the intensity.
    chromaticity = rgb / input_intensity[:, :, None]

    # Split log intensity into base and detail parts.
    log_intensity = np.log(input_intensity)
    log_base = bilateral_filter(log_intensity, kernel_radius, sigma_space, sigma_range)
    log_details = log_intensity - log_base

    # Compress the base, then add the details back.
    compression = np.log(output_range) / (log_base.max() - log_base.min())
    log_offset = -log_base.max() * compression

    output_intensity = np.exp(log_base * compression + log_offset + log_details)
    result = chromaticity * output_intensity[:, :, None]

    return result


def save_image(image, path):
    # Use percentile display scaling so a few extreme pixels do not dominate.
    image = np.clip(image, 0, None)
    low = np.percentile(image, 0.01)
    high = np.percentile(image, 99.9)
    image = (image - low) / (high - low)
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(path, quality=98)


# Demosaic and white-balance once. Only iCAM06 settings change after this.
red_mask, green_mask, blue_mask = make_bayer_masks(pattern)

red = demosaic_channel(red_mask)
green = demosaic_channel(green_mask)
blue = demosaic_channel(blue_mask)

rgb = np.dstack([red, green, blue])
rgb = gray_world(rgb)
rgb = np.clip(rgb, 0, None)

# These are the parameter sets we compare visually.
settings = [
    {
        "name": "setting_1_default",
        "output_range": 4,
        "kernel_radius": 4,
        "sigma_space": 4,
        "sigma_range": 0.5,
    },
    {
        "name": "setting_2_brighter",
        "output_range": 6,
        "kernel_radius": 4,
        "sigma_space": 4,
        "sigma_range": 0.5,
    },
    {
        "name": "setting_3_smoother",
        "output_range": 4,
        "kernel_radius": 5,
        "sigma_space": 5,
        "sigma_range": 0.8,
    },
    {
        "name": "setting_4_less_halo",
        "output_range": 6,
        "kernel_radius": 5,
        "sigma_space": 6,
        "sigma_range": 1.0,
    },
    {
        "name": "setting_5_more_contrast",
        "output_range": 3,
        "kernel_radius": 4,
        "sigma_space": 4,
        "sigma_range": 0.4,
    },
]

for setting in settings:
    print("iCAM06 parameters:", setting)

    result = icam06(
        rgb,
        setting["output_range"],
        setting["kernel_radius"],
        setting["sigma_space"],
        setting["sigma_range"],
    )

    output_name = "assignment 7 icam06 " + setting["name"] + ".jpg"
    save_image(result, output_name)

    print("Saved", output_name)
