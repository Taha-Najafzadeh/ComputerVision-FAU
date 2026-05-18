import sys

import numpy as np
from PIL import Image


sys.path.insert(0, ".codex_deps")
import rawpy


# Assignment 8: process one raw CR3 file into a JPG image.
#
# The required function is process_raw(raw_path, output_path).
# It uses the same course-compatible steps as the previous exercises:
# raw data -> Bayer masks -> lecture demosaicing -> luminosity correction
# -> gray-world white balance -> high-quality JPG.


def load_raw_sensor_data(raw):
    # Remove the camera black offset before doing any image processing.
    image = np.array(raw.raw_image_visible, dtype=np.float32)
    black_levels = np.array(raw.black_level_per_channel, dtype=np.float32)
    black_image = black_levels[raw.raw_colors_visible]

    image = image - black_image
    image = np.clip(image, 0, None)

    return image


def make_bayer_masks(raw):
    # raw_colors_visible tells us which color each visible pixel belongs to.
    colors = raw.raw_colors_visible
    color_description = raw.color_desc.decode("ascii")

    red_indices = [index for index, color in enumerate(color_description) if color == "R"]
    green_indices = [index for index, color in enumerate(color_description) if color == "G"]
    blue_indices = [index for index, color in enumerate(color_description) if color == "B"]

    red_mask = np.isin(colors, red_indices).astype(np.float32)
    green_mask = np.isin(colors, green_indices).astype(np.float32)
    blue_mask = np.isin(colors, blue_indices).astype(np.float32)

    return red_mask, green_mask, blue_mask


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


def demosaic_channel(raw_image, mask):
    # Lecture demosaicing formula for one channel.
    weighted_sum = convolution_3x3(raw_image * mask)
    weights = convolution_3x3(mask)
    return weighted_sum / weights


def normalize_with_percentiles(image):
    # Normalize brightness while ignoring very extreme pixel values.
    low = np.percentile(image, 0.01)
    high = np.percentile(image, 99.99)

    image = (image - low) / (high - low)
    image = np.clip(image, 0, 1)

    return image


def gray_world(image):
    # Correct color cast by making the channel averages similar.
    image_mean = image.mean()

    red_mean = image[:, :, 0].mean()
    green_mean = image[:, :, 1].mean()
    blue_mean = image[:, :, 2].mean()

    image[:, :, 0] = image[:, :, 0] * image_mean / red_mean
    image[:, :, 1] = image[:, :, 1] * image_mean / green_mean
    image[:, :, 2] = image[:, :, 2] * image_mean / blue_mean

    image = np.clip(image, 0, 1)

    return image


def save_image(image, output_path):
    # Convert to 8-bit only when saving the final JPG.
    image = np.clip(image, 0, 1)
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(output_path, quality=98)


def process_raw(raw_path, output_path):
    # This is the function requested by Exercise 8.
    raw = rawpy.imread(raw_path)

    # Load raw data and build masks from the CR3 metadata.
    raw_image = load_raw_sensor_data(raw)
    red_mask, green_mask, blue_mask = make_bayer_masks(raw)

    print("Input:", raw_path)
    print("Raw Bayer pattern:")
    print(raw.raw_pattern)
    print("Color description:", raw.color_desc.decode("ascii"))
    print("Black levels:", raw.black_level_per_channel)

    # Demosaic the raw Bayer image into RGB.
    red = demosaic_channel(raw_image, red_mask)
    green = demosaic_channel(raw_image, green_mask)
    blue = demosaic_channel(raw_image, blue_mask)

    rgb = np.dstack([red, green, blue])

    # Improve brightness, then apply gray-world white balance.
    rgb = normalize_with_percentiles(rgb)
    rgb = rgb**0.3
    rgb = gray_world(rgb)

    save_image(rgb, output_path)
    print("Saved", output_path)


if __name__ == "__main__":
    # With two command-line arguments, process the given input and output paths.
    # Without arguments, run a default test on the provided raw file.
    if len(sys.argv) == 3:
        process_raw(sys.argv[1], sys.argv[2])
    else:
        process_raw("exercise_2_data/02/IMG_4782.CR3", "assignment 8 process raw.jpg")
