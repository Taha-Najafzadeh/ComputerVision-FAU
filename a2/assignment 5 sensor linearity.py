import sys

import numpy as np
from PIL import Image, ImageDraw


sys.path.insert(0, ".codex_deps")
import rawpy


# Assignment 5: Show that sensor data is linear
#
# Images IMG_3044 to IMG_3049 show the same scene with different exposure times.
# Each exposure time is half of the previous one:
#
# IMG_3044: 1/10 s
# IMG_3045: 1/20 s
# IMG_3046: 1/40 s
# IMG_3047: 1/80 s
# IMG_3048: 1/160 s
# IMG_3049: 1/320 s
#
# We compute ONE average value for each whole raw image.
# We do not compute channel-wise averages, because the exercise says that would fail.


files = [
    "exercise_2_data/05/IMG_3044.CR3",
    "exercise_2_data/05/IMG_3045.CR3",
    "exercise_2_data/05/IMG_3046.CR3",
    "exercise_2_data/05/IMG_3047.CR3",
    "exercise_2_data/05/IMG_3048.CR3",
    "exercise_2_data/05/IMG_3049.CR3",
]

exposure_times = np.array([1 / 10, 1 / 20, 1 / 40, 1 / 80, 1 / 160, 1 / 320])
average_values = []

# For each raw image, use the mean of the whole sensor image.
# This shows how brightness changes with exposure time.
for filename in files:
    raw = rawpy.imread(filename)
    raw_image = np.array(raw.raw_image_visible, dtype=np.float32)
    average_values.append(raw_image.mean())

average_values = np.array(average_values)

print("Exposure time and average raw value:")
for filename, exposure, average in zip(files, exposure_times, average_values):
    print(filename, " exposure =", exposure, " average =", average)


# Draw a simple plot without extra libraries.

width = 900
height = 600
margin = 80

image = Image.new("RGB", (width, height), "white")
draw = ImageDraw.Draw(image)

x_min = exposure_times.min()
x_max = exposure_times.max()
y_min = 0
y_max = average_values.max() * 1.1


# Convert exposure/value coordinates into pixel positions in the plot image.
def plot_x(x):
    return margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)


def plot_y(y):
    return height - margin - (y - y_min) / (y_max - y_min) * (height - 2 * margin)


# Axes
draw.line((margin, height - margin, width - margin, height - margin), fill="black", width=2)
draw.line((margin, margin, margin, height - margin), fill="black", width=2)

draw.text((width // 2 - 70, height - 40), "Exposure time (seconds)", fill="black")
draw.text((10, 20), "Average raw value", fill="black")

# Store the points first, then draw one line through them.
points = []
for exposure, average in zip(exposure_times, average_values):
    points.append((plot_x(exposure), plot_y(average)))

draw.line(points, fill="blue", width=3)

for x, y in points:
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red")

image.save("assignment 5 sensor linearity.png")

print("Saved plot as assignment 5 sensor linearity.png")
