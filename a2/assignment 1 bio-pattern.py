import numpy as np


# Assignment 1: Investigate Bayer Pattern
#
# The file IMG_9939.JPG shows three pens on white paper:
# red at the top, green in the middle, blue at the bottom.
#
# The raw sensor data in IMG_9939.npy is a single-channel Bayer mosaic.
# We inspect the four possible positions in the repeating 2x2 pattern:
#   (even row, even col), (even row, odd col),
#   (odd row, even col),  (odd row, odd col)
#
# Lecture hint:
# green sensor values are usually higher, even on non-green pixels.


raw_data = np.load("exercise_2_data/01/IMG_9939.npy")

# First we print basic information, so we know what kind of raw array we have.
print("Loaded raw sensor data from exercise_2_data/01/IMG_9939.npy")
print("Shape:", raw_data.shape)
print("Data type:", raw_data.dtype)
print("Minimum value:", raw_data.min())
print("Maximum value:", raw_data.max())
print()

print("Raw NumPy array:")
print(raw_data)
print()

# These names make the four positions of the 2x2 Bayer block easier to read.
parity_names = {
    (0, 0): "even row, even col",
    (0, 1): "even row, odd col",
    (1, 0): "odd row, even col",
    (1, 1): "odd row, odd col",
}

print("Mean raw values for each Bayer position:")

# Every Bayer pattern repeats every 2 rows and 2 columns.
# Here we collect the values belonging to each of the four positions.
for row_parity in (0, 1):
    for col_parity in (0, 1):
        values = raw_data[row_parity::2, col_parity::2]
        print(
            f"{parity_names[(row_parity, col_parity)]:18s}: "
            f"mean={values.mean():.2f}, "
            f"min={values.min()}, "
            f"max={values.max()}"
        )

print()

# From the printed means, the two high-value positions are:
#   (even row, even col) and (odd row, odd col)
# According to the lecture hint, these two positions are the green samples.
#
# Comparing local regions of the red and blue pens:
#   - the red pen responds more strongly at (odd row, even col)
#   - the blue pen responds more strongly at (even row, odd col)
#
# Therefore the 2x2 Bayer pattern is:
#   G B
#   R G

print("Conclusion:")
print("The Bayer pattern is:")
print("G B")
print("R G")
print()
print("Pattern name: GBRG")
