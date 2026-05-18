import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import importlib
import cv2

# rgb data from HDR after demosaicing
hdr_module = importlib.import_module('6_hdr_implementation')
rgb = hdr_module.rgb_wb

R = rgb[:, :, 0]
G = rgb[:, :, 1]
B = rgb[:, :, 2]

# input_intensity
input_intensity = (1/61) * (20*R + 40*G + 1*B)
input_intensity = np.clip(input_intensity, 1e-6, None)  # avoid log(0)

# separate color from brightness
r = R / input_intensity
g = G / input_intensity
b = B / input_intensity

# intensity log
log_intensity = np.log(input_intensity)

# bilateral filter
log_base = cv2.bilateralFilter(log_intensity.astype(np.float32),d=3,sigmaColor=0.1,sigmaSpace=3)


# detail remaining after base removal
log_details = log_intensity - log_base

# compression
output_range = 4
compression = np.log(output_range) / (log_base.max() - log_base.min())

# offset
log_offset = -log_base.max() * compression

output_intensity = np.exp(log_base * compression + log_offset + log_details)

# add color back
R_out = r * output_intensity
G_out = g * output_intensity
B_out = b * output_intensity

rgb_out = np.stack([R_out, G_out, B_out], axis=2)

# normalization
a = np.percentile(rgb_out, 0.01)
b = np.percentile(rgb_out, 99.99)
rgb_out = np.clip((rgb_out - a) / (b - a), 0, 1) * 255
Image.fromarray(rgb_out.astype(np.uint8)).save('icam06_result.jpg')
print('Saved icam06_result.jpg')