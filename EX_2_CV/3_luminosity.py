import numpy as np
from PIL import Image
import importlib

# to take rgb variable from 2nd excercise script 
demosaicing = importlib.import_module("2_demosaicing_algo")
rgb = demosaicing.rgb

# Normalizing using percentile as manual suggested
a = np.percentile(rgb, 0.01)
b = np.percentile(rgb, 99.99)
rgb_normalized = (rgb - a) / (b - a)

# as suggested in manual to make negativce value 0 and >1 to 1
rgb_normalized[rgb_normalized<0] = 0
rgb_normalized[rgb_normalized>1] = 1

# gamma correction
gamma = 0.3
gamma_corrected_rgb = rgb_normalized ** gamma

# only for preview purpose we are scaling it to (0,255)
gamma_corrected_rgb_preview = gamma_corrected_rgb * 255

# invert normalization back to original range
gamma_corrected_rgb = gamma_corrected_rgb * (b - a) + a

del rgb

# testing other curve: log_curve
log_curve = np.log1p(rgb_normalized * 9) / np.log(10)
log_curve_preview = log_curve * 255

if __name__ == "__main__":
    # only for preview purpose
    preview = gamma_corrected_rgb_preview.astype(np.uint8)
    Image.fromarray(preview).save('gamma_0.3.jpg')
    print('saved gamma_0.3.jpg')
    del gamma_corrected_rgb_preview, preview

    # only for preview purpose
    preview = log_curve_preview.astype(np.uint8)
    Image.fromarray(preview).save('log_curve.jpg')
    print('saved log_curve.3.jpg')

    del log_curve, log_curve_preview, preview
