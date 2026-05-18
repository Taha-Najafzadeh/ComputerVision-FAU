import numpy as np
import rawpy
from scipy.ndimage import convolve
from PIL import Image

# Load raw file
raw = rawpy.imread('exercise_2_data/02/IMG_4782.CR3')
X = np.array(raw.raw_image_visible).astype(np.float32)
# print(X[:8,:8])

# Build masks with Bayer pattern(R,G,G,B) or (B,G,G,R)
Mr = np.zeros_like(X)
Mg = np.zeros_like(X)
Mb = np.zeros_like(X)

Mr[0::2, 0::2] = 1
Mg[1::2, 0::2] = 1
Mg[0::2, 1::2] = 1
Mb[1::2, 1::2] = 1

# Convolution kernel
K = np.ones((3, 3))

# The convolution formula 
R = convolve(Mr * X, K) / convolve(Mr, K)
G = convolve(Mg * X, K) / convolve(Mg, K)
B = convolve(Mb * X, K) / convolve(Mb, K)

# Stack into RGB image
rgb = np.stack([R, G, B], axis=2)

#to free space because of killed(memory full) error occurs some time
del R, G, B, Mr, Mb, Mg, X, raw 

if __name__ == "__main__":
    print('Done. Shape:', rgb.shape, '  dtype:', rgb.dtype)
    print('Value range: min=', rgb.min(), ' max=', rgb.max())

    preview = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
    Image.fromarray(preview.astype(np.uint8)).save('demosaiced.jpg')

    del preview