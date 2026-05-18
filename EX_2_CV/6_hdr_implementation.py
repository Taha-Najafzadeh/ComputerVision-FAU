import numpy as np
import rawpy
from scipy.ndimage import convolve
from PIL import Image

image_folder_path = "exercise_2_data/06"
files = ["00.CR3","01.CR3","02.CR3","03.CR3","04.CR3","05.CR3","06.CR3","07.CR3","08.CR3","09.CR3","10.CR3"]

# longest exposure img as a base
raw = rawpy.imread(image_folder_path+"/"+files[0])
h = np.array(raw.raw_image_visible).astype(np.float32)

print(f'Loaded {files[0]}, shape {h.shape}, max={h.max():.0f}')

# Loop replace pixels
for idx in range(1, len(files)):
    raw = rawpy.imread(image_folder_path+"/"+files[idx])
    i = np.array(raw.raw_image_visible).astype(np.float32)

    # Each image has half the exposure multiply by 2,4,8,... to bring to same scale
    exposure_ratio = 2 ** idx
    i_scaled = i * exposure_ratio

    # Replace pixels in h that are overexposed 
    t = 0.8 * h.max()
    h[h > t] = i_scaled[h > t]

    print(f'Loaded {files[idx]}, ratio={exposure_ratio}, replaced pixels above {t:.0f}')

# only for preview
rgb_norm = (h - h.min()) / (h.max() - h.min()) * 255
Image.fromarray(rgb_norm.astype(np.uint8)).save('hdr_result.jpg')
print('Saved hdr_result.jpg')

#---------------------demosaicing-------------------------------
# Build masks with Bayer pattern(R,G,G,B) or (B,G,G,R)
Mr = np.zeros_like(h)
Mg = np.zeros_like(h)
Mb = np.zeros_like(h)

Mr[0::2, 0::2] = 1
Mg[1::2, 0::2] = 1
Mg[0::2, 1::2] = 1
Mb[1::2, 1::2] = 1

# Convolution kernel
K = np.ones((3, 3))

# The convolution formula 
R = convolve(Mr * h, K) / convolve(Mr, K)
G = convolve(Mg * h, K) / convolve(Mg, K)
B = convolve(Mb * h, K) / convolve(Mb, K)

# Stack into RGB image
rgb = np.stack([R, G, B], axis=2)

#to free space because of killed(memory full) error occurs some time
del R, G, B, Mr, Mb, Mg, raw 

print('Done. Shape:', rgb.shape, '  dtype:', rgb.dtype)
print('Value range: min=', rgb.min(), ' max=', rgb.max())

preview = (rgb - rgb.min()) / (rgb.max() - rgb.min()) * 255
Image.fromarray(preview.astype(np.uint8)).save('hdr_demosaiced.jpg')

del preview

# ---------------------- white balance --------------------------

mi = rgb.mean()   # image mean

# Red,green,blue channel mean
mr = rgb[:, :, 0].mean()     
mg = rgb[:, :, 1].mean()     
mb = rgb[:, :, 2].mean()      

# multiply values of channel c by mi / mc
rgb_wb = rgb.copy()
rgb_wb[:, :, 0] = rgb[:, :, 0] * (mi / mr)
rgb_wb[:, :, 1] = rgb[:, :, 1] * (mi / mg)
rgb_wb[:, :, 2] = rgb[:, :, 2] * (mi / mb)

print(f'Scale factors R: {mi/mr:.3f}  G: {mi/mg:.3f}  B: {mi/mb:.3f}')


# Now again normalize for preview
a = np.percentile(rgb_wb, 0.01)
b = np.percentile(rgb_wb, 99.99)
normalized = np.clip((rgb_wb - a) / (b - a), 0, 1)

del mg, mb, mi, a, b

preview = np.clip(normalized * 255, 0, 255).astype(np.uint8)
Image.fromarray(preview).save('hdr_white_balanced.jpg')
print('Saved hdr_white_balanced.jpg')

del normalized, preview

# ------------------ Log scale to compress HDR range (human vision) -------------------------------
rgb_log = np.log(rgb_wb + 1)  # +1 to avoid log(0)

# ----------------- Normalize to [0, 255] -------------------------
rgb_norm = (rgb_log - rgb_log.min()) / (rgb_log.max() - rgb_log.min()) * 255
Image.fromarray(rgb_norm.astype(np.uint8)).save('hdr_result_final.jpg')
print('Saved hdr_result_final.jpg')
