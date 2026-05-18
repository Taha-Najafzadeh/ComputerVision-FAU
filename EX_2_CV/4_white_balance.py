import numpy as np
import importlib
from PIL import Image

luminosity = importlib.import_module('3_luminosity')
gamma_corrected_rgb = luminosity.gamma_corrected_rgb
mi = gamma_corrected_rgb.mean()    # image mean

# Red,green,blue channel mean
mr = gamma_corrected_rgb[:, :, 0].mean()     
mg = gamma_corrected_rgb[:, :, 1].mean()     
mb = gamma_corrected_rgb[:, :, 2].mean()      

# multiply values of channel c by mi / mc
rgb_wb = gamma_corrected_rgb.copy()
rgb_wb[:, :, 0] = gamma_corrected_rgb[:, :, 0] * (mi / mr)
rgb_wb[:, :, 1] = gamma_corrected_rgb[:, :, 1] * (mi / mg)
rgb_wb[:, :, 2] = gamma_corrected_rgb[:, :, 2] * (mi / mb)

print(f'Scale factors R: {mi/mr:.3f}  G: {mi/mg:.3f}  B: {mi/mb:.3f}')

if __name__ == "__main__":
    # Now again normalize for preview
    a = np.percentile(rgb_wb, 0.01)
    b = np.percentile(rgb_wb, 99.99)
    normalized = np.clip((rgb_wb - a) / (b - a), 0, 1)

    del gamma_corrected_rgb, mr, mg, mb, mi, a, b

    preview = (normalized * 255).astype(np.uint8)
    Image.fromarray(preview).save('white_balanced.jpg')
    print('Saved white_balanced.jpg')

    del normalized, preview