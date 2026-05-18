import numpy as np
import rawpy
from scipy.ndimage import convolve
from PIL import Image

def process_raw(raw_path, output_path):

    raw = rawpy.imread(raw_path)
    X = np.array(raw.raw_image_visible, dtype=np.float32)

    #---------------------demosaicing-------------------------------
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
    del R, G, B, Mr, Mb, Mg, raw 
    print('Demosaicing done. Shape:', rgb.shape, '  dtype:', rgb.dtype)
    
    # --------------- Gamma Correction  ----------------------
    a = np.percentile(rgb, 0.01)
    b = np.percentile(rgb, 99.99)
    normalized = np.clip((rgb - a) / (b - a), 0, 1)
    corrected = normalized ** 0.45   

    # ---------------------- white balance --------------------------
    mi = corrected.mean()      # image mean

    # Red,green,blue channel mean
    mr = corrected[:, :, 0].mean()     
    mg = corrected[:, :, 1].mean()     
    mb = corrected[:, :, 2].mean()      

    # multiply values of channel c by mi / mc
    rgb_wb = corrected.copy()
    rgb_wb[:, :, 0] = corrected[:, :, 0] * (mi / mr)
    rgb_wb[:, :, 1] = corrected[:, :, 1] * (mi / mg)
    rgb_wb[:, :, 2] = corrected[:, :, 2] * (mi / mb)

    print(f'Scale factors R: {mi/mr:.3f}  G: {mi/mg:.3f}  B: {mi/mb:.3f}')

    # save img
    out = np.clip(rgb_wb * 255, 0, 255).astype(np.uint8)
    Image.fromarray(out).save(output_path, quality=100)
    print(f'Saved {output_path}')

if __name__ == "__main__":
    process_raw('exercise_2_data/02/IMG_4782.CR3', '8_result.jpg')