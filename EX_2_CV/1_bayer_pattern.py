import numpy as np
array = np.load('exercise_2_data/01/IMG_9939.npy')
print('Shape:', array.shape)
print(array[4008:,3002:3010])

'''As discussed in class, human eyes are more sensitive to green light, so the bayer pattern here has 2 blocks of higest values which are 
green pixels, and the other 2 blocks are red and blue pixels. Assuming from the image data, observing the top-left corner which is orangish 
color, which is more closer to red, we can guess the pattern is RGGB. 
Observing the center portion of the last few rows, we can see the paper which has the color closer to bluish color, so we can confirm the 
pattern is RGGB.
print(array[4008:,3002:3010])'''