import numpy as np
import rawpy
import matplotlib.pyplot as plt

image_folder_path = "exercise_2_data/05"

# Exposure times in seconds
exposure_times = [1/10, 1/20, 1/40, 1/80, 1/160, 1/320]
files = ['IMG_3044.CR3', 'IMG_3045.CR3', 'IMG_3046.CR3',
         'IMG_3047.CR3', 'IMG_3048.CR3', 'IMG_3049.CR3']

avg_values = []

for file in files:
    raw = rawpy.imread(image_folder_path+"/"+file)
    array = np.array(raw.raw_image_visible)
    avg_values.append(array.mean())
    print(f'{file}    mean = {avg_values[-1]:.1f}')

plt.plot(exposure_times, avg_values, 'o-')
plt.xlabel('Exposure time (seconds)')
plt.ylabel('Average pixel value')
plt.savefig('graph.png')
plt.show()
print('Saved graph.png')