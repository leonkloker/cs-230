import numpy as np
import os
from PIL import Image

os.mkdir('Train')
os.mkdir('Validation')
os.mkdir('Test')

# initialize and sort data
rgb = os.listdir('Dataset3/RGB/')
elevation = os.listdir('Dataset3/Elevation/')
rgb.sort(key = lambda x : int(x[4:-4]))
elevation.sort(key = lambda x : int(x[2:-4]))

# process rgb data
i = 0; c = 1
for image in rgb:
    with Image.open('Dataset3/RGB/' + image) as im:
        width, height = im.size

        for col in range(0, width-255, 256):
            for row in range(0, height-255, 256):
                if i < 25000:
                    crop = im.crop((col, row, col + 256, row + 256))
                    if c == 24:
                        save_to = os.path.join('Validation', "rgb_{}.png")
                    elif c == 25:
                        save_to = os.path.join('Test', "rgb_{}.png")
                        c = 0
                    else:
                        save_to = os.path.join('Train', "rgb_{}.png")
                    crop.save(save_to.format(i))
                    i += 1; c += 1

# process elevation data
scale_factors = np.load('scale_factors.npy')
i = 0; c = 1; s = 0
for image in elevation:
    with Image.open('Dataset3/Elevation/' + image) as im:
        width, height = im.size

        for col in range(0, width-255, 256):
            for row in range(0, height-255, 256):
                if i < 25000:
                    crop = im.crop((col, row, col + 256, row + 256))
                    data = np.asarray(crop)

                    min_i = np.amin(data)
                    scaled = (data - min_i) * (scale_factors[s])
                    if c == 24:
                        save_to = os.path.join('Validation', "elevation_{}.npy")
                    elif c == 25:
                        save_to = os.path.join('Test', "elevation_{}.npy")
                        c = 0
                    else:
                        save_to = os.path.join('Train', "elevation_{}.npy")
                    np.save(save_to.format(i), scaled)
                    i += 1; c += 1
    s += 1
