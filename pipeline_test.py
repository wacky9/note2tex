from preprocessing import *
import skimage
from skimage import io
import numpy as np


image_path = 'pipelineTD/TD0_1.png'

IMAGE = io.imread(image_path)
frames = preprocess(IMAGE)
a = 0
for frame in frames:
    for i in range(0,len(frame),10):
        io.imsave('img_output/TD0_' + str(a) + '.png',frame[i])
        a += 1