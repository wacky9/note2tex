import skimage
from skimage import io
import numpy as np

#Potential problem: need to detect if image is [0,1] or [0,255]
def binarize(Im):
    grayIm = skimage.color.rgb2gray(Im[:,:,0:3])
    if np.max(grayIm)>1:
        grayIm = grayIm/255
    return grayIm