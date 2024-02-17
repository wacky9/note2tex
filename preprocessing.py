import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import scipy

#Determines how to binarize data
#Assumes images is between [0,1] and grayscale
threshold = 0.95

#Potential problem: need to detect if image is [0,1] or [0,255]
def binarize(Im,rgb=True):
    grayIm = skimage.color.rgb2gray(Im[:,:,0:3])
    return grayIm

#Detects information about dividers
def divider_info(Im):
    first_divider = -1
    divider_height = -1
    spacing = -1
    #detection method: sum each row. Rows that are mostly pixels are dividers
    density = (Im.shape[1]-np.sum(Im,axis=1))/Im.shape[1]
    divider_density = np.max(density)
    print(divider_density)
    return (first_divider,divider_height,spacing)

#Detects horizontal lines and returns an array of lines
def detect_lines(Im):
    first_divider, divider_thickness,spacing = divider_info(Im)
    return 1

def main():
    test = io.imread('Test_Data.png')
    bin = binarize(test)>threshold
    detect_lines(bin)
    io.imshow(bin); io.show()
    return 0

if __name__=="__main__": 
    main() 