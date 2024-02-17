import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import cv2
import scipy

#Determines how to binarize data
#Assumes images is between [0,1] and grayscale
THRESHOLD = 0.95
TOLERANCE = 0.05
SIZE = (32,32)
#Potential problem: need to detect if image is [0,1] or [0,255]
def binarize(Im):
    grayIm = skimage.color.rgb2gray(Im[:,:,0:3])
    return grayIm

#Detects information about dividers
def divider_info(Im):
    #detection method: sum each row. Rows that are mostly pixels are dividers
    density = (Im.shape[1]-np.sum(Im,axis=1))/Im.shape[1]
    #The maximum density is going to be a divider
    divider_density = np.max(density)
    #Detect which lines are close to the max density, within a tolerance
    compare = np.full((density.shape),divider_density)
    dividers = np.isclose(density,compare,rtol=TOLERANCE)
    #Get indices of dividers and use to find info
    divider_indices = np.argwhere(dividers)
    divider_indices = divider_indices[:,0]
    divider_indices[1::2]+=2
    return divider_indices

#Segments horizontal lines and returns an array of lines
def segment_lines(Im):
    indices = divider_info(Im)
    segments = np.split(Im,indices)
    lines = segments[::2]
    return lines

def CC(Im):
    from skimage.measure import label, regionprops
    labels = label(Im,background=1)
    regions = regionprops(labels)
    return regions

#standardizes each image to a certain size and color scheme
def standardize(Im,size):
    Im = Im.astype(np.uint8)*255
    Im = cv2.resize(Im,size,interpolation=cv2.INTER_CUBIC)
    return Im


def main():
    test = io.imread('White_Data.png')
    bin = binarize(test)>THRESHOLD
    lines = segment_lines(bin)
    io.imshow(lines[1]); io.show()
    regions = CC(lines[1])
    standard = standardize(regions[0].image,SIZE)
    io.imshow(standard); io.show()
    return 0

if __name__=="__main__": 
    main() 