#histogram method:
#Threshold to only capture black pixels
#Count number of pixels per line and put line markers at local minima

import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from skimage.measure import label, regionprops
import cv2
from constants import *
from utilities import *
from component_extraction import extract
THRESHOLD = 0.95
COMPRESSION_FACTOR = 5
TOLERANCE = 0.8
def segment_lines(bin):
    density = np.sum(bin,axis=1)/bin.shape[1]
    # Compress density by COMPRESSION_FACTOR
    avg_density = np.pad(density, (0, COMPRESSION_FACTOR - density.shape[0] % COMPRESSION_FACTOR), 'edge')
    np.mean(avg_density.reshape(-1,COMPRESSION_FACTOR), axis=1)
    #The maximum density is going to be a divider
    divider_density = np.max(avg_density)
    #Detect which lines are close to the max density, within a tolerance
    compare = np.full((avg_density.shape),divider_density)
    dividers = np.isclose(avg_density,compare,rtol=TOLERANCE)
    #eliminate adjacent dividers
    for i in range(len(dividers)-1):
        if dividers[i+1]-dividers[i] != 1:
            dividers[i] = 0
    #Get indices of dividers and use to find info
    divider_indices = np.argwhere(dividers)
    divider_indices = divider_indices[:,0]
    print("hello")

def detect(Image):
    IM = Image
    bin = binarize(IM)>THRESHOLD
    lines = segment_lines(bin)
    frames = extract(lines)
    return frames