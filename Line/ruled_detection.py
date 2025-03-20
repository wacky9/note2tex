# Detects lines assuming user submits an image of ruled paper and writes between the lines
# Finds the lines on the paper and uses them to isolate Lines of math


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

#Determines how to binarize data
#Assumes images is between [0,1] and grayscale
THRESHOLD = 0.95
TOLERANCE = 0.05
SIZE = (28,28)

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
    #Prevents dividers from being caught in lines
    for i in range(len(divider_indices)-1):
        if divider_indices[i+1]-divider_indices[i] != 1:
            divider_indices[i] += 1
    return divider_indices

#Segments horizontal lines and returns an array of lines
def segment_lines(Im):
    indices = divider_info(Im)
    segments = np.split(Im,indices)
    #filters out all lines that are too small
    lines = list(filter(lambda x: x.shape[0]>10,segments))
    return lines

def get_boxes_list(regions):
    boxes = []
    for i in range(len(regions)):
        boxes.append(regions[i].bbox)
    return boxes

#Saves image of line and image of resulting frames
def debug(lines,frames):
    #debugging step
    TEST_LINE = 1
    io.imsave(OUTPUT_PATH + '/line_test.png',lines[TEST_LINE],check_contrast=False)
     #debugging step
    for i in range(len(frames[TEST_LINE])):
        io.imsave(OUTPUT_PATH + '/frame' +str(i)+'.png',frames[TEST_LINE][i],check_contrast=False)

def detect(Image):
    IM = Image
    bin = binarize(IM)>THRESHOLD
    lines = segment_lines(bin)
    frames = extract(lines)
    debug(lines,frames)
    return frames