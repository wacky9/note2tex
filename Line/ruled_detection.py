# Detects lines assuming user submits an image of ruled paper and writes between the lines
# Finds the lines on the paper and uses them to isolate Lines of math


import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from skimage.measure import label, regionprops
from collections import namedtuple
import cv2
from constants import *

#Determines how to binarize data
#Assumes images is between [0,1] and grayscale
THRESHOLD = 0.95
TOLERANCE = 0.05
SIZE = (28,28)
Y_OVERLAP = 0.6
MODE = 'TEST'


#Potential problem: need to detect if image is [0,1] or [0,255]
def binarize(Im):
    grayIm = skimage.color.rgb2gray(Im[:,:,0:3])
    if np.max(grayIm)>1:
        grayIm = grayIm/255
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

def get_components(Im):
    labels = label(Im,background=1)
    regions = regionprops(labels)
    regions.sort(key=lambda x: x.centroid[1], reverse=False)
    return regions

#Returns array of all bounding boxes for line
def get_line_bounding_boxes(regions):
    #Get all the bounding boxes
    bounding_boxes = []
    skip_current=False
    for i in range(0,len(regions)):
        if skip_current:
            skip_current=False
            continue
        if i == len(regions)-1:
            bounding_boxes.append(regions[i].bbox)
        else:
            region_curr = regions[i]
            region_next = regions[i+1]
            #If two regions are overtop one another
            if overlap(region_curr.bbox,region_next.bbox)>Y_OVERLAP:
                bounding_boxes.append(combine_bounding_boxes(region_curr.bbox,region_next.bbox))
                skip_current=True
            else:
                bounding_boxes.append(region_curr.bbox)
    return bounding_boxes

def get_boxes_list(regions):
    boxes = []
    for i in range(len(regions)):
        boxes.append(regions[i].bbox)
    return boxes

#Combine two small bounding boxes into a big one
def combine_bounding_boxes(A,B):
    minrow_1, mincol_1, maxrow_1, maxcol_1 = A
    minrow_2, mincol_2, maxrow_2, maxcol_2 = B
    return ((min(minrow_1, minrow_2),
             min(mincol_1, mincol_2),
             max(maxrow_1, maxrow_2),
             max(maxcol_1, maxcol_2)))    

#standardizes each image to a certain size and color scheme
def standardize(Im,size):
    Im = expand(Im,size)
    Im = Im.astype(np.uint8)*255
    Im = cv2.resize(Im,size,interpolation=cv2.INTER_CUBIC)
    return Im

#Add whitespace so all dimensions are at least size dimensions
def expand(Im,size):
    #row expand
    rowdiff = Im.shape[0]-size[0]
    if rowdiff<0:
        top = -rowdiff//2
        bottom = -rowdiff//2+rowdiff%2
        Im = np.pad(Im,pad_width=((top,bottom),(0,0)),mode='constant',constant_values=1)
    #column expand
    coldiff = Im.shape[1]-size[1]
    if coldiff<0:
        left = -coldiff//2
        right = -coldiff//2+coldiff%2
        Im = np.pad(Im,pad_width=((0,0),(left,right)),mode='constant',constant_values=1)
    return Im


def getIm(box,line):
    return standardize(line[box[0]:box[2],box[1]:box[3]],SIZE)

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
    frames = []
    for line_num in range(len(lines)):
        line = []
        components = get_components(lines[line_num])
        boxes = get_line_bounding_boxes(components)
        for box in boxes:
            Im = standardize(lines[line_num][box[0]:box[2],box[1]:box[3]],SIZE)
            line.append(Im)
        frames.append(line.copy())
        line.clear()
    debug(lines,frames)
    return frames