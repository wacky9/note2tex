# extracts components from a line image
from skimage.measure import label, regionprops
import numpy as np
import cv2
from constants import *

Y_OVERLAP = 0.6


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


#Determines the % of overlap between A & B, with reference to A
#That is, if 75% of A overlaps with B, the value will be 0.75.
def overlap(boxA,boxB,H=True):
    size = 0
    if H:
        diffOne = boxB[3]-boxA[1]
        diffTwo = boxA[3]-boxB[1]
        size = boxA[3]-boxA[1]
    else:
        diffOne = boxB[2]-boxA[0]
        diffTwo = boxA[2]-boxB[0]
        size = boxA[2]-boxA[0]
    return max(0,min(diffOne,diffTwo)/size)

def symmetric_overlap(boxA,boxB,H=True):
    K = overlap(boxA,boxB,H)
    L = overlap(boxB,boxA,H)
    return max((K,L))


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

def extract(lines):
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