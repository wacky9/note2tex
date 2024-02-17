import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
import cv2

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

#Returns array of all bounding boxes for line
def get_line_bounding_boxes(Im):

    #Get regions and sort by their x centroid values
    from skimage.measure import label, regionprops
    labels = label(Im,background=1)
    regions = regionprops(labels)
    regions.sort(key=lambda x: x.centroid[1], reverse=False)

    #Get all the bounding boxes
    x_threshold = 10
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
            if region_next.centroid[1] - region_curr.centroid[1] < 10:
                minrow_1, mincol_1, maxrow_1, maxcol_1 = region_curr.bbox
                minrow_2, mincol_2, maxrow_2, maxcol_2 = region_next.bbox
                bounding_boxes.append((min(minrow_1, minrow_2),
                                       min(mincol_1, mincol_2),
                                       max(maxrow_1, maxrow_2),
                                       max(maxcol_1, maxcol_2)))
                skip_current=True
            else:
                bounding_boxes.append(region_curr.bbox)
                #io.imshow(region.image, cmap='gray'); io.show()

    return bounding_boxes

#standardizes each image to a certain size and color scheme
def standardize(Im,size):
    Im = expand(Im,size)
    Im = Im.astype(np.uint8)*255
    Im = cv2.resize(Im,size,interpolation=cv2.INTER_CUBIC)
    return Im

#Add whitespace so all dimensions are at least size
def expand(Im,size):
    #row expand
    rowdiff = Im.shape[0]-size[0]
    if rowdiff<0:
        top = -rowdiff//2
        bottom = -rowdiff//2-rowdiff%2
        Im = np.pad(Im,pad_width=((top,bottom),(0,0)),mode='constant',constant_values=1)
    #column expand
    coldiff = Im.shape[1]-size[1]
    if coldiff<0:
        left = -coldiff//2
        right = -coldiff//2-coldiff%2
        Im = np.pad(Im,pad_width=((0,0),(left,right)),mode='constant',constant_values=1)
    return Im

def get_supersubscript_labling(boxes):
    starting_y = (boxes[0][0] + boxes[0][2] / 2)
    tolerance = 10
    labels = []
    for i in range(len(boxes)):
        curr_y = (boxes[i][0] + boxes[i][2] / 2)
        print(curr_y)
        if abs(starting_y - curr_y) > tolerance and curr_y > starting_y:
            labels.append("Super")
        elif abs(starting_y - curr_y) > tolerance and curr_y < starting_y:
            labels.append("Sub")
        else:
            labels.append("Normal")
    return labels


def main():
    line_num = 3
    test = io.imread('White_Data.png')
    bin = binarize(test)>THRESHOLD
    lines = segment_lines(bin)
    io.imshow(lines[line_num],cmap='gray'); io.show()
    boxes = get_line_bounding_boxes(lines[line_num])
    labels = get_supersubscript_labling(boxes)
    counter = 0
    for box in boxes:
        print(box)
        print(labels[counter])
        Im = standardize(lines[line_num][box[0]:box[2],box[1]:box[3]],SIZE)
        io.imshow(Im); io.show()
        counter += 1
    return 0


if __name__=="__main__": 
    main() 