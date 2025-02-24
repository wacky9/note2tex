import skimage
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from skimage.measure import label, regionprops
from collections import namedtuple
import cv2

#Determines how to binarize data
#Assumes images is between [0,1] and grayscale
THRESHOLD = 0.95
TOLERANCE = 0.05
SIZE = (28,28)
Y_OVERLAP = 0.6
MODE = 'TEST'

class Group(Enum):
    FRAC = 1
    SQRT = 2
    COMB = 3

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

def create_groups(boxes):
    length = len(boxes)
    grouped = np.zeros((length,1))
    group_index = 1
    for i in range(length):
        if grouped[i] != 0:
            continue
        #box should expand as it encounters overlapping objects
        center_box = boxes[i]
        #expand left until stopping
        left = i-1
        end = False
        while not left<0 and grouped[left]==0 and not end:
           if symmetric_overlap(center_box,boxes[left]) > Y_OVERLAP: 
               grouped[left] = group_index
               grouped[i] = group_index
               center_box = combine_bounding_boxes(center_box,boxes[left])
               left -= 1
           else:
               end = True

        #expand right until stopping
        right = i+1
        end = False
        while not right>=length and grouped[right]== 0 and not end:
            if symmetric_overlap(center_box,boxes[right]) > Y_OVERLAP: 
               grouped[right] = group_index
               grouped[i] = group_index 
               center_box = combine_bounding_boxes(center_box,boxes[right])
               right += 1
            else:
               end = True
        group_index += 1
    return grouped

#Create a list of a list of tuples that represents groups
def fill_groups(groups,boxes):
    all_groups = []
    last = 0
    one_group = []
    for i in range(len(groups)):
        curr = groups[i]
        box = boxes[i]
        if curr == last and last != 0:
            one_group.append(box)
        elif curr != last and last == 0:
            one_group.append(box)
        elif curr != last and last != 0 and curr != 0:
            all_groups.append((last,one_group.copy()))
            one_group.clear()
            one_group.append(box)
        elif curr != last and curr == 0:
            all_groups.append((last,one_group.copy()))
            one_group.clear()
        last = curr
    if last != 0:
        all_groups.append((last,one_group.copy()))
    return all_groups

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


def classifyGroup(group):
    #case: group_size = 2 -> sqrt, =, i/j
    if len(group) == 2: return two_case(group)
    else:
        #if there is substantial vertical overlap between the first symbol and the others, SQRT
        first_sym = group[0]
        overlapping = True
        for i in range(1,len(group),1):
            if not symmetric_overlap(first_sym,group[i],H=False) > Y_OVERLAP:
                overlapping = False
        if overlapping: return Group.SQRT
    return Group.FRAC

def two_case(group):
    if symmetric_overlap(group[0],group[1],H=False)>Y_OVERLAP:
        return Group.SQRT
    else: 
        return Group.COMB

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

def get_supersubscript_labling(boxes):
    starting_y = (boxes[0][0] + boxes[0][2] / 2)
    tolerance = 10
    labels = []
    for i in range(len(boxes)):
        curr_y = (boxes[i][0] + boxes[i][2] / 2)
        if abs(starting_y - curr_y) > tolerance and curr_y > starting_y:
            labels.append("Super")
        elif abs(starting_y - curr_y) > tolerance and curr_y < starting_y:
            labels.append("Sub")
        else:
            labels.append("Normal")
    return labels


def getIm(box,line):
    return standardize(line[box[0]:box[2],box[1]:box[3]],SIZE)


def resolveGroups(groups,groups_indices,boxes,components,line):
    images = []
    for index,group in groups:
        label = classifyGroup(group)
        group_box = np.where(groups_indices==index)[0]
        if label == Group.COMB:
           big_box = combine_bounding_boxes(boxes[group_box[0]],boxes[group_box[1]])
           im = getIm(big_box,line)
           images.append(frame(im,"NONE"))

        if label == Group.SQRT:
            sqrt_box = boxes[group_box[0]]
            sqrt_image = 0
            #set 1-image to flip values
            for i in components:
                if sqrt_box == i.bbox: sqrt_image = 1-i.image
            images.append(frame(sqrt_image,"SQRT"))
            for k in range(1,len(group_box)):
                im = getIm(boxes[group_box[k]],line)
                images.append(frame(im,"SQRT"))

        if label == Group.FRAC:
            print("cry")
    return images

frame = namedtuple('frame','image label')

def preprocess(Image):
    if MODE == 'COMPLICATED':
        line_num = 0
        test = Image
        bin = binarize(test)>THRESHOLD
        lines = segment_lines(bin)
        #io.imshow(lines[line_num],cmap='gray'); io.show()
        components = get_components(lines[line_num])
        boxes = get_boxes_list(components)
        print(len(boxes))
        groups_indices = create_groups(boxes)
        groups_indices = groups_indices[:,0]
        print(len(groups_indices))
        groups = fill_groups(groups_indices,boxes)
        print(len(groups))
        frames = resolveGroups(groups,groups_indices,boxes,components,lines[line_num])
        return frames
    elif MODE == 'TEST':
        IM = Image
        bin = binarize(IM)>THRESHOLD
        lines = segment_lines(bin)
        #debugging step
        TEST_LINE = 0
        io.imsave('img_output/line_test.png',lines[TEST_LINE],check_contrast=False)
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
        #debugging step
        for i in range(len(frames[TEST_LINE])):
            io.imsave('img_output/frame' +str(i)+'.png',frames[TEST_LINE][i],check_contrast=False)
        return frames