from Line.line_detection import preprocess
from Detect.detection import *
from Display.latex_conv import *
import skimage
from skimage import io
import torchvision
import torchvision.transforms.v2.functional as functional
import numpy as np
import sys
import csv
import Display.latex_conv as latex_conv
from constants import *

def transform_frame(frame):
    #rescale frames to [0,1]
    frame = functional.to_image(frame)
    frame = functional.to_dtype(frame,torch.float32, scale=True)
    #invert frame to match MNIST:
    frame = 1-frame
    #Turn add batch dimension of 1 
    frame = frame.unsqueeze(0)
    #frame = functional.normalize(frame,[0.1307],[0.3081])
    return frame

def main(image_path):
    IMAGE = io.imread(image_path)
    #frames is an array of matrices. Each element is a line from the OG file and each matrix corresponds to a frame
    frames = preprocess(IMAGE)
    preds = []
    TEST_LINE = 1
    for i in range(len(frames[TEST_LINE])):
        frame = frames[TEST_LINE][i]
        frame = transform_frame(frame)
        frame = frame.squeeze(0)
        frame = frame.squeeze(0)
        frame = frame.numpy()
        frame = frame*255//1
        frame = frame.astype('uint8')
        io.imsave(OUTPUT_PATH +'/frame' +str(i)+'B.png',frame,check_contrast=False)
    for line in frames:
        pred_line = []
        model = load_model()
        for frame in line:
            frame = transform_frame(frame)
            predicted_label = detect(model,frame)
            pred_line.append(predicted_label)
        preds.append(pred_line.copy())
        pred_line.clear()

    preds = list(filter(None, preds))

    with open('support/intermediate.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(preds)
    performLatexGen() 


if __name__ == '__main__':
    main(sys.argv[1])
