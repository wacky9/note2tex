from preprocessing import *
from detection import *
from latex_conv import *
import skimage
from skimage import io
from torchvision.transforms.v2.functional import normalize
import numpy as np
import sys
import csv
import latex_conv

def transform_frame(frame):
    frame = torch.from_numpy(frame).float()
    #rescale frames to [0,1]
    if torch.max(frame)>1:
        frame = frame/255.0
    #invert frame to match MNIST:
    frame = 1-frame
    #Turn 2D frame into 4D tensor with 
    frame = frame.unsqueeze(0)
    frame = frame.unsqueeze(0)
    frame = normalize(frame,[0.5],[1])
    return frame

def main(image_path):
    IMAGE = io.imread(image_path)
    #frames is an array of matrices. Each element is a line from the OG file and each matrix corresponds to a frame
    frames = preprocess(IMAGE)
    preds = []
    for line in frames:
        pred_line = []
        model = load_model()
        for frame in line:
            frame = transform_frame(frame)
            predicted_label = detect(model,frame)
            pred_line.append(predicted_label)
        preds.append(pred_line.copy())
        pred_line.clear()
    print(preds)

    preds = list(filter(None, preds))

    with open('support/intermediate.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(preds)
    performLatexGen()


if __name__ == '__main__':
    main(sys.argv[1])
