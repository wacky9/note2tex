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

def main(image_path):
    IMAGE = io.imread(image_path)
    frames = preprocess(IMAGE)

    preds = []

    for line in frames:
        pred_line = []
        model = load_model()
        for frame in line:
            frame = torch.from_numpy(frame).float()
            #channels = 1
            frame = frame.unsqueeze(0)
            #batch_size = 1
            frame = frame.unsqueeze(0)
            frame = normalize(frame,[0.5],[1])
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
