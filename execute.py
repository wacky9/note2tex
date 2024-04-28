from preprocessing import *
from recognition import *
from latex_conv import *
import skimage
from skimage import io
import numpy as np
import sys
import csv
import latex_conv

def main(image_path):
    IMAGE = io.imread(image_path)
    frames = preprocess(IMAGE)

    labels = sorted(get_label_list())
    preds = []
    

    for line in frames:
        pred_line = []
        for frame in line:
            model = init_model()
            predicted_label, confidence = predict_label(frame, model, labels)
            pred_line.append(predicted_label)
        preds.append(pred_line.copy())
        pred_line.clear()

    preds = list(filter(None, preds))

    with open('support\intermediate.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(preds)
    performLatexGen()

if __name__ == '__main__':
    main(sys.argv[1])
