from constants import *
from ruled_detection import detect

def preprocess(image, MODE):
    frames = []
    if MODE == RULED_MODE:
        frames = detect(image)
    return frames