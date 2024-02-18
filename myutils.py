import os
import datetime
import numpy as np
from skimage import io
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torchvision import transforms

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.datasets import mnist

NUM_LABELS = 70     # number of unique classes
NUM_CNNS=3      # count of CNNs in the ensemble

# read all imgs in dir.
def images_to_numpy_array(folder_path):
    image_arrays = []
    labels = []

    # Iterate through each folder (assuming each folder represents a label)
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if not os.path.isdir(label_path):
            continue

        # Read images from each folder
        for filename in os.listdir(label_path):
            image_path = os.path.join(label_path, filename)
            if os.path.isfile(image_path) and filename.endswith(('.png')):
                # Open image using scikit-image
                image = io.imread(image_path, as_gray=True)

                # Convert image to numpy array
                image_array = np.array(image)

                # Check if image values are in the range of 0-255
                if np.max(image_array) > 1.0:
                    # Normalize image values to the range of 0-1
                    image_array = image_array / 255.0
                            

                # Append to list
                image_arrays.append(image_array)
                labels.append(label.replace('_', ''))
                # print(f'Image label: {label.replace("_", "")}, image_file: {image_path}')

    # Convert lists to numpy arrays
    image_arrays = np.array(image_arrays)
    labels = np.array(labels)

    return image_arrays, labels



# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(32*32, 128)
        self.fc2 = nn.Linear(128, NUM_LABELS)

    def forward(self, x):
        x = x.view(-1, 32*32)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def create_net():
    return Net()

# build a single model
def create_TF_CNN():
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size = 3, activation='relu', input_shape = (32, 32, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, kernel_size = 5, strides=2, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, kernel_size = 4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(70, activation='softmax'))


    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def save_ensemble_to_files(ensemble):
    for ind, mod in enumerate(ensemble):
        mod.save_weights(f'tf_models/model{ind}.ckpkt')
