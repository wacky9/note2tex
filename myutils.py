import os
import datetime
import numpy as np
import torch
import torch.nn as nn

import os
import datetime
import numpy as np
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
