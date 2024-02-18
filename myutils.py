import os
import datetime
import numpy as np
import torch
import torch.nn as nn

NUM_LABELS = 70

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

def get_label_list():
    li=[]
    with open('data/labels.list') as f:
        li=eval(f.read().strip())
    return li