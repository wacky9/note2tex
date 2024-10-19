import torch
from torch import nn
import torch.nn.functional as F

INTERNAL = 128
FILTERS = 3

class CNN(nn.Module):
    # classes = # of classses. size = number of img pixels

    def __init__(self,classes,size):
        self.size = size
        super().__init__()
        #Should have a 6x6 img left over after applying convolutions and pooling
        self.input = nn.Linear(2*FILTERS*6*6,INTERNAL)
        self.internal = nn.Linear(INTERNAL,INTERNAL)
        # in_channels = 1 since grayscale
        # out_channels = FILTERS (number of different filters to use)
        self.conv1 = nn.Conv2d(1,FILTERS,3)
        self.conv2 = nn.Conv2d(3,2*FILTERS,3)
        self.dropout = nn.Dropout1d()
        self.out = nn.Linear(INTERNAL,classes)
        self.soft = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x = torch.flatten(x,1)
        #input layer
        x = F.relu(self.input(x))
        #internal layer
        x = F.relu(self.internal(x))
        #internal layer
        x = F.relu(self.internal(x))
        x = self.soft(self.out(x))
        return x
