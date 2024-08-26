import os
import torch
from torch import nn
import torch.nn.functional as F

internal = 128

class NN(nn.Module):
    # classes = # of classses. size = number of img pixels

    def __init__(self,classes,size):
        super().__init__()
        self.input = nn.Linear(size,128)
        self.internal = nn.Linear(128,128)
        self.dropout = nn.Dropout1d()
        self.out = nn.Linear(128,classes)
        self.soft = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = torch.flatten(x,1)
        #input layer
        x = self.input(x)
        #internal stuff
        x = self.dropout(x)
        x = F.relu(x)
        #internal layer
        x = self.internal(x)
        x = self.dropout(x)
        x = F.relu(x)
        #internal layer
        x = self.internal(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.out(x)
        x = self.soft(x)
        return x
