import torch
from torch import nn
import torch.nn.functional as F
from Detect.model_conv import compute_output_size

IN_CHANNEL_1 = 1
FILTER_1 = 32
FILTER_2 = 64
FILTER_3 = 128
FILTER_4 = 256
FILTER_5 = 512
KERNEL_SIZE = 3
class LineCounter(nn.module):

    def __init__(self):
        super().__init__()
        #a layers downsample via stride, b layers do not
        self.conv1a = nn.Conv2d(IN_CHANNEL_1,FILTER_1,KERNEL_SIZE,stride=2)
        self.conv1b = nn.Conv2d(FILTER_1,FILTER_1,KERNEL_SIZE)
        self.conv2a = nn.Conv2d(FILTER_1,FILTER_2,KERNEL_SIZE,stride=2)
        self.conv2b = nn.Conv2d(FILTER_1,FILTER_2,KERNEL_SIZE)
        self.conv3a = nn.Conv2d(FILTER_2,FILTER_2,KERNEL_SIZE,stride=2)
        self.conv3b = nn.Conv2d(FILTER_2,FILTER_3,KERNEL_SIZE)
        self.conv4a = nn.Conv2d(FILTER_3,FILTER_3,KERNEL_SIZE,stride=2)
        self.conv4b = nn.Conv2d(FILTER_3,FILTER_4,KERNEL_SIZE)
        self.conv5a = nn.Conv2d(FILTER_4,FILTER_4,KERNEL_SIZE,stride=2)
        self.conv5b = nn.Conv2d(FILTER_4,FILTER_5,KERNEL_SIZE)

        self.vGru = nn.GRU(FILTER_5,1,num_layers=32,bidirectional=True)
        self.hGru = nn.GRU(FILTER_5,1,num_layers=32,bidirectional=True)



    def forward(self,x):
        #Encoder
        x = F.relu(F.batch_norm2d(self.conv1a(x)))
        x = F.relu(F.batch_norm2d(self.conv1b(x)))
        x = F.relu(F.batch_norm2d(self.conv2a(x)))
        x = F.relu(F.batch_norm2d(self.conv2b(x)))
        x = F.relu(F.batch_norm2d(self.conv3a(x)))
        x = F.relu(F.batch_norm2d(self.conv3b(x)))
        x = F.relu(F.batch_norm2d(self.conv4a(x)))
        x = F.relu(F.batch_norm2d(self.conv4b(x)))
        x = F.relu(F.batch_norm2d(self.conv5a(x)))
        x = F.relu(F.batch_norm2d(self.conv5b(x)))

        #Counter
        x = F.relu(F.batch_norm2d(self.conv1b(x)))
        x = F.hard_sigmoid(self.hGru(x))
        x = F.relu(F.batch_norm2d(self.conv1b(x)))
        x = F.hard_sigmoid(self.vGru(x))
        x = self.conv1b(x)
        #Decoder

        return x
