#more advanced convolution neural network
#Replace max pooling with strided convolution
# Shift more learning from internal layers to convolutional layers
import torch
from torch import nn
import torch.nn.functional as F

INTERNAL = 128
FILTER_1 = 3
FILTER_2 = 6
FILTER_3 = 9
IN_CHANNEL_1 = 1
KERNEL_SIZE = 3
class CNN2(nn.Module):
    # classes = # of classses. size = number of img pixels

    def __init__(self,classes,size):
        self.size = size
        super().__init__()
        #Should have a 6x6 img left over after applying convolutions and pooling
        #self.input = nn.Linear(2*FILTERS*6*6,INTERNAL)
        #dimensions of a single filter
        filter_1 = compute_output_size((28,28),(KERNEL_SIZE,KERNEL_SIZE),stride=2, operation="conv")
        filter_2 = compute_output_size(filter_1,(KERNEL_SIZE,KERNEL_SIZE), operation="conv")
        filter_3 = compute_output_size(filter_2,(KERNEL_SIZE,KERNEL_SIZE), operation="conv")
        print(filter_1)
        print(filter_2)
        print(filter_3)
        #dimension of a single filter times number of filters
        input_size = filter_3[0] * filter_3[1] * 2 * FILTER_3
        print(input_size)
        self.input = nn.Linear(input_size,INTERNAL)
        self.internal = nn.Linear(INTERNAL,INTERNAL)
        # in_channels = 1 since grayscale
        # out_channels = FILTERS (number of different filters to use)
        self.conv1 = nn.Conv2d(IN_CHANNEL_1,FILTER_1,KERNEL_SIZE)
        self.conv2 = nn.Conv2d(FILTER_1,FILTER_2,KERNEL_SIZE)
        self.conv3 = nn.Conv2d(FILTER_2,FILTER_3,KERNEL_SIZE)
        self.out = nn.Linear(INTERNAL,classes)
        self.soft = nn.Softmax(dim=1)
    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x,1)
        #input layer
        x = F.relu(self.input(x))
        #internal layer
        x = F.relu(self.internal(x))
        x = self.soft(self.out(x))
        return x
    
def compute_output_size(input_size, kernel_size, stride=1, padding=0, operation="conv"):
    """
    Compute the output size of a matrix after a convolution or max pooling operation.

    Parameters:
        input_size (tuple): (height, width) of the input matrix.
        kernel_size (tuple): (kernel_height, kernel_width) of the filter.
        stride (int): Stride value.
        padding (int, optional): Padding applied to the input. Default is 0.
        operation (str): Either "conv" for convolution or "pool" for max pooling.

    Returns:
        tuple: (output_height, output_width)
    """
    H, W = input_size
    K_h, K_w = kernel_size
    S = stride
    P = padding

    if operation == "conv":
        H_out = ((H - (K_h-1) -1 + 2 * P) // S) + 1
        W_out = ((W - (K_w-1) -1 + 2 * P) // S) + 1
    elif operation == "pool":
        H_out = ((H - (K_h-1)-1) // S) + 1
        W_out = ((W - (K_w-1)-1) // S) + 1
    else:
        raise ValueError("Operation must be 'conv' or 'pool'")

    return H_out, W_out

   
