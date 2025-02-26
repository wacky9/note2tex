import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import torchvision as vision
from skimage import io
from data import CustomDataset
from data import SmallDataset
from model_nn_basic import NN
from model_conv import CNN
from matplotlib.pylab import plt
import time
import numpy as np
criterion = nn.CrossEntropyLoss()
LR = 0.001
BATCH = 64
EPOCH = 300
PATH = 'dataset2'
SIZE = 32
IMG_PATH = 'img_output/'
class_num = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Note to self: transforms are applied to each image every single batch. Dataset is the same size but each epoch it sees different data
# Augmentations: rotation, blur, resizing... others?
# Do I need larger images to safely do some of these transforms?
AUGMENT_TRANSFORM = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.RandomRotation(degrees=[-15,15]),
    v2.GaussianNoise(),
    v2.GaussianBlur(3,(0.1,1.0)),
    #Just arbitrary values to get the data between [-1,1]. Not sure about this: may be harmful
    v2.Normalize([0.1307],[0.3081])
])

PLAIN_TRANSFORM = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32)
])

def full_train(data, model,class_num):
    #test-train split
    train_data,test_data = torch.utils.data.random_split(data,[0.8,0.2])
    #To test: num_workers, pin_memory, prefetch_factor, persistent_workers
    train = DataLoader(train_data,batch_size = BATCH,num_workers=8, pin_memory=True, persistent_workers=True)
    normalize(train)
    test = DataLoader(test_data, num_workers=8, pin_memory=True, persistent_workers=True)    
    optimizer = optim.Adam(model.parameters(),lr=LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # train
    loss_over_time = np.zeros((EPOCH,1))
    for epoch in range(EPOCH):
        loss = 0.0
        for data, labels in train:
            loss = train_batch(data.to(device),labels.to(device),model.to(device),optimizer,class_num)
        loss_over_time[epoch,0] = loss
        if epoch % 50 == 1:
            print(loss)
        scheduler.step()
    
    # test
    correct = 0 
    total = 0
    correct_classes = np.zeros((10,)) 
    for data,label in test:
        pred = model(data.to(device))
        # Find maximum value in prediction and return that index as predicted value
        prediction = torch.argmax(pred)
        if prediction.item() == label.item():
            correct += 1
            correct_classes[label.item()] = correct_classes[label.item()]+1
        total += 1
    print(correct/total)
    print(correct_classes)
    return loss_over_time

# Subtract mean from data to center it at 0
def normalize(loader):
    mean = get_mean(loader)
    # test to see if more effective; currently doing nothing
    #loader = loader-mean

def plot_loss(loss):
    epochs = range(1,len(loss)+1)
    plt.plot(epochs,loss,label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(IMG_PATH + 'losscurve.png')


def get_mean(loader):
    # Compute the mean and standard deviation of all pixels in the dataset
    num_pixels = 0
    mean = 0.0
    std = 0.0
    for images, _ in loader:
        batch_size, num_channels, height, width = images.shape
        num_pixels += batch_size * height * width
        mean += images.mean(axis=(0, 2, 3)).sum()
    mean /= num_pixels
    return mean

# train a single mini-batch of data
def train_batch(data,labels,model,optimizer,class_num):
    optimizer.zero_grad()
    # Convert to a 1-hot vector for loss
    labels = torch.nn.functional.one_hot(labels,num_classes=class_num).to(dtype=torch.float32)
    preds = model(data)
    loss = criterion(preds,labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def create_dataset():
    dataset = vision.datasets.MNIST(root="mnist",download=True,transform=AUGMENT_TRANSFORM)
    return dataset

def main():
    #which dataset to use
    mode = 'small'
    if mode == 'mnist':
        dataset = vision.datasets.MNIST(root="mnist",download=True,transform=AUGMENT_TRANSFORM)
        global SIZE
        SIZE = 28
    elif mode == 'tiny':
        dataset = SmallDataset('single_dataset',file_list = set(['forall','equals']),transform=AUGMENT_TRANSFORM)
    elif mode == 'small':
        dataset = SmallDataset(PATH,file_list = set(['arrow2','dot','epsilon','less','greater','lowerg','upperA','rational','three','lparen']),transform=AUGMENT_TRANSFORM)
    elif mode == 'full':
        dataset = CustomDataset(PATH, transform=AUGMENT_TRANSFORM)
    
    global class_num
    class_num = len(dataset.classes)
    net = CNN(class_num,SIZE*SIZE)
    loss = full_train(dataset,net)
    plot_loss(loss)

if __name__ == '__main__':
    main()
