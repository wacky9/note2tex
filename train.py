import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from skimage import io
from data import CustomDataset
from data import SmallDataset
from model_nn_basic import NN
from model_conv import CNN
from matplotlib.pylab import plt
criterion = nn.CrossEntropyLoss()
LR = 0.001
BATCH = 4
EPOCH = 1000
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
    #Just arbitrary values to get the data between [-1,1]
    v2.Normalize([0.5],[1])
])

PLAIN_TRANSFORM = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32)
])

def full_train(data, model):
    #test-train split
    #train_data,test_data = torch.utils.data.random_split(data,[0.8,0.2])
    #no split
    train_data = data
    test_data = data

    train = DataLoader(train_data,batch_size = BATCH)
    normalize(train)
    test = DataLoader(test_data)    
    optimizer = optim.Adam(model.parameters(),lr=LR)
    # train
    loss_over_time = []
    for epoch in range(EPOCH):
        loss = 0.0
        for data,labels in train:
            loss = train_batch(data.to(device),labels.to(device),model.to(device),optimizer)
        loss_over_time.append(loss)
    
    # test
    correct = 0 
    total = 0
    correct_classes = [] 
    for data,label in test:
        pred = model(data.to(device))
        # Find maximum value in prediction and return that index as predicted value
        prediction = torch.argmax(pred)
        if prediction.item() == label.item():
            correct += 1
            correct_classes.append(label.item())
        total += 1
    print(correct/total)
    print(correct_classes)
    return loss_over_time

# Subtract mean from data to center it at 0
def normalize(loader):
    mean= get_mean(loader)

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
def train_batch(data,labels,model,optimizer):
    optimizer.zero_grad()
    # Convert to a 1-hot vector for loss
    labels = torch.nn.functional.one_hot(labels,num_classes=class_num).to(dtype=torch.float32)
    preds = model(data)
    loss = criterion(preds,labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def main():
    global class_num
    #dataset = CustomDataset(PATH, transform=AUGMENT_TRANSFORM)

    #small dataset
    dataset = SmallDataset(PATH,file_list = set(['arrow2','dot','epsilon','less','greater','lowerg','upperA','rational','three','lparen']),transform=AUGMENT_TRANSFORM)
    #Extremely small dataset
    #dataset = SmallDataset('single_dataset',file_list = set(['forall','equals']),transform=PLAIN_TRANSFORM)
    class_num = len(dataset.classes)
    net = NN(class_num,SIZE*SIZE)
    loss = full_train(dataset,net)
    plot_loss(loss)

    #dataset = SmallDataset('single_dataset',file_list = set(['forall','equals']),transform=PLAIN_TRANSFORM)
    class_num = len(dataset.classes)
    net = CNN(class_num,SIZE*SIZE)
    loss = full_train(dataset,net)
    plot_loss(loss)

if __name__ == '__main__':
    main()
