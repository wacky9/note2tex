import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from torchvision.transforms import v2
from skimage import io
from data import CustomDataset
from model_nn_basic import NN
criterion = nn.MSELoss()
LR = 0.01
BATCH = 16
EPOCH = 500
PATH = 'dataset2'
SIZE = 32
class_num = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Note to self: transforms are applied to each image every single batch. Dataset is the same size but each epoch it sees different data
# Augmentations: rotation, blur, resizing... others?
# Do I need larger images to safely do some of these transforms?
TRANSFORM = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32),
    v2.RandomRotation(degrees=[-15,15]),
    v2.GaussianNoise(),
    v2.GaussianBlur(3,(0.1,1.0)),
])

def full_train(data, model):
    #test-train split
    train_data,test_data,validate_data = torch.utils.data.random_split(data,[0.8,0.1,0.1])
    train = DataLoader(train_data,batch_size = BATCH)
    test = DataLoader(test_data)    
    validate = DataLoader(validate_data,batch_size = BATCH)    
    optimizer = optim.Adam(model.parameters(),lr=LR)
    # train
    for epoch in range(EPOCH):
        for data,labels in train:
            batch(data.to(device),labels.to(device),model.to(device),optimizer)
    
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
    return 0

# train a single mini-batch of data
def batch(data,labels,model,optimizer):
    optimizer.zero_grad()
    # Convert to a 1-hot vector for loss
    labels = torch.nn.functional.one_hot(labels,num_classes=class_num).to(dtype=torch.float32)
    preds = model(data)
    loss = criterion(preds,labels)
    loss.backward()
    optimizer.step()
    #print(loss.item())


def main():
    global class_num
    dataset = CustomDataset(PATH, transform=TRANSFORM)
    class_num = len(dataset.classes)
    net = NN(class_num,SIZE*SIZE)
    full_train(dataset,net)

if __name__ == '__main__':
    main()
