import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torchvision import transforms

from data import CustomDataset
from model_nn_basic import NN
criterion = nn.MSELoss()
LR = 0.01
BATCH = 64
EPOCH = 4
PATH = 'dataset2'
SIZE = 224
class_num = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
])

def full_train(data, model):
    #test-train split
    train_data,test_data,validate_data = torch.utils.data.random_split(data,[0.8,0.1,0.1])
    train = DataLoader(train_data,batch_size = BATCH)
    test = DataLoader(test_data,batch_size = BATCH)    
    validate = DataLoader(validate_data,batch_size = BATCH)    
    optimizer = optim.Adam(model.parameters(),lr=LR)
    for epoch in range(EPOCH):
        for data,labels in train:
            batch(data.to(device),labels.to(device),model.to(device),optimizer)
    return 0

# train a single mini-batch of data
def batch(data,labels,model,optimizer):
    optimizer.zero_grad()
    labels = torch.nn.functional.one_hot(labels,num_classes=class_num)
    preds = model(data)
    print(preds[0])
    #TODO: Preds are vectors, labels are just the number. Fix
    loss = criterion(preds,labels)
    loss.backward()
    optimizer.step()
    print(loss.item())


def main():
    global class_num
    dataset = CustomDataset(PATH, transform=TRANSFORM)
    class_num = len(dataset.classes)
    print(class_num)
    net = NN(class_num,SIZE*SIZE)
    full_train(dataset,net)

if __name__ == '__main__':
    main()
