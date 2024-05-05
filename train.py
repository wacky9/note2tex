import torch
import torch.nn as nn
import torch.optim as optim

criterion = nn.MSELoss()
LR = 0.01


def full_train(data, labels,model):
        
    optimizer = optim.Adam(model.parameters(),lr=LR)

    return 0

# train a single mini-batch of data
def batch(data,labels,model,optimizer):
    preds = model(data)
    loss = criterion(preds,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()