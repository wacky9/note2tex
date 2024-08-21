import torch
import torch.nn as nn
import torch.optim as optim

criterion = nn.MSELoss()
LR = 0.01
BATCH = 64

def full_train(data, labels,model):
    #test-train split
    train_data,test_data,validate_data = torch.utils.random_split(data,[0.8,0.1,0.1])
    train = DataLoader(train_data,batch_size = BATCH)
    test = DataLoader(test_data,batch_size = BATCH)    
    validate = DataLoader(validate_data,batch_size = BATCH)    
    
    optimizer = optim.Adam(model.parameters(),lr=LR)
    
    return 0

# train a single mini-batch of data
def batch(data,labels,model,optimizer):
    preds = model(data)
    loss = criterion(preds,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()