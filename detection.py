# Handles the neural network aspect of the project
from enum import Enum
from model_nn_basic import NN
from train import full_train
import torch
import torch.nn as nn
import torch.optim as optim


class Model(Enum):
    neural = 1
    conv = 2

def create_model(name, params):
    model
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    if name == Model.neural:
        classes,size = params
        model = NN(classes,size).to(device)
        LR = 0.01
    return model

def detect():
    raise NotImplemented

def training(data,labels, params):
    model = create_model(Model.neural,params)
    full_train(data,labels,model)

def benchmark():
    raise NotImplemented