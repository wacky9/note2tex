# Handles the neural network aspect of the project
from enum import Enum
from model_nn_basic import NN
import torch

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
    return model

def detect():
    raise NotImplemented

def train(params):
    model = create_model(Model.neural,params)
    
    return 0

def benchmark():
    raise NotImplemented