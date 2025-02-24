# Handles the neural network aspect of the project
from enum import Enum
from model_nn_basic import NN
from model_conv import CNN
from train import full_train, create_dataset
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'model/basic_mnist.pth'
class Model(Enum):
    neural = 1
    conv = 2

def detect(model, frame):
    pred = model(frame.to(device))
    return torch.argmax(pred).item()

def training():
    #MNIST dataset
    dataset = create_dataset()
    class_num = len(dataset.classes)
    #img_size
    SIZE = 28
    net = CNN(class_num,SIZE*SIZE)
    full_train(dataset,net,class_num)
    #save trained model for inference
    torch.save(net.state_dict(),model_path)

def load_model():
    dataset = create_dataset()
    class_num = len(dataset.classes)
    #img_size
    SIZE = 28
    model = CNN(class_num,SIZE*SIZE)
    model.load_state_dict(torch.load(model_path,weights_only=True))
    model.eval()
    return model.to(device)

def benchmark():
    raise NotImplemented

if __name__ == '__main__':
    training()