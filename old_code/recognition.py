#old
import torch
import os
from torchvision import transforms
from PIL import Image
from skimage import io

from old_code.myutils import *


# Function to predict label from a frame
def predict_label(frame, model, labels):
    # Move the frame to the appropriate device
    # frame = frame.to(device)
    # Define image preprocessing

    frame = frame/max(frame)

    preprocess = transforms.Compose([
        #transforms.Resize((32,32)),  # Resize to match model input size
        transforms.ToTensor(),         # Convert to tensor
    ])
    frame = preprocess(Image.fromarray(frame))

    # Run the model
    with torch.no_grad():
        outputs = model(frame)

    probabilities = torch.softmax(outputs, dim=1)    

    _, predicted = torch.max(outputs, 1)
    predicted_label = predicted.item()
    ans = labels[predicted_label]

    confidence = probabilities[0, predicted_label].item()
    
    return ans, confidence

def get_label_list(filepath='data/labels.list'):
    li=[]
    with open(filepath) as f:
        li=eval(f.read().strip())
    return sorted(li)

# pytorch model from file
def init_model(model_filepath= 'data/model_20240217164634.pth'):
    # load model
    model_state = torch.load(model_filepath)
    device = torch.device("cpu")
    model = create_net()
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

def init_cnn_from_files():
    ensemble = []
    dir_path = 'tf_models'
    for i in range(NUM_CNNS):
        model = create_TF_CNN()
        model.load_weights(dir_path+f'/model{i}.ckpkt')
        ensemble.append(model)
    return ensemble

    
# inference on frame with tf ensemble.
# returns label and confidence.
def predict_ensemble_label(frame, ensemble, labels):
    print('range of frame is ',np.max(frame),np.min(frame))
    frame = np.array(frame)/255.0
    print('NEW range of frame is ',np.max(frame),np.min(frame))
    results=[0 for _ in range(len(ensemble))]
    frame = frame.reshape((1, 32, 32, 1))
    ans=[0 for _ in range(len(ensemble))]
    for i,mod in enumerate(ensemble):
        # predict
        results[i] = mod.predict(frame)
        print(f'mod.predict(frame) is {results[i].shape}')
        print(f'mod.predict(frame) is {results[i]}')
        print('found argmax to be ',np.argmax(results[i], axis = 1))
        ans[i] = np.argmax(results[i], axis = 1)
    ans = np.array(ans)
    print(ans)
    # ans=ans.flatten()
    print(f'shape of ans is {ans.shape}')
    comm = np.bincount(ans[1]).argmax()
    return labels[comm], results[0][0][comm]