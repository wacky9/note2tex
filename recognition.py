import torch
from torchvision import transforms
from PIL import Image
from skimage import io

from myutils import *

# Function to predict label from a frame
def predict_label(frame, model, labels):
    # Move the frame to the appropriate device
    # frame = frame.to(device)
    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((32,32)),  # Resize to match model input size
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

def init_model(model_filepath= 'data/model_20240217164634.pth'):
    # load model
    model_state = torch.load(model_filepath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_net()
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model