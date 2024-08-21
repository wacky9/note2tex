import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import os

PATH = 'dataset2'

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [file for file in os.listdir(root_dir) if not file.startswith('.')]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._make_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, target = self.images[idx]
        # Open image and convert to grayscale
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, target

    def _make_dataset(self):
        images = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.root_dir, class_name)
            for file_name in os.listdir(class_dir):
                images.append((os.path.join(class_dir, file_name), class_idx))
        return images

# Example usage
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# Test Code Below
dataset = CustomDataset(PATH, transform=transform)

# Accessing an image and its label
image, label = dataset[100]
print(image.shape, label)

train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img)
plt.show()
print(f"Label: {label}")