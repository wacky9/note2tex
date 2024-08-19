import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

PATH = 'dataset2'

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = [file for file in os.listdir(root_dir) if not file.startswith('.')]
        print(self.classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._make_dataset()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, target = self.images[idx]
        image = Image.open(img_path).convert("RGB")
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

dataset = CustomDataset(PATH, transform=transform)

# Accessing an image and its label
image, label = dataset[100]
print(image.shape, label)
