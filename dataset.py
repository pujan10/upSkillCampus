
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class AgriDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, self.image_files[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        label_name = os.path.join(self.labels_dir, self.image_files[idx].replace('.jpeg', '.txt'))
        if not os.path.exists(label_name):
            raise FileNotFoundError(f"Label file {label_name} not found.")
        
        with open(label_name, 'r') as file:
            label = int(file.readline().strip().split()[0])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
