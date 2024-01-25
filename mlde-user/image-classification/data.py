import torch
import os
from PIL import Image

from skimage import io
from torch.utils.data import Dataset

# ======================================================================================================================

class CatDogDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files     = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = io.imread(img_name)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = 0 if 'dog' in img_name else 1
        sample = (image, label)
        #print(f'Fetched image: {idx} / {img_name}')
        return sample

# ======================================================================================================================
