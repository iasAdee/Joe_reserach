import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

class EvARESTDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.img_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]
        self.transforms = transforms

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        ann_name = img_name.rsplit('.', 1)[0] + '.txt'

        img_path = os.path.join(self.root_dir, img_name)
        ann_path = os.path.join(self.root_dir, ann_name)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if os.path.exists(ann_path):
            with open(ann_path, 'r', encoding='utf-8') as file:
                for line in file:
                    coords = list(map(int, line.strip().split(',')[:8]))
                    points = np.array([(coords[i], coords[i+1]) for i in range(0, 8, 2)], np.int32)
                    cv2.fillPoly(mask, [points], color=1)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.transpose(2, 0, 1) 
        return torch.tensor(image, dtype=torch.float32) / 255.0, torch.tensor(mask, dtype=torch.float32)
