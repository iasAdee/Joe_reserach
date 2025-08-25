import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import fcn_resnet50
from sklearn.metrics import precision_score, recall_score, f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.dataset import EvARESTDataset


# src/detection/train_fcn_detector.py
def train_fcn_model(data_dir, device, batch_size=4, epochs=10, lr=1e-4):
    print(data_dir)

    transform = A.Compose([
        A.Resize(256, 256),
        A.RandomBrightnessContrast(),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    ])

    # DATA
    dataset = EvARESTDataset(data_dir, transforms=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # MODEL
    model = fcn_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)  # binary output
    model.to(device)

    # LOSS & OPTIMIZER
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # TRAIN LOOP
    print("Training FCN Model on EvArEST Dataset...\n")
    model.train()
    scores = []
    for epoch in range(epochs):
        total_loss = 0
        y_true, y_pred = [], []
        for images, masks in dataloader:
            images, masks = images.to(device), masks.unsqueeze(1).to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int().cpu().numpy().flatten()
            labels = masks.int().cpu().numpy().flatten()
            y_true.extend(labels)
            y_pred.extend(preds)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
        scores.append([epoch+1, total_loss, precision, recall, f1])

    print("Training Complete.")

    MODEL_PATH = "../models/fcn_evarest.pth"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")



