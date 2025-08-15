# src/main.py

import torch
import os
from train_fcn_detector import train_fcn_model

if __name__ == "__main__":
    
    # Resolve dataset path
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = "../data/Det_train/"
    #DATA_DIR = os.path.join(BASE_DIR, "../data/Det_train")

    # Auto-select MPS or CPU
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print("Using device:", DEVICE)

    # Call training
    train_fcn_model(
        data_dir=DATA_DIR,
        device=DEVICE,
        batch_size=4,
        epochs=10,
        lr=1e-4
    )
