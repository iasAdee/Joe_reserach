import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from torchvision.models.segmentation import fcn_resnet50
import albumentations as A
import matplotlib.pyplot as plt


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Define model structure
# No need to download pretrained weights
model = fcn_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1)  # binary output  
model.load_state_dict(torch.load("../models/fcn_evarest.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("Model loaded and ready for inference.")


transform = A.Compose([
    A.Resize(256, 256),
])

def preprocess_image(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    augmented = transform(image=image)
    image = augmented["image"]
    image = image.transpose(2, 0, 1)  # HWC → CHW
    image = torch.tensor(image, dtype=torch.float32) / 255.0
    return image.unsqueeze(0)  # Add batch dimension

# ========================
# 3. Run inference
# ========================
image_tensor = preprocess_image("../data/Det_train/img_13.jpg").to(DEVICE)

with torch.no_grad():
    output = model(image_tensor)["out"]
    pred_mask = (torch.sigmoid(output) > 0.5).float().squeeze().cpu().numpy()

"""
# ========================
# 4. Visualize the prediction
# ========================
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(cv2.imread("../data/Det_train/img_105.jpg"), cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Text Mask")
plt.axis("off")
plt.show()"""


# ========================
# Function to create ground truth mask from .txt
# ========================
def load_ground_truth_mask(image_path, annotation_dir):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)

    # Get corresponding annotation file
    img_name = os.path.basename(image_path)
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(annotation_dir, txt_name)

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Annotation file not found: {txt_path}")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            coords = list(map(int, line.strip().split(",")[:8]))
            points = np.array([(coords[i], coords[i+1]) for i in range(0, 8, 2)], np.int32)
            cv2.fillPoly(mask, [points], color=1)

    return mask


image_path = "../data/Det_train/img_13.jpg"
annotation_dir = "../data/Det_train"

#gt_mask = load_ground_truth_mask(image_path, annotation_dir)

# ========================
# Visualization
# ========================
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

"""plt.subplot(1, 3, 2)
plt.imshow(gt_mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")
"""
plt.subplot(1, 3, 3)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.show()


output_dir = "../data/fcn_masks/"
os.makedirs(output_dir, exist_ok=True)

# Resize pred_mask to original image size
orig = cv2.imread(image_path)
mask_resized = cv2.resize(pred_mask.astype(np.uint8) * 255, (orig.shape[1], orig.shape[0]))

# Save mask
mask_path = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", "_mask.png"))
cv2.imwrite(mask_path, mask_resized)

print(f"Saved predicted mask: {mask_path}")