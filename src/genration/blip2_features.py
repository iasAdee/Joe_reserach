import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ========================
# CONFIG
# ========================
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Load BLIP
print("Loading BLIP...")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

# ========================
# Helper: Overlay FCN Mask
# ========================
def apply_fcn_mask(image_path, mask_path, alpha=0.5):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Resize mask to match image
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Highlight text regions in red
    red_mask = cv2.merge([mask, mask*0, mask*0])
    blended = cv2.addWeighted(img, 1, red_mask, alpha, 0)

    return blended

# ========================
# BLIP Inference
# ========================
def generate_caption(image_array):
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# ========================
# Example Run
# ========================
if __name__ == "__main__":
    # Example paths (replace with your dataset + FCN predicted mask)
    image_path = "../../data/Det_train/img_13.jpg"
    mask_path = "../../data/fcn_masks/img_13_mask.png"  # save FCN mask earlier

    # Create guided image (original + mask overlay)
    guided_img = apply_fcn_mask(image_path, mask_path)

    # Get captions
    caption_original = generate_caption(cv2.imread(image_path))
    caption_guided = generate_caption(guided_img)

    # Show comparison
    print("\nOriginal BLIP caption:", caption_original)
    print("FCN-guided BLIP caption:", caption_guided)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
    plt.title(f"Original: {caption_original}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(guided_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Guided: {caption_guided}")
    plt.axis("off")
    plt.show()
