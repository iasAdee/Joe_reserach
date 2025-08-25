# Everyday Arabic-English Scene Text Recognition & Generation

This repository implements the research study **"Arabic Scene Text Recognition in the Deep Learning Era: Analysis on a Novel Dataset"**.  
We build and evaluate deep learning pipelines for text detection, multimodal feature extraction, and Arabic scene description generation.  

---

## 📌 Research Phases

### **Phase 1: Text Detection (FCN)**
- Implemented **Fully Convolutional Network (FCN)** based on ResNet-50 backbone.
- Dataset: [Everyday Arabic-English Scene Text (EvArEST)]([https://arxiv.org/abs/2306.09255](https://github.com/HGamal11/EvArEST-dataset-for-Arabic-scene-text)<img width="468" height="15" alt="image" src="https://github.com/user-attachments/assets/d8597d54-93cb-46a9-96cb-e531c6615c3a" />
)  
  - 510 images, annotated with 4-point polygons per word.
- Pipeline:
  1. Load dataset & parse annotations (`.txt` files with polygon coords).
  2. Apply augmentations (resize, rotation, brightness).
  3. Train FCN for **binary segmentation (text vs background)**.
  4. Evaluate with **Precision, Recall, F1**.
  5. Save predicted **binary masks**.

➡️ Output: Masks showing **where text is located** in natural scenes.

---

### **Phase 2: Multimodal Feature Extraction**
- Integrated **Vision-Language Models** to extract high-level scene features:
  - **BLIP** (used instead of BLIP-2 due to model size).
  - Captions are generated both on **raw images** and **FCN-guided masked images**.
- Workflow:
  1. Use **FCN output mask** to highlight text regions.
  2. Pass both **original image** and **guided image** into BLIP.
  3. Compare captions (with vs. without FCN guidance).

➡️ Output: Text-aware captions for images.

---

### **Phase 3: Arabic Scene Description Generation (Planned)**
- Use LLMs fine-tuned for Arabic (HuggingFace Arabic GPT models, or OpenAI GPT with Arabic prompts).
- Generate fluent scene-level descriptions with **Arabic text content** included.
- Evaluation:
  - Automatic: BLEU, ROUGE, METEOR.
  - Human: Fluency, contextual accuracy, storytelling ability.

---

## 🚀 Getting Started

### 1. Clone Repo
```bash
git clone https://github.com/iasAdee/Joe_reserach.git
cd Joe_reserach
