# object-detection
# Object Detection Using PyTorch

This repository contains an object detection project implemented using PyTorch and `torchvision`. It demonstrates how to use pre-trained models such as **SSD** and **Faster R-CNN** on a custom dataset, visualize detections, and train the model.

## 📁 Project Structure
object-detection/
│
├── data/
│ ├── VOCtrainval_06-Nov-2007.tar # [LARGE] Pascal VOC dataset
│
├── models/
│ ├── ssd_resnet_pascal.pth # Pre-trained weights (if any)
│
├── utils/
│ ├── dataset_utils.py # Custom dataset/dataloader utilities
│ ├── visualization.py # Functions for plotting boxes
│
├── train.py # Script to train the model
├── demo.py # Script to run inference on images
├── requirements.txt # Python dependencies
└── README.md # Project description


---

## 🚀 Features

- Pre-trained **SSD** and **Faster R-CNN** models from `torchvision`
- Custom dataset support (Pascal VOC format or COCO)
- Real-time inference and bounding box visualization
- Easily switch models and backbones

---

## 🛠️ Installation

1. **Clone the repo**:

```bash
git clone https://github.com/AgarwalBhavya/object-detection.git
cd object-detection

2. **Set up virtual environment (optional but recommended):**
python -m venv .venv
source .venv/Scripts/activate   # On Windows

3. **Install dependencies:**
pip install -r requirements.txt

Training

To train the model:
python train.py

Inference Demo
Run the following command to test detection on a sample image:
python demo.py

Requirements
Python 3.8+
PyTorch
torchvision
matplotlib
numpy
opencv-python (optional)
