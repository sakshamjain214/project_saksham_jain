# project_saksham_jain

# Deepfake Detector CNN

This project implements a custom Convolutional Neural Network (CNN) built from scratch in PyTorch to classify images as either **Real** or **Fake**. The architecture is specifically designed to detect microscopic artifacts common in AI-generated media.

## 📂 Directory Structure

This repository is formatted to strictly adhere to the project submission guidelines. 

```text
project_saksham_jain/
│
├── checkpoints/
│   └── final_weights.pth      # Pre-trained model weights (State Dictionary)
│
├── data/                       # Sample dataset for inference testing
│   ├── fake1.jpg
│   ├── ...
│   ├── real1.jpg
│   └── ...
│
├── config.py                   # Global hyperparameters and dataset paths
├── dataset.py                  # Custom PyTorch Dataloaders and image transforms
├── interface.py                # Centralized namespace hub for evaluation scripts
├── model.py                    # CNN Architecture (DeepfakeDetectorCNN)
├── predict.py                  # Inference script for evaluating new images
├── train.py                    # Training loop with validation and checkpoint saving
└── README.md                   # Project documentation
