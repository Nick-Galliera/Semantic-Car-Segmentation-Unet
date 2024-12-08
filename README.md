# Semantic Car Segmentation with U-Net

This project implements semantic car segmentation using a U-Net architecture with ResNet (18, 34, and 101) backbones. The goal is to achieve precise pixel-level segmentation of car images for computer vision applications like autonomous driving.

### Features
U-Net Architecture for accurate semantic segmentation.
ResNet Backbones (18, 34, and 101) for robust feature extraction.
Easy configuration through the config.py file where all model parameters are defined.

### Configuration
The model parameters, training settings, and paths are defined in the config.py file. The file includes the following configurable parameters:
-  Backbone selection: Choose from ResNet18, ResNet34, or ResNet101.
-  Batch size: Set the desired batch size for training.
-  Learning rate: Control the learning rate during model training.
-  Epochs: Set the number of epochs for training.
-  Training/Testing Mode: Easily switch between training and testing modes.
-  Ensure you review and update the config.py file before starting the training process..

### Usage
1. setup runtime mode in config.py
2. Run the model with: _python main.py_

## Weights
Important: download weights from [drive](https://drive.google.com/drive/folders/1udwkYJ82kRoGECNK9OxzzxyfT1LfaNle?usp=share_link) and place them in the same director as the project

## Dataset
The dataset is available on Kaggle and can be accessed [here](https://www.kaggle.com/datasets/intelecai/car-segmentation).Please place the dataset in the same directory as the project for proper loading.

<img width="989" alt="mask_image_overlap" src="https://github.com/user-attachments/assets/dcbe1722-2ef5-47a8-b7b0-dbb90971f3b5">


## Results
The best result achieved with the U-Net model and ResNet101 backbone produced an average **Intersection Over Union (IoU) score of 0.932**. Additional experiments are ongoing to improve performance.


![out_unet_resnet101_top](https://github.com/user-attachments/assets/229855eb-b868-430f-a0dd-d147d4530da6)
