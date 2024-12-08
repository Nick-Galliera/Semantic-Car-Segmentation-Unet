**Semantic Car Segmentation with U-Net**

This project implements semantic car segmentation using a U-Net architecture with ResNet (18, 34, and 101) backbones. The goal is to achieve precise pixel-level segmentation of car images for computer vision applications like autonomous driving.

**Features**
U-Net Architecture for accurate semantic segmentation.
ResNet Backbones (18, 34, and 101) for robust feature extraction.
Easy configuration through the config.py file where all model parameters are defined.

**Configuration**
The model parameters, training settings, and paths can be easily adjusted in the config.py file. This file contains all the necessary parameters for:
-  Backbone selection (ResNet18, ResNet34, ResNet101)
-  Batch size
-  Learning rate
-  Epochs and more
-  Training or Testing mode
  
Make sure to review and update the config.py file before training.

**Usage**
setup runtime mode in config.py
run with python main.py

