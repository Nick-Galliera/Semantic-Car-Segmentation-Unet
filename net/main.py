from config import *
from utility import *
from custom_transformations import *
from car_segmentation_dataset import *
from models import Pretrained_U_net
from custom_tensorboard import TensorboardVisualizer

import matplotlib.pyplot as plt
import numpy as np
import random

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Subset
import torch.optim as optim
import torchvision.transforms.v2 as T

from segmentation_models_pytorch.encoders import get_preprocessing_fn


if __name__ == '__main__':

    if TENSORBOARD_VISUALIZE:
        tensorboard = TensorboardVisualizer(log_dir=LOG_DIR)
        tensorboard.prepare_folder()

# ---------------------- DATASET DEFINITION ---------------------- # 

    preprocess_input = get_preprocessing_fn(encoder_name=PRETRAINED_BACKBONE, pretrained='imagenet')

    image_transform = T.Compose([
        ResizeWithPadding(target_size=224, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=preprocess_input.keywords['mean'], std=preprocess_input.keywords['std'])
    ])

    mask_transform = T.Compose([
        ResizeWithPadding(target_size=224, interpolation=T.InterpolationMode.NEAREST),
        MaskToTensor()
    ])

    dataset = CarSegmentationDataset(images_dir=IMAGE_FOLDER, masks_dir=MASKS_FOLDER, image_transform=image_transform, mask_transform=mask_transform)

    train_size = int(TRAIN_SIZE * len(dataset))  
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    if AUGMENT:
        augmented_indices = random.sample(range(len(train_dataset)), int(AUGMENT_SIZE * len(train_dataset)))
        subset_for_augmentation = Subset(train_dataset, augmented_indices)
        augmented_data = []

        for image, mask in subset_for_augmentation:
            rotation_angle = random.uniform(-180, 180)

            augmented_image = T.functional.rotate(image, angle=rotation_angle, interpolation=T.InterpolationMode.BILINEAR)
            
            augmented_mask = mask.unsqueeze(0)
            augmented_mask = T.functional.rotate(augmented_mask, angle=rotation_angle, interpolation=T.InterpolationMode.NEAREST)
            augmented_mask = augmented_mask.squeeze(0)

            augmented_data.append((augmented_image, augmented_mask))

        train_dataset = train_dataset + augmented_data

    train = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    if SHOW_DATASET_PREVIEW:
        get_dataset_inspection(dataset=train_dataset, json_dir=JSON_FILE, index=len(train_dataset)-1)
    
    print(f"[?] Train len: {len(train_dataset)}")
    print(f"[?] Test len: {test_size}")
    
# ---------------------- MODEL DEFINITION - TRANSFER LEARNING MODEL ---------------------- # 
    
    pretrained_model = Pretrained_U_net(encoder_name=PRETRAINED_BACKBONE, requiregrad_encoder=False, num_classes=NUM_CLASSES, activation=None).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_model.model.decoder.parameters(), lr=PRETRAINED_LR) 
    
    if TENSORBOARD_VISUALIZE:
        imgs, _ = next(iter(test))
        tensorboard.plot_graph(pretrained_model, imgs)
    
    if TRAIN_PRETRAINED:
        train_model(model=pretrained_model, train_loader=train, optimizer=optimizer, criterion=criterion, num_epochs=PRETRAINED_EPOCHS, device=DEVICE, test_loader=test, tensorboard=tensorboard)
        torch.save(pretrained_model.state_dict(), PRETRAINED_WEIGHTS)
    else:
        print(f"[?] Loading {PRETRAINED_WEIGHTS}")
        pretrained_model.load_state_dict(torch.load(PRETRAINED_WEIGHTS, weights_only=True))


    tensorboard.close()