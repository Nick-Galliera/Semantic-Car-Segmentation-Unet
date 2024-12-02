import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import json
from matplotlib.patches import Patch
from config import *
from torchvision.transforms import ToPILImage
import time
from torchmetrics import MeanMetric
from tqdm import tqdm
from torchvision.transforms import ToTensor

def get_class_from_number(number):
    return CLASSES.get(number, "Unknow")


def get_number_from_class(value):
    return next((key for key, val in CLASSES.items() if val == value), None) 


def hex_to_rgb(hex_value):
    hex_value = hex_value.lstrip('#')
    r = int(hex_value[0:2], 16)  
    g = int(hex_value[2:4], 16)  
    b = int(hex_value[4:6], 16) 
    return (r, g, b)


''' allinea classi [0..4] e colori descritti nel json '''
def align_class_colors(mask, json_dir=JSON_FILE):

    with open(json_dir, 'r') as f:
        json_description = json.load(f)

    classes_colors = {tag['name']: tag['color'] for tag in json_description['tags']}
    mask_array = np.array(mask)
    rgb_image = np.ones((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8) * 255 
    print(f"[?] Aligning mask to json colors. Unique values in mask_array: {np.unique(mask_array)}")
    for class_name, hex_color in classes_colors.items():

        rgb = hex_to_rgb(hex_color)
        class_value = get_number_from_class(class_name)
        rgb_image[mask_array == class_value] = rgb

    final_image = Image.fromarray(rgb_image)
    return final_image, classes_colors


''' meglio togliere norm e std da transformazione '''
def get_dataset_inspection(dataset, json_dir=JSON_FILE, index=0):

    print(f"[?] Dataset len: {len(dataset)}")

    if len(dataset) != 0:

        image, mask = dataset[index]
        image = image.permute(1, 2, 0).cpu().numpy()
        
        mask_aligned, classes_colors = align_class_colors(mask, json_dir)

        print(f"[?] Img[{index}] shape: {image.shape}")
        print(f"[?] Mask[{index}] shape: {mask.shape}")

        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        
        axs[0].imshow(image)
        axs[0].axis("off")
        axs[0].set_title(f"Image")

        axs[1].imshow(mask_aligned) 
        axs[1].axis("off")
        axs[1].set_title(f"Mask")

        axs[2].imshow(overlap_mask_img(image, mask, json_dir, show=False)) 
        axs[2].axis("off")
        axs[2].set_title(f"Overlap")

        legend_elements = [Patch(facecolor=color, label=name) for name, color in classes_colors.items()]
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=False)

        plt.tight_layout()
        plt.show()


''' crea immagine con overlapping di immagine e maschera '''
def overlap_mask_img(img, mask, json_dir=JSON_FILE, alpha=0.3, show=True):

    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)

    mask_aligned, _ = align_class_colors(mask, json_dir)
    
    mask_aligned = np.array(mask_aligned)

    overlapped_img = img.copy()

    overlapped_img = (
        (1 - alpha) * img + alpha * mask_aligned
    ).astype(np.uint8)

    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(overlapped_img)
        plt.axis("off")
        plt.show()
    else:
        return overlapped_img


def train_model(model, train_loader, optimizer, criterion, num_epochs, device, test_loader=None, tensorboard=None):
    
    loss_record = MeanMetric()
    valid_loss_record = MeanMetric()

    training_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[?] Training: {model.__class__.__name__} to {device} at {training_start_time}")

    for epoch in range(num_epochs):

        model.train()

        for imgs, masks in tqdm(train_loader):

            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            reconstructed = model(imgs)
            loss = criterion(reconstructed, masks)
            loss.backward()
            optimizer.step()

            loss_record.update(loss.detach().item())

        train_loss = loss_record.compute().item()

        if tensorboard:
            tensorboard.plot_loss(torch.tensor(train_loss), epoch, f"Loss {model.__class__.__name__}")

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

        if test_loader:
            model.eval()
            to_tensor = ToTensor()
            for test_batch, _ in test_loader:
                test_batch = test_batch.to(device)
                with torch.no_grad():
                    model_outs = model.predict(test_batch)
                out = []
                for model_out in model_outs:
                    out_aligned, _ = align_class_colors(model_out)
                    if not isinstance(out_aligned, torch.Tensor):
                        out_aligned = to_tensor(out_aligned)  
                    out.append(out_aligned)
                tensorboard.visualize_image_batch(out, epoch, f"{model.__class__.__name__} training reconstruction") 
