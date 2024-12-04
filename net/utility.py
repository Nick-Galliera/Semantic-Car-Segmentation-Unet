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


def compute_iou(pred_mask, true_mask, num_classes=NUM_CLASSES):

    pred_mask = torch.from_numpy(pred_mask)  
    true_mask = torch.from_numpy(true_mask)  

    ious = []
    for cls in range(num_classes):
        # Crea maschere binarie per la classe corrente
        pred_cls = (pred_mask == cls)
        true_cls = (true_mask == cls)

        # Calcola l'intersezione e l'unione
        intersection = torch.sum(pred_cls & true_cls).float()
        union = torch.sum(pred_cls | true_cls).float()

        # Calcola l'IoU, evitando la divisione per zero
        iou = intersection / union if union > 0 else torch.tensor(0.0)
        ious.append(iou)

    return torch.tensor(ious)

''' allinea classi [0..4] e colori descritti nel json '''
def align_class_colors(mask, json_dir=JSON_FILE, on_gpu=True):

    with open(json_dir, 'r') as f:
        json_description = json.load(f)

    classes_colors = {tag['name']: tag['color'] for tag in json_description['tags']}
    if on_gpu:
        mask_array = np.array(mask.cpu())
    else:
        mask_array = np.array(mask)
    rgb_image = np.ones((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8) * 255 
    #print(f"[?] Aligning mask to json colors. Unique values in mask_array: {np.unique(mask_array)}")
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


def train_model(model, train_loader, optimizer, criterion, num_epochs, device, test_loader=None, tensorboard=None, scheduler=None):
    
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

        if test_loader:
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for imgs, masks in test_loader:
                    imgs, masks = imgs.to(device), masks.to(device)

                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                    valid_loss += loss.item()

            valid_loss = valid_loss / len(test_loader)
            valid_loss_record.update(valid_loss)

            if tensorboard:
                tensorboard.plot_loss(torch.tensor(valid_loss), epoch, f"Validation Loss {model.__class__.__name__}")

        if scheduler:
            scheduler.step(valid_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")


def test_model(model, val_loader, show_samples=True, show_number=3):

    all_imgs = []
    all_masks = []
    for imgs, masks in val_loader:
        all_imgs.append(imgs)
        all_masks.append(masks)
    all_imgs = torch.cat(all_imgs, dim=0) 
    all_masks = torch.cat(all_masks, dim=0)

    model.eval()

    with torch.no_grad():
        model_masks = model.predict(all_imgs)

    if show_samples:
        fig, axes = plt.subplots(show_number, 3, figsize=(10, 3 * show_number))
        
        for i in range(show_number):

            iou = compute_iou(model_masks[i].cpu().numpy(), all_masks[i].cpu().numpy())

            print(f"[out] Mean Iou val {i}: {iou.mean()} Iou by class: {iou}")

            # Immagine originale
            axes[i, 0].imshow(all_imgs[i].permute(1, 2, 0).cpu().numpy())
            axes[i, 0].set_title("Original Image")
            axes[i, 0].axis('off')

            # Maschera originale (allineamento colori opzionale)
            real_mask, _ = align_class_colors(all_masks[i].cpu().numpy(), on_gpu=False)
            axes[i, 1].imshow(real_mask)
            axes[i, 1].set_title("True Mask")
            axes[i, 1].axis('off')

            # Maschera predetta (allineamento colori opzionale)
            pred_mask, _ = align_class_colors(model_masks[i].cpu().numpy(), on_gpu=False)
            axes[i, 2].imshow(pred_mask)
            axes[i, 2].set_title("Predicted Mask")
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.show()
    
    tot_iou_mean = []
    tot_iou_class = np.zeros(NUM_CLASSES) 

    for i in range(len(all_imgs)):
        
        iou = compute_iou(model_masks[i].cpu().numpy(), all_masks[i].cpu().numpy(), num_classes=NUM_CLASSES)

        tot_iou_class += iou.cpu().numpy()
        tot_iou_mean.append(iou.mean())

    mean_iou = np.mean(tot_iou_mean)

    mean_iou_class = tot_iou_class / len(all_imgs)

    print(f"\n[out] Model: {model.__class__.__name__}:")
    print(f"Mean IoU on the entire dataset: {mean_iou}")
    print(f"Mean IoU by class on the entire dataset: {mean_iou_class}\n")


