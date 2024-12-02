import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from config import *


class Pretrained_U_net(nn.Module):

    def __init__(self, encoder_name=PRETRAINED_BACKBONE, requiregrad_encoder=False, num_classes=NUM_CLASSES, activation=None, visualize_model=PRETRAINED_VISUALIZE_MODEL):

        super(Pretrained_U_net, self).__init__()

        self.model = smp.Unet(encoder_name=encoder_name, classes=num_classes, activation=None)
        self.num_classes = num_classes
        self.activation = activation
        
        if visualize_model:
            print(f"[?] Pretrained Unet:\n{self.model}")

        for param in self.model.encoder.parameters():
            param.requires_grad = requiregrad_encoder

    def forward(self, x):

        logits = self.model(x) 
        
        if self.activation == "softmax":
            return F.softmax(logits, dim=1)
        elif self.activation == "sigmoid":
            return torch.sigmoid(logits)
        return logits

    def predict(self, x):
        probs = self.forward(x)
        return torch.argmax(probs, dim=1)
