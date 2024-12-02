import torch
import numpy as np
from torchvision import transforms as T
from torchvision.transforms.functional import pad

class MaskToTensor:
    def __call__(self, mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(np.array(mask), dtype=torch.int64)
        return mask


class ResizeWithPadding:
    def __init__(self, target_size, interpolation):
        self.target_size = target_size
        self.interpolation = interpolation

    def __call__(self, img):
        original_width, original_height = img.size
        scale = self.target_size / max(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        img = T.Resize((new_height, new_width), interpolation=self.interpolation)(img)

        pad_left = (self.target_size - new_width) // 2
        pad_right = self.target_size - new_width - pad_left
        pad_top = (self.target_size - new_height) // 2
        pad_bottom = self.target_size - new_height - pad_top

        img = pad(img, padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0)

        return img


