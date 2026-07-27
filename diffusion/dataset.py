"""Dataset utilities for mask-conditioned diffusion pretraining (MDP).

Every training sample is a triple of
    (histopathology image, pseudo mask, multi-hot image-level label)

Pseudo masks are colour PNGs produced by the Stage-1 CAM pipeline (see
``stage1_pseudo_masks/``); they are converted to per-pixel class indices
on the fly. The image-level label vector is parsed from the file name,
which follows the dataset convention ``<id>-...-[l1 l2 l3 l4].png``.
"""

import os
import re

import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 256  # all images and pseudo masks are resized to 256 x 256

# Colour palette of the pseudo masks, as RGB triplets. All readers in this
# repository convert masks to RGB before matching, so the palette is defined
# in RGB order (identical to the ground-truth palette of the datasets).
COLOR_MAP = {
    (205, 51, 51): 0,    # TE / TUM
    (0, 255, 0): 1,      # NEC
    (65, 105, 225): 2,   # LYM
    (255, 165, 0): 3,    # TAS / STR
    (255, 255, 255): 4,  # background
}

# Class-index to RGB palette (used when saving generated masks).
CLASS_TO_RGB = {
    0: (205, 51, 51),
    1: (0, 255, 0),
    2: (65, 105, 225),
    3: (255, 165, 0),
    4: (255, 255, 255),
}


def get_images_list(path, k=None):
    """List image files sorted by their numeric prefix."""
    supported = (".png", ".jpg", ".jpeg", ".bmp")
    names = [f for f in os.listdir(path) if f.lower().endswith(supported)]
    names = sorted(names, key=lambda x: int(x.split('-')[0]))
    if k is not None:
        names = names[:k]
    return np.array(names)


def parse_category_label(filename):
    """Parse the multi-hot image-level label from the file name."""
    pattern = r'\[(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\]'
    match = re.search(pattern, filename)
    if match:
        return torch.FloatTensor([float(match.group(i)) for i in range(1, 5)])
    return torch.FloatTensor([0., 0., 0., 0.])


def mask_to_class_index(mask_img):
    """Convert a colour pseudo mask (H, W, 3) to class indices (H, W)."""
    h, w, _ = mask_img.shape
    class_idx = np.zeros((h, w), dtype=np.int64)
    for color, idx in COLOR_MAP.items():
        match = np.all(mask_img == color, axis=-1)
        class_idx[match] = idx
    return class_idx


def class_index_to_rgb(class_idx):
    """Convert class indices (H, W) back to an RGB mask (H, W, 3)."""
    rgb = np.full((*class_idx.shape, 3), 255, dtype=np.uint8)
    for idx, color in CLASS_TO_RGB.items():
        rgb[class_idx == idx] = color
    return rgb


def _image_normalization(x):
    """Scale [0, 1] image tensors to [-1, 1] for the diffusion process."""
    return (x * 2) - 1


def get_mask_transforms():
    """Mask transforms: keep spatially aligned with the image transforms."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ])


class HistoDataset(Dataset):
    """Image / pseudo-mask / label triples for MDP."""

    def __init__(self, image_dir, mask_dir, image_list,
                 transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        name = self.image_list[index]

        image = io.imread(os.path.join(self.image_dir, name))
        mask_img = io.imread(os.path.join(self.mask_dir, name))
        cat_label = parse_category_label(name)

        if self.transform is not None:
            image = self.transform(image)

        if self.mask_transform is not None:
            mask_t = self.mask_transform(mask_img)
            mask_np = mask_t.permute(1, 2, 0).numpy().astype(np.uint8)
            mask_class = torch.LongTensor(mask_to_class_index(mask_np))
        else:
            mask_class = torch.LongTensor(mask_to_class_index(mask_img))

        return image, mask_class, cat_label


def load_transformed_dataset(image_dir, mask_dir, val_ratio=0.1, seed=2023,
                             max_images=None):
    """Build the MDP train/validation split (90% / 10% by default)."""
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Lambda(_image_normalization),
    ])
    mask_transform = get_mask_transforms()

    image_list = get_images_list(image_dir, k=max_images)
    idxs = np.random.RandomState(seed).permutation(len(image_list))
    split = int(len(image_list) * (1.0 - val_ratio))
    train_index, valid_index = idxs[:split], idxs[split:]

    train_dataset = HistoDataset(image_dir, mask_dir, image_list[train_index],
                                 transform=data_transform,
                                 mask_transform=mask_transform)
    eval_dataset = HistoDataset(image_dir, mask_dir, image_list[valid_index],
                                transform=data_transform,
                                mask_transform=mask_transform)
    return train_dataset, eval_dataset


def reverse_transforms_image(image):
    """Undo the [-1, 1] normalisation and return a uint8 HWC array."""
    reverse = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
    ])
    if len(image.shape) == 4:
        image = image[0, ...]
    return reverse(image)
