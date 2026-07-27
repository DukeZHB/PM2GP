"""Datasets for the Stage-1 multi-label classifier and CAM inference.

Image-level labels are parsed from the file name, which follows the
dataset convention ``<id>-...-[l1 l2 l3 l4].png`` (LUAD-HistoSeg uses
space-separated digits; BCSS-WSSS uses adjacent digits).
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class Stage1_InferDataset(Dataset):
    """Plain image folder used for CAM inference."""

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.object = self.path_label()

    def __getitem__(self, index):
        fn = self.object[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return fn.split('/')[-1][:-4], img

    def __len__(self):
        return len(self.object)

    def path_label(self):
        path_list = []
        for root, dirname, filename in os.walk(self.data_path):
            for f in filename:
                path_list.append(os.path.join(root, f))
        return path_list


class Stage1_TrainDataset(Dataset):
    """Training images with parsed multi-hot image-level labels."""

    def __init__(self, data_path, transform=None, dataset=None):
        self.data_path = data_path
        self.transform = transform
        self.dataset = dataset
        self.object = self.path_label()

    def __getitem__(self, index):
        fn, label = self.object[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return fn.split('/')[-1][:-4], img, label

    def __len__(self):
        return len(self.object)

    def path_label(self):
        path_label = []
        for root, dirname, filename in os.walk(self.data_path):
            for f in filename:
                image_path = os.path.join(root, f)
                fname = f[:-4]
                label_str = fname.split(']')[0].split('[')[-1]
                if self.dataset == 'luad':
                    # "[1 0 0 1]" space-separated digits
                    image_label = torch.Tensor([int(label_str[0]), int(label_str[2]),
                                                int(label_str[4]), int(label_str[6])])
                elif self.dataset == 'bcss':
                    # "[1001]" adjacent digits
                    image_label = torch.Tensor([int(label_str[0]), int(label_str[1]),
                                                int(label_str[2]), int(label_str[3])])
                else:
                    raise ValueError(f'unknown dataset: {self.dataset}')
                path_label.append((image_path, image_label))
        return path_label
