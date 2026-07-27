"""Segmentation dataset from the legacy PSPNet pipeline.

Loads image/mask pairs for train / val / test splits. Not used by the
PM2GP training scripts (see ``diffusion/dataset.py`` and
``segmentation/train_seg.py``); preserved for reproducing the baseline
results.
"""

import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from legacy import custom_transforms as tr


class Stage2_Dataset(Dataset):
    def __init__(self, args, base_dir, split):

        super().__init__()
        self._base_dir = base_dir
        self.split = split

        if self.split == "train":
            self._image_dir = os.path.join(self._base_dir, 'train/img/')
            self._cat_dir = os.path.join(self._base_dir, 'train_PM/')
        elif self.split == 'val':
            self._image_dir = os.path.join(self._base_dir, 'val/img/')
            self._cat_dir = os.path.join(self._base_dir, 'val/mask/')
        elif self.split == 'test':
            self._image_dir = os.path.join(self._base_dir, 'test/img/')
            self._cat_dir = os.path.join(self._base_dir, 'test/mask/')
        elif self.split == 'pred':
            self._image_dir = os.path.join(self._base_dir, 'test/img/')
            self._cat_dir = os.path.join(self._base_dir, 'test/mask/')
        self.args = args
        self.filenames = [os.path.splitext(file)[0] for file in os.listdir(self._image_dir) if not file.startswith('.')]
        self.images = [os.path.join(self._image_dir, fn + '.png') for fn in self.filenames]
        self.categories = [os.path.join(self._cat_dir, fn + '.png') for fn in self.filenames]

        assert (len(self.images) == len(self.categories))
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if self.split == "train":
            _img, _target = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target}
        elif (self.split == 'val') or (self.split == 'test'):
            _img, _target, = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target}
            image_dir = self.images[index]
        elif (self.split == 'pred'):
            _img, _target, = self._make_img_gt_point_pair(index)
            sample = {'image': _img, 'label': _target}
            image_dir = self.images[index]
        if self.split == "train":
            return self.transform_tr_ab(sample)
        elif (self.split == 'val') or (self.split == 'test'):
            return self.transform_val(sample), image_dir
        elif self.split == 'pred':
            return self.transform_val(sample), image_dir

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        return _img, _target

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomGaussianBlur(),
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_tr_ab(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip_ab(),
            tr.RandomGaussianBlur_ab(),
            tr.Normalize_ab(),
            tr.ToTensor_ab()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.Normalize(),
            tr.ToTensor()])
        return composed_transforms(sample)

    def __str__(self):
        return None


def make_data_loader(args, **kwargs):

    train_set = Stage2_Dataset(args, base_dir=args.dataroot, split='train')
    val_set = Stage2_Dataset(args, base_dir=args.dataroot, split='val')
    test_set = Stage2_Dataset(args, base_dir=args.dataroot, split='test')

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader


def make_pred_loader(args, **kwargs):
    pred_set = Stage2_Dataset(args, base_dir=args.dataroot, split='pred')
    pred_loader = DataLoader(pred_set, batch_size=1, shuffle=False, **kwargs)
    return pred_loader
