"""Test-time evaluation of the fine-tuned segmentation network.

Reports per-class IoU, mean IoU and pixel accuracy over the foreground
tissue classes (background excluded), matching the assessment criteria
of the paper (Section 4.2). Dice scores are printed as a supplement.

Run from the repository root, e.g.:

    python -m segmentation.evaluate \
        --checkpoint ./checkpoints/luad_seg/8_0.0001_100_0.12345.pth \
        --image_dir  ./datasets/LUAD-HistoSeg/test/img \
        --mask_dir   ./datasets/LUAD-HistoSeg/test/mask
"""

import argparse
import os

import numpy as np
import torch
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusion.dataset import CLASS_TO_RGB, mask_to_class_index
from segmentation.evaluator import Evaluator
from segmentation.segnet import SegNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class HistoTestDataset(Dataset):
    """(image, ground-truth mask) pairs of the test split."""

    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.names = sorted(
            [f for f in os.listdir(image_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda x: int(x.split('-')[0]))
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = io.imread(os.path.join(self.image_dir, name))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        mask = mask_to_class_index(io.imread(os.path.join(self.mask_dir, name)))
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.from_numpy(mask).long(), name


def apply_color_map(pred):
    """Class-index map -> RGB overlay using the dataset palette."""
    color_pred = np.zeros((*pred.shape, 3), dtype=np.uint8)
    for label, color in CLASS_TO_RGB.items():
        color_pred[pred == label] = color
    return color_pred


def main():
    parser = argparse.ArgumentParser(
        description='Segmentation evaluation (foreground mIoU / PA)')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image_dir', required=True)
    parser.add_argument('--mask_dir', required=True)
    parser.add_argument('--num_classes', type=int, default=5,
                        help='classes including background')
    parser.add_argument('--save_vis', default=None,
                        help='optional directory for colourised predictions')
    args = parser.parse_args()

    dataset = HistoTestDataset(args.image_dir, args.mask_dir,
                               transform=transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = SegNet(dim=64, channels=3, num_classes=args.num_classes).to(DEVICE)
    snapshot = torch.load(args.checkpoint, map_location=DEVICE)
    model.load_state_dict(snapshot['model_state_dict'])
    model.eval()
    print(f'Loaded {args.checkpoint}; evaluating {len(dataset)} patches ...')

    num_fg = args.num_classes - 1  # foreground classes (background excluded)
    evaluator_all = Evaluator(args.num_classes)
    evaluator_fg = Evaluator(num_fg)

    if args.save_vis:
        os.makedirs(args.save_vis, exist_ok=True)

    with torch.no_grad():
        for image, target, name in tqdm(loader):
            image = image.to(DEVICE)
            t = torch.zeros((image.shape[0],), dtype=torch.long, device=DEVICE)
            pred = model(image, t)
            pred_idx = torch.argmax(pred, dim=1).cpu().numpy()
            target_np = target.numpy()

            evaluator_all.add_batch(target_np, pred_idx)

            # foreground-only view: pixels whose ground truth is a tissue class
            fg = target_np != (args.num_classes - 1)
            if fg.sum() > 0:
                evaluator_fg.add_batch(
                    np.clip(target_np[fg], 0, num_fg - 1),
                    np.clip(pred_idx[fg], 0, num_fg - 1))

            if args.save_vis:
                vis = apply_color_map(pred_idx[0])
                io.imsave(os.path.join(args.save_vis, name[0]), vis,
                          check_contrast=False)

    class_names = ['TE/TUM', 'NEC', 'LYM', 'TAS/STR', 'background']

    ious_all = evaluator_all.Intersection_over_Union()
    print('\n[All classes]')
    for i, iou in enumerate(ious_all):
        print(f'  {class_names[i]:10s} IoU {iou:.4f}')

    ious_fg = evaluator_fg.Intersection_over_Union()
    mean_dice_fg, dice_fg = evaluator_fg.Dice_Score()
    print('\n[Foreground classes - paper metrics]')
    for i, iou in enumerate(ious_fg):
        print(f'  {class_names[i]:10s} IoU {iou:.4f}   Dice {dice_fg[i]:.4f}')
    print(f'\n  mIoU (foreground): {ious_fg.mean():.4f}')
    print(f'  PA   (foreground): {evaluator_fg.Pixel_Accuracy():.4f}')
    print(f'  mDice(foreground): {mean_dice_fg:.4f}')


if __name__ == '__main__':
    main()
