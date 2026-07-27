"""Encoder transfer and downstream segmentation fine-tuning, Section 3.4.

The encoder of the pretrained mask-conditioned diffusion model (MDP) is
transferred to the segmentation network and the whole pipeline is
fine-tuned with the Region-Enhanced Segmentation Loss (RESL). When RACD
synthetic pairs are provided, the training set is the augmented dataset
D_aug = D_train + D_synth (Section 3.3).

Weight transfer follows the paper exactly: every pretrained diffusion
weight is copied except (i) the segmentation head (final_conv), which is
trained from scratch, and (ii) the first convolutional layer, whose
4-channel weights (noisy image + mask) are truncated to the 3 RGB
channels. All remaining parameters are fine-tuned (strict=False load).

Paper settings (Section 4.2): 100 epochs, batch size 8, Adam with
constant lr 1e-4, RESL with alpha = 0.5.

Run from the repository root, e.g.:

    python -m segmentation.train_seg \
        --image_dir ./datasets/LUAD-HistoSeg/train/img \
        --mask_dir  ./datasets/LUAD-HistoSeg/train/pseudo_mask \
        --synth_image_dir ./generated/luad_racd/img \
        --synth_mask_dir  ./generated/luad_racd/mask \
        --pretrained_ckpt ./checkpoints/luad_mdp/8_0.0001_100_0.02731.pth \
        --exp_name luad_seg
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from skimage import io
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from diffusion.dataset import mask_to_class_index
from diffusion.early_stopping import EarlyStopping
from segmentation.resl import RESLoss
from segmentation.segnet import SegNet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def augment(image, mask):
    """Spatial and photometric augmentation shared by image and mask."""
    if random.random() > 0.5:
        image, mask = TF.vflip(image), TF.vflip(mask)
    if random.random() > 0.5:
        image, mask = TF.hflip(image), TF.hflip(mask)
    if random.random() > 0.7:
        image = TF.gaussian_blur(image, [3, 3], [1.0, 2.0])
    if random.random() > 0.7:
        jitter = transforms.ColorJitter(brightness=.5, contrast=.4)
        image = jitter(image)
    return image, mask


class HistoSegDataset(Dataset):
    """(image, pseudo mask) pairs for segmentation fine-tuning.

    ``image_dirs`` / ``mask_dirs`` accept one or more directory pairs so
    that RACD synthetic pairs can be appended to the real training set.
    """

    def __init__(self, image_dirs, mask_dirs, transform=None):
        self.pairs = []  # (image_path, mask_path)
        for image_dir, mask_dir in zip(image_dirs, mask_dirs):
            names = sorted(
                [f for f in os.listdir(image_dir)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
                key=lambda x: x.split('-')[0])
            for name in names:
                mask_path = os.path.join(mask_dir, name)
                if os.path.exists(mask_path):
                    self.pairs.append((os.path.join(image_dir, name), mask_path))
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        image = io.imread(image_path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        mask = io.imread(mask_path)
        mask = mask_to_class_index(mask)  # colour mask -> class indices

        if self.transform is not None:
            image = self.transform(image)
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        image, mask = augment(image, mask)
        return image, mask.squeeze(0).long()


def load_pretrained_encoder(model, checkpoint_path, device):
    """Transfer the MDP encoder weights into the segmentation network."""
    print(f'Loading pretrained diffusion weights from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint['model_state_dict']

    # (i) drop the diffusion output head; the segmentation head is
    # trained from scratch
    state = {k: v for k, v in state.items() if 'final_conv' not in k}

    # (ii) drop the category-conditioning branch (cat_emb_layer / mlp_cat);
    # it is not part of the visual encoder and SegNet is built with
    # with_cat_emb=False
    state = {k: v for k, v in state.items() if 'cat_emb' not in k}

    # (iii) adapt the first convolution from 4 input channels
    # (noisy image + mask) to the 3 RGB channels
    key = 'net.init_conv.weight'
    if key in state and state[key].shape[1] == 4:
        state[key] = state[key][:, :3, :, :]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Encoder transfer done. Missing keys: {len(missing)}, '
          f'unexpected keys: {len(unexpected)}')


def run_epoch(dataloader, model, criterion, num_classes, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    losses = []
    for image, true_label in tqdm(dataloader):
        image = image.to(DEVICE)
        true_label = true_label.to(DEVICE)
        target_onehot = F.one_hot(true_label, num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        # constant timestep 0: the backbone is used as a plain feed-forward net
        t = torch.full((image.shape[0],), 0, dtype=torch.long, device=DEVICE)
        pred = model(image, t)
        loss = criterion(pred, target_onehot)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())

    avg = float(np.mean(losses))
    print(f"{'train' if train else 'val'} loss: {avg:.4f}")
    return avg


def main():
    parser = argparse.ArgumentParser(
        description='Segmentation fine-tuning with encoder transfer + RESL')
    parser.add_argument('--image_dir', required=True,
                        help='training images (real patches)')
    parser.add_argument('--mask_dir', required=True,
                        help='pseudo masks of the real training patches')
    parser.add_argument('--synth_image_dir', default=None,
                        help='optional RACD synthetic images')
    parser.add_argument('--synth_mask_dir', default=None,
                        help='optional RACD synthetic masks')
    parser.add_argument('--pretrained_ckpt', default=None,
                        help='MDP checkpoint for encoder transfer; if omitted '
                             'the network is trained from random initialisation')
    parser.add_argument('--exp_name', default='seg')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='classes including background')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='RESL balancing weight (Table 5)')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()

    checkpoint_dir = os.path.join('./checkpoints', args.exp_name)
    plot_dir = os.path.join('./plots', args.exp_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    image_dirs = [args.image_dir]
    mask_dirs = [args.mask_dir]
    if args.synth_image_dir and args.synth_mask_dir:
        image_dirs.append(args.synth_image_dir)
        mask_dirs.append(args.synth_mask_dir)

    full_dataset = HistoSegDataset(
        image_dirs, mask_dirs, transform=transforms.ToTensor())
    print(f'{len(full_dataset)} image-mask pairs for fine-tuning '
          f'({len(image_dirs)} source(s))')

    idxs = np.random.RandomState(args.seed).permutation(len(full_dataset))
    split = int(len(full_dataset) * (1.0 - args.val_ratio))
    train_set = torch.utils.data.Subset(full_dataset, idxs[:split])
    val_set = torch.utils.data.Subset(full_dataset, idxs[split:])

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, drop_last=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = SegNet(dim=64, channels=3, num_classes=args.num_classes).to(DEVICE)
    print(f"Num params: {sum(p.numel() for p in model.parameters())}")

    if args.pretrained_ckpt:
        load_pretrained_encoder(model, args.pretrained_ckpt, DEVICE)

    criterion = RESLoss(alpha=args.alpha)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # constant lr
    early_stopping = EarlyStopping(
        patience=args.patience, verbose=True,
        path=os.path.join(checkpoint_dir, f'{args.batch_size}_{args.lr}.pth'))

    train_losses, val_losses = [], []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f'epoch {epoch}/{args.epochs}')
        train_losses.append(
            run_epoch(train_loader, model, criterion, args.num_classes,
                      optimizer=optimizer))
        with torch.no_grad():
            val_losses.append(
                run_epoch(val_loader, model, criterion, args.num_classes))

        early_stopping(val_losses[-1], model, epoch)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    print(f'Total time elapsed: {time.time() - start_time:.1f} seconds')

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'tab:blue', label='Train Loss')
    plt.plot(epochs, val_losses, 'tab:orange', label='Val Loss')
    plt.title('Segmentation fine-tuning (RESL)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'seg_loss.jpg'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
