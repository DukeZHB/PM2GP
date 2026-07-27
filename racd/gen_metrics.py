"""Generative quality metrics for the synthetic image-mask pairs.

Reports the four metrics used in the paper (Section 4.2 and Table 6):
Fréchet Inception Distance (FID, ref [35]), Inception Score (IS,
ref [36]) and Improved Precision / Improved Recall (IP / IR, ref [37],
Kynkaanniemi et al., NeurIPS 2019). All metrics are computed on the
generated set against an equally sized set of real training patches
(5,000 vs 5,000 in the paper).

Run from the repository root, e.g.:

    python -m racd.gen_metrics \
        --gen_dir  ./generated/luad_racd/img \
        --real_dir ./datasets/LUAD-HistoSeg/train/img
"""

import argparse
import os
import ssl

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

IMG_SIZE = 256


class ImageListDataset(Dataset):
    """Plain folder of images, resized to IMG_SIZE and scaled to [0, 1]."""

    def __init__(self, folder, transform):
        self.imgs = sorted(f for f in os.listdir(folder)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        path = os.path.join(self.folder, self.imgs[idx])
        return self.transform(Image.open(path).convert('RGB'))


def get_inception_pool3(device):
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT,
                                transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval()
    return model.to(device)


@torch.no_grad()
def extract_pool3(dataloader, model, device):
    feats = []
    for batch in tqdm(dataloader, desc='pool3'):
        img = batch[0] if isinstance(batch, (list, tuple)) else batch
        img = F.interpolate(img.to(device), (299, 299), mode='bilinear')
        feats.append(model(img).cpu().numpy())
    return np.concatenate(feats, 0)


@torch.no_grad()
def extract_softmax(dataloader, model, device):
    probs = []
    for batch in tqdm(dataloader, desc='softmax'):
        img = batch[0] if isinstance(batch, (list, tuple)) else batch
        img = F.interpolate(img.to(device), (299, 299), mode='bilinear')
        probs.append(F.softmax(model(img), dim=1).cpu().numpy())
    return np.concatenate(probs, 0)


def compute_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet Inception Distance between two Gaussian feature statistics."""
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_is(probs, splits=10):
    """Inception Score (mean +/- std over splits)."""
    n = probs.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    scores = []
    for i in range(splits):
        p_yx = probs[idx[i::splits]]
        p_y = np.mean(p_yx, 0)
        kl = np.sum(p_yx * (np.log(p_yx + 1e-10) - np.log(p_y + 1e-10)), 1)
        scores.append(np.exp(np.mean(kl)))
    return float(np.mean(scores)), float(np.std(scores))


def _knn_radii(feats, k):
    """Distance from every sample to its k-th nearest neighbour within
    the same set (the manifold radius of Kynkaanniemi et al.)."""
    n = feats.shape[0]
    # pairwise squared distances via the (a-b)^2 expansion
    sq = np.sum(feats ** 2, axis=1, keepdims=True)
    d2 = sq + sq.T - 2 * feats @ feats.T
    np.fill_diagonal(d2, np.inf)  # exclude self
    kth = np.partition(d2, kth=k - 1, axis=1)[:, k - 1]
    return np.sqrt(np.maximum(kth, 0.0))


def compute_improved_precision_recall(real_feats, gen_feats, k=3):
    """Improved Precision and Recall (ref [37]).

    Each set is approximated by the union of k-NN balls around its
    samples. Precision is the fraction of generated samples falling
    inside the real manifold; recall is the fraction of real samples
    covered by the generated manifold.
    """
    real_radii = _knn_radii(real_feats, k)
    gen_radii = _knn_radii(gen_feats, k)

    sq_r = np.sum(real_feats ** 2, axis=1, keepdims=True)
    sq_g = np.sum(gen_feats ** 2, axis=1, keepdims=True)
    d2_rg = sq_g.T + sq_r - 2 * real_feats @ gen_feats.T  # [n_real, n_gen]

    # precision: generated sample inside the ball of its nearest real sample
    nearest_real = np.argmin(d2_rg, axis=0)
    precision = np.mean(
        d2_rg[nearest_real, np.arange(len(nearest_real))]
        <= real_radii[nearest_real] ** 2)

    # recall: real sample inside the ball of its nearest generated sample
    nearest_gen = np.argmin(d2_rg, axis=1)
    recall = np.mean(
        d2_rg[np.arange(len(nearest_gen)), nearest_gen]
        <= gen_radii[nearest_gen] ** 2)

    return float(precision), float(recall)


def main():
    parser = argparse.ArgumentParser(
        description='Generative quality metrics (FID / IS / IP / IR)')
    parser.add_argument('--gen_dir', required=True,
                        help='folder with generated images')
    parser.add_argument('--real_dir', required=True,
                        help='folder with real training patches')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--knn_k', type=int, default=3,
                        help='neighbourhood size k for IP/IR (default 3)')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    real_loader = DataLoader(ImageListDataset(args.real_dir, trans),
                             args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    gen_loader = DataLoader(ImageListDataset(args.gen_dir, trans),
                            args.batch_size, shuffle=False,
                            num_workers=args.num_workers)

    print('Extracting Inception pool3 features ...')
    pool3 = get_inception_pool3(device)
    real_feat = extract_pool3(real_loader, pool3, device)
    gen_feat = extract_pool3(gen_loader, pool3, device)

    fid = compute_fid(np.mean(real_feat, 0), np.cov(real_feat.T),
                      np.mean(gen_feat, 0), np.cov(gen_feat.T))
    print(f'FID: {fid:.4f}')

    print('Computing Inception Score ...')
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT,
        transform_input=False).eval().to(device)
    gen_probs = extract_softmax(gen_loader, inception, device)
    is_mean, is_std = compute_is(gen_probs)
    print(f'IS: {is_mean:.4f} +/- {is_std:.4f}')

    print('Computing Improved Precision / Recall ...')
    ip, ir = compute_improved_precision_recall(real_feat, gen_feat, k=args.knn_k)
    print(f'Improved Precision: {ip:.4f}')
    print(f'Improved Recall:    {ir:.4f}')


if __name__ == '__main__':
    main()
