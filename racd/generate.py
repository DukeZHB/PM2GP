"""Rare-Category Augmentation via Conditional Diffusion (RACD), Section 3.3.

Samples new histopathology images from the reverse diffusion process,
conditioned on exemplar pseudo masks (Eq. (9)). Two kinds of conditioning
masks are used (Section 3.3):

    (i)  original exemplar masks, each reused with different random seeds;
    (ii) blended masks that union the rare-class regions of two exemplars
         (mask blending, Eq. (10)).

The class-prioritised sampler below draws exemplar masks with a
probability proportional to how much of the requested rare categories
they contain, so that under-represented tissue types dominate the
synthetic set. In the paper, 5,000 synthetic image-mask pairs are
generated per dataset (Table 4).

Run from the repository root, e.g.:

    python -m racd.generate \
        --checkpoint ./checkpoints/luad_mdp/8_0.0001_100_0.02731.pth \
        --mask_dir   ./datasets/LUAD-HistoSeg/train/pseudo_mask \
        --out_dir    ./generated/luad_racd \
        --num_generate 5000 --rare_classes 1 2
"""

import argparse
import os
import random

import cv2
import numpy as np
import torch
from tqdm import tqdm

from diffusion import schedules
from diffusion.dataset import (IMG_SIZE, get_images_list, mask_to_class_index,
                               class_index_to_rgb, parse_category_label,
                               reverse_transforms_image)
from diffusion.unet import DiffusionNet


@torch.no_grad()
def sample_timestep(model, x, t, cat_label, mask, betas, betas_schedule):
    """One reverse-diffusion step conditioned on the pseudo mask (Eq. (9))."""
    x_input = torch.cat([x, mask], dim=1)
    noise_pred = model(x_input, t, cat_label)

    betas_t = schedules.get_index_from_list(betas, t, x.shape)
    sqrt_recip_alphas_t = schedules.get_index_from_list(
        betas_schedule['sqrt_recip_alphas'], t, x.shape)
    sqrt_one_minus = schedules.get_index_from_list(
        betas_schedule['sqrt_one_minus_alphas_cumprod'], t, x.shape)
    posterior_variance_t = schedules.get_index_from_list(
        betas_schedule['posterior_variance'], t, x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus)
    if t == 0:
        return model_mean
    # stochastic term: repeated sampling from the same mask yields
    # visually distinct yet anatomically plausible images (Section 3.3)
    return model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)


@torch.no_grad()
def sample_image(model, cat_label, mask, betas, betas_schedule, device,
                 img_size=IMG_SIZE, timesteps=1000):
    """Generate one image conditioned on ``mask`` starting from pure noise."""
    if cat_label.dim() == 1:
        cat_label = cat_label.unsqueeze(0)
    img = torch.randn((1, 3, img_size, img_size), device=device)
    for i in range(0, timesteps)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(model, img, t, cat_label, mask, betas, betas_schedule)
    return img


def load_mask_as_tensor(mask_path, device):
    """Read a colour pseudo mask and return it as a [1, 1, H, W] tensor."""
    mask_img = cv2.imread(mask_path)
    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
    mask_class = mask_to_class_index(mask_img)
    return torch.LongTensor(mask_class).unsqueeze(0).unsqueeze(1).float().to(device)


def blend_masks(mask_a, mask_b, rare_class):
    """Mask blending (Eq. (10)).

    Unions the rare-class regions of two exemplar masks without altering
    their original geometry: every pixel that belongs to ``rare_class``
    in either exemplar becomes ``rare_class`` in the blended mask; all
    other pixels keep their label from the first exemplar.
    """
    blended = mask_a.copy()
    blended[mask_b == rare_class] = rare_class
    return blended


def load_exemplar_masks(mask_dir, rare_classes):
    """Load exemplar pseudo masks as class-index arrays together with the
    multi-hot label parsed from each file name.

    Returns a list of (class_index_array, label_tensor) and a parallel
    list of rarity scores (number of rare-class pixels + 1) used for
    class-prioritised sampling.
    """
    names = get_images_list(mask_dir)
    exemplars, scores = [], []
    for name in names:
        mask_img = cv2.imread(os.path.join(mask_dir, name))
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)
        mask_class = mask_to_class_index(mask_img)
        label = parse_category_label(name)
        rare_pixels = sum(int((mask_class == c).sum()) for c in rare_classes)
        exemplars.append((mask_class, label))
        scores.append(rare_pixels + 1)
    return exemplars, np.array(scores, dtype=np.float64)


def pick_exemplar(exemplars, probs):
    idx = np.random.choice(len(exemplars), p=probs)
    return exemplars[idx]


def main():
    parser = argparse.ArgumentParser(
        description='Rare-Category Augmentation via Conditional Diffusion')
    parser.add_argument('--checkpoint', required=True,
                        help='MDP checkpoint (DiffusionNet weights)')
    parser.add_argument('--mask_dir', required=True,
                        help='directory with training pseudo masks (exemplars)')
    parser.add_argument('--out_dir', required=True,
                        help='output directory for synthetic image-mask pairs')
    parser.add_argument('--num_generate', type=int, default=5000,
                        help='number of synthetic pairs (5,000 in the paper)')
    parser.add_argument('--rare_classes', type=int, nargs='+', default=[1, 2],
                        help='class indices prioritised during generation '
                             '(default 1 2, i.e. NEC and LYM)')
    parser.add_argument('--blend_ratio', type=float, default=0.5,
                        help='fraction of samples conditioned on blended masks')
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--img_size', type=int, default=IMG_SIZE)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img_out = os.path.join(args.out_dir, 'img')
    mask_out = os.path.join(args.out_dir, 'mask')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    betas = schedules.quadratic_beta_schedule(timesteps=args.timesteps)
    betas_schedule = schedules.get_beta_schedule(betas)

    model = DiffusionNet(dim=64, channels=4).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    exemplars, scores = load_exemplar_masks(args.mask_dir, args.rare_classes)
    probs = scores / scores.sum()
    print(f'Loaded {len(exemplars)} exemplar masks; '
          f'rare classes prioritised: {args.rare_classes}')

    for i in tqdm(range(args.num_generate), desc='RACD generating'):
        mask_class, cat_label = pick_exemplar(exemplars, probs)

        # mask blending (Eq. (10)): transfer the rare-class region of a
        # second exemplar into the context of the first one
        if random.random() < args.blend_ratio:
            mask_class_b, _ = pick_exemplar(exemplars, probs)
            rare = random.choice(args.rare_classes)
            mask_class = blend_masks(mask_class, mask_class_b, rare)
            # the blended mask advertises the rare class
            cat_label = cat_label.clone()
            cat_label[rare] = 1.0

        mask_tensor = (torch.LongTensor(mask_class)
                       .unsqueeze(0).unsqueeze(1).float().to(device))
        img_tensor = sample_image(model, cat_label.to(device), mask_tensor,
                                  betas, betas_schedule, device,
                                  img_size=args.img_size,
                                  timesteps=args.timesteps)
        out_img = reverse_transforms_image(img_tensor.cpu())
        out_mask = class_index_to_rgb(mask_class)

        label_str = ' '.join(str(int(x)) for x in cat_label.tolist())
        fname = f"racd_{i:05d}_[{label_str}].png"
        cv2.imwrite(os.path.join(img_out, fname),
                    cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(mask_out, fname),
                    cv2.cvtColor(out_mask, cv2.COLOR_RGB2BGR))

    print(f'Done. {args.num_generate} synthetic pairs written to {args.out_dir}')


if __name__ == '__main__':
    main()
