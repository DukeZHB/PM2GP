"""Mask-conditioned Diffusion Pretraining (MDP), Section 3.2.

Trains the conditional denoising network on (image, pseudo mask) pairs
with the Mask-Grouped Diffusion Loss (MGDL). The network input is the
concatenation of the noised image (3 channels) and the pseudo mask
(1 channel); only the 3 image channels are noised and reconstructed.

Paper settings (Section 4.2): U-Net base dimension 64, T = 1000 quadratic
schedule, 100 epochs, batch size 8, Adam with constant lr 1e-4, images and
masks resized to 256 x 256, MGDL with lambda = 0.5.

Run from the repository root, e.g.:

    python -m diffusion.train_mdp \
        --image_dir ./datasets/LUAD-HistoSeg/train/img \
        --mask_dir  ./datasets/LUAD-HistoSeg/train/pseudo_mask \
        --exp_name  luad_mdp
"""

import argparse
import os
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion import schedules
from diffusion.dataset import load_transformed_dataset, reverse_transforms_image
from diffusion.early_stopping import EarlyStopping
from diffusion.mgdl import MDPLoss
from diffusion.unet import DiffusionNet


@torch.no_grad()
def sample_timestep(model, x, t, betas, betas_schedule):
    """One reverse-diffusion step p(x_{t-1} | x_t) (Eq. (9)).

    ``x`` is the 4-channel tensor (noisy image + mask); the model predicts
    the 3-channel image noise and the mask channel is carried through
    unchanged.
    """
    betas_t = schedules.get_index_from_list(betas, t, x.shape)
    sqrt_one_minus = schedules.get_index_from_list(
        betas_schedule['sqrt_one_minus_alphas_cumprod'], t, x.shape)
    sqrt_recip_alphas_t = schedules.get_index_from_list(
        betas_schedule['sqrt_recip_alphas'], t, x.shape)

    batch_size = x.shape[0]
    cat_label = torch.zeros((batch_size, 4), device=x.device)  # placeholder
    noise_pred_3ch = model(x, t, cat_label)
    noise_pred_4ch = torch.cat(
        [noise_pred_3ch, torch.zeros_like(x[:, 3:4, :, :])], dim=1)

    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred_4ch / sqrt_one_minus)
    posterior_variance_t = schedules.get_index_from_list(
        betas_schedule['posterior_variance'], t, x.shape)

    if t == 0:
        return model_mean
    noise = torch.randn_like(x)
    return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, epoch, args, betas, betas_schedule):
    """Denoise a random sample and save a 10x10 grid of intermediate steps."""
    device = args.device
    img = torch.randn((1, 3, args.img_size, args.img_size), device=device)
    mask = torch.zeros((1, 1, args.img_size, args.img_size), device=device)
    x = torch.cat([img, mask], dim=1)

    num_images = 100
    stepsize = int(args.timesteps / num_images)
    all_images = []
    for i in range(0, args.timesteps)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        x = sample_timestep(model, x, t, betas, betas_schedule)
        if i % stepsize == 0:
            all_images.append(x[:, :3, :, :])

    fig, axs = plt.subplots(10, 10)
    idx = 0
    for r in range(10):
        for c in range(10):
            axs[r, c].imshow(reverse_transforms_image(all_images[idx].detach().cpu()))
            axs[r, c].axis('off')
            idx += 1
    plt.savefig(os.path.join(args.image_out_dir, f'image_{epoch}.png'), dpi=300)
    plt.close()


def run_epoch(dataloader, model, optimizer, criterion, betas_schedule, args,
              train=True):
    model.train() if train else model.eval()
    losses = []
    p_bar = tqdm(dataloader)
    for img_batch, mask_batch, cat_batch in p_bar:
        img_batch = img_batch.to(args.device)
        mask_batch = mask_batch.unsqueeze(1).to(args.device)  # [B,1,H,W]
        cat_batch = cat_batch.to(args.device)

        t = torch.randint(0, args.timesteps, (img_batch.shape[0],)).long().to(args.device)
        # forward diffusion on the 3 image channels only
        x_noisy, noise = schedules.forward_diffusion_sample(
            img_batch, t, betas_schedule, args.device)
        # 4-channel model input: noisy image + pseudo mask
        x_noisy_input = torch.cat([x_noisy, mask_batch], dim=1)

        noise_pred = model(x_noisy_input, t, cat_batch)
        loss, parts = criterion(noise, noise_pred, mask_batch.squeeze(1),
                                t=t, betas_schedule=betas_schedule)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        p_bar.set_postfix(loss=f"{loss.item():.4f}", mgdl_cls=f"{parts['l_cls']:.4f}")

    avg = float(np.mean(losses))
    print(f"{'train' if train else 'eval'} loss: {avg:.4f}")
    return avg


def main():
    parser = argparse.ArgumentParser(
        description='Mask-conditioned Diffusion Pretraining (MDP)')
    parser.add_argument('--image_dir', required=True,
                        help='directory with training images (renamed patches)')
    parser.add_argument('--mask_dir', required=True,
                        help='directory with the corresponding pseudo masks')
    parser.add_argument('--exp_name', default='mdp',
                        help='experiment name used for output directories')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lambda_reg', type=float, default=0.5,
                        help='MGDL regularisation weight lambda (Table 3)')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='margin m of the inter-class separation penalty')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='pseudo-mask categories including background')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', default=None,
                        help='optional checkpoint to resume from')
    args = parser.parse_args()

    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    checkpoint_dir = os.path.join('./checkpoints', args.exp_name)
    args.image_out_dir = os.path.join('./generated', args.exp_name, 'previews')
    plot_dir = os.path.join('./plots', args.exp_name)
    for d in (checkpoint_dir, args.image_out_dir, plot_dir):
        os.makedirs(d, exist_ok=True)

    betas = schedules.quadratic_beta_schedule(timesteps=args.timesteps)
    betas_schedule = schedules.get_beta_schedule(betas)

    train_dataset, eval_dataset = load_transformed_dataset(
        args.image_dir, args.mask_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size,
                             shuffle=False, drop_last=False,
                             num_workers=args.num_workers, pin_memory=True)

    # 4-channel input: noisy image (3 ch) + pseudo mask (1 ch)
    model = DiffusionNet(dim=64, channels=4).to(args.device)
    print(f"Num params: {sum(p.numel() for p in model.parameters())}")

    if args.resume is not None:
        print(f'Loading checkpoint from: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])

    criterion = MDPLoss(num_classes=args.num_classes,
                        lambda_reg=args.lambda_reg, margin=args.margin)
    optimizer = Adam(model.parameters(), lr=args.lr)
    early_stopping = EarlyStopping(
        patience=args.patience, verbose=True,
        path=os.path.join(checkpoint_dir, f'{args.batch_size}_{args.lr}.pth'))

    train_losses, eval_losses = [], []
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        print(f'epoch {epoch}/{args.epochs}')
        train_losses.append(
            run_epoch(train_loader, model, optimizer, criterion,
                      betas_schedule, args, train=True))
        eval_losses.append(
            run_epoch(eval_loader, model, optimizer, criterion,
                      betas_schedule, args, train=False))

        if epoch % 20 == 0:
            sample_plot_image(model, epoch, args, betas, betas_schedule)

        early_stopping(eval_losses[-1], model, epoch)
        if early_stopping.early_stop:
            print('Early stopping')
            break

    print(f'Total time elapsed: {time.time() - start_time:.1f} seconds')

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'tab:blue', label='Train Loss')
    plt.plot(epochs, eval_losses, 'tab:orange', label='Eval Loss')
    plt.title('MDP training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'mdp_loss.jpg'), dpi=300)
    plt.close()


if __name__ == '__main__':
    main()
