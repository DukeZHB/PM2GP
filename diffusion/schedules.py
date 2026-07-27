"""Noise schedules and forward diffusion process (Section 3.2.1, Eq. (1)-(2)).

The paper uses the quadratic variance schedule with T = 1000 diffusion
steps. Alternative schedules are kept for reference but are not used in
the experiments.
"""

import torch
import torch.nn.functional as F


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule (Nichol & Dhariwal). Not used in the paper."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """Linear schedule (Ho et al.). Not used in the paper."""
    return torch.linspace(start, end, timesteps)


def quadratic_beta_schedule(timesteps):
    """Quadratic schedule used for all MDP experiments (T = 1000)."""
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    """Sigmoid schedule. Not used in the paper."""
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def get_beta_schedule(betas):
    """Pre-compute every quantity derived from the variance schedule."""
    schedule = {}
    schedule['alphas'] = 1. - betas
    schedule['alphas_cumprod'] = torch.cumprod(schedule['alphas'], dim=0)
    schedule['alphas_cumprod_prev'] = F.pad(schedule['alphas_cumprod'][:-1], (1, 0), value=1.0)
    schedule['sqrt_recip_alphas'] = torch.sqrt(1.0 / schedule['alphas'])
    schedule['sqrt_alphas_cumprod'] = torch.sqrt(schedule['alphas_cumprod'])
    schedule['sqrt_one_minus_alphas_cumprod'] = torch.sqrt(1. - schedule['alphas_cumprod'])
    schedule['posterior_variance'] = betas * (1. - schedule['alphas_cumprod_prev']) / (
            1. - schedule['alphas_cumprod'])
    return schedule


def get_index_from_list(vals, t, x_shape):
    """Gather vals[t] for every batch element and reshape for broadcasting."""
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, betas_schedule, device="cpu"):
    """Closed-form sampling of x_t from x_0 (Eq. (2)).

    Returns the noised image x_t and the Gaussian noise that was added.
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(betas_schedule['sqrt_alphas_cumprod'], t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        betas_schedule['sqrt_one_minus_alphas_cumprod'], t, x_0.shape
    )
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
           + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)
