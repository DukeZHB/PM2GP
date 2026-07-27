"""Denoising U-Net for mask- and category-conditioned diffusion (MDP).

The network takes a 4-channel input (3-channel noised image + 1-channel
pseudo-mask map) and predicts the 3-channel image noise. Two kinds of
conditioning are injected into every ResNet block:

    - the diffusion timestep (sinusoidal embedding + MLP);
    - the multi-hot image-level category label (CategoryEmbedding).

``DiffusionNet`` is the denoiser used by MDP pretraining
(``diffusion/train_mdp.py``) and RACD generation (``racd/generate.py``);
``Unet`` doubles as the backbone of the segmentation network
(``segmentation/segnet.py``, with ``with_cat_emb=False``).

Architecture and parameter names follow the original experiment code
exactly, so pretrained checkpoints remain fully compatible.
"""

import math
from functools import partial
from inspect import isfunction

import torch
import torch.nn as nn
from einops import rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CategoryEmbedding(nn.Module):
    """Embeds the 4-dim multi-hot image-level label vector."""

    def __init__(self, cat_dim=4, embed_dim=64):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(cat_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, cat_label):
        return self.embed(cat_label)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, cat_emb_dim=None, groups=8):
        super().__init__()
        # timestep embedding projection
        self.mlp_time = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )
        # category embedding projection
        self.mlp_cat = (
            nn.Sequential(nn.SiLU(), nn.Linear(cat_emb_dim, dim_out))
            if exists(cat_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, cat_emb=None):
        h = self.block1(x)

        # fuse the timestep embedding
        if exists(self.mlp_time) and exists(time_emb):
            time_emb = self.mlp_time(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        # fuse the category embedding
        if exists(self.mlp_cat) and exists(cat_emb):
            cat_emb = self.mlp_cat(cat_emb)
            h = rearrange(cat_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale
        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=4,  # 3 (noised image) + 1 (mask) = 4 channels
            with_time_emb=True,
            with_cat_emb=True,  # category-embedding switch
            resnet_block_groups=8,
            convnext_mult=2,
            encoder_only=False
    ):
        super().__init__()
        self.encoder_only = encoder_only
        self.with_cat_emb = with_cat_emb
        self.channels = channels

        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # timestep embedding
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # category embedding
        if with_cat_emb:
            self.cat_emb_layer = CategoryEmbedding(cat_dim=4, embed_dim=dim * 4)
            cat_dim = dim * 4
        else:
            cat_dim = None

        # downsampling path
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim, cat_emb_dim=cat_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim, cat_emb_dim=cat_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, cat_emb_dim=cat_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim, cat_emb_dim=cat_dim)

        if self.encoder_only is False:
            for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
                is_last = ind >= (num_resolutions - 1)
                self.ups.append(
                    nn.ModuleList(
                        [
                            block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim, cat_emb_dim=cat_dim),
                            block_klass(dim_in, dim_in, time_emb_dim=time_dim, cat_emb_dim=cat_dim),
                            Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                            Upsample(dim_in) if not is_last else nn.Identity(),
                        ]
                    )
                )
            out_dim = default(out_dim, 3)  # output stays 3-channel (image noise / logits)
            self.final_conv = nn.Sequential(
                block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
            )

    def forward(self, x, time, cat_label=None):
        # x: (B, 4, H, W) = 3-channel image + 1-channel mask
        x = self.init_conv(x)

        # timestep embedding
        t_emb = self.time_mlp(time) if exists(self.time_mlp) else None
        # category embedding
        c_emb = self.cat_emb_layer(cat_label) if (self.with_cat_emb and exists(cat_label)) else None

        h = []

        # downsampling
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t_emb, c_emb)
            x = block2(x, t_emb, c_emb)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t_emb, c_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb, c_emb)

        # upsampling
        if self.encoder_only is False:
            for block1, block2, attn, upsample in self.ups:
                x = torch.cat((x, h.pop()), dim=1)
                x = block1(x, t_emb, c_emb)
                x = block2(x, t_emb, c_emb)
                x = attn(x)
                x = upsample(x)
            x = self.final_conv(x)

        return x


class DiffusionNet(nn.Module):
    """Mask- and category-conditioned denoiser used by MDP and RACD."""

    def __init__(self, dim, channels=4):  # default 4 channels (3 image + 1 mask)
        super(DiffusionNet, self).__init__()
        self.net = Unet(
            dim=dim,
            channels=channels,
            dim_mults=(1, 2, 4, 8),
            with_cat_emb=True  # category conditioning enabled
        )

    def forward(self, x, time_stamps, cat_label=None):
        # x: concatenated 4-channel input
        e = self.net(x, time_stamps, cat_label)
        return e
