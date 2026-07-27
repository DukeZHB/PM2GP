"""Segmentation network (Section 3.4).

SegNet shares the same U-Net backbone as the diffusion denoiser, which
enables direct encoder transfer: the weights of the pretrained diffusion
encoder are copied into this network, except for the first convolutional
layer which is adapted to the RGB (3-channel) input.
"""

import torch.nn as nn

from diffusion.unet import Unet


class SegNet(nn.Module):
    """U-Net segmentation network initialisable from the MDP encoder.

    Args:
        dim: base channel width of the U-Net backbone (64 in the paper).
        channels: number of input image channels (3 for RGB).
        num_classes: number of output classes, including background.
    """

    def __init__(self, dim=64, channels=3, num_classes=5):
        super().__init__()
        # with_cat_emb=False: the category-conditioning branch of the
        # diffusion denoiser is not part of the visual encoder, so it is
        # neither built here nor transferred from the MDP checkpoint.
        self.net = Unet(dim=dim, channels=channels, out_dim=num_classes,
                        dim_mults=(1, 2, 4, 8), with_cat_emb=False)

    def forward(self, x, time_stamps):
        # time_stamps is kept in the signature so that the module can be
        # called exactly like the denoising network; during fine-tuning a
        # constant timestep of 0 is used.
        return self.net(x, time_stamps)
