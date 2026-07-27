# Stage 1 — Pseudo-Mask Generation (standard CAM pipeline)

> **This stage is NOT a contribution of the paper.** It follows the
> conventional CAM-based weakly supervised pipeline and exists only to
> produce the pseudo masks that condition the diffusion pretraining
> (MDP). The contributions of PM2GP live in [`diffusion/`](../diffusion),
> [`racd/`](../racd) and [`segmentation/`](../segmentation).

## What it does

1. Trains a ResNet-38 multi-label classifier with **image-level labels
   only** (`train_classifier.py`, backbone in `resnet38d.py` /
   `resnet38_cls.py`).
2. Extracts class activation maps from three branches (`ic1`, `ic2`,
   `fc8`), fuses them with fixed dataset-specific weights, thresholds
   them by the image-level prediction and argmaxes into a pseudo-mask
   label map (`cam_inference.py`, `cam_utils.py`).
3. Writes one color palette PNG per training image; these PNGs are the
   conditional input of [`diffusion/train_mdp.py`](../diffusion/train_mdp.py).

## Usage

```bash
python -m stage1_pseudo_masks.train_classifier \
    --dataset luad \
    --trainroot ./datasets/LUAD-HistoSeg/train/img/ \
    --testroot  ./datasets/LUAD-HistoSeg/test/ \
    --pmroot    ./datasets/LUAD-HistoSeg/train_pseudo_mask/ \
    --weights   ./init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
```

Key defaults (from the original experiment code): batch size 20,
20 epochs, poly learning rate 0.01, three-branch loss weights
0.2 / 0.3 / 0.5, CAM fusion weights (0.47, 0.06, 0.47) for LUAD and
(0.11, 0.78, 0.11) for BCSS.

Optionally, CAM overlays can be fused for visual inspection with
[`scripts/fuse_cam_overlays.py`](../scripts/fuse_cam_overlays.py).

## Notes

- ImageNet pretraining uses the MXNet ResNet-38 weights
  (`ilsvrc-cls_rna-a1_cls1000_ep-0001.params`); `mxnet` is only needed
  for the one-off weight conversion in
  `resnet38d.convert_mxnet_to_torch`.
- The feature-refinement blocks in `resnet38d.py` originate from the
  authors' previous work and are kept unchanged for reproducibility.
