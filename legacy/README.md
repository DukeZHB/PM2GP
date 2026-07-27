# Legacy — baseline and experimental code

This directory preserves code from the original research code base that
is **not part of the PM2GP pipeline**. It is kept for reproducibility of
the baselines and ablations reported in the paper, and for reference
only.

| File | Origin | Purpose |
| --- | --- | --- |
| `inference_pspnet.py` | `inference.py` | PSPNet (timm-resnest200e) baseline inference; includes the baseline's white-background post-processing |
| `stage2_dataset.py` | `GenDataset.py` (Stage-2 part) | image/mask dataset + loaders for the PSPNet pipeline |
| `custom_transforms.py` | `custom_transforms.py` | image+label transforms for the PSPNet pipeline |
| `seg_transformers.py` | `seg_transformers.py` | image-only transforms for PSPNet inference |
| `lr_scheduler.py` | `lr_scheduler.py` | step/cosine/poly LR scheduler (unused by PM2GP, which follows the paper's constant lr) |
| `experimental_losses.py` | `losses.py` | segmentation losses explored during development (Focal/CE/SS/Tversky/Cosine/FocalLog/LogMax/PolyLog); PM2GP uses RESL in `segmentation/resl.py` |

If you only want to train or evaluate PM2GP, you do not need anything in
this directory. `segmentation_models_pytorch` is only required by
`inference_pspnet.py`.
