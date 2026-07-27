# PM2GP: From Pseudo Masks to Generative Priors

Official implementation of **"From Pseudo Masks to Generative Priors: A
Diffusion-Based Framework for Weakly Supervised Histopathology Image
Segmentation"**.

PM2GP turns the noisy pseudo masks of a conventional CAM pipeline into
*generative priors*: a mask-conditioned diffusion model is first
pretrained to model the joint distribution of histopathology images and
their masks (MDP), then used (a) as a generative augmenter that
synthesizes image/mask pairs with rare classes prioritized (RACD) and
(b) as a pretrained backbone whose encoder initializes the segmentation
network, trained with a region-enhanced loss (RESL).

## Method overview

```
image-level labels
       │
       ▼  (standard CAM pipeline - NOT a contribution)
pseudo masks ──────────────┐
       │                   │ condition
       ▼                   ▼
┌─────────────────────────────────┐
│  MDP: Mask-conditioned Diffusion │  L_MDP = L_simple + λ · L_MGDL
│  Pretraining (4-ch input U-Net)  │  (λ = 0.5)
└─────────────────────────────────┘
       │                    │
       ▼                    ▼
  RACD: rare-class      encoder weights
  conditional           transferred to the
  augmentation          segmentation network
  (5000 pairs,                 │
   mask blending)              ▼
                        RESL training
                  L_RESL = α·CE + (1-α)·Dice (α = 0.5)
                               │
                               ▼
                     segmentation masks
```

- **MDP (Mask-conditioned Diffusion Pretraining)** — a U-Net takes a
  4-channel input (3-channel noised image + 1-channel mask map) and is
  trained at 256×256, T = 1000 with a quadratic β schedule, 100 epochs,
  batch size 8, Adam with constant lr 1e-4. In addition to the timestep
  and the mask channel, the denoiser receives the image-level multi-hot
  label vector through a `CategoryEmbedding` branch (an auxiliary
  conditioning input of the implementation; the manuscript describes
  timestep + mask conditioning).
- **MGDL (Mask-Grouped Diffusion Loss)** — groups pixels by their mask
  label in the denoising error map and adds intra-group compactness and
  inter-group separability regularization
  (`L_MGDL = L_cls + λ(L_intra + L_inter)`), steering the generative
  capacity toward tissue-discriminative structure.
- **RACD (Rare-class-Aware Conditional Diffusion augmentation)** —
  samples 5000 image/mask pairs with rare classes prioritized by pixel
  count; a mask-blending step mixes generated and pseudo masks to
  correct label noise.
- **RESL (Region-Enhanced Segmentation Loss)** — CE + Dice hybrid
  (α = 0.5) for the downstream segmentation network, whose encoder is
  initialized from the diffusion U-Net.

Stage 1 (pseudo-mask generation with a standard CAM classifier) is a
*supporting* step and is deliberately isolated under
[`stage1_pseudo_masks/`](stage1_pseudo_masks/README.md); it is **not** a
contribution of this paper.

## Results

Segmentation (foreground classes):

| Dataset | mIoU | PA |
| --- | --- | --- |
| LUAD-HistoSeg | **78.83** | **87.43** |
| BCSS-WSSS | **70.99** | – |

Generation quality (5000 generated vs. 5000 real):

| Dataset | FID ↓ | IS ↑ | Improved Precision ↑ | Improved Recall ↑ |
| --- | --- | --- | --- | --- |
| LUAD-HistoSeg | 19.57 | 2.76 | 0.83 | 0.62 |
| BCSS-WSSS | 23.67 | 2.26 | 0.78 | 0.50 |

## Repository layout

```
PM2GP/
├── diffusion/             # MDP pretraining (core contribution)
│   ├── unet.py            #   U-Net backbone + DiffusionNet wrapper
│   ├── schedules.py       #   quadratic β schedule & q/p sampling
│   ├── mgdl.py            #   MGDL + MDP loss
│   ├── dataset.py         #   image/mask dataset (RGB palette decoding)
│   ├── early_stopping.py
│   └── train_mdp.py       #   MDP training entry point
├── racd/                  # generative augmentation (core contribution)
│   ├── generate.py        #   rare-class conditional sampling + mask blending
│   └── gen_metrics.py     #   FID / IS / Improved Precision & Recall
├── segmentation/          # downstream segmentation (core contribution)
│   ├── segnet.py          #   segmentation network (shared U-Net backbone)
│   ├── resl.py            #   RESL (α·CE + (1-α)·Dice)
│   ├── train_seg.py       #   encoder transfer + RESL training
│   ├── evaluator.py       #   Evaluator / AJI / HD95
│   └── evaluate.py        #   foreground mIoU / PA / Dice
├── stage1_pseudo_masks/   # standard CAM pipeline (NOT a contribution)
├── scripts/               # small utilities (CAM overlay fusion)
├── legacy/                # PSPNet baseline & experimental losses (reference only)
└── checkpoints/           # place trained weights here
```

## Setup

Tested with Python 3.11.11, PyTorch 2.4.0, single RTX A6000 (48 GB).

```bash
pip install -r requirements.txt
```

## Data preparation

- [LUAD-HistoSeg](https://github.com/DukeZHB/LUAD-HistoSeg):
  17,285 patches (16,678 train / 300 val / 307 test), foreground classes
  TE / NEC / LYM / TAS.
- BCSS-WSSS: 31,826 patches (23,422 train / 3,418 val / 4,986 test),
  foreground classes TUM / STR / LYM / NEC.

Masks are stored as RGB palette PNGs:
TE (205,51,51) · NEC (0,255,0) · LYM (65,105,225) · TAS (255,165,0) ·
background (255,255,255).

## Usage

### 0. (Supporting) pseudo-mask generation — standard CAM pipeline

```bash
python -m stage1_pseudo_masks.train_classifier \
    --dataset luad \
    --trainroot ./datasets/LUAD-HistoSeg/train/img/ \
    --testroot  ./datasets/LUAD-HistoSeg/test/ \
    --pmroot    ./datasets/LUAD-HistoSeg/train_pseudo_mask/ \
    --weights   ./init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
```

### 1. MDP — mask-conditioned diffusion pretraining

```bash
python -m diffusion.train_mdp \
    --image_dir ./datasets/LUAD-HistoSeg/train/img/ \
    --mask_dir  ./datasets/LUAD-HistoSeg/train_pseudo_mask/ \
    --exp_name  mdp_luad \
    --epochs 100 --batch_size 8 --lr 1e-4 --lambda_reg 0.5
```

### 2. RACD — rare-class conditional augmentation

```bash
python -m racd.generate \
    --checkpoint ./checkpoints/mdp_luad/best.pth \
    --mask_dir   ./datasets/LUAD-HistoSeg/train_pseudo_mask/ \
    --out_dir    ./datasets/LUAD-HistoSeg/racd/ \
    --num_generate 5000 --rare_classes 1 2 --blend_ratio 0.5
```

(`--rare_classes` takes the palette indices of the rare classes of the
dataset; 1 = NEC, 2 = LYM for LUAD-HistoSeg.)

Generation quality (FID / IS / Improved Precision & Recall):

```bash
python -m racd.gen_metrics \
    --gen_dir  ./datasets/LUAD-HistoSeg/racd/img/ \
    --real_dir ./datasets/LUAD-HistoSeg/train/img/
```

### 3. Segmentation with generative priors + RESL

```bash
python -m segmentation.train_seg \
    --image_dir ./datasets/LUAD-HistoSeg/train/img/ \
    --mask_dir  ./datasets/LUAD-HistoSeg/train_pseudo_mask/ \
    --synth_image_dir ./datasets/LUAD-HistoSeg/racd/img/ \
    --synth_mask_dir  ./datasets/LUAD-HistoSeg/racd/mask/ \
    --pretrained_ckpt ./checkpoints/mdp_luad/best.pth \
    --exp_name seg_luad --epochs 100 --batch_size 8 --lr 1e-4 --alpha 0.5
```

### 4. Evaluation

```bash
python -m segmentation.evaluate \
    --checkpoint ./checkpoints/seg_luad/best.pth \
    --image_dir  ./datasets/LUAD-HistoSeg/test/img/ \
    --mask_dir   ./datasets/LUAD-HistoSeg/test/mask/
```

## Notes

- The diffusion U-Net and the segmentation network share the same
  backbone definition (`diffusion/unet.py`); the segmentation encoder is
  initialized from the MDP checkpoint with the 4→3 channel input
  adaptation handled automatically by `segmentation/train_seg.py`.
- The PSPNet baseline and experimental losses live in
  [`legacy/`](legacy/README.md) and are only needed to reproduce the
  baselines.

## Citation

If you find this code useful, please cite:

```bibtex
@article{zhao2026pm2gp,
  title   = {From Pseudo Masks to Generative Priors: A Diffusion-Based
             Framework for Weakly Supervised Histopathology Image
             Segmentation},
  author  = {Zhao, Hongbo and Zhang, Miao and Chen, Yifei and Fu, Yao and Shen, Yi},
  year    = {2026}
}
```
