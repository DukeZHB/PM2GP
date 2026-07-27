"""CAM inference and pseudo-mask export for the Stage-1 classifier.

The classifier exposes three CAM branches (two intermediate ``ic1`` /
``ic2`` convolutions and the final ``fc8`` head, see ``resnet38_cls.py``).
The per-image CAM is the weighted sum of the three branches:

    CAM = w1 * cam1 + w2 * cam2 + w3 * cam3

with dataset-specific fusion weights below. The fused CAM is masked by
the thresholded image-level prediction, combined with a heuristic
background score, and argmaxed into the final pseudo-mask label map.

Note: the original experiment code used 0.06 for the middle branch in
the evaluation path but 0.05 in the mask-export path for LUAD; both are
unified here to the evaluation value (0.06). The paper itself does not
report these weights - Stage 1 follows the standard CAM pipeline and is
not a contribution of this work.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from stage1_pseudo_masks import cam_utils, iou_metrics, pyutils
from stage1_pseudo_masks.datasets import Stage1_InferDataset

cudnn.enabled = True

# dataset-specific CAM fusion weights (cam1, cam2, cam3)
CAM_FUSION_WEIGHTS = {
    'luad': (0.47, 0.06, 0.47),
    'bcss': (0.11, 0.78, 0.11),
}

# threshold on the sigmoid image-level prediction
PRED_THRESHOLDS = {
    'luad': 0.3,
    'bcss': 0.5,
}

# 5-class palettes (4 foreground classes + background) used when saving
# pseudo masks as palette-indexed PNGs
PALETTES = {
    'luad': [205, 51, 51, 0, 255, 0, 65, 105, 225, 255, 165, 0, 255, 255, 255],
    'bcss': [255, 0, 0, 0, 255, 0, 0, 0, 255, 153, 0, 255, 255, 255, 255],
}


def _fuse_cams(cam1, cam2, cam3, dataset):
    w1, w2, w3 = CAM_FUSION_WEIGHTS[dataset]
    return w1 * cam1 + w2 * cam2 + w3 * cam3


def infer(model, dataroot, n_class, args):
    """Evaluate CAM-derived label maps against the test ground truth."""
    model.eval()
    model = model.cuda()
    cam_list, gt_list = [], []
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot, 'img/'),
                                        transform=transform)
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=8,
                                   pin_memory=False)
    thr = PRED_THRESHOLDS[args.dataset]
    # BCSS pseudo masks do not use the background channel
    use_bg = (args.dataset != 'bcss')

    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]
        img_path = os.path.join(dataroot + 'img/' + img_name + '.png')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img, thr=thr):
            with torch.no_grad():
                img = img.cuda()
                cam1, cam2, cam3, y = model.forward_cam(img)
                y = y.cpu().detach().numpy().tolist()[0]
                label = torch.tensor([1.0 if j > thr else 0.0 for j in y])
                cam3 = F.interpolate(cam3, orig_img_size, mode='bilinear',
                                     align_corners=False)[0]
                cam1 = F.interpolate(cam1, orig_img_size, mode='bilinear',
                                     align_corners=False)[0]
                cam2 = F.interpolate(cam2, orig_img_size, mode='bilinear',
                                     align_corners=False)[0]
                cam = _fuse_cams(cam1, cam2, cam3, args.dataset)
                cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()
                return cam, label

        thread_pool = pyutils.BatchThreader(
            _work, list(enumerate(img_list.unsqueeze(0))),
            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        sum_cam_min, sum_cam_max = np.min(sum_cam), np.max(sum_cam)
        if np.isclose(sum_cam_max, sum_cam_min, atol=1e-8):
            norm_cam = np.zeros_like(sum_cam)
        else:
            norm_cam = (sum_cam - sum_cam_min) / (sum_cam_max - sum_cam_min)
        cam_dict = cam_utils.cam_npy_to_cam_dict(norm_cam, label)

        cam_score, bg_score = cam_utils.dict2npy(cam_dict, label, orig_img)
        if use_bg:
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
            seg_map = cam_utils.cam_npy_to_label_map(bgcam_score)
        else:
            seg_map = cam_utils.cam_npy_to_label_map(cam_score)

        if iter % 100 == 0:
            print(iter)
        cam_list.append(seg_map)
        gt_map_path = os.path.join(os.path.join(dataroot, 'mask/'), img_name + '.png')
        gt_list.append(np.array(Image.open(gt_map_path)))
    return iou_metrics.scores(gt_list, cam_list, n_class=n_class)


def get_mask(model, dataroot, args, save_path):
    """Export the fused pseudo mask of every training image as a PNG."""
    os.makedirs(save_path, exist_ok=True)
    palette = PALETTES[args.dataset]
    thr = PRED_THRESHOLDS[args.dataset]
    use_bg = (args.dataset != 'bcss')

    model.eval()
    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot),
                                        transform=transform)
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=8,
                                   pin_memory=False)
    model = model.cuda()

    for iter, (img_name, img_list) in enumerate(infer_data_loader):
        img_name = img_name[0]
        img_path = os.path.join(dataroot + img_name + '.png')
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img, thr=thr):
            with torch.no_grad():
                img = img.cuda()
                cam1, cam2, cam3, y = model.forward_cam(img)
                y = y.cpu().detach().numpy().tolist()[0]
                label = torch.tensor([1.0 if j > thr else 0.0 for j in y])
                cam3 = F.interpolate(cam3, orig_img_size, mode='bilinear',
                                     align_corners=False)[0]
                cam1 = F.interpolate(cam1, orig_img_size, mode='bilinear',
                                     align_corners=False)[0]
                cam2 = F.interpolate(cam2, orig_img_size, mode='bilinear',
                                     align_corners=False)[0]

                cam1_np = cam1.cpu().numpy()
                cam2_np = cam2.cpu().numpy()
                cam3_np = cam3.cpu().numpy()

                generate_cam_heatmap(cam1_np, orig_img_size, save_path, img_name,
                                     "cam1", dataroot)
                generate_cam_heatmap(cam2_np, orig_img_size, save_path, img_name,
                                     "cam2", dataroot)
                generate_cam_heatmap(cam3_np, orig_img_size, save_path, img_name,
                                     "cam3", dataroot)

                label_np = label.clone().view(4, 1, 1).numpy()
                cam1_masked = cam1_np * label_np
                cam2_masked = cam2_np * label_np
                cam3_masked = cam3_np * label_np

                # per-branch pseudo masks (useful for visual inspection)
                generate_mask_from_cam(cam1_masked, label, orig_img, palette,
                                       save_path, img_name, suffix="_cam1",
                                       use_bg=use_bg)
                generate_mask_from_cam(cam2_masked, label, orig_img, palette,
                                       save_path, img_name, suffix="_cam2",
                                       use_bg=use_bg)
                generate_mask_from_cam(cam3_masked, label, orig_img, palette,
                                       save_path, img_name, suffix="_cam3",
                                       use_bg=use_bg)

                cam = _fuse_cams(cam1, cam2, cam3, args.dataset)
                cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()
                return cam, label

        thread_pool = pyutils.BatchThreader(
            _work, list(enumerate(img_list.unsqueeze(0))),
            batch_size=12, prefetch_size=0, processes=8)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        norm_cam = (sum_cam - np.min(sum_cam)) / (np.max(sum_cam) - np.min(sum_cam))
        cam_dict = cam_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = cam_utils.dict2npy(cam_dict, label, orig_img)

        if use_bg:
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
            seg_map = cam_utils.cam_npy_to_label_map(bgcam_score)
        else:
            seg_map = cam_utils.cam_npy_to_label_map(cam_score)

        visualimg = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        save_file = os.path.join(save_path, img_name + '.png')
        visualimg.save(save_file, format='PNG')
        if iter % 100 == 0:
            print(iter)


def generate_mask_from_cam(cam, label, orig_img, palette, save_dir, img_name,
                           suffix="", use_bg=True):
    """Build and save a pseudo mask from a single CAM branch."""
    cam_min, cam_max = cam.min(), cam.max()
    if np.isclose(cam_max, cam_min, atol=1e-8):
        norm_cam = np.zeros_like(cam)
    else:
        norm_cam = (cam - cam_min) / (cam_max - cam_min)

    label_np = label.cpu().numpy() if isinstance(label, torch.Tensor) else np.array(label)
    cam_dict = cam_utils.cam_npy_to_cam_dict(norm_cam, label_np)
    cam_score, bg_score = cam_utils.dict2npy(cam_dict, label_np, orig_img)

    if use_bg:
        bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        seg_map = cam_utils.cam_npy_to_label_map(bgcam_score)
    else:
        seg_map = cam_utils.cam_npy_to_label_map(cam_score)

    mask_dir = os.path.join(save_dir, "per_layer_masks")
    os.makedirs(mask_dir, exist_ok=True)
    save_file = os.path.join(mask_dir, img_name + suffix + '.png')
    visualimg = Image.fromarray(seg_map.astype(np.uint8), "P")
    visualimg.putpalette(palette)
    visualimg.save(save_file, format='PNG')


def generate_cam_heatmap(cam_np, orig_img_size, save_path, img_name, cam_level,
                         dataroot):
    """Save per-class CAM heatmaps and fused heatmaps (with/without overlay)."""
    heatmap_single_dir = os.path.join(save_path, f"cam_heatmaps/{cam_level}/single_cls")
    heatmap_single_overlay_dir = os.path.join(save_path, f"cam_heatmaps/{cam_level}/single_cls_overlay")
    heatmap_fusion_dir = os.path.join(save_path, f"cam_heatmaps/{cam_level}/fusion_cls")
    heatmap_fusion_overlay_dir = os.path.join(save_path, f"cam_heatmaps/{cam_level}/fusion_cls_overlay")
    for d in (heatmap_single_dir, heatmap_single_overlay_dir,
              heatmap_fusion_dir, heatmap_fusion_overlay_dir):
        os.makedirs(d, exist_ok=True)

    orig_img_path = os.path.join(dataroot, f"{img_name}.png")
    orig_img = cv2.imread(orig_img_path)
    if orig_img is None:
        raise FileNotFoundError(f"original image not found: {orig_img_path}")
    orig_img = cv2.resize(orig_img, (orig_img_size[1], orig_img_size[0]))

    for cls_idx in range(cam_np.shape[0]):
        cam_single = cam_np[cls_idx]
        cam_single = cv2.resize(cam_single, (orig_img_size[1], orig_img_size[0]))
        cam_single = (cam_single - cam_single.min()) / (
                cam_single.max() - cam_single.min() + 1e-8)
        cam_single = (cam_single * 255).astype(np.uint8)

        heatmap_single = cv2.applyColorMap(cam_single, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(heatmap_single_dir,
                                 f"{img_name}_cls{cls_idx}.png"), heatmap_single)
        overlay = cv2.addWeighted(orig_img, 0.5, heatmap_single, 0.5, 0)
        cv2.imwrite(os.path.join(heatmap_single_overlay_dir,
                                 f"{img_name}_cls{cls_idx}_overlay.png"), overlay)

    cam_fusion = np.sum(cam_np, axis=0)
    cam_fusion = cv2.resize(cam_fusion, (orig_img_size[1], orig_img_size[0]))
    cam_fusion = (cam_fusion - cam_fusion.min()) / (
            cam_fusion.max() - cam_fusion.min() + 1e-8)
    cam_fusion = (cam_fusion * 255).astype(np.uint8)

    heatmap_fusion = cv2.applyColorMap(cam_fusion, cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(heatmap_fusion_dir,
                             f"{img_name}_fusion_all_cls.png"), heatmap_fusion)
    overlay = cv2.addWeighted(orig_img, 0.5, heatmap_fusion, 0.5, 0)
    cv2.imwrite(os.path.join(heatmap_fusion_overlay_dir,
                             f"{img_name}_fusion_all_cls_overlay.png"), overlay)
