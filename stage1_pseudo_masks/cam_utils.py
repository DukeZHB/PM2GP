"""Helpers that turn raw CAM arrays into pseudo-mask label maps."""

import cv2
import numpy as np
from skimage import morphology


def cam_npy_to_cam_dict(cam_np, label):
    """Keep only the CAM channels whose image-level label is active."""
    cam_dict = {}
    for i in range(len(label)):
        if label[i] > 1e-5:
            cam_dict[i] = cam_np[i]
    return cam_dict


def dict2npy(cam_dict, gt_label, orig_img):
    """Re-assemble a full 4-channel CAM array and a background score map."""
    gt_cat = np.where(gt_label == 1)[0]
    bg_score = [gen_bg_mask(orig_img)]
    if len(gt_cat) == 0:
        # no foreground class predicted: empty CAM plus background only
        orig_img_size = orig_img.shape[:2]
        cam_npy = np.zeros((4, orig_img_size[0], orig_img_size[1]))
        return cam_npy, bg_score
    orig_img_size = cam_dict[gt_cat[0]].shape
    cam_npy = np.zeros((4, orig_img_size[0], orig_img_size[1]))
    for gt in gt_cat:
        cam_npy[gt] = cam_dict[gt]
    return cam_npy, bg_score


def gen_bg_mask(orig_img):
    """Heuristic background map: bright, near-white regions become background."""
    img_array = np.array(orig_img).astype(np.uint8)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    binary = np.uint8(binary)
    dst = morphology.remove_small_objects(binary == 255, min_size=50, connectivity=1)
    bg_mask = np.zeros(orig_img.shape[:2])
    bg_mask[dst == True] = 1.000001
    return bg_mask


def cam_npy_to_label_map(cam_npy):
    """Argmax over the channel dimension -> per-pixel label map."""
    seg_map = cam_npy.transpose(1, 2, 0)
    return np.asarray(np.argmax(seg_map, axis=2), dtype=np.int64)
