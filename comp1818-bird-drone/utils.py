"""Shared geometry and deterministic IoU matching utilities."""
from scipy.optimize import linear_sum_assignment
import numpy as np


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    return ((xc - w / 2) * img_w, (yc - h / 2) * img_h,
            (xc + w / 2) * img_w, (yc + h / 2) * img_h)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    area_a = max(0, ax2-ax1)*max(0, ay2-ay1)
    area_b = max(0, bx2-bx1)*max(0, by2-by1)
    inter = max(0, min(ax2,bx2)-max(ax1,bx1))*max(0, min(ay2,by2)-max(ay1,by1))
    den = area_a + area_b - inter
    return inter/den if den > 0 else 0.0


def match(gt_boxes, pred_boxes, iou_thr=0.5):
    """Maximum-total-IoU Hungarian assignment; discard matches below threshold."""
    if not gt_boxes or not pred_boxes:
        return []
    scores = np.array([[iou(g,p) for p in pred_boxes] for g in gt_boxes])
    rows, cols = linear_sum_assignment(1-scores)
    return [(int(r),int(c)) for r,c in zip(rows,cols) if scores[r,c] >= iou_thr]
