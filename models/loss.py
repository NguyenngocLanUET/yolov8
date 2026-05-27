import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.anchors import make_anchors
from utils.bbox import bbox_iou


class YOLOLoss(nn.Module):

    def __init__(self, num_classes=5, reg_max=16, device='cuda'):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.device = device

        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.proj = torch.arange(reg_max, dtype=torch.float32, device=device)

    def forward(self, preds, feats, targets):

        cls_preds, reg_preds = preds

        batch_size, num_anchors, _ = cls_preds.shape

        anchor_points, stride_tensor = make_anchors(feats)

        pred_bboxes = self.decode(reg_preds, anchor_points, stride_tensor)
        pred_scores = torch.sigmoid(cls_preds)

        gt_labels = targets[:, :, 0].long()
        gt_bboxes = targets[:, :, 1:5]
        gt_mask = targets[:, :, 5] > 0

        target_scores = torch.zeros(
            (batch_size, num_anchors, self.num_classes),
            device=self.device
        )

        fg_mask = torch.zeros((batch_size, num_anchors), dtype=torch.bool, device=self.device)

        # simplified assignment (kept same structure)
        for i in range(batch_size):
            if not gt_mask[i].any():
                continue

            gt_b = gt_bboxes[i][gt_mask[i]]
            gt_l = gt_labels[i][gt_mask[i]]

            ious = bbox_iou(pred_bboxes[i].unsqueeze(1), gt_b.unsqueeze(0), CIoU=False)

            cls_scores = pred_scores[i][:, gt_l]

            metrics = (cls_scores ** 0.5) * (ious ** 6)

            best_gt = metrics.argmax(dim=1)

            fg_mask[i] = metrics.max(dim=1).values > 0

            idx = fg_mask[i]

            target_scores[i, idx, gt_l[best_gt[idx]]] = 1.0

        loss_cls = self.bce(cls_preds, target_scores).mean()

        loss_iou = torch.tensor(0.0, device=self.device)
        loss_dfl = torch.tensor(0.0, device=self.device)

        total = loss_cls + 2.5 * loss_iou + 0.5 * loss_dfl

        return total, loss_cls, loss_iou, loss_dfl

    def decode(self, reg_preds, anchor_points, stride_tensor):
        b, a, _ = reg_preds.shape
        reg_preds = reg_preds.view(b, a, 4, self.reg_max)

        dist = F.softmax(reg_preds, dim=-1)
        proj = self.proj

        ltrb = torch.sum(dist * proj, dim=-1) * stride_tensor

        lt, rb = torch.chunk(ltrb, 2, dim=-1)

        return torch.cat([anchor_points - lt, anchor_points + rb], dim=-1)