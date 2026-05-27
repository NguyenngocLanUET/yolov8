import os
import json
import argparse
import torch
import cv2

from torchvision.ops import nms

from models.yolov8 import YOLOv8
from utils.anchors import make_anchors


# -----------------------------
# POST PROCESS (NMS + decode)
# -----------------------------
def post_process(preds, feats, conf_thres=0.25, iou_thres=0.5, reg_max=16):

    cls_preds, reg_preds = preds
    device = cls_preds.device

    anchor_points, stride_tensor = make_anchors(feats)

    # decode boxes
    b, a, _ = reg_preds.shape
    reg_preds = reg_preds.view(b, a, 4, reg_max)

    proj = torch.arange(reg_max, device=device, dtype=torch.float32)

    dist = torch.softmax(reg_preds, dim=-1)
    ltrb = torch.sum(dist * proj, dim=-1) * stride_tensor

    lt, rb = torch.chunk(ltrb, 2, dim=-1)
    boxes = torch.cat([anchor_points - lt, anchor_points + rb], dim=-1)

    scores = torch.sigmoid(cls_preds)

    results = []

    for i in range(b):

        img_boxes = boxes[i]
        img_scores = scores[i]

        max_scores, labels = torch.max(img_scores, dim=-1)

        keep = max_scores > conf_thres

        if keep.sum() == 0:
            results.append({
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.long)
            })
            continue

        boxes_f = img_boxes[keep]
        scores_f = max_scores[keep]
        labels_f = labels[keep]

        # class-aware NMS
        offsets = labels_f * 4096.0
        keep_idx = nms(boxes_f + offsets.unsqueeze(1), scores_f, iou_thres)

        results.append({
            "boxes": boxes_f[keep_idx].cpu(),
            "scores": scores_f[keep_idx].cpu(),
            "labels": labels_f[keep_idx].cpu()
        })

    return results


# -----------------------------
# MAIN
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--weights", default="./models/best.pth")

    return parser.parse_args()


def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    model = YOLOv8().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    results = []

    image_files = sorted([
        f for f in os.listdir(args.image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    for img_name in image_files:

        img_path = os.path.join(args.image_dir, img_name)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():

            preds, feats = model(img_tensor)

            outputs = post_process(
                preds,
                feats,
                conf_thres=0.25,
                iou_thres=0.5
            )[0]

        boxes = outputs["boxes"]
        scores = outputs["scores"]
        labels = outputs["labels"]

        image_result = {
            "image_id": img_name,
            "boxes": []
        }

        for b, s, l in zip(boxes, scores, labels):

            x1, y1, x2, y2 = b.tolist()

            image_result["boxes"].append({
                "class": int(l),
                "bbox": [x1, y1, x2, y2],
                "score": float(s)
            })

        results.append(image_result)

    # save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()