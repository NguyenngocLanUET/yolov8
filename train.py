import os
import argparse
import torch
from torch.utils.data import DataLoader

from models.yolov8 import YOLOv8
from models.loss import YOLOLoss
from utils.dataset import YOLODataset, yolo_collate_fn
from utils.trainer import train_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_data')
    parser.add_argument('--val_data')
    parser.add_argument('--image_dir')
    parser.add_argument('--val_image_dir')
    parser.add_argument('--checkpoint_dir')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)

    return parser.parse_args()


def main():

    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_ds = YOLODataset(args.train_data, args.image_dir)
    val_ds = YOLODataset(args.val_data, args.val_image_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=yolo_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=yolo_collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = YOLOv8().to(device)
    criterion = YOLOLoss(device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    train_model(model, train_loader, val_loader, criterion, optimizer, args.checkpoint_dir, args.epochs, device)


if __name__ == "__main__":
    main()