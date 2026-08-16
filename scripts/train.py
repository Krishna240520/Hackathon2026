"""
train.py

Generic image-classification trainer. Run it once per stage:

    python scripts/train.py --data_dir data/splits/species       --output models/species_model.pt
    python scripts/train.py --data_dir data/splits/breed_cattle  --output models/cattle_breed_model.pt
    python scripts/train.py --data_dir data/splits/breed_buffalo --output models/buffalo_breed_model.pt

Expects data_dir/{train,val}/<class_name>/*.jpg (standard torchvision ImageFolder layout,
produced by data_prep.py). Saves a checkpoint containing the model weights, the
class_to_idx mapping, and the arch name, so inference.py can load it standalone.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import build_model, get_input_size

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(input_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        if train:
            optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Folder with train/ and val/ subfolders")
    ap.add_argument("--output", required=True, help="Path to save the trained checkpoint (.pt)")
    ap.add_argument("--arch", default="efficientnet_b0", choices=["efficientnet_b0", "resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=5, help="Early-stopping patience (epochs w/o val improvement)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    input_size = get_input_size(args.arch)
    train_tf, eval_tf = build_transforms(input_size)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)

    if train_ds.classes != val_ds.classes:
        raise ValueError("train/ and val/ folders must contain the same set of class subfolders")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    num_classes = len(train_ds.classes)
    print(f"Classes ({num_classes}): {train_ds.classes}")

    model = build_model(num_classes, arch=args.arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    epochs_without_improve = 0
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "num_classes": num_classes,
                "class_to_idx": train_ds.class_to_idx,
            }, output_path)
            print(f"  -> saved new best checkpoint (val_acc={val_acc:.4f}) to {output_path}")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                print(f"Early stopping: no val improvement for {args.patience} epochs")
                break

    # also drop a plain json of the label map next to the checkpoint for easy inspection
    ckpt = torch.load(output_path, map_location="cpu")
    with open(output_path.with_suffix(".labels.json"), "w") as f:
        json.dump(ckpt["class_to_idx"], f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}. Checkpoint: {output_path}")


if __name__ == "__main__":
    main()
