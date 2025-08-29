
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json

from models.dataset import build_datasets

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running = 0.0
    ys, ps = [], []
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        running += loss.item() * x.size(0)
        ys.append(y.cpu())
        ps.append(out.argmax(1).cpu())
    ys = torch.cat(ys).numpy()
    ps = torch.cat(ps).numpy()
    return running / len(loader.dataset), ys, ps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--out", type=str, default="weights/model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = build_datasets(args.data_dir, args.img_size)
    import os
    nw = int(os.getenv("NUM_WORKERS", "0"))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=nw)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw)

    num_classes = len(train_ds.classes)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    best_val = 1e9
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, ys, ps = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}: train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": args.img_size,
            }, args.out)
            print(f"Saved best to {args.out}")

    print("\nValidation report:")
    print(classification_report(ys, ps, target_names=train_ds.classes))
    print("Confusion matrix:")
    print(confusion_matrix(ys, ps))

if __name__ == "__main__":
    main()
