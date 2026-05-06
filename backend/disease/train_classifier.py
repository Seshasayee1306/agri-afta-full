import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def parse_args():
    p = argparse.ArgumentParser(description="Train disease classifier from PlantVillage-style dataset")
    p.add_argument("--data-dir", required=True, help="Dataset root with class subfolders")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--out-model", default="backend/disease_model.pt")
    p.add_argument("--out-classes", default="backend/disease_class_names.json")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def make_loaders(data_dir: str, batch_size: int, val_split: float, num_workers: int):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(data_dir, transform=tfm)

    if len(ds) < 10:
        raise ValueError("Dataset too small. Please provide a proper PlantVillage extraction.")

    val_size = max(1, int(len(ds) * val_split))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return ds.classes, train_loader, val_loader


def build_model(num_classes: int):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return correct / max(total, 1)


def main():
    args = parse_args()
    print(f"loading dataset from: {args.data_dir}")
    classes, train_loader, val_loader = make_loaders(
        args.data_dir, args.batch_size, args.val_split, args.num_workers
    )
    print(f"dataset ready: classes={len(classes)} train_batches={len(train_loader)} val_batches={len(val_loader)}")

    device = pick_device()
    print(f"using device: {device}")
    model = build_model(len(classes)).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if step % 50 == 0 or step == len(train_loader):
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={loss.item():.4f}")

        val_acc = evaluate(model, val_loader, device)
        avg_loss = running_loss / max(len(train_loader), 1)
        print(f"epoch={epoch} loss={avg_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    Path(os.path.dirname(args.out_model) or ".").mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args.out_classes) or ".").mkdir(parents=True, exist_ok=True)

    torch.save(best_state, args.out_model)
    with open(args.out_classes, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print(f"saved model: {args.out_model}")
    print(f"saved classes: {args.out_classes}")
    print(f"best_val_acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
