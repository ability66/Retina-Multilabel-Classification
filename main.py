
import argparse, os, math, time, json, random
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

import monai
from monai.networks.nets import DenseNet121
import timm
import torch.nn as nn
import torch.optim as optim

LABELS = ["DR","NORMAL","MH","ODC","TSLN","ARMD","DN","MYA","BRVO","ODP","CRVO","CNV","RS","ODE","LS","CSR","HTR","ASR","CRS","OTHER"]

class FundusCSVDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=512, is_train=False):
        self.df = pd.read_csv(csv_path)
        assert "path" in self.df.columns, "CSV must contain a 'path' column"
        self.img_dir = Path(img_dir)
        self.is_train = is_train
        self.labels = LABELS
        self.img_size = img_size

        # Albumentations-like transforms using OpenCV + torchvision
        # Keep it simple to avoid extra deps
        self.train_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
        self.val_tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fp = self.img_dir / str(row["path"])
        img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {fp}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tf = self.train_tf if self.is_train else self.val_tf
        img = tf(img)

        # Collect multi-label targets
        y = torch.tensor(row[self.labels].values.astype(np.float32))
        return img, y

def build_model(model_name, num_classes):
    if model_name.lower() == "densenet121":
        # MONAI DenseNet121 with ImageNet weights
        model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=num_classes, pretrained=True)
    else:
        # Use timm for other backbones (ImageNet-pretrained)
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        y_prob.append(probs)
        y_true.append(ys.numpy())
    y_true = np.vstack(y_true)
    y_prob = np.vstack(y_prob)

    # Threshold at 0.5 for F1 micro; use ROC-AUC for ranking
    try:
        auc_macro = roc_auc_score(y_true, y_prob, average="macro")
        auc_micro = roc_auc_score(y_true, y_prob, average="micro")
    except ValueError:
        auc_macro, auc_micro = float("nan"), float("nan")
    y_pred = (y_prob >= 0.5).astype(np.float32)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    return {"auc_macro": auc_macro, "auc_micro": auc_micro, "f1_micro": f1_micro}, y_pred

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(LABELS)

    # Datasets & loaders
    train_ds = FundusCSVDataset(args.train_csv, args.data_dir, img_size=args.img_size, is_train=True)
    val_ds = FundusCSVDataset(args.val_csv, args.data_dir, img_size=args.img_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = build_model(args.model, num_classes).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = -1
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, ys in train_loader:
            imgs = imgs.to(device)
            ys = ys.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        metrics, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"AUC(macro)={metrics['auc_macro']:.4f} | AUC(micro)={metrics['auc_micro']:.4f} | "
              f"F1(micro)={metrics['f1_micro']:.4f}")

        if metrics["auc_micro"] > best_auc:
            best_auc = metrics["auc_micro"]
            torch.save({"state_dict": model.state_dict(),
                        "model": args.model,
                        "num_classes": num_classes,
                        "img_size": args.img_size},
                       "checkpoints/best.pt")
            print(f"✓ Saved new best checkpoint with AUC(micro)={best_auc:.4f}")

def eval_only(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(LABELS)
    model = build_model(args.model, num_classes).to(device)

    assert args.ckpt and os.path.exists(args.ckpt), "Provide a valid --ckpt"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    ds = FundusCSVDataset(args.test_csv, args.data_dir, img_size=args.img_size, is_train=False)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    probs_list = []
    with torch.no_grad():
        for imgs, _ in dl:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()  # [B, K]
            probs_list.append(probs)
    y_prob = np.vstack(probs_list)                     # [N, K]
    y_pred = (y_prob >= 0.5).astype(int)

    df = pd.read_csv(args.val_csv)
    for i, lab in enumerate(LABELS):
        if lab not in df.columns:
            df[lab] = 0
        df[lab] = y_pred[:, i].astype(int)

    out_csv = getattr(args, "out_csv", None) or args.val_csv
    df.to_csv(out_csv, index=False)
    print(f"✓ Wrote multi-label predictions (threshold={0.5}) to {out_csv}")
    print(df.head().to_string(index=False))
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--train-csv", type=str, default=None)
    p.add_argument("--val-csv", type=str, required=True)
    p.add_argument("--test-csv", type=str, default="test_sample.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", type=str, default="densenet121",
                   help="densenet121 (MONAI) or any timm model name, e.g., resnet50")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--out-csv", type=str, default="pred_test_sample.csv", help="Write predictions to this CSV; default is --val-csv (in-place).")
    p.add_argument("--ckpt", type=str, default=None)
    args = p.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        if args.train_csv is None:
            raise SystemExit("--train-csv is required for training (omit only in --eval-only mode)")
        train(args)

if __name__ == "__main__":
    main()
