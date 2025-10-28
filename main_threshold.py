
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

# ----------------------------
# Dataset
# ----------------------------
class FundusCSVDataset(Dataset):
    def __init__(self, csv_path, img_dir, img_size=512, is_train=False):
        self.df = pd.read_csv(csv_path)
        assert "path" in self.df.columns, "CSV must contain a 'path' column"
        self.img_dir = Path(img_dir)
        self.is_train = is_train
        self.labels = LABELS
        self.img_size = img_size

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

        y = torch.tensor(row[self.labels].values.astype(np.float32))
        return img, y

# ----------------------------
# Model + Learnable Threshold
# ----------------------------
def build_model(model_name, num_classes):
    if model_name.lower() == "densenet121":
        model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=num_classes, pretrained=True)
    else:
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model

class LearnableThreshold(nn.Module):
    """
    Per-class learnable decision boundary tau_i.
    We adjust logits as: (logits - tau) / T, then use BCEWithLogitsLoss on adjusted logits.
    """
    def __init__(self, num_classes, init_value=0.0, temperature=1.0):
        super().__init__()
        self.tau = nn.Parameter(torch.full((num_classes,), float(init_value)))
        self.T = float(temperature)

    def forward(self, logits):
        # returns adjusted logits for BCEWithLogitsLoss
        # shape: [B, K] - tau will broadcast across batch
        return (logits - self.tau) / self.T

# ----------------------------
# Evaluation
# ----------------------------
@torch.no_grad()
def evaluate(model, threshold_layer, loader, device, sweep_thresholds=True):
    model.eval()
    if threshold_layer is not None:
        threshold_layer.eval()

    y_true, y_prob_raw = [], []
    for imgs, ys in loader:
        imgs = imgs.to(device)
        logits = model(imgs)                     # raw logits
        probs_raw = torch.sigmoid(logits).cpu().numpy()  # use RAW probs for AUC (submission-consistent)
        y_prob_raw.append(probs_raw)
        y_true.append(ys.numpy())
    y_true = np.vstack(y_true)
    y_prob_raw = np.vstack(y_prob_raw)

    # AUC on RAW probabilities (consistent with test-time & leaderboard)
    try:
        auc_macro = roc_auc_score(y_true, y_prob_raw, average="macro")
        auc_micro = roc_auc_score(y_true, y_prob_raw, average="micro")
    except ValueError:
        auc_macro, auc_micro = float("nan"), float("nan")

    # F1 with per-class threshold sweep (optional, for monitoring)
    if sweep_thresholds:
        best_thrs = []
        for i in range(y_true.shape[1]):
            best_t, best_f1 = 0.5, 0.0
            # coarse sweep 0.1~0.9
            for t in np.linspace(0.1, 0.9, 17):
                y_pred_i = (y_prob_raw[:, i] >= t).astype(int)
                f1_i = f1_score(y_true[:, i], y_pred_i, zero_division=0)
                if f1_i > best_f1:
                    best_t, best_f1 = t, f1_i
            best_thrs.append(best_t)
        y_pred = (y_prob_raw >= np.array(best_thrs)[None, :]).astype(int)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    else:
        y_pred = (y_prob_raw >= 0.5).astype(int)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        best_thrs = [0.5] * y_true.shape[1]

    return {
        "auc_macro": auc_macro,
        "auc_micro": auc_micro,
        "f1_micro": f1_micro,
        "best_thrs": np.array(best_thrs),
    }

# ----------------------------
# Train
# ----------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(LABELS)

    # Datasets & loaders
    train_ds = FundusCSVDataset(args.train_csv, args.data_dir, img_size=args.img_size, is_train=True)
    val_ds = FundusCSVDataset(args.val_csv, args.data_dir, img_size=args.img_size, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Model + learnable thresholds
    model = build_model(args.model, num_classes).to(device)
    thr = LearnableThreshold(num_classes, init_value=args.tau_init, temperature=args.temperature).to(device)

    # Loss on ADJUSTED logits
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        list(model.parameters()) + list(thr.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = -1.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train(); thr.train()
        running_loss = 0.0
        for imgs, ys in train_loader:
            imgs = imgs.to(device)
            ys = ys.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            adj_logits = thr(logits)                   # (logits - tau) / T
            loss = criterion(adj_logits, ys)          # BCEWithLogitsLoss expects logits
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Evaluate with RAW sigmoid probabilities (submission-consistent)
        metrics = evaluate(model, thr, val_loader, device, sweep_thresholds=True)
        tau_vals = thr.tau.detach().cpu().numpy()
        print(f"Epoch {epoch:03d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"AUC(macro)={metrics['auc_macro']:.4f} | AUC(micro)={metrics['auc_micro']:.4f} | "
              f"F1(micro)={metrics['f1_micro']:.4f}")
        # Print per-class taus with 3 decimals
        taus_str = np.array2string(np.round(tau_vals, 3), separator=", ")
        print(f"Learned per-class tau (3dp): {taus_str}")
        # Also print threshold sweep result summary
        print(f"Per-class F1-opt thresholds (3dp): {np.round(metrics['best_thrs'], 3)}")

        if float(metrics["auc_micro"]) > best_auc:
            best_auc = float(metrics["auc_micro"])
            torch.save({
                "state_dict": model.state_dict(),
                "tau": thr.tau.detach().cpu(),     # save learned taus
                "model": args.model,
                "num_classes": num_classes,
                "img_size": args.img_size,
                "temperature": args.temperature,
            }, "checkpoints/best.pt")
            print(f"✓ Saved new best checkpoint with AUC(micro)={best_auc:.4f}")

# ----------------------------
# Eval-only (submission: probabilities, no tau applied)
# ----------------------------
@torch.no_grad()
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
    for imgs, _ in dl:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()  # NOTE: raw probabilities (no tau), for AUC submission
        probs_list.append(probs)

    y_prob = np.vstack(probs_list)  # [N, K]

    # 读取 CSV 并写入概率而非0/1
    df = pd.read_csv(args.val_csv)
    for i, lab in enumerate(LABELS):
        df[lab] = y_prob[:, i]

    out_csv = getattr(args, "out_csv", None) or args.val_csv
    df.to_csv(out_csv, index=False, float_format="%.6f")

    print(f"✓ Wrote probability predictions (for AUC evaluation) to {out_csv}")
    print("示例输出（前5行）：")
    print(df.head().to_string(index=False))
    return df

# ----------------------------
# Main
# ----------------------------
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
    p.add_argument("--out-csv", type=str, default="pred_test_sample.csv",
                   help="Write predictions to this CSV; default is --val-csv (in-place).")
    p.add_argument("--ckpt", type=str, default=None)
    # learnable-threshold args
    p.add_argument("--temperature", type=float, default=1.0, help="Temperature T in (logits - tau)/T")
    p.add_argument("--tau-init", type=float, default=0.0, help="Initial value for per-class tau")

    args = p.parse_args()

    if args.eval_only:
        eval_only(args)
    else:
        if args.train_csv is None:
            raise SystemExit("--train-csv is required for training (omit only in --eval-only mode)")
        train(args)

if __name__ == "__main__":
    main()


    '''
    train:
    python main_threshold.py \
    --data-dir ./images \
    --train-csv ./train.csv \
    --val-csv ./val.csv \
    --epochs 30 --batch-size 32 --lr 3e-4 \
    --model densenet121 \
    --temperature 1.0 --tau-init 0.0 \
    2>&1 | tee log.txt
    

    评测/导出概率 CSV（不加载 τ_i）：
    python main_threshold.py \
    --data-dir ./images \
    --val-csv ./test_sample.csv \
    --eval-only \
    --ckpt checkpoints/best.pt \
    --out-csv ./pred_test_sample.csv \
    2>&1 | tee log.txt
    '''