# -*- coding: utf-8 -*-
"""
MNIST Pairwise Comparison (End-to-End, from scratch)
ResNet (small) + Extreme Augmentation + Windows CUDA + AMP + EMA
- 输入：28x56 的单通道图像（整图，不拆左右塔）；严格端到端二分类
- 训练：AdamW + 余弦退火（含warmup）+ 混合精度 + EMA + 梯度裁剪
- 增强：仿射/透视/模糊/对比度/擦除（为鲁棒的“极限增强”，但不使用会破坏笔画结构的夸张操作）
- 输出：保存 best.pt；可对 public/private 生成 CSV 预测（id,label）

数据要求：.npz 内像素为 [0,255] 的 uint8；键：
- train.npz: x (N,28,56), y (N,)
- val.npz:   x (N,28,56), y (N,)
- test_public.npz:  x (M,28,56), id (M,)
- test_private.npz: x (K,28,56), id (K,)
"""

import os, math, argparse, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 2025):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):
        self.sum = 0.0; self.cnt = 0
    def update(self, val, n=1):
        self.sum += float(val) * n; self.cnt += n
    @property
    def avg(self): return self.sum / max(1, self.cnt)


@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5) -> float:
    preds = (logits.sigmoid() >= thr).long()
    return (preds == targets.long()).float().mean().item()


class ModelEMA:
    """简洁的 EMA 封装：ema = decay*ema + (1-decay)*model"""
    def __init__(self, model: nn.Module, decay=0.999):
        self.ema = deepcopy_model(model).eval()
        for p in self.ema.parameters(): p.requires_grad_(False)
        self.decay = decay
    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        msd, esd = model.state_dict(), self.ema.state_dict()
        for k in esd.keys():
            esd[k].copy_(esd[k]*d + msd[k]*(1.0-d))

def deepcopy_model(model: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(model)


# -----------------------------
# Dataset
# -----------------------------
class NpzEnd2EndDataset(Dataset):
    """直接读取整幅 28x56；训练/验证统一接口"""
    def __init__(self, npz_path: str, is_train: bool, transform=None):
        pack = np.load(npz_path)
        self.x = pack["x"]        # (N,28,56) uint8 [0,255]
        self.is_train = is_train
        self.tf = transform
        if is_train or ("y" in pack):
            self.y = pack["y"].astype(np.int64)
            self.ids = None
        else:
            self.y = None
            self.ids = pack["id"]
        print(f"[Dataset] {os.path.basename(npz_path)} -> {len(self.x)} samples")

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        img = self.x[idx]  # (28,56)
        # 转 PIL 灰度图以便使用 torchvision 增强
        img = Image.fromarray(img.astype(np.uint8), mode="L")
        if self.tf: img = self.tf(img)
        if self.y is None:
            return img, -1, self.ids[idx]
        else:
            return img, int(self.y[idx]), None


# -----------------------------
# Augmentations
# -----------------------------
def get_transforms(train=True):
    # MNIST 常见均值方差
    mean, std = (0.1307,), (0.3081,)

    if train:
        tf = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=20, translate=(0.2, 0.2),
                    scale=(0.8, 1.2), shear=10, fill=0
                )
            ], p=0.95),
            transforms.RandomApply([
                transforms.RandomPerspective(distortion_scale=0.5, p=1.0)
            ], p=0.4),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.2),
            transforms.ColorJitter(brightness=0.4, contrast=0.4),  # 对灰度图有效
            transforms.ToTensor(),  # -> [0,1]
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        ])
    else:
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return tf


# -----------------------------
# Model: small-ResNet for 28x56
# -----------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.act   = nn.ReLU(inplace=True)
        self.down  = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
    def forward(self, x):
        identity = self.down(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.act(out + identity)
        return out


class ResNetSmall(nn.Module):
    """输入 1x28x56，输出二分类 logit"""
    def __init__(self, width=32, dropout=0.2):
        super().__init__()
        c1, c2, c3, c4 = width, width*2, width*4, width*4
        self.stem = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            BasicBlock(c1, c1, stride=1),
            BasicBlock(c1, c1, stride=1),
        )
        self.stage2 = nn.Sequential(
            BasicBlock(c1, c2, stride=2),   # 14x28
            BasicBlock(c2, c2, stride=1),
        )
        self.stage3 = nn.Sequential(
            BasicBlock(c2, c3, stride=2),   # 7x14
            BasicBlock(c3, c3, stride=1),
        )
        self.stage4 = nn.Sequential(
            BasicBlock(c3, c4, stride=2),   # 4x7（下取整）
            BasicBlock(c4, c4, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c4, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x); x = self.stage2(x); x = self.stage3(x); x = self.stage4(x)
        x = self.pool(x)
        logit = self.head(x).squeeze(1)
        return logit


# -----------------------------
# Train / Validate / Infer
# -----------------------------
def create_loaders(data_dir, batch_size, num_workers=4):
    tf_tr = get_transforms(train=True)
    tf_va = get_transforms(train=False)

    ds_tr = NpzEnd2EndDataset(os.path.join(data_dir, "train.npz"), is_train=True,  transform=tf_tr)
    ds_va = NpzEnd2EndDataset(os.path.join(data_dir, "val.npz"),   is_train=True,  transform=tf_va)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers,
                       pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                       pin_memory=True, drop_last=False)
    return dl_tr, dl_va


def build_optimizer_scheduler(model, steps_per_epoch, epochs, base_lr=1e-3, weight_decay=1e-4, warmup_epochs=5):
    opt = AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    # 余弦退火 + 线性 warmup
    cosine = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6)
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            return float(current_epoch + 1) / float(max(1, warmup_epochs))
        # 之后由 cosine 接管（我们组合一个 “两段式” 调度器）
        return 1.0
    warmup = LambdaLR(opt, lr_lambda)
    return opt, warmup, cosine


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[Device] Using {device.type}")

    dl_tr, dl_va = create_loaders(args.data_dir, args.batch_size, args.num_workers)
    model = ResNetSmall(width=32, dropout=0.2).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer, sched_warm, sched_cos = build_optimizer_scheduler(
        model, steps_per_epoch=len(dl_tr), epochs=args.epochs,
        base_lr=args.lr, weight_decay=args.weight_decay, warmup_epochs=args.warmup
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    ema = ModelEMA(model, decay=0.999)

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc, best_ep = 0.0, -1
    best_path = os.path.join(args.out_dir, "best.pt")

    print("[Train] Start training...")
    for epoch in range(args.epochs):
        model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(dl_tr, desc=f"Epoch {epoch+1}/{args.epochs}")

        for (x, y, _) in pbar:
            x = x.to(device, non_blocking=True)
            y = y.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA
            ema.update(model)

            loss_meter.update(loss.item(), x.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        # 调度器：先 warmup（前 warmup 轮），再 cosine
        if epoch < args.warmup: sched_warm.step()
        else: sched_cos.step()

        # 验证（用 EMA 权重评估更稳）
        val_acc = evaluate(ema.ema, dl_va, device)
        print(f"[Val] epoch={epoch+1} acc={val_acc:.4f} lr={optimizer.param_groups[0]['lr']:.3e}")

        if val_acc > best_acc:
            best_acc, best_ep = val_acc, epoch + 1
            torch.save(ema.ema.state_dict(), best_path)
            print(f"  >> New best! acc={best_acc:.4f} @epoch {best_ep}")

    print(f"[Done] Best val_acc={best_acc:.4f} @epoch {best_ep}. Saved: {best_path}")


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    acc_meter = AverageMeter()
    for (x, y, _) in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        acc = accuracy_from_logits(logits, y, thr=0.5)
        acc_meter.update(acc, x.size(0))
    return acc_meter.avg


@torch.no_grad()
def infer_to_csv(args, split: str):
    """split in {"public","private"}，将 test_{split}.npz -> pred_{split}.csv"""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"[Device] Using {device.type}")

    # 构建模型并加载 best
    model = ResNetSmall(width=32, dropout=0.0).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.eval()

    # DataLoader
    tf = get_transforms(train=False)
    npz_path = os.path.join(args.data_dir, f"test_{split}.npz")
    ds = NpzEnd2EndDataset(npz_path, is_train=False, transform=tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers, pin_memory=True)

    # 推理
    ids_all, preds_all = [], []
    for (x, _, ids) in tqdm(dl, desc=f"Infer {split}"):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = logits.sigmoid()
        preds = (probs >= 0.5).long().cpu().numpy().tolist()
        preds_all.extend(preds)
        ids_all.extend(ids)

    # 写 CSV
    out = Path(args.out_dir) / f"pred_{split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for _id, _p in zip(ids_all, preds_all):
            w.writerow([_id, int(_p)])
    print(f"[Infer] Wrote: {out}")


def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    ap_train = sub.add_parser("train", help="Train model on train.npz, validate on val.npz")
    ap_train.add_argument("--data_dir", type=str, required=True)
    ap_train.add_argument("--out_dir", type=str, default="./outputs/resnet")
    ap_train.add_argument("--epochs", type=int, default=80)
    ap_train.add_argument("--warmup", type=int, default=5)
    ap_train.add_argument("--batch_size", type=int, default=512)
    ap_train.add_argument("--num_workers", type=int, default=4)
    ap_train.add_argument("--lr", type=float, default=1e-3)
    ap_train.add_argument("--weight_decay", type=float, default=1e-4)
    ap_train.add_argument("--seed", type=int, default=2025)

    # infer public/private
    for sp in ("public", "private"):
        ap_inf = sub.add_parser(f"infer_{sp}", help=f"Export predictions for test_{sp}.npz")
        ap_inf.add_argument("--data_dir", type=str, required=True)
        ap_inf.add_argument("--ckpt", type=str, required=True, help="Path to best.pt")
        ap_inf.add_argument("--out_dir", type=str, default="./outputs/resnet")
        ap_inf.add_argument("--batch_size", type=int, default=1024)
        ap_inf.add_argument("--num_workers", type=int, default=4)

    return ap.parse_args()


def main():
    args = parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "infer_public":
        infer_to_csv(args, split="public")
    elif args.cmd == "infer_private":
        infer_to_csv(args, split="private")


if __name__ == "__main__":
    main()
