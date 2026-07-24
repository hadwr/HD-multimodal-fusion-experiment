#!/usr/bin/env python3
"""
Multi-modal fusion baseline for classification & regression.

Fuses pre-extracted audio + video embeddings (and optionally MemTrax features)
via simple concatenation, then trains an MLP classifier.

Tasks (configurable via config.yaml):
- ``pre`` vs ``(1, 2)`` binary classification
- HD vs HC binary classification (when HC data is available)
- Regression on scale scores (future)

Usage::

    python fusion_baseline.py
    python fusion_baseline.py --epochs 200 --lr 0.0005
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from utils import load_config, load_embeddings, merge_embeddings


# ============================================================
# Model
# ============================================================

class MLPClassifier(nn.Module):
    """Simple MLP with dropout for multi-modal fusion classification."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dims=(256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Training
# ============================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(yb.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


# ============================================================
# Main
# ============================================================

def prepare_data(
    cfg: dict,
    audio_emb: dict,
    video_emb: dict,
    labels_df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor, LabelEncoder]:
    """
    Merge embeddings, align with labels, build classification target.

    Returns
    -------
    X : torch.Tensor  (N, D) fused feature matrix
    y : torch.Tensor  (N,) integer class labels
    le : LabelEncoder
    """
    positive = cfg["fusion"]["labels"]["positive"]   # e.g. [1, 2]
    negative = cfg["fusion"]["labels"]["negative"]   # e.g. ["pre"]

    # Align subjects
    common_ids, X_dict = merge_embeddings(audio_emb, video_emb)

    # Align labels
    labels_aligned = labels_df.loc[labels_df.index.isin(common_ids), "stages"]
    common_ids = [sid for sid in common_ids if sid in labels_aligned.index]

    print(f"  Subjects with embeddings + labels: {len(common_ids)}")

    # Build feature matrix
    audio_mat = np.stack([audio_emb[sid] for sid in common_ids])
    video_mat = np.stack([video_emb[sid] for sid in common_ids])
    X = np.concatenate([audio_mat, video_mat], axis=1)

    # Build binary target: negative=0, positive=1
    label_values = labels_aligned.loc[common_ids].values
    y_str = np.array([
        1 if str(v) in [str(p) for p in positive] else 0
        for v in label_values
    ])

    # Filter out any labels not in {positive, negative}
    in_set = np.array([
        str(v) in [str(p) for p in positive] + [str(n) for n in negative]
        for v in label_values
    ])
    X = X[in_set]
    y_str = y_str[in_set]
    common_ids_final = [sid for sid, keep in zip(common_ids, in_set) if keep]

    print(f"  After filtering to {negative} vs {positive}: {len(common_ids_final)} subjects")
    pos_count = (y_str == 1).sum()
    neg_count = (y_str == 0).sum()
    print(f"  Class distribution — negative({negative}): {neg_count}, positive({positive}): {pos_count}")

    if len(common_ids_final) < 4:
        print("[ERROR] Too few subjects for train/test split.")
        sys.exit(1)

    le = LabelEncoder()
    y = torch.tensor(le.fit_transform(y_str), dtype=torch.long)

    return torch.tensor(X, dtype=torch.float32), y, le


def main():
    parser = argparse.ArgumentParser(description="Multimodal fusion baseline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = args.device or cfg.get("device", "cuda")
    output_dir = Path(cfg["paths"]["output_dir"])
    emb_dir = output_dir / "emb"
    data_dir = output_dir / "data"
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs or cfg["fusion"].get("epochs", 100)
    lr = args.lr or cfg["fusion"].get("lr", 0.001)
    batch_size = args.batch_size or cfg["fusion"].get("batch_size", 32)
    hidden_dims = cfg["fusion"].get("hidden_dims", [256, 128])
    dropout = cfg["fusion"].get("dropout", 0.3)
    test_size = cfg["fusion"].get("test_size", 0.2)
    val_size = cfg["fusion"].get("val_size", 0.1)
    patience = cfg["fusion"].get("early_stopping_patience", 15)
    random_state = cfg["fusion"].get("random_state", 42)

    # ---- load ----
    print("=== Fusion Baseline ===\n")

    audio_emb = load_embeddings(str(emb_dir / "audio"))
    video_emb = load_embeddings(str(emb_dir / "video"))
    print(f"  Audio embeddings : {len(audio_emb)} subjects")
    print(f"  Video embeddings : {len(video_emb)} subjects")

    csv_path = data_dir / "merged.csv"
    if not csv_path.exists():
        print(f"[ERROR] merged.csv not found at {csv_path}")
        print("[HINT] Run load_tabular.py first.")
        sys.exit(1)
    labels_df = pd.read_csv(str(csv_path), index_col=0)

    # ---- prepare ----
    X, y, le = prepare_data(cfg, audio_emb, video_emb, labels_df)
    num_classes = len(le.classes_)
    print(f"  Classes: {le.classes_.tolist()}  (encoded as {list(range(num_classes))})")
    print(f"  Feature dim: {X.shape[1]}")

    # ---- split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )
    # Further split train → train / val
    if val_size > 0:
        val_frac = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_frac,
            random_state=random_state, stratify=y_train,
        )
    else:
        X_val, y_val = X_test, y_test

    # Fit preprocessing on training subjects only.  The previous version fit
    # StandardScaler before the split, leaking test-set means and variances.
    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train.numpy())
    X_val_np = scaler.transform(X_val.numpy())
    X_test_np = scaler.transform(X_test.numpy())
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ---- data loaders ----
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    # ---- model ----
    model = MLPClassifier(
        input_dim=X.shape[1],
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
    ).to(device)
    print(f"\n  Model: {sum(p.numel() for p in model.parameters())} params")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ---- train ----
    print(f"\n  Training {epochs} epochs (lr={lr}, batch={batch_size})...\n")
    best_val_acc = float("-inf")
    best_state = None
    stale = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f}  {'*' if stale == 0 else ''}")

        if stale >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # ---- test ----
    model.load_state_dict(best_state)
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

    print(f"\n{'='*50}")
    print(f"  Test Results")
    print(f"{'='*50}")
    print(f"  Accuracy  : {test_acc:.4f}")
    print(f"  F1 (macro): {f1_score(labels, preds, average='macro'):.4f}")
    print(f"  F1 (weighted): {f1_score(labels, preds, average='weighted'):.4f}")
    print(f"\n  Classification Report:")
    target_names = [str(c) for c in le.classes_]
    print(classification_report(labels, preds, target_names=target_names))
    print("  Confusion Matrix:")
    print(confusion_matrix(labels, preds))

    # ---- save model ----
    model_path = model_dir / "fusion_mlp.pt"
    torch.save(
        {
            "state_dict": best_state,
            "config": cfg["fusion"],
            "le_classes": le.classes_.tolist(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
        },
        str(model_path),
    )
    print(f"\n  Model saved to: {model_path}")


if __name__ == "__main__":
    main()
