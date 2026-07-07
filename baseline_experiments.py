#!/usr/bin/env python3
"""
Comprehensive ML baseline experiments for multimodal fusion.

Runs single-modality (audio, video) and fusion baselines:
- Classifiers: SVM, Logistic Regression, Random Forest
- Dimensionality reduction: PCA (0.95 variance retained)
- Early fusion: concatenation then classifier
- Late fusion: mean probability, majority voting

Produces:
1. ``output/results/metrics.csv`` — all method metrics
2. ``output/results/comparison_bar.png`` — bar chart comparison
3. ``output/results/confusion_matrices.png`` — confusion matrices
4. ``output/results/report.txt`` — text summary

Usage::

    python baseline_experiments.py
    python baseline_experiments.py --test-size 0.3
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import load_config, load_embeddings, extract_subject_id


# ============================================================
# Data loading
# ============================================================

def load_data(cfg: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load audio/video embeddings, align with labels, return feature matrices.

    Returns
    -------
    X_audio : np.ndarray  (N, 768)
    X_video : np.ndarray  (N, 768)
    y : np.ndarray        (N,) binary: 0=pre, 1=stage 1-4
    subject_ids : list[str]
    """
    output_dir = Path(cfg["paths"]["output_dir"])
    emb_dir = output_dir / "emb"
    data_dir = output_dir / "data"

    # Load embeddings
    audio_emb = load_embeddings(str(emb_dir / "audio"))
    video_emb = load_embeddings(str(emb_dir / "video"))
    print(f"  Audio subjects : {len(audio_emb)}")
    print(f"  Video subjects : {len(video_emb)}")

    # Load labels
    csv_path = data_dir / "merged.csv"
    if not csv_path.exists():
        print(f"[ERROR] merged.csv not found at {csv_path}")
        print("[HINT] Run load_tabular.py first.")
        sys.exit(1)
    labels_df = pd.read_csv(str(csv_path), index_col=0)

    # Align
    common_ids = sorted(set(audio_emb.keys()) & set(video_emb.keys()) & set(labels_df.index))
    print(f"  Subjects with all modalities + labels : {len(common_ids)}")

    if len(common_ids) < 10:
        print(f"[ERROR] Only {len(common_ids)} subjects found. Need more data.")
        sys.exit(1)

    X_audio = np.stack([audio_emb[sid] for sid in common_ids])
    X_video = np.stack([video_emb[sid] for sid in common_ids])

    # Build binary labels
    positive = [str(p) for p in cfg["fusion"]["labels"]["positive"]]
    negative = [str(n) for n in cfg["fusion"]["labels"]["negative"]]
    label_values = labels_df.loc[common_ids, "stages"].values
    y = np.array([1 if str(v) in positive else 0 for v in label_values])

    # Filter to only positive/negative
    keep = np.array([
        str(v) in positive + negative for v in label_values
    ])
    X_audio = X_audio[keep]
    X_video = X_video[keep]
    y = y[keep]
    common_ids = [sid for sid, k in zip(common_ids, keep) if k]

    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    print(f"  Labels: {n_neg} (pre) vs {n_pos} (stage 1-4)")
    print(f"  Total samples: {len(y)}")

    return X_audio, X_video, y, common_ids


# ============================================================
# Classifier factory
# ============================================================

def make_classifier(name: str, random_state: int = 42):
    """Create a classifier by name."""
    if name == "svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   random_state=random_state)
    elif name == "logistic":
        return LogisticRegression(max_iter=2000, random_state=random_state)
    elif name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {name}")


def make_classifiers(config: dict) -> List[Tuple[str, object]]:
    """Build all configured classifiers."""
    names = config["baseline"].get("classifiers", ["svm", "logistic", "random_forest"])
    rs = config["baseline"].get("random_state", 42)
    return [(name, make_classifier(name, rs)) for name in names]


# ============================================================
# Evaluation
# ============================================================

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> dict:
    """Compute all metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def cv_evaluate(clf, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> dict:
    """Stratified K-fold cross-validation evaluation."""
    cv = StratifiedKFold(n_splits=min(cv_folds, min(np.bincount(y))), shuffle=True,
                         random_state=42)
    y_pred = cross_val_predict(clf, X, y, cv=cv, method="predict")
    try:
        y_prob = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    except Exception:
        y_prob = None
    return evaluate(y, y_pred, y_prob)


# ============================================================
# Single modality experiments
# ============================================================

def run_single_modality(
    X: np.ndarray,
    y: np.ndarray,
    modality_name: str,
    cfg: dict,
) -> List[dict]:
    """Run all classifiers on a single modality (raw + PCA)."""
    results = []
    test_size = cfg["baseline"]["test_size"]
    pca_var = cfg["baseline"]["pca_variance"]
    cv_folds = cfg["baseline"]["cv_folds"]
    rs = cfg["baseline"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=rs, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- PCA ---
    pca = PCA(n_components=pca_var, random_state=rs)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    pca_dim = X_train_pca.shape[1]
    print(f"\n{modality_name}: raw dim={X_train.shape[1]}, PCA dim={pca_dim} ({pca_var:.0%} variance)")

    for clf_name, clf in make_classifiers(cfg):
        # Raw
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        try:
            y_prob = clf.predict_proba(X_test)
        except Exception:
            y_prob = None
        metrics = evaluate(y_test, y_pred, y_prob)
        metrics.update({
            "modality": modality_name,
            "classifier": clf_name,
            "features": "raw",
            "dim": X_train.shape[1],
        })
        # CV
        cv_m = cv_evaluate(clf, X, y, cv_folds)
        metrics["cv_accuracy"] = cv_m["accuracy"]
        metrics["cv_f1_weighted"] = cv_m["f1_weighted"]
        results.append(metrics)
        print(f"  {clf_name:>16s}  raw   | acc={metrics['accuracy']:.4f}  f1={metrics['f1_weighted']:.4f}  cv_acc={metrics['cv_accuracy']:.4f}")

        # PCA
        clf_pca = make_classifier(clf_name, rs)
        clf_pca.fit(X_train_pca, y_train)
        y_pred_pca = clf_pca.predict(X_test_pca)
        try:
            y_prob_pca = clf_pca.predict_proba(X_test_pca)
        except Exception:
            y_prob_pca = None
        metrics_pca = evaluate(y_test_pca, y_pred_pca, y_prob_pca)
        metrics_pca.update({
            "modality": modality_name,
            "classifier": clf_name,
            "features": f"pca{dim}",
            "dim": pca_dim,
        })
        cv_pca = cv_evaluate(clf_pca, np.vstack([X_train_pca, X_test_pca]),
                             np.hstack([y_train, y_test]), cv_folds)
        metrics_pca["cv_accuracy"] = cv_pca["accuracy"]
        metrics_pca["cv_f1_weighted"] = cv_pca["f1_weighted"]
        results.append(metrics_pca)
        print(f"  {clf_name:>16s}  PCA   | acc={metrics_pca['accuracy']:.4f}  f1={metrics_pca['f1_weighted']:.4f}  cv_acc={metrics_pca['cv_accuracy']:.4f}")

    # Store test predictions for late fusion
    return results


def get_best_clf_predictions(
    X_train, y_train, X_test, clf_name: str, rs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a classifier and return (predictions, probabilities) on test set."""
    clf = make_classifier(clf_name, rs)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    try:
        y_prob = clf.predict_proba(X_test)
    except Exception:
        y_prob = None
    return y_pred, y_prob, clf


# ============================================================
# Early fusion (concatenation)
# ============================================================

def run_early_fusion(
    X_audio: np.ndarray,
    X_video: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> List[dict]:
    """Concatenate audio+video then classify."""
    print("\n=== Early Fusion (Concat) ===")

    X = np.concatenate([X_audio, X_video], axis=1)
    results = run_single_modality(X, y, "audio+video (early fusion)", cfg)
    return results


# ============================================================
# Late fusion (mean probability, voting)
# ============================================================

def run_late_fusion(
    X_audio: np.ndarray,
    X_video: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> List[dict]:
    """
    Late fusion: train separate audio & video classifiers,
    then combine via mean probability and majority voting.
    """
    print("\n=== Late Fusion ===")

    test_size = cfg["baseline"]["test_size"]
    rs = cfg["baseline"]["random_state"]
    clf_name = "svm"  # use SVM as the base classifier for late fusion

    # Split
    _, X_a_test, _, X_v_test, _, y_test, X_a_train, X_v_train, y_train = _split(
        X_audio, X_video, y, test_size, rs
    )
    scaler_a, scaler_v = StandardScaler(), StandardScaler()
    X_a_train = scaler_a.fit_transform(X_a_train)
    X_a_test = scaler_a.transform(X_a_test)
    X_v_train = scaler_v.fit_transform(X_v_train)
    X_v_test = scaler_v.transform(X_v_test)

    # Train separate SVMs
    a_pred, a_prob, _ = get_best_clf_predictions(X_a_train, y_train, X_a_test, clf_name, rs)
    v_pred, v_prob, _ = get_best_clf_predictions(X_v_train, y_train, X_v_test, clf_name, rs)

    results = []

    # --- Mean probability ---
    if a_prob is not None and v_prob is not None:
        mean_prob = (a_prob + v_prob) / 2.0
        mean_pred = mean_prob.argmax(axis=1)
        m = evaluate(y_test, mean_pred)
        m.update({"modality": "audio+video (late)", "classifier": "mean_prob_svm", "features": "late", "dim": 0})
        print(f"  mean_prob_svm      | acc={m['accuracy']:.4f}  f1={m['f1_weighted']:.4f}")
        results.append(m)

    # --- Majority voting ---
    votes = np.column_stack([a_pred, v_pred])
    vote_pred = np.array([np.bincount(v).argmax() for v in votes])
    m = evaluate(y_test, vote_pred)
    m.update({"modality": "audio+video (late)", "classifier": "voting_svm", "features": "late", "dim": 0})
    print(f"  voting_svm         | acc={m['accuracy']:.4f}  f1={m['f1_weighted']:.4f}")
    results.append(m)

    return results


def _split(X_a, X_v, y, test_size, rs):
    """Stratified train/test split that preserves alignment."""
    n = len(y)
    indices = np.arange(n)
    train_idx, test_idx = train_test_split(indices, test_size=test_size,
                                           random_state=rs, stratify=y)
    return (
        X_a[train_idx], X_a[test_idx],
        X_v[train_idx], X_v[test_idx],
        y[train_idx], y[test_idx],
        X_a[train_idx], X_v[train_idx], y[train_idx],
    )


# ============================================================
# Plotting
# ============================================================

def plot_results(all_results: List[dict], output_dir: Path):
    """Generate comparison bar chart and confusion matrices."""
    df = pd.DataFrame(all_results)
    metrics_csv = output_dir / "metrics.csv"
    df.to_csv(str(metrics_csv), index=False)
    print(f"\nMetrics saved to {metrics_csv}")

    # --- Bar chart: accuracy by method ---
    fig, ax = plt.subplots(figsize=(14, 6))
    df_plot = df.copy()
    df_plot["label"] = df_plot["modality"] + " | " + df_plot["classifier"] + " | " + df_plot["features"]
    df_plot = df_plot.sort_values("accuracy", ascending=True)

    colors = []
    for _, row in df_plot.iterrows():
        if "late" in str(row["features"]):
            colors.append("#e74c3c")
        elif "early" in str(row["modality"]) or "audio+video" in str(row["modality"]):
            colors.append("#3498db")
        elif "audio" in str(row["modality"]):
            colors.append("#2ecc71")
        else:
            colors.append("#f39c12")

    bars = ax.barh(range(len(df_plot)), df_plot["accuracy"], color=colors)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["label"], fontsize=8)
    ax.set_xlabel("Accuracy")
    ax.set_title("Baseline Comparison: pre vs stage 1-4 Classification", fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Audio only"),
        Patch(facecolor="#f39c12", label="Video only"),
        Patch(facecolor="#3498db", label="Early Fusion (Concat)"),
        Patch(facecolor="#e74c3c", label="Late Fusion"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(output_dir / "comparison_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'comparison_bar.png'}")


def plot_confusion_matrices(all_results: List[dict], X_audio, X_video, y, cfg, output_dir: Path):
    """Generate confusion matrices for top-3 methods."""
    from sklearn.model_selection import train_test_split as tts

    df = pd.DataFrame(all_results)
    top3 = df.nlargest(3, "accuracy")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    rs = cfg["baseline"]["random_state"]
    test_size = cfg["baseline"]["test_size"]

    for ax, (_, row) in zip(axes, top3.iterrows()):
        mod = row["modality"]
        clf_name = row["classifier"]
        feat = row["features"]

        # Build X
        if "audio+video" in mod and "late" not in str(feat):
            if "early" in mod:
                X = np.concatenate([X_audio, X_video], axis=1)
            else:
                X = np.concatenate([X_audio, X_video], axis=1)
        elif "audio" in mod:
            X = X_audio
        else:
            X = X_video

        X_tr, X_te, y_tr, y_te = tts(X, y, test_size=test_size, random_state=rs, stratify=y)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        # PCA if needed
        if "pca" in str(feat):
            pca = PCA(n_components=cfg["baseline"]["pca_variance"], random_state=rs)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

        clf = make_classifier(clf_name, rs)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        cm = confusion_matrix(y_te, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["pre", "stage 1-4"],
                    yticklabels=["pre", "stage 1-4"])
        title = f"{mod} | {clf_name} | {feat}"
        if len(title) > 40:
            title = title[:37] + "..."
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    fig.tight_layout()
    fig.savefig(str(output_dir / "confusion_matrices.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'confusion_matrices.png'}")


# ============================================================
# Report
# ============================================================

def write_report(all_results: List[dict], output_dir: Path):
    """Write a brief text report."""
    df = pd.DataFrame(all_results)
    report_path = output_dir / "report.txt"

    best = df.loc[df["accuracy"].idxmax()]
    audio_best = df[df["modality"] == "audio"].loc[df[df["modality"] == "audio"]["accuracy"].idxmax()]
    video_best = df[df["modality"] == "video"].loc[df[df["modality"] == "video"]["accuracy"].idxmax()]

    lines = [
        "=" * 60,
        "  Multimodal Fusion — Baseline Experiment Report",
        "=" * 60,
        "",
        f"  Task: pre vs stage (1,2,3,4) binary classification",
        f"  Total methods evaluated: {len(df)}",
        "",
        "-" * 40,
        "  Best overall",
        "-" * 40,
        f"  Method    : {best['modality']} | {best['classifier']} | {best['features']}",
        f"  Accuracy  : {best['accuracy']:.4f}",
        f"  F1 (wtd)  : {best['f1_weighted']:.4f}",
        f"  CV Acc    : {best.get('cv_accuracy', 'N/A')}",
        "",
        "-" * 40,
        "  Best audio-only",
        "-" * 40,
        f"  Method    : {audio_best['classifier']} | {audio_best['features']}",
        f"  Accuracy  : {audio_best['accuracy']:.4f}",
        f"  F1 (wtd)  : {audio_best['f1_weighted']:.4f}",
        "",
        "-" * 40,
        "  Best video-only",
        "-" * 40,
        f"  Method    : {video_best['classifier']} | {video_best['features']}",
        f"  Accuracy  : {video_best['accuracy']:.4f}",
        f"  F1 (wtd)  : {video_best['f1_weighted']:.4f}",
        "",
        "-" * 40,
        "  Full results table",
        "-" * 40,
        "",
    ]

    # Format as table
    cols = ["modality", "classifier", "features", "accuracy", "f1_weighted", "cv_accuracy"]
    tbl = df[cols].sort_values("accuracy", ascending=False)
    lines.append(tbl.to_string(index=False))
    lines.append("")
    lines.append("=" * 60)

    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport saved to {report_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Baseline ML experiments")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--test-size", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.test_size is not None:
        cfg["baseline"]["test_size"] = args.test_size

    output_dir = Path(cfg["paths"]["output_dir"])
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Multimodal Fusion — Baseline Experiments")
    print("=" * 60)
    print(f"  Output: {results_dir}")

    # ---- Load data ----
    print("\n--- Loading Data ---")
    X_audio, X_video, y, subject_ids = load_data(cfg)

    all_results = []

    # ---- Single modality: Audio ----
    print("\n--- Audio-Only ---")
    all_results.extend(run_single_modality(X_audio, y, "audio", cfg))

    # ---- Single modality: Video ----
    print("\n--- Video-Only ---")
    all_results.extend(run_single_modality(X_video, y, "video", cfg))

    # ---- Early fusion ----
    all_results.extend(run_early_fusion(X_audio, X_video, y, cfg))

    # ---- Late fusion ----
    all_results.extend(run_late_fusion(X_audio, X_video, y, cfg))

    # ---- Plots & report ----
    print("\n--- Generating Plots & Report ---")
    plot_results(all_results, results_dir)
    plot_confusion_matrices(all_results, X_audio, X_video, y, cfg, results_dir)
    write_report(all_results, results_dir)

    print(f"\n{'='*60}")
    print(f"  Done. All results in: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
