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

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.cross_decomposition import CCA
from sklearn.pipeline import Pipeline
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
        from sklearn.calibration import CalibratedClassifierCV
        base = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
        return CalibratedClassifierCV(base, ensemble=False)
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
    """Compute metrics. Primary: AUC-ROC and PR-AUC (require y_prob)."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
        pos_prob = y_prob[:, 1]
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, pos_prob)
        except ValueError:
            metrics["auc_roc"] = float("nan")
        try:
            metrics["pr_auc"] = average_precision_score(y_true, pos_prob)
        except ValueError:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["auc_roc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def cv_evaluate(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    cv_folds: int = 5,
    pca_variance: Optional[float] = None,
) -> dict:
    """Leakage-safe CV with scaling/PCA fitted independently in every fold."""
    cv = StratifiedKFold(n_splits=min(cv_folds, min(np.bincount(y))), shuffle=True,
                         random_state=42)
    steps = [("scale", StandardScaler())]
    if pca_variance is not None:
        steps.append(("pca", PCA(n_components=pca_variance, random_state=42)))
    steps.append(("classifier", clf))
    estimator = Pipeline(steps)
    y_pred = cross_val_predict(estimator, X, y, cv=cv, method="predict")
    try:
        y_prob = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
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
        metrics["cv_auc_roc"] = cv_m["auc_roc"]
        metrics["cv_pr_auc"] = cv_m["pr_auc"]
        results.append(metrics)
        print(f"  {clf_name:>16s}  raw   | AUC={metrics['auc_roc']:.4f}  PR={metrics['pr_auc']:.4f}  CV_AUC={metrics['cv_auc_roc']:.4f}")

        # PCA
        clf_pca = make_classifier(clf_name, rs)
        clf_pca.fit(X_train_pca, y_train)
        y_pred_pca = clf_pca.predict(X_test_pca)
        try:
            y_prob_pca = clf_pca.predict_proba(X_test_pca)
        except Exception:
            y_prob_pca = None
        metrics_pca = evaluate(y_test, y_pred_pca, y_prob_pca)
        metrics_pca.update({
            "modality": modality_name,
            "classifier": clf_name,
            "features": f"pca{pca_dim}",
            "dim": pca_dim,
        })
        cv_pca = cv_evaluate(
            make_classifier(clf_name, rs),
            X,
            y,
            cv_folds,
            pca_variance=pca_var,
        )
        metrics_pca["cv_auc_roc"] = cv_pca["auc_roc"]
        metrics_pca["cv_pr_auc"] = cv_pca["pr_auc"]
        results.append(metrics_pca)
        print(f"  {clf_name:>16s}  PCA   | AUC={metrics_pca['auc_roc']:.4f}  PR={metrics_pca['pr_auc']:.4f}  CV_AUC={metrics_pca['cv_auc_roc']:.4f}")

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
        m = evaluate(y_test, mean_pred, mean_prob)
        m.update({"modality": "audio+video (late)", "classifier": "mean_prob_svm", "features": "late", "dim": 0})
        print(f"  mean_prob_svm      | AUC={m['auc_roc']:.4f}  PR={m['pr_auc']:.4f}")
        results.append(m)

    print("  hard voting omitted: two modalities have no valid majority tie-break")

    return results


def run_cca_fusion(
    X_audio: np.ndarray,
    X_video: np.ndarray,
    y: np.ndarray,
    cfg: dict,
) -> List[dict]:
    """
    CCA fusion: PCA-reduce each modality, then CCA to find maximally
    correlated projections. Fuse via concatenation of CCA components,
    then classify.
    """
    print("\n=== CCA Fusion ===")

    results = []
    test_size = cfg["baseline"]["test_size"]
    pca_var = cfg["baseline"]["pca_variance"]
    cv_folds = cfg["baseline"]["cv_folds"]
    rs = cfg["baseline"]["random_state"]

    # Stratified split
    _, X_a_test, _, X_v_test, _, y_test, X_a_train, X_v_train, y_train = _split(
        X_audio, X_video, y, test_size, rs
    )
    scaler_a, scaler_v = StandardScaler(), StandardScaler()
    X_a_train = scaler_a.fit_transform(X_a_train)
    X_a_test = scaler_a.transform(X_a_test)
    X_v_train = scaler_v.fit_transform(X_v_train)
    X_v_test = scaler_v.transform(X_v_test)

    # PCA first (reduce to manageable dim)
    pca_a = PCA(n_components=pca_var, random_state=rs)
    pca_v = PCA(n_components=pca_var, random_state=rs)
    X_a_train_pca = pca_a.fit_transform(X_a_train)
    X_a_test_pca = pca_a.transform(X_a_test)
    X_v_train_pca = pca_v.fit_transform(X_v_train)
    X_v_test_pca = pca_v.transform(X_v_test)

    # CCA — find correlated subspace
    max_cca = int(cfg["baseline"].get("cca_components", 5))
    n_cca = min(
        max_cca,
        X_a_train_pca.shape[1],
        X_v_train_pca.shape[1],
        len(y_train) - 1,
    )
    cca = CCA(n_components=n_cca, scale=False, max_iter=2000)
    cca.fit(X_a_train_pca, X_v_train_pca)

    # Transform
    a_train_cca, v_train_cca = cca.transform(X_a_train_pca, X_v_train_pca)
    a_test_cca, v_test_cca = cca.transform(X_a_test_pca, X_v_test_pca)

    # Report held-out correlations. Training correlations are optimistically
    # biased, particularly when feature dimension approaches sample count.
    can_corrs = np.array([
        np.corrcoef(a_test_cca[:, i], v_test_cca[:, i])[0, 1]
        if np.std(a_test_cca[:, i]) > 0 and np.std(v_test_cca[:, i]) > 0
        else np.nan
        for i in range(n_cca)
    ])
    top5_corr = ", ".join(
        f"{c:.3f}" for c in sorted(can_corrs[np.isfinite(can_corrs)], reverse=True)[:5]
    )
    print(f"  PCA dims: audio={X_a_train_pca.shape[1]}, video={X_v_train_pca.shape[1]}")
    print(f"  CCA components: {n_cca}")
    print(f"  Held-out canonical correlations: {top5_corr}")
    print(f"  Held-out mean canonical corr: {np.nanmean(can_corrs):.4f}")

    # Fuse: concatenate CCA projections
    X_train_fused = np.concatenate([a_train_cca, v_train_cca], axis=1)
    X_test_fused = np.concatenate([a_test_cca, v_test_cca], axis=1)
    fused_dim = X_train_fused.shape[1]

    for clf_name, clf in make_classifiers(cfg):
        clf.fit(X_train_fused, y_train)
        y_pred = clf.predict(X_test_fused)
        try:
            y_prob = clf.predict_proba(X_test_fused)
        except Exception:
            y_prob = None
        metrics = evaluate(y_test, y_pred, y_prob)
        metrics.update({
            "modality": "audio+video (CCA)",
            "classifier": clf_name,
            "features": f"cca{fused_dim}",
            "dim": fused_dim,
            "canonical_corr_mean": np.nanmean(can_corrs),
        })
        # A correct CCA CV requires fitting scaling, PCA and CCA inside every
        # fold. The legacy helper cannot express paired-modality transforms, so
        # leave these fields empty rather than report leaked estimates.
        metrics["cv_auc_roc"] = float("nan")
        metrics["cv_pr_auc"] = float("nan")
        results.append(metrics)
        print(f"  {clf_name:>16s}  CCA   | AUC={metrics['auc_roc']:.4f}  PR={metrics['pr_auc']:.4f}  CV_AUC={metrics['cv_auc_roc']:.4f}")

    # Also store canonical correlations for plotting
    results.append({
        "modality": "audio+video (CCA)",
        "classifier": "_cca_stats",
        "features": f"cca{n_cca}",
        "dim": n_cca,
        "auc_roc": np.nanmean(can_corrs),
        "pr_auc": 0,
        "accuracy": 0,
        "cv_auc_roc": 0,
        "cv_pr_auc": 0,
        "canonical_corr_mean": np.nanmean(can_corrs),
    })

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

def plot_results(all_results: List[dict], X_audio, X_video, y, cfg, output_dir: Path):
    """Generate comparison bar chart, ROC curves, and PR curves."""
    df = pd.DataFrame(all_results)
    metrics_csv = output_dir / "metrics.csv"
    df.to_csv(str(metrics_csv), index=False)
    print(f"\nMetrics saved to {metrics_csv}")
    rs = cfg["baseline"]["random_state"]
    test_size = cfg["baseline"]["test_size"]

    # --- Bar chart: AUC-ROC by method ---
    fig, ax = plt.subplots(figsize=(14, 7))
    df_plot = df[df["classifier"] != "_cca_stats"].copy()
    df_plot["label"] = df_plot["modality"] + " | " + df_plot["classifier"] + " | " + df_plot["features"]
    df_plot = df_plot.sort_values("auc_roc", ascending=True)

    colors = []
    for _, row in df_plot.iterrows():
        if "late" in str(row["features"]):
            colors.append("#e74c3c")
        elif "CCA" in str(row["modality"]):
            colors.append("#9b59b6")
        elif "early" in str(row["modality"]) or "audio+video" in str(row["modality"]):
            colors.append("#3498db")
        elif "audio" in str(row["modality"]):
            colors.append("#2ecc71")
        else:
            colors.append("#f39c12")

    ax.barh(range(len(df_plot)), df_plot["auc_roc"], color=colors)
    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["label"], fontsize=8)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Baseline Comparison: pre vs stage 1-4 (AUC-ROC)", fontweight="bold")
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Audio only"),
        Patch(facecolor="#f39c12", label="Video only"),
        Patch(facecolor="#3498db", label="Early Fusion (Concat)"),
        Patch(facecolor="#e74c3c", label="Late Fusion"),
        Patch(facecolor="#9b59b6", label="CCA Fusion"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(str(output_dir / "comparison_bar.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'comparison_bar.png'}")

    # --- ROC + PR curves for top-6 *non-late* methods ---
    top6 = df[(df["features"] != "late") & (df["classifier"] != "_cca_stats")].nlargest(6, "auc_roc")
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, curve_name, curve_fn, baseline_val in [
        (ax_roc, "ROC", roc_curve, None),
        (ax_pr, "Precision-Recall", precision_recall_curve, y.mean()),
    ]:
        for _, row in top6.iterrows():
            mod = row["modality"]
            clf_name = row["classifier"]
            feat = row["features"]

            # Build X
            if "audio+video" in mod and "late" not in str(feat):
                X = np.concatenate([X_audio, X_video], axis=1)
            elif "audio" in mod:
                X = X_audio
            else:
                X = X_video

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=rs, stratify=y)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            if "pca" in str(feat):
                from sklearn.decomposition import PCA as PCA2
                pca_m = PCA2(n_components=cfg["baseline"]["pca_variance"], random_state=rs)
                X_tr = pca_m.fit_transform(X_tr)
                X_te = pca_m.transform(X_te)

            clf = make_classifier(clf_name, rs)
            clf.fit(X_tr, y_tr)
            try:
                y_prob = clf.predict_proba(X_te)[:, 1]
            except Exception:
                continue

            if curve_name == "ROC":
                fpr, tpr, _ = curve_fn(y_te, y_prob)
                label = f"{mod[:20]} | {clf_name} | {feat}"
                if len(label) > 35:
                    label = label[:32] + "..."
                ax.plot(fpr, tpr, linewidth=1.5, label=label)
            else:
                prec, rec, _ = curve_fn(y_te, y_prob)
                ax.plot(rec, prec, linewidth=1.5)

        if curve_name == "ROC":
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curves (top-6)", fontweight="bold")
            ax.legend(fontsize=7, loc="lower right")
        else:
            ax.axhline(y=baseline_val, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curves (top-6)", fontweight="bold")

    fig.tight_layout()
    fig.savefig(str(output_dir / "roc_pr_curves.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir / 'roc_pr_curves.png'}")

    # --- CCA canonical correlation ---
    cca_info = df[df["classifier"] == "_cca_stats"]
    if len(cca_info) > 0:
        mean_corr = cca_info.iloc[0]["canonical_corr_mean"]
        print(f"  CCA mean canonical correlation: {mean_corr:.4f}")


def plot_confusion_matrices(all_results: List[dict], X_audio, X_video, y, cfg, output_dir: Path):
    """Generate confusion matrices for top-3 *non-late* methods."""
    from sklearn.model_selection import train_test_split as tts

    df = pd.DataFrame(all_results)
    # Skip late fusion and CCA — they aren't simple classifier models
    df_trainable = df[(df["features"] != "late") & (~df["features"].str.startswith("cca"))]
    top_n = min(3, len(df_trainable))
    if top_n == 0:
        print("  [Warn] No trainable methods for confusion matrices.")
        return
    top = df_trainable.nlargest(top_n, "auc_roc")

    fig, axes = plt.subplots(1, top_n, figsize=(5 * top_n, 4.5))
    if top_n == 1:
        axes = [axes]
    rs = cfg["baseline"]["random_state"]
    test_size = cfg["baseline"]["test_size"]

    for ax, (_, row) in zip(axes, top.iterrows()):
        mod = row["modality"]
        clf_name = row["classifier"]
        feat = row["features"]

        # Build X
        if "audio+video" in str(mod):
            X = np.concatenate([X_audio, X_video], axis=1)
        elif "audio" in str(mod):
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
    # Exclude metadata marker rows
    df_report = df[df["classifier"] != "_cca_stats"].copy()
    report_path = output_dir / "report.txt"

    best = df_report.loc[df_report["auc_roc"].idxmax()]
    audio_best = df_report[df_report["modality"] == "audio"].loc[
        df_report[df_report["modality"] == "audio"]["auc_roc"].idxmax()]
    video_best = df_report[df_report["modality"] == "video"].loc[
        df_report[df_report["modality"] == "video"]["auc_roc"].idxmax()]

    # CCA stats
    cca_info = df[df["classifier"] == "_cca_stats"]
    cca_corr = cca_info.iloc[0]["canonical_corr_mean"] if len(cca_info) > 0 else None

    lines = [
        "=" * 60,
        "  Multimodal Fusion — Baseline Experiment Report",
        "=" * 60,
        "",
        f"  Task: pre vs stage (1,2,3,4) binary classification",
        f"  Total methods evaluated: {len(df_report)}",
    ]

    if cca_corr is not None:
        lines.append(f"  CCA mean canonical correlation: {cca_corr:.4f}")

    lines += [
        "",
        "-" * 40,
        "  Best overall (by AUC-ROC)",
        "-" * 40,
        f"  Method    : {best['modality']} | {best['classifier']} | {best['features']}",
        f"  AUC-ROC   : {best['auc_roc']:.4f}",
        f"  PR-AUC    : {best['pr_auc']:.4f}",
        f"  CV AUC    : {best.get('cv_auc_roc', 'N/A')}",
        "",
        "-" * 40,
        "  Best audio-only (by AUC-ROC)",
        "-" * 40,
        f"  Method    : {audio_best['classifier']} | {audio_best['features']}",
        f"  AUC-ROC   : {audio_best['auc_roc']:.4f}",
        f"  PR-AUC    : {audio_best['pr_auc']:.4f}",
        "",
        "-" * 40,
        "  Best video-only (by AUC-ROC)",
        "-" * 40,
        f"  Method    : {video_best['classifier']} | {video_best['features']}",
        f"  AUC-ROC   : {video_best['auc_roc']:.4f}",
        f"  PR-AUC    : {video_best['pr_auc']:.4f}",
        "",
        "-" * 40,
        "  Full results table",
        "-" * 40,
        "",
    ]

    cols = ["modality", "classifier", "features", "auc_roc", "pr_auc", "cv_auc_roc", "cv_pr_auc"]
    tbl = df_report[cols].sort_values("auc_roc", ascending=False)
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

    # ---- CCA fusion ----
    all_results.extend(run_cca_fusion(X_audio, X_video, y, cfg))

    # ---- Plots & report ----
    print("\n--- Generating Plots & Report ---")
    plot_results(all_results, X_audio, X_video, y, cfg, results_dir)
    plot_confusion_matrices(all_results, X_audio, X_video, y, cfg, results_dir)
    write_report(all_results, results_dir)

    print(f"\n{'='*60}")
    print(f"  Done. All results in: {results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
