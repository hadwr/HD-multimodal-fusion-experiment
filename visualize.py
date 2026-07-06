#!/usr/bin/env python3
"""
Visualize multimodal embeddings with t-SNE / UMAP.

Loads pre-extracted audio/video embeddings and merged tabular data,
then produces dimensionality-reduction plots for:
- Single modality: audio only, video only, tabular only (if available)
- Fused: concatenation of all available modalities

All plots are colored by stage label and saved as PNGs.

Usage::

    python visualize.py
    python visualize.py --method both   # run both UMAP and t-SNE
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from utils import load_config, load_embeddings, merge_embeddings


def run_umap(X: np.ndarray, n_neighbors: int = 15, random_state: int = 42) -> np.ndarray:
    """Run UMAP and return 2-D embedding."""
    import umap
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, metric="cosine",
                        random_state=random_state, verbose=False)
    return reducer.fit_transform(X)


def run_tsne(X: np.ndarray, perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """Run t-SNE and return 2-D embedding."""
    reducer = TSNE(n_components=2, perplexity=perplexity, metric="cosine",
                   random_state=random_state, verbose=False)
    return reducer.fit_transform(X)


def plot_embedding(
    coords: np.ndarray,
    labels: pd.Series,
    title: str,
    save_path: str,
):
    """Draw a 2-D scatter plot colored by stage label."""
    fig, ax = plt.subplots(figsize=(8, 6))

    palette = sns.color_palette("Set2", n_colors=labels.nunique())
    for i, stage in enumerate(sorted(labels.unique())):
        mask = labels == stage
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[palette[i]], label=str(stage),
            alpha=0.7, s=40, edgecolors="none",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title="Stage", frameon=True)
    sns.despine()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize multimodal embeddings")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--method", default=None,
                        choices=["umap", "tsne", "both"],
                        help="Override vis method from config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    output_dir = Path(cfg["paths"]["output_dir"])
    emb_dir = output_dir / "emb"
    data_dir = output_dir / "data"
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    method = args.method or cfg["visualization"].get("method", "umap")
    methods = ["umap", "tsne"] if method == "both" else [method]

    n_neighbors = cfg["visualization"].get("n_neighbors", 15)
    perplexity = cfg["visualization"].get("perplexity", 30)
    random_state = cfg["visualization"].get("random_state", 42)

    # ---- load embeddings ----
    print("=== Visualization ===\n")

    audio_emb = load_embeddings(str(emb_dir / "audio"))
    video_emb = load_embeddings(str(emb_dir / "video"))
    print(f"  Audio embeddings : {len(audio_emb)} subjects")
    print(f"  Video embeddings : {len(video_emb)} subjects")

    # ---- load labels ----
    csv_path = data_dir / "merged.csv"
    if not csv_path.exists():
        print(f"[ERROR] merged.csv not found at {csv_path}")
        print("[HINT] Run load_tabular.py first.")
        sys.exit(1)

    df = pd.read_csv(str(csv_path), index_col=0)
    print(f"  Tabular data     : {len(df)} subjects, columns={list(df.columns)}")

    # ---- merge into aligned arrays ----
    common_ids, X_dict = merge_embeddings(audio_emb, video_emb)
    if not common_ids:
        print("[ERROR] No subjects have both audio and video embeddings.")
        sys.exit(1)

    # Align labels
    labels = df.loc[df.index.isin(common_ids), "stages"]
    common_ids = [sid for sid in common_ids if sid in labels.index]
    print(f"  With labels      : {len(common_ids)} subjects\n")

    if len(common_ids) < 2:
        print("[ERROR] Fewer than 2 labeled subjects. Nothing to visualize.")
        sys.exit(1)

    # Re-index embeddings to aligned order
    audio_mat = np.stack([audio_emb[sid] for sid in common_ids])
    video_mat = np.stack([video_emb[sid] for sid in common_ids])
    labels_aligned = labels.loc[common_ids]

    # ---- run reductions ----
    for meth in methods:
        reducer_fn = run_umap if meth == "umap" else run_tsne
        reducer_kwargs = (
            {"n_neighbors": min(n_neighbors, len(common_ids) - 1), "random_state": random_state}
            if meth == "umap" else
            {"perplexity": min(perplexity, len(common_ids) - 1), "random_state": random_state}
        )

        print(f"--- {meth.upper()} ---")

        # Audio only
        print("  Audio...")
        audio_coords = reducer_fn(audio_mat, **reducer_kwargs)
        plot_embedding(audio_coords, labels_aligned,
                       f"Audio embedding ({meth.upper()})",
                       str(fig_dir / f"audio_{meth}.png"))

        # Video only
        print("  Video...")
        video_coords = reducer_fn(video_mat, **reducer_kwargs)
        plot_embedding(video_coords, labels_aligned,
                       f"Video embedding ({meth.upper()})",
                       str(fig_dir / f"video_{meth}.png"))

        # Fused (concat)
        print("  Fused (concat)...")
        fused = np.concatenate([audio_mat, video_mat], axis=1)
        fused = StandardScaler().fit_transform(fused)
        fused_coords = reducer_fn(fused, **reducer_kwargs)
        plot_embedding(fused_coords, labels_aligned,
                       f"Fused audio+video ({meth.upper()})",
                       str(fig_dir / f"fused_{meth}.png"))

    print(f"\nDone. Figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
