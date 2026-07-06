#!/usr/bin/env python3
"""
Extract video embeddings using VideoMAE (via ModelScope → transformers).

For each ``HDXXX/4.mp4`` under the video directory, this script:
1. Uniformly samples 16 frames, resizes to 224×224
2. Runs VideoMAE-base, takes the CLS token (or mean-pools all patches)
3. Saves the resulting vector as ``output/emb/video/HDXXX.npy``

Usage::

    python extract_video.py                    # use config.yaml defaults
    python extract_video.py --device cpu       # force CPU
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from utils import (
    load_config,
    resolve_model_path,
    extract_subject_id,
    save_embedding,
)


def load_videomae_local(local_path: str, device: str = "cuda"):
    """Load VideoMAE from a local ModelScope-downloaded directory."""
    from transformers import VideoMAEImageProcessor, VideoMAEModel

    processor = VideoMAEImageProcessor.from_pretrained(local_path)
    model = VideoMAEModel.from_pretrained(local_path)
    model = model.to(device)
    model.eval()
    return model, processor


def sample_frames_from_video(
    video_path: str,
    num_frames: int = 16,
    img_size: int = 224,
) -> np.ndarray:
    """
    Uniformly sample ``num_frames`` frames from a video using OpenCV.

    Returns
    -------
    frames : np.ndarray  shape (num_frames, H, W, C), dtype uint8, RGB
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No video frames found in {video_path}")

    # Uniform indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
        while len(indices) < num_frames:
            indices.append(indices[-1])
    else:
        step = total_frames / num_frames
        indices = [int(i * step) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        # BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if frame.shape[0] != img_size or frame.shape[1] != img_size:
            frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)

    cap.release()

    # Pad if we couldn't read enough frames
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((img_size, img_size, 3), dtype=np.uint8))

    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    return frames


@torch.no_grad()
def extract_video_embedding(
    video_path: str,
    model,
    processor,
    device: str = "cuda",
    num_frames: int = 16,
    img_size: int = 224,
    pool_mode: str = "cls",
) -> np.ndarray:
    """
    Load a video, run VideoMAE, return the embedding vector.

    Parameters
    ----------
    video_path : str
    model : VideoMAEModel
    processor : VideoMAEImageProcessor
    device : str
    num_frames : int
    img_size : int
    pool_mode : str
        "cls" → use the CLS token (first position);
        "mean" → mean-pool all patch tokens.

    Returns
    -------
    embedding : np.ndarray  shape (hidden_dim,)
    """
    # ---- sample frames ----
    frames = sample_frames_from_video(video_path, num_frames=num_frames, img_size=img_size)

    # ---- preprocess ----
    # processor expects a list of (num_frames, H, W, C) or already-preprocessed arrays
    inputs = processor(list(frames), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # (1, num_frames, 3, H, W)

    # ---- forward ----
    outputs = model(pixel_values)
    last_hidden = outputs.last_hidden_state  # (1, num_patches+1, hidden_dim)

    # ---- pool ----
    if pool_mode == "cls":
        embedding = last_hidden[:, 0, :]  # CLS token
    else:
        embedding = last_hidden[:, 1:, :].mean(dim=1)  # mean over patches

    return embedding.squeeze(0).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Extract VideoMAE video embeddings")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--pool", default="cls", choices=["cls", "mean"],
                        help="Pooling mode: cls token or mean over patches")
    parser.add_argument("--model-path", default=None,
                        help="Pre-downloaded model directory (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg.get("device", "cuda")

    video_dir = cfg["paths"]["video_dir"]
    output_dir = Path(cfg["paths"]["output_dir"])
    emb_out = output_dir / "emb" / "video"

    # ---- resolve model path (local > ModelScope) ----
    model_local = args.model_path or cfg["models"]["video"].get("local_path")
    ms_name = cfg["models"]["video"]["ms_name"]
    num_frames = cfg["models"]["video"]["frame_count"]
    img_size = cfg["models"]["video"]["img_size"]

    print(f"=== Video Embedding Extraction ===")
    print(f"  Video dir : {video_dir}")
    print(f"  Model     : {ms_name}  (via ModelScope)")
    print(f"  Frames    : {num_frames} @ {img_size}×{img_size}")
    print(f"  Pooling   : {args.pool}")
    print(f"  Device    : {device}")
    print(f"  Output    : {emb_out}")

    if not Path(video_dir).is_dir():
        print(f"[ERROR] Video directory not found: {video_dir}")
        print("[HINT] Are you running on the server with the data mounted?")
        sys.exit(1)

    local_path = resolve_model_path(model_local, ms_name)
    model, processor = load_videomae_local(local_path, device=device)

    # ---- iterate over HDXXX/ subdirs ----
    subdirs = sorted(
        d for d in Path(video_dir).iterdir()
        if d.is_dir() and extract_subject_id(d.name) is not None
    )
    if not subdirs:
        print(f"[ERROR] No HDXXX/ directories found in {video_dir}")
        sys.exit(1)

    print(f"  Subjects  : {len(subdirs)} directories")

    for subdir in tqdm(subdirs, desc="Extracting video"):
        sid = extract_subject_id(subdir.name)
        mp4_path = subdir / "4.mp4"
        if not mp4_path.exists():
            print(f"  [Skip] {sid}: 4.mp4 not found")
            continue

        try:
            emb = extract_video_embedding(
                str(mp4_path), model, processor,
                device=device, num_frames=num_frames, img_size=img_size,
                pool_mode=args.pool,
            )
            save_embedding(str(emb_out), sid, emb)
        except Exception as e:
            print(f"  [Error] {sid}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"Done. Embeddings saved to {emb_out}")


if __name__ == "__main__":
    main()
