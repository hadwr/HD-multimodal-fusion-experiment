#!/usr/bin/env python3
"""Extract corrected global and sliding-window VideoMAE embeddings.

VideoMAE does not prepend a CLS token.  The old implementation used token 0,
which is only the first spatiotemporal patch.  This implementation mean-pools
all patch tokens and samples *continuous temporal clips* for windowed output.

Outputs
-------
``output/emb/video/HDXXX.npy``
    Corrected whole-recording baseline (16 frames spread over the recording).
``output/emb_windows/video/HDXXX.npz``
    Window embeddings plus start/end timestamps and window sizes.

Examples
--------
python extract_video.py --mode both
python extract_video.py --mode windows --window-sizes 2,4,8 --overlap 0.5
"""

import argparse
from contextlib import nullcontext
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from utils import (
    extract_subject_id,
    load_config,
    resolve_model_path,
    save_embedding,
)
from window_utils import (
    WindowEmbeddings,
    evenly_spaced_subset,
    make_sliding_windows,
    parse_window_sizes,
    save_window_embeddings,
)


def load_videomae_local(
    local_path: str,
    device: str = "cuda",
    checkpoint: Optional[str] = None,
    num_frames: int = 16,
    image_size: int = 224,
):
    """Load either a Transformers model or OpenGVLab's raw VideoMAE V2 bundle."""
    from transformers import VideoMAEImageProcessor

    root = Path(local_path)
    raw_checkpoint = (
        root / checkpoint
        if checkpoint
        else root / "mae-b" / "pytorch_model.bin"
    )
    if raw_checkpoint.is_file() and not (root / "config.json").is_file():
        from videomaev2_native import load_native_videomaev2_base

        # The raw checkpoint bundle has no preprocessing metadata. These are
        # the standard VideoMAE ImageNet normalization/crop defaults.
        processor = VideoMAEImageProcessor(
            do_resize=True,
            size={"shortest_edge": image_size},
            do_center_crop=True,
            crop_size={"height": image_size, "width": image_size},
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )
        model, report = load_native_videomaev2_base(
            str(raw_checkpoint),
            device=device,
            num_frames=num_frames,
            image_size=image_size,
        )
        return model, processor, report

    try:
        processor = VideoMAEImageProcessor.from_pretrained(
            local_path, local_files_only=True
        )
    except OSError:
        processor = VideoMAEImageProcessor(
            do_resize=True,
            size={"shortest_edge": image_size},
            do_center_crop=True,
            crop_size={"height": image_size, "width": image_size},
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

    with open(root / "config.json", encoding="utf-8") as handle:
        import json

        model_config = json.load(handle)
    if model_config.get("auto_map"):
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=True, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            local_path,
            config=config,
            trust_remote_code=True,
            local_files_only=True,
        )
        model._hd_pixel_layout = "bcthw"
        backend = "videomaev2_transformers_custom"
    else:
        from transformers import VideoMAEModel

        model = VideoMAEModel.from_pretrained(
            local_path, local_files_only=True
        )
        model._hd_pixel_layout = "btchw"
        backend = "videomae_transformers"
    model = model.to(device)
    model.eval()
    report = {
        "backend": backend,
        "checkpoint": str(root),
        "parameter_coverage": 1.0,
        "missing_keys": [],
        "unexpected_keys": [],
        "hidden_size": getattr(model.config, "hidden_size", "unknown"),
    }
    return model, processor, report


def read_video_metadata(video_path: str) -> Tuple[int, float, float]:
    """Return ``(total_frames, fps, duration_sec)`` using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if total_frames <= 0:
        raise ValueError(f"No video frames found in {video_path}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"Invalid FPS ({fps}) in {video_path}. Re-encode the video or fix metadata."
        )
    return total_frames, fps, total_frames / fps


def _frame_indices(
    total_frames: int,
    fps: float,
    num_frames: int,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> np.ndarray:
    """Generate exactly ``num_frames`` indices inside one temporal interval."""
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    duration_sec = total_frames / fps
    start_sec = max(0.0, float(start_sec))
    end_sec = duration_sec if end_sec is None else min(float(end_sec), duration_sec)
    if end_sec <= start_sec:
        raise ValueError(f"empty video interval [{start_sec}, {end_sec}]")

    first = min(int(np.floor(start_sec * fps)), total_frames - 1)
    # end_sec is exclusive; subtracting a tiny value prevents selecting the next interval.
    last = min(int(np.ceil(end_sec * fps - 1e-8)) - 1, total_frames - 1)
    last = max(first, last)
    return np.rint(np.linspace(first, last, num_frames)).astype(np.int64)


def sample_frames_from_video(
    video_path: str,
    num_frames: int = 16,
    img_size: int = 224,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    metadata: Optional[Tuple[int, float, float]] = None,
) -> np.ndarray:
    """Sample frames from a continuous interval and return RGB uint8 frames."""
    import cv2

    total_frames, fps, _ = metadata or read_video_metadata(video_path)
    indices = _frame_indices(
        total_frames=total_frames,
        fps=fps,
        num_frames=num_frames,
        start_sec=start_sec,
        end_sec=end_sec,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames: List[np.ndarray] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise ValueError(
            f"Could not decode frames from {video_path} "
            f"within [{start_sec}, {end_sec}]"
        )
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    # VideoMAEImageProcessor performs its own resize/crop and normalization.
    return np.stack(frames[:num_frames], axis=0)


def pool_video_hidden(last_hidden: torch.Tensor, pool_mode: str = "mean") -> torch.Tensor:
    """Pool VideoMAE patch tokens. VideoMAE has no CLS token."""
    if last_hidden.ndim != 3:
        raise ValueError(
            f"expected hidden states with shape (B, tokens, dim), got {last_hidden.shape}"
        )
    if pool_mode == "mean":
        return last_hidden.mean(dim=1)
    if pool_mode == "mean_std":
        return torch.cat(
            [last_hidden.mean(dim=1), last_hidden.std(dim=1, unbiased=False)], dim=-1
        )
    raise ValueError(f"Unsupported pool mode: {pool_mode}")


@torch.inference_mode()
def encode_video_frames(
    frames: np.ndarray,
    model,
    processor,
    device: str = "cuda",
    pool_mode: str = "mean",
    use_amp: bool = False,
) -> np.ndarray:
    """Encode already sampled frames."""
    inputs = processor(list(frames), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    if getattr(model, "_hd_pixel_layout", "btchw") == "bcthw":
        pixel_values = pixel_values.permute(0, 2, 1, 3, 4).contiguous()
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and str(device).startswith("cuda")
        else nullcontext()
    )
    with amp_context:
        outputs = model(pixel_values=pixel_values)
        if isinstance(outputs, torch.Tensor):
            if outputs.ndim == 3:
                embedding = pool_video_hidden(outputs, pool_mode=pool_mode)
            elif outputs.ndim == 2 and pool_mode == "mean":
                embedding = outputs
            else:
                raise ValueError(
                    f"model returned {tuple(outputs.shape)}; "
                    f"pool={pool_mode} is not supported"
                )
        else:
            embedding = pool_video_hidden(
                outputs.last_hidden_state, pool_mode=pool_mode
            )
    return embedding.squeeze(0).float().cpu().numpy()


def extract_video_embedding(
    video_path: str,
    model,
    processor,
    device: str = "cuda",
    num_frames: int = 16,
    img_size: int = 224,
    pool_mode: str = "mean",
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
    metadata: Optional[Tuple[int, float, float]] = None,
    use_amp: bool = False,
) -> np.ndarray:
    """Sample one interval, encode it, and pool all VideoMAE patch tokens."""
    frames = sample_frames_from_video(
        video_path,
        num_frames=num_frames,
        img_size=img_size,
        start_sec=start_sec,
        end_sec=end_sec,
        metadata=metadata,
    )
    return encode_video_frames(
        frames,
        model=model,
        processor=processor,
        device=device,
        pool_mode=pool_mode,
        use_amp=use_amp,
    )


def extract_video_windows(
    video_path: str,
    model,
    processor,
    window_sizes: List[float],
    overlap: float,
    device: str,
    num_frames: int,
    img_size: int,
    pool_mode: str,
    use_amp: bool,
    max_windows_per_size: Optional[int],
) -> Tuple[WindowEmbeddings, Tuple[int, float, float]]:
    """Extract all requested window sizes for one video."""
    metadata = read_video_metadata(video_path)
    _, _, duration_sec = metadata
    embeddings, starts, ends, sizes, valid_ratios = [], [], [], [], []

    for requested_size in window_sizes:
        intervals = make_sliding_windows(
            duration_sec, requested_size, overlap=overlap
        )
        intervals = evenly_spaced_subset(intervals, max_windows_per_size)
        for start_sec, end_sec in intervals:
            embeddings.append(
                extract_video_embedding(
                    video_path,
                    model=model,
                    processor=processor,
                    device=device,
                    num_frames=num_frames,
                    img_size=img_size,
                    pool_mode=pool_mode,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    metadata=metadata,
                    use_amp=use_amp,
                )
            )
            starts.append(start_sec)
            ends.append(end_sec)
            sizes.append(requested_size)
            valid_ratios.append(min(1.0, (end_sec - start_sec) / requested_size))

    record = WindowEmbeddings(
        embeddings=np.stack(embeddings),
        start_sec=np.asarray(starts),
        end_sec=np.asarray(ends),
        window_sec=np.asarray(sizes),
        valid_ratio=np.asarray(valid_ratios),
    ).validate()
    return record, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Extract corrected global and windowed VideoMAE embeddings"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu")
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow ModelScope access if no local checkpoint is found",
    )
    parser.add_argument(
        "--window-output-dir",
        default=None,
        help="Override output/emb_windows/video for checkpoint ablations",
    )
    parser.add_argument("--mode", choices=["global", "windows", "both"], default=None)
    parser.add_argument("--pool", choices=["mean", "mean_std"], default=None)
    parser.add_argument("--window-sizes", default=None, help="Seconds, e.g. 2,4,8")
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--amp", action="store_true", help="Use CUDA float16 autocast")
    parser.add_argument("--subject-ids", default=None, help="Comma-separated HD IDs")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N subjects")
    parser.add_argument("--max-windows-per-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    video_cfg = cfg["models"]["video"]
    window_cfg = cfg.get("windowing", {}).get("video", {})
    device = args.device or cfg.get("device", "cuda")
    mode = args.mode or window_cfg.get("mode", "both")
    pool_mode = args.pool or window_cfg.get("pool", "mean")
    overlap = args.overlap
    if overlap is None:
        overlap = float(window_cfg.get("overlap", 0.5))
    sizes_value = args.window_sizes or ",".join(
        str(value) for value in window_cfg.get("sizes_sec", [2, 4, 8])
    )
    window_sizes = parse_window_sizes(sizes_value)
    max_windows_per_size = (
        args.max_windows_per_size
        if args.max_windows_per_size is not None
        else int(window_cfg.get("max_windows_per_size", 64))
    )

    video_dir = cfg["paths"]["video_dir"]
    output_dir = Path(cfg["paths"]["output_dir"])
    global_out = output_dir / "emb" / "video"
    window_out = (
        Path(args.window_output_dir)
        if args.window_output_dir
        else output_dir / "emb_windows" / "video"
    )
    num_frames = int(video_cfg["frame_count"])
    img_size = int(video_cfg["img_size"])

    print("=== Corrected VideoMAE Embedding Extraction ===")
    print(f"  Video dir       : {video_dir}")
    print(f"  Mode            : {mode}")
    print(f"  Pooling         : {pool_mode} (all patch tokens; no CLS)")
    print(f"  Window sizes    : {window_sizes} sec, overlap={overlap:.2f}")
    print(f"  Max windows     : {max_windows_per_size} per size and subject")
    print(f"  Frames per clip : {num_frames}")
    print(f"  Device / AMP    : {device} / {args.amp}")

    if not Path(video_dir).is_dir():
        print(f"[ERROR] Video directory not found: {video_dir}")
        sys.exit(1)

    model_local = args.model_path or video_cfg.get("local_path")
    local_path = resolve_model_path(
        model_local,
        video_cfg["ms_name"],
        cache_dir=cfg["models"].get("cache_dir"),
        allow_download=args.allow_model_download,
    )
    model, processor, load_report = load_videomae_local(
        local_path,
        device=device,
        checkpoint=video_cfg.get("checkpoint"),
        num_frames=num_frames,
        image_size=img_size,
    )
    print(f"  Backend         : {load_report['backend']}")
    print(f"  Checkpoint      : {load_report['checkpoint']}")
    print(f"  Weight coverage : {load_report['parameter_coverage']:.1%}")
    print(f"  Hidden size     : {load_report['hidden_size']}")
    if load_report["missing_keys"]:
        print(f"  Missing keys    : {load_report['missing_keys'][:8]}")

    subdirs = sorted(
        path
        for path in Path(video_dir).iterdir()
        if path.is_dir() and extract_subject_id(path.name) is not None
    )
    if args.subject_ids:
        requested_ids = {
            value.strip().upper()
            for value in args.subject_ids.split(",")
            if value.strip()
        }
        subdirs = [
            path for path in subdirs if extract_subject_id(path.name) in requested_ids
        ]
    if args.limit is not None:
        subdirs = subdirs[: args.limit]
    if not subdirs:
        print(f"[ERROR] No HDXXX directories found in {video_dir}")
        sys.exit(1)

    failures = 0
    for subdir in tqdm(subdirs, desc="Extracting video"):
        sid = extract_subject_id(subdir.name)
        video_path = subdir / "4.mp4"
        if not video_path.exists():
            print(f"  [Skip] {sid}: 4.mp4 not found")
            failures += 1
            continue
        try:
            metadata = read_video_metadata(str(video_path))
            if mode in {"global", "both"}:
                global_embedding = extract_video_embedding(
                    str(video_path),
                    model=model,
                    processor=processor,
                    device=device,
                    num_frames=num_frames,
                    img_size=img_size,
                    pool_mode=pool_mode,
                    metadata=metadata,
                    use_amp=args.amp,
                )
                save_embedding(str(global_out), sid, global_embedding)

            if mode in {"windows", "both"}:
                record, metadata = extract_video_windows(
                    str(video_path),
                    model=model,
                    processor=processor,
                    window_sizes=window_sizes,
                    overlap=overlap,
                    device=device,
                    num_frames=num_frames,
                    img_size=img_size,
                    pool_mode=pool_mode,
                    use_amp=args.amp,
                    max_windows_per_size=max_windows_per_size,
                )
                _, fps, duration_sec = metadata
                destination = save_window_embeddings(
                    str(window_out),
                    sid,
                    record,
                    metadata={
                        "source": str(video_path),
                        "fps": fps,
                        "duration_sec": duration_sec,
                        "pool": pool_mode,
                        "num_frames": num_frames,
                        "max_windows_per_size": max_windows_per_size,
                    },
                )
                print(f"  Saved windows: {destination} ({len(record.embeddings)})")
        except Exception as exc:
            failures += 1
            print(f"  [Error] {sid}: {exc}")
            import traceback

            traceback.print_exc()

    print(f"Done. Subjects={len(subdirs)}, failures={failures}")
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
