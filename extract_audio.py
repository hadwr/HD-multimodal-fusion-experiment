#!/usr/bin/env python3
"""Extract global and sliding-window wav2vec2/WavLM embeddings.

The windowed format keeps timestamps and a simple energy-based speech ratio.
Windows remain grouped by subject in one ``.npz`` file so evaluation can be
performed at subject level without leakage.
"""

import argparse
from contextlib import nullcontext
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchaudio
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


def load_audio_model_local(local_path: str, device: str = "cuda"):
    """Load any local checkpoint compatible with ``AutoModel`` for raw speech."""
    from transformers import AutoFeatureExtractor, AutoModel

    feature_extractor = AutoFeatureExtractor.from_pretrained(local_path)
    model = AutoModel.from_pretrained(local_path)
    model = model.to(device)
    model.eval()
    return model, feature_extractor


# Backwards-compatible alias for code importing the previous function.
load_wav2vec2_local = load_audio_model_local


def load_mono_audio(audio_path: str, target_sr: int) -> Tuple[torch.Tensor, int]:
    """Load, mono-convert and resample a file; return a contiguous 1-D tensor."""
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.ndim != 2 or waveform.shape[1] == 0:
        raise ValueError(f"invalid waveform shape in {audio_path}: {waveform.shape}")
    waveform = waveform.mean(dim=0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr
    return waveform.contiguous(), sample_rate


def estimate_speech_activity(
    waveform: torch.Tensor,
    sample_rate: int,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    top_db: float = 35.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return energy-frame center times and a conservative active-speech mask."""
    frame_length = max(1, round(sample_rate * frame_ms / 1000.0))
    hop_length = max(1, round(sample_rate * hop_ms / 1000.0))
    if waveform.numel() < frame_length:
        padded = torch.nn.functional.pad(waveform, (0, frame_length - waveform.numel()))
    else:
        padded = waveform
    frames = padded.unfold(0, frame_length, hop_length)
    rms = frames.float().pow(2).mean(dim=1).sqrt().clamp_min(1e-10)
    db = 20.0 * torch.log10(rms)
    if float(db.max()) < -60.0:
        active = torch.zeros_like(db, dtype=torch.bool)
    else:
        active = db >= (db.max() - top_db)
    centers = (
        torch.arange(len(frames), dtype=torch.float32) * hop_length + frame_length / 2
    ) / sample_rate
    return centers.numpy(), active.numpy()


def interval_speech_ratio(
    frame_centers: np.ndarray,
    activity: np.ndarray,
    start_sec: float,
    end_sec: float,
) -> float:
    """Compute active-frame fraction for one interval."""
    selected = (frame_centers >= start_sec) & (frame_centers < end_sec)
    if not selected.any():
        return 0.0
    return float(activity[selected].mean())


def pool_audio_hidden(
    hidden: torch.Tensor,
    pool_mode: str = "mean",
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pool frame-level speech representations."""
    if hidden.ndim != 3:
        raise ValueError(f"expected (B, time, dim), got {hidden.shape}")

    # Single unpadded windows are the normal path.  This branch also supports
    # future batched/padded extraction when a feature-vector attention mask is
    # supplied at the same temporal resolution as ``hidden``.
    if attention_mask is not None and attention_mask.shape[1] == hidden.shape[1]:
        weights = attention_mask.to(hidden.dtype).unsqueeze(-1)
        count = weights.sum(dim=1).clamp_min(1.0)
        mean = (hidden * weights).sum(dim=1) / count
        variance = ((hidden - mean.unsqueeze(1)).pow(2) * weights).sum(dim=1) / count
        std = variance.clamp_min(0.0).sqrt()
    else:
        mean = hidden.mean(dim=1)
        std = hidden.std(dim=1, unbiased=False)

    if pool_mode == "mean":
        return mean
    if pool_mode == "mean_std":
        return torch.cat([mean, std], dim=-1)
    raise ValueError(f"Unsupported pool mode: {pool_mode}")


@torch.inference_mode()
def encode_audio_segment(
    waveform: torch.Tensor,
    model,
    processor,
    sample_rate: int,
    device: str,
    pool_mode: str = "mean",
    layer: int = -1,
    use_amp: bool = False,
) -> np.ndarray:
    """Encode one waveform segment and return a fixed-length vector."""
    audio_np = waveform.detach().cpu().numpy()
    inputs = processor(audio_np, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    output_hidden_states = layer != -1
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp and str(device).startswith("cuda")
        else nullcontext()
    )
    with amp_context:
        outputs = model(**inputs, output_hidden_states=output_hidden_states)
        if layer == -1:
            hidden = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states
            if not -len(hidden_states) <= layer < len(hidden_states):
                raise ValueError(
                    f"layer {layer} outside available range "
                    f"[-{len(hidden_states)}, {len(hidden_states) - 1}]"
                )
            hidden = hidden_states[layer]
        embedding = pool_audio_hidden(hidden, pool_mode=pool_mode)
    return embedding.squeeze(0).float().cpu().numpy()


def extract_audio_embedding(
    audio_path: str,
    model,
    processor,
    device: str = "cuda",
    target_sr: int = 16000,
    pool_mode: str = "mean",
    layer: int = -1,
    use_amp: bool = False,
) -> np.ndarray:
    """Backwards-compatible whole-recording extraction."""
    waveform, sample_rate = load_mono_audio(audio_path, target_sr)
    return encode_audio_segment(
        waveform,
        model=model,
        processor=processor,
        sample_rate=sample_rate,
        device=device,
        pool_mode=pool_mode,
        layer=layer,
        use_amp=use_amp,
    )


def _slice_audio(
    waveform: torch.Tensor,
    sample_rate: int,
    start_sec: float,
    end_sec: float,
) -> torch.Tensor:
    start = max(0, int(round(start_sec * sample_rate)))
    end = min(waveform.numel(), int(round(end_sec * sample_rate)))
    if end <= start:
        raise ValueError(f"empty audio interval [{start_sec}, {end_sec}]")
    return waveform[start:end]


def extract_audio_windows(
    waveform: torch.Tensor,
    sample_rate: int,
    model,
    processor,
    window_sizes: List[float],
    overlap: float,
    min_speech_ratio: float,
    vad_top_db: float,
    device: str,
    pool_mode: str,
    layer: int,
    use_amp: bool,
    max_windows_per_size: Optional[int],
) -> WindowEmbeddings:
    """Extract multiple window scales from one in-memory waveform."""
    duration_sec = waveform.numel() / sample_rate
    frame_centers, activity = estimate_speech_activity(
        waveform, sample_rate, top_db=vad_top_db
    )
    embeddings, starts, ends, sizes, speech_ratios = [], [], [], [], []

    for requested_size in window_sizes:
        candidates = []
        for start_sec, end_sec in make_sliding_windows(
            duration_sec, requested_size, overlap=overlap
        ):
            ratio = interval_speech_ratio(
                frame_centers, activity, start_sec=start_sec, end_sec=end_sec
            )
            candidates.append((ratio, start_sec, end_sec))

        selected = [item for item in candidates if item[0] >= min_speech_ratio]
        if not selected:
            # Never silently lose a subject. Keep the least-silent window and
            # preserve its low speech ratio for downstream filtering/sensitivity.
            selected = [max(candidates, key=lambda item: item[0])]
        selected = evenly_spaced_subset(selected, max_windows_per_size)

        for ratio, start_sec, end_sec in selected:
            segment = _slice_audio(waveform, sample_rate, start_sec, end_sec)
            embeddings.append(
                encode_audio_segment(
                    segment,
                    model=model,
                    processor=processor,
                    sample_rate=sample_rate,
                    device=device,
                    pool_mode=pool_mode,
                    layer=layer,
                    use_amp=use_amp,
                )
            )
            starts.append(start_sec)
            ends.append(end_sec)
            sizes.append(requested_size)
            speech_ratios.append(ratio)

    return WindowEmbeddings(
        embeddings=np.stack(embeddings),
        start_sec=np.asarray(starts),
        end_sec=np.asarray(ends),
        window_sec=np.asarray(sizes),
        valid_ratio=np.asarray(speech_ratios),
    ).validate()


def main():
    parser = argparse.ArgumentParser(
        description="Extract global and windowed wav2vec2/WavLM embeddings"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow ModelScope access if no local checkpoint is found",
    )
    parser.add_argument(
        "--window-output-dir",
        default=None,
        help="Override output/emb_windows/audio (useful for layer/backbone ablations)",
    )
    parser.add_argument("--mode", choices=["global", "windows", "both"], default=None)
    parser.add_argument("--pool", choices=["mean", "mean_std"], default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--window-sizes", default=None, help="Seconds, e.g. 4,8,16")
    parser.add_argument("--overlap", type=float, default=None)
    parser.add_argument("--min-speech-ratio", type=float, default=None)
    parser.add_argument("--vad-top-db", type=float, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--subject-ids", default=None, help="Comma-separated HD IDs")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files")
    parser.add_argument("--max-windows-per-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    audio_cfg = cfg["models"]["audio"]
    window_cfg = cfg.get("windowing", {}).get("audio", {})
    device = args.device or cfg.get("device", "cuda")
    mode = args.mode or window_cfg.get("mode", "both")
    pool_mode = args.pool or window_cfg.get("pool", "mean")
    layer = args.layer if args.layer is not None else int(window_cfg.get("layer", -1))
    overlap = (
        args.overlap
        if args.overlap is not None
        else float(window_cfg.get("overlap", 0.5))
    )
    min_speech_ratio = (
        args.min_speech_ratio
        if args.min_speech_ratio is not None
        else float(window_cfg.get("min_speech_ratio", 0.2))
    )
    vad_top_db = (
        args.vad_top_db
        if args.vad_top_db is not None
        else float(window_cfg.get("vad_top_db", 35.0))
    )
    sizes_value = args.window_sizes or ",".join(
        str(value) for value in window_cfg.get("sizes_sec", [4, 8, 16])
    )
    window_sizes = parse_window_sizes(sizes_value)
    max_windows_per_size = (
        args.max_windows_per_size
        if args.max_windows_per_size is not None
        else int(window_cfg.get("max_windows_per_size", 128))
    )

    audio_dir = cfg["paths"]["audio_dir"]
    output_dir = Path(cfg["paths"]["output_dir"])
    global_out = output_dir / "emb" / "audio"
    window_out = (
        Path(args.window_output_dir)
        if args.window_output_dir
        else output_dir / "emb_windows" / "audio"
    )
    target_sr = int(audio_cfg["sample_rate"])

    print("=== Windowed Speech Embedding Extraction ===")
    print(f"  Audio dir       : {audio_dir}")
    print(f"  Mode            : {mode}")
    print(f"  Pool / layer    : {pool_mode} / {layer}")
    print(f"  Window sizes    : {window_sizes} sec, overlap={overlap:.2f}")
    print(f"  Max windows     : {max_windows_per_size} per size and subject")
    print(f"  Speech filter   : ratio>={min_speech_ratio}, top_db={vad_top_db}")
    print(f"  Device / AMP    : {device} / {args.amp}")

    if not Path(audio_dir).is_dir():
        print(f"[ERROR] Audio directory not found: {audio_dir}")
        sys.exit(1)

    model_local = args.model_path or audio_cfg.get("local_path")
    local_path = resolve_model_path(
        model_local,
        audio_cfg["ms_name"],
        cache_dir=cfg["models"].get("cache_dir"),
        allow_download=args.allow_model_download,
    )
    model, processor = load_audio_model_local(local_path, device=device)
    print(f"  Architecture    : {getattr(model.config, 'architectures', None)}")
    print(f"  Hidden size     : {getattr(model.config, 'hidden_size', 'unknown')}")

    wav_files = sorted(Path(audio_dir).glob("*.wav"))
    if args.subject_ids:
        requested_ids = {
            value.strip().upper()
            for value in args.subject_ids.split(",")
            if value.strip()
        }
        wav_files = [
            path for path in wav_files if extract_subject_id(path.stem) in requested_ids
        ]
    if args.limit is not None:
        wav_files = wav_files[: args.limit]
    if not wav_files:
        print(f"[ERROR] No .wav files found in {audio_dir}")
        sys.exit(1)

    failures = 0
    for wav_path in tqdm(wav_files, desc="Extracting audio"):
        sid = extract_subject_id(wav_path.stem)
        if sid is None:
            print(f"  [Skip] Cannot parse subject ID from {wav_path.name}")
            failures += 1
            continue
        try:
            waveform, sample_rate = load_mono_audio(str(wav_path), target_sr)
            duration_sec = waveform.numel() / sample_rate
            if mode in {"windows", "both"}:
                record = extract_audio_windows(
                    waveform,
                    sample_rate=sample_rate,
                    model=model,
                    processor=processor,
                    window_sizes=window_sizes,
                    overlap=overlap,
                    min_speech_ratio=min_speech_ratio,
                    vad_top_db=vad_top_db,
                    device=device,
                    pool_mode=pool_mode,
                    layer=layer,
                    use_amp=args.amp,
                    max_windows_per_size=max_windows_per_size,
                )
                destination = save_window_embeddings(
                    str(window_out),
                    sid,
                    record,
                    metadata={
                        "source": str(wav_path),
                        "sample_rate": sample_rate,
                        "duration_sec": duration_sec,
                        "pool": pool_mode,
                        "layer": layer,
                        "min_speech_ratio": min_speech_ratio,
                        "vad_top_db": vad_top_db,
                        "max_windows_per_size": max_windows_per_size,
                    },
                )
                print(f"  Saved windows: {destination} ({len(record.embeddings)})")

            # Whole-recording inference can use much more memory than windowed
            # inference, so run it second. If it fails, the valuable window
            # file for this subject has already been saved.
            if mode in {"global", "both"}:
                global_embedding = encode_audio_segment(
                    waveform,
                    model=model,
                    processor=processor,
                    sample_rate=sample_rate,
                    device=device,
                    pool_mode=pool_mode,
                    layer=layer,
                    use_amp=args.amp,
                )
                save_embedding(str(global_out), sid, global_embedding)
        except Exception as exc:
            failures += 1
            print(f"  [Error] {sid}: {exc}")
            import traceback

            traceback.print_exc()

    print(f"Done. Subjects={len(wav_files)}, failures={failures}")
    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
