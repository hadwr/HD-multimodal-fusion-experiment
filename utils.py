"""
Shared utilities for the HD multimodal fusion experiment.

- Config loading
- ModelScope → transformers bridge
- Subject ID extraction & matching
- Embedding save/load helpers
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml


# ============================================================
# Config
# ============================================================

def load_config(config_path: str = "config.yaml") -> dict:
    """Load YAML config and resolve relative paths against the config file's directory."""
    config_path = Path(config_path)
    if not config_path.exists():
        # Try relative to the project root (parent of this file)
        alt = Path(__file__).resolve().parent / "config.yaml"
        if alt.exists():
            config_path = alt
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Resolve output_dir relative to config location
    out = cfg["paths"].get("output_dir", "./output")
    if not Path(out).is_absolute():
        cfg["paths"]["output_dir"] = str(config_path.parent.resolve() / out)

    return cfg


# ============================================================
# ModelScope → transformers bridge
# ============================================================

def resolve_model_path(local_path: Optional[str], ms_model_id: str, cache_dir: Optional[str] = None) -> str:
    """
    Resolve model path: use ``local_path`` if provided, otherwise download from ModelScope.

    Parameters
    ----------
    local_path : str or None
        Pre-downloaded model directory. If set, returned as-is.
    ms_model_id : str
        ModelScope model ID, used only if local_path is None.
    cache_dir : str, optional

    Returns
    -------
    str
        Absolute path to the model directory.
    """
    if local_path and Path(local_path).is_dir():
        print(f"[Model] Using local path: {local_path}")
        return local_path
    if local_path:
        print(f"[Model] Local path not found: {local_path}, falling back to ModelScope.")
    return download_model_from_modelscope(ms_model_id, cache_dir=cache_dir)


def download_model_from_modelscope(ms_model_id: str, cache_dir: Optional[str] = None) -> str:
    """
    Download a model from ModelScope and return the local path
    that can be passed to ``transformers.AutoModel.from_pretrained(local_path)``.

    Parameters
    ----------
    ms_model_id : str
        ModelScope model ID, e.g. "iic/wav2vec2-base".
    cache_dir : str, optional
        Where to cache downloaded models. Defaults to ``~/.cache/modelscope/hub``.

    Returns
    -------
    local_path : str
        Absolute path to the downloaded model directory.
    """
    from modelscope.hub.snapshot_download import snapshot_download

    local_path = snapshot_download(ms_model_id, cache_dir=cache_dir)
    print(f"[ModelScope] {ms_model_id} → {local_path}")
    return local_path


# ============================================================
# Subject ID helpers
# ============================================================

SUBJECT_ID_PATTERN = re.compile(r"(HD\d{3})", re.IGNORECASE)


def extract_subject_id(path: str) -> Optional[str]:
    """
    Extract HDXXX subject ID from a file or directory path.

    Examples
    --------
    >>> extract_subject_id("/data/HDacoustic/HD001.wav")
    'HD001'
    >>> extract_subject_id("/data/preprocessed_audio/HD042/4.mp4")
    'HD042'
    """
    name = str(path)
    m = SUBJECT_ID_PATTERN.search(name)
    return m.group(1).upper() if m else None


def match_subjects(audio_dir: str, video_dir: str) -> Dict[str, dict]:
    """
    Scan audio and video directories, match subjects by HDXXX ID.

    Returns
    -------
    dict[str, dict]
        {"HD001": {"audio": "/path/HD001.wav", "video": "/path/HD001/4.mp4"}, ...}
        Only subjects with BOTH modalities present are returned.
    """
    subjects = {}

    # Scan audio files
    audio_path = Path(audio_dir)
    if audio_path.is_dir():
        for f in audio_path.glob("*.wav"):
            sid = extract_subject_id(f.stem)
            if sid:
                subjects.setdefault(sid, {})["audio"] = str(f)

    # Scan video dirs
    video_path = Path(video_dir)
    if video_path.is_dir():
        for subdir in video_path.iterdir():
            if not subdir.is_dir():
                continue
            sid = extract_subject_id(subdir.name)
            if sid is None:
                continue
            mp4 = subdir / "4.mp4"
            if mp4.exists():
                subjects.setdefault(sid, {})["video"] = str(mp4)

    # Keep only subjects with both modalities
    matched = {sid: paths for sid, paths in subjects.items() if "audio" in paths and "video" in paths}
    skipped = {sid: paths for sid, paths in subjects.items() if sid not in matched}

    print(f"[Match] {len(matched)} subjects with both audio+video found.")
    if skipped:
        missing = []
        for sid, p in skipped.items():
            mm = []
            if "audio" not in p:
                mm.append("audio")
            if "video" not in p:
                mm.append("video")
            missing.append(f"  {sid}: missing {', '.join(mm)}")
        print(f"[Match] {len(skipped)} subjects skipped (missing one modality):")
        for line in missing:
            print(line)

    return matched


# ============================================================
# Embedding I/O
# ============================================================

def save_embedding(emb_dir: str, subject_id: str, vector: np.ndarray):
    """Save a single embedding vector as .npy."""
    out = Path(emb_dir)
    out.mkdir(parents=True, exist_ok=True)
    fpath = out / f"{subject_id}.npy"
    np.save(str(fpath), vector)
    print(f"  Saved: {fpath}")


def load_embeddings(emb_dir: str, subject_ids: Optional[list] = None) -> dict:
    """
    Load all .npy embeddings from a directory.

    Parameters
    ----------
    emb_dir : str
        Directory containing ``HDXXX.npy`` files.
    subject_ids : list[str], optional
        If given, only load these subjects.

    Returns
    -------
    dict[str, np.ndarray]
        {subject_id: embedding_vector}
    """
    emb_dir = Path(emb_dir)
    if not emb_dir.is_dir():
        print(f"[Warn] Embedding directory not found: {emb_dir}")
        return {}

    embeddings = {}
    for f in emb_dir.glob("*.npy"):
        sid = extract_subject_id(f.stem)
        if sid is None:
            continue
        if subject_ids is not None and sid not in subject_ids:
            continue
        embeddings[sid] = np.load(str(f))
    return embeddings


def merge_embeddings(
    audio_emb: dict,
    video_emb: dict,
    tabular_feats: Optional[dict] = None,
) -> tuple:
    """
    Merge audio, video, and optional tabular embeddings into aligned arrays.

    Only subjects present in ALL provided modalities are kept.

    Returns
    -------
    (subject_ids, X_dict, y)
        subject_ids: list[str]
        X_dict: dict[str, np.ndarray] with keys "audio", "video", "tabular" (if provided)
        y: np.ndarray or None
    """
    ids_audio = set(audio_emb.keys())
    ids_video = set(video_emb.keys())
    common = ids_audio & ids_video

    if tabular_feats is not None:
        common &= set(tabular_feats.keys())

    common = sorted(common)
    print(f"[Merge] {len(common)} subjects with all modalities present.")

    result = {}
    result["audio"] = np.stack([audio_emb[sid] for sid in common])
    result["video"] = np.stack([video_emb[sid] for sid in common])
    if tabular_feats is not None:
        result["tabular"] = np.stack([tabular_feats[sid] for sid in common])

    return common, result
