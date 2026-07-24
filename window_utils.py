"""Utilities shared by windowed audio/video extraction and evaluation.

Each subject is stored as one compressed ``.npz`` file.  Keeping all windows
for a subject in one file makes it harder to accidentally split windows from
the same subject across train and test sets.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass
class WindowEmbeddings:
    """Window-level embeddings and their source time intervals."""

    embeddings: np.ndarray
    start_sec: np.ndarray
    end_sec: np.ndarray
    window_sec: np.ndarray
    valid_ratio: np.ndarray

    def validate(self) -> "WindowEmbeddings":
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.start_sec = np.asarray(self.start_sec, dtype=np.float32)
        self.end_sec = np.asarray(self.end_sec, dtype=np.float32)
        self.window_sec = np.asarray(self.window_sec, dtype=np.float32)
        self.valid_ratio = np.asarray(self.valid_ratio, dtype=np.float32)

        if self.embeddings.ndim != 2:
            raise ValueError(
                f"embeddings must have shape (num_windows, dim), got {self.embeddings.shape}"
            )
        n = self.embeddings.shape[0]
        for name in ("start_sec", "end_sec", "window_sec", "valid_ratio"):
            value = getattr(self, name)
            if value.shape != (n,):
                raise ValueError(f"{name} must have shape ({n},), got {value.shape}")
        if n == 0:
            raise ValueError("at least one window is required")
        if not np.isfinite(self.embeddings).all():
            raise ValueError("embeddings contain NaN or infinity")
        if np.any(self.end_sec <= self.start_sec):
            raise ValueError("each window must have end_sec > start_sec")
        return self


def parse_window_sizes(value: str) -> List[float]:
    """Parse a comma-separated window-size argument."""
    sizes = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("window sizes must be positive, e.g. '4,8,16'")
    return sizes


def make_sliding_windows(
    duration_sec: float,
    window_sec: float,
    overlap: float = 0.5,
) -> List[Tuple[float, float]]:
    """Return windows that cover the recording, including a final tail window."""
    if duration_sec <= 0:
        raise ValueError(f"duration_sec must be positive, got {duration_sec}")
    if window_sec <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec}")
    if not 0 <= overlap < 1:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    if duration_sec <= window_sec:
        return [(0.0, duration_sec)]

    stride = window_sec * (1.0 - overlap)
    last_start = duration_sec - window_sec
    starts = list(np.arange(0.0, last_start + 1e-8, stride, dtype=np.float64))
    if not starts or last_start - starts[-1] > 1e-6:
        starts.append(last_start)

    # Rounding avoids tiny duplicate intervals caused by floating point steps.
    unique_starts = sorted({round(float(start), 6) for start in starts})
    return [(start, min(start + window_sec, duration_sec)) for start in unique_starts]


def evenly_spaced_subset(items: Sequence[T], max_items: Optional[int]) -> List[T]:
    """Deterministically retain coverage of the full recording when capped."""
    values = list(items)
    if max_items is None or len(values) <= max_items:
        return values
    if max_items <= 0:
        raise ValueError("max_items must be positive or None")
    indices = np.rint(np.linspace(0, len(values) - 1, max_items)).astype(int)
    return [values[index] for index in indices]


def save_window_embeddings(
    emb_dir: str,
    subject_id: str,
    data: WindowEmbeddings,
    metadata: Optional[Dict[str, object]] = None,
) -> Path:
    """Atomically save all window embeddings for one subject."""
    data.validate()
    out_dir = Path(emb_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{subject_id}.npz"

    arrays: Dict[str, np.ndarray] = {
        "embeddings": data.embeddings,
        "start_sec": data.start_sec,
        "end_sec": data.end_sec,
        "window_sec": data.window_sec,
        "valid_ratio": data.valid_ratio,
    }
    for key, value in (metadata or {}).items():
        if value is not None:
            arrays[f"meta_{key}"] = np.asarray(value)

    handle, temp_name = tempfile.mkstemp(
        prefix=f".{subject_id}.", suffix=".npz", dir=str(out_dir)
    )
    os.close(handle)
    try:
        np.savez_compressed(temp_name, **arrays)
        os.replace(temp_name, destination)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return destination


def load_window_embeddings(path: str) -> WindowEmbeddings:
    """Load one subject-level ``.npz`` file."""
    with np.load(path, allow_pickle=False) as data:
        n = len(data["embeddings"])
        valid_ratio = data["valid_ratio"] if "valid_ratio" in data else np.ones(n)
        result = WindowEmbeddings(
            embeddings=data["embeddings"],
            start_sec=data["start_sec"],
            end_sec=data["end_sec"],
            window_sec=data["window_sec"],
            valid_ratio=valid_ratio,
        )
    return result.validate()


def available_window_sizes(records: Iterable[WindowEmbeddings]) -> List[float]:
    """Return window sizes present in every supplied subject record."""
    common: Optional[set] = None
    for record in records:
        sizes = {round(float(value), 6) for value in record.window_sec}
        common = sizes if common is None else common & sizes
    return sorted(common or set())


def aggregate_windows(
    data: WindowEmbeddings,
    window_sec: float,
    method: str = "mean_std",
    min_valid_ratio: float = 0.0,
) -> np.ndarray:
    """Aggregate a subject's selected windows into one fixed-length vector."""
    selected = np.isclose(data.window_sec, window_sec, rtol=0.0, atol=1e-4)
    selected &= data.valid_ratio >= min_valid_ratio
    values = data.embeddings[selected]
    if len(values) == 0:
        raise ValueError(
            f"no windows remain for window_sec={window_sec}, "
            f"min_valid_ratio={min_valid_ratio}"
        )

    if method == "mean":
        result = values.mean(axis=0)
    elif method == "mean_std":
        result = np.concatenate([values.mean(axis=0), values.std(axis=0)])
    elif method == "median":
        result = np.median(values, axis=0)
    elif method == "max":
        result = values.max(axis=0)
    else:
        raise ValueError(f"unknown aggregation method: {method}")
    return np.asarray(result, dtype=np.float32)
