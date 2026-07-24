"""Video dataset discovery across cohort/collection directories."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence


VIDEO_SUBJECT_PATTERN = re.compile(r"(HD|HC)\d{3}", re.IGNORECASE)
VIDEO_TAKE_PATTERN = re.compile(
    r"^4\s*(?:[（(]\s*(\d+)\s*[）)])?\.mp4$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VideoSource:
    subject_id: str
    video_path: Path
    collection: str


def video_subject_id(path: Path) -> Optional[str]:
    """Find the nearest HDXXX/HCXXX identifier in a path."""
    for part in reversed(path.parts):
        match = VIDEO_SUBJECT_PATTERN.search(part)
        if match:
            return match.group(0).upper()
    return None


def discover_video_sources(
    base_dir: Path,
    collections: Sequence[str],
) -> List[VideoSource]:
    """Recursively select the first ``4`` recording for each subject.

    ``4.mp4`` is first, followed by duplicate names such as ``4（1）.mp4`` and
    ``4(1).mp4`` in numeric order. If a subject occurs in multiple collections,
    the configured collection order is the deterministic tie-breaker. Subjects
    with no matching ``4`` recording are logged and skipped.
    """
    grouped: Dict[str, List[tuple]] = {}
    missing = []
    for collection_index, collection in enumerate(collections):
        collection_dir = base_dir / collection
        if not collection_dir.is_dir():
            missing.append(str(collection_dir))
            continue
        for path in sorted(collection_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".mp4":
                continue
            subject_id = video_subject_id(path.parent)
            if subject_id:
                grouped.setdefault(subject_id, []).append(
                    (path, collection, collection_index)
                )

    if missing:
        raise FileNotFoundError(
            "Configured video collection directories not found: "
            + ", ".join(missing)
        )
    if not grouped:
        raise ValueError(
            f"No MP4 files below configured collections in {base_dir}: "
            f"{list(collections)}"
        )

    selected: List[VideoSource] = []
    for subject_id, candidates in sorted(grouped.items()):
        unique = list(
            {
                (str(path), collection, collection_index): (
                    path,
                    collection,
                    collection_index,
                )
                for path, collection, collection_index in candidates
            }.values()
        )
        matching = []
        for path, collection, collection_index in unique:
            match = VIDEO_TAKE_PATTERN.fullmatch(path.name)
            if match:
                # Exact 4.mp4 ranks before copy 1, copy 2, ...
                copy_index = int(match.group(1) or 0)
                matching.append(
                    (copy_index, collection_index, str(path), path, collection)
                )
        if not matching:
            print(
                f"  [Skip] {subject_id}: no 4.mp4/4(n).mp4 recording; "
                "available MP4 files: "
                + ", ".join(path.name for path, _, _ in unique)
            )
            continue
        matching.sort(key=lambda item: item[:3])
        _, _, _, chosen_path, chosen_collection = matching[0]
        if len(matching) > 1:
            print(
                f"  [Select] {subject_id}: using {chosen_path}; "
                f"ignored {len(matching) - 1} duplicate 4 recording(s)"
            )
        selected.append(
            VideoSource(
                subject_id=subject_id,
                video_path=chosen_path,
                collection=chosen_collection,
            )
        )

    return selected
