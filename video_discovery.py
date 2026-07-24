"""Video dataset discovery across cohort/collection directories."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence


VIDEO_SUBJECT_PATTERN = re.compile(r"(HD|HC)\d{3}", re.IGNORECASE)


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
    preferred_filename: str = "4.mp4",
) -> List[VideoSource]:
    """Recursively locate one unambiguous MP4 per subject.

    If a subject directory contains multiple files, ``preferred_filename`` is
    selected only when it uniquely resolves the subject. Duplicate preferred
    files across collections are treated as an error instead of silently
    overwriting one recording with another.
    """
    grouped: Dict[str, List[tuple]] = {}
    missing = []
    for collection in collections:
        collection_dir = base_dir / collection
        if not collection_dir.is_dir():
            missing.append(str(collection_dir))
            continue
        for path in sorted(collection_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() != ".mp4":
                continue
            subject_id = video_subject_id(path.parent)
            if subject_id:
                grouped.setdefault(subject_id, []).append((path, collection))

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
    ambiguous = []
    preferred_lower = preferred_filename.lower()
    for subject_id, candidates in sorted(grouped.items()):
        unique = sorted(set(candidates), key=lambda item: str(item[0]))
        preferred = [
            item for item in unique if item[0].name.lower() == preferred_lower
        ]
        if len(preferred) == 1:
            chosen = preferred[0]
        elif len(unique) == 1:
            chosen = unique[0]
        else:
            ambiguous.append(
                f"{subject_id}: " + ", ".join(str(path) for path, _ in unique)
            )
            continue
        selected.append(
            VideoSource(
                subject_id=subject_id,
                video_path=chosen[0],
                collection=chosen[1],
            )
        )

    if ambiguous:
        raise ValueError(
            "Multiple MP4 files found for the same subject and no unique "
            f"'{preferred_filename}' could resolve them:\n  "
            + "\n  ".join(ambiguous[:20])
        )
    return selected
