#!/usr/bin/env python3
"""
Load and merge tabular data: MemTrax metrics + stage labels.

1. Reads the MemTrax xlsx (one row per subject, columns = cognitive metrics).
2. Reads the label xlsx (columns: ``sample_id``, ``stages``).
3. Matches subjects by HDXXX ID.
4. Saves a merged CSV: ``output/data/merged.csv``.

Usage::

    python load_tabular.py
    python load_tabular.py --config my.yaml
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from utils import load_config, extract_subject_id


def load_labels(label_path: str) -> pd.DataFrame:
    """
    Load the label xlsx.

    Auto-detects the subject-ID column (looks for ``sample_id``, ``filename``, or
    any column containing "id").  Requires a ``stages`` column.

    Returns DataFrame with index = normalized subject ID, columns = [stage, ...].
    """
    df = pd.read_excel(label_path)

    if "stages" not in df.columns:
        raise KeyError(f"Label file must have a 'stages' column. Found: {list(df.columns)}")

    # --- auto-detect subject-ID column ---
    id_col = _find_id_column(df)
    if id_col is None:
        raise KeyError(
            f"Could not find a subject-ID column (tried: sample_id, filename, "
            f"or any column with 'id' in name).  Found: {list(df.columns)}"
        )

    # Normalize subject IDs
    df["subject_id"] = df[id_col].apply(lambda x: extract_subject_id(str(x)))
    df = df.dropna(subset=["subject_id"])
    df = df.set_index("subject_id")
    return df[["stages"]]


def _find_id_column(df: pd.DataFrame) -> str | None:
    """Scan columns for a likely subject-ID column."""
    def _has_hd_pattern(col: str) -> bool:
        """Check if any value in this column matches HDXXX."""
        for val in df[col].head(10):
            if extract_subject_id(str(val)):
                return True
        return False

    # Priority order
    for candidate in ["sample_id", "filename"]:
        if candidate in df.columns:
            if _has_hd_pattern(candidate):
                return candidate
    # Fallback: any column with "id" or "name" in name
    for col in df.columns:
        lower = str(col).lower()
        if "id" in lower or "name" in lower:
            if _has_hd_pattern(col):
                return col
    return None


def load_memtrax(memtrax_path: str) -> pd.DataFrame:
    """
    Load the MemTrax xlsx.

    Expected: one row per subject, columns are metric names (accuracy, reaction time, etc.).

    Returns DataFrame with index = normalized subject ID.
    """
    df = pd.read_excel(memtrax_path)

    # Try to find the subject ID column
    id_col = _find_id_column(df)
    if id_col is None:
        raise KeyError(
            f"Could not find a subject-ID column in MemTrax file. "
            f"Columns: {list(df.columns)}"
        )

    df["subject_id"] = df[id_col].apply(lambda x: extract_subject_id(str(x)))
    df = df.dropna(subset=["subject_id"])
    df = df.set_index("subject_id")
    df = df.drop(columns=[id_col])
    return df


def main():
    parser = argparse.ArgumentParser(description="Load & merge tabular data")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    label_xlsx = cfg["paths"]["label_xlsx"]
    memtrax_xlsx = cfg["paths"]["memtrax_xlsx"]
    output_dir = Path(cfg["paths"]["output_dir"])
    data_out = output_dir / "data"
    data_out.mkdir(parents=True, exist_ok=True)

    # ---- load labels ----
    print(f"=== Tabular Data Loading ===")
    print(f"  Labels   : {label_xlsx}")

    if not Path(label_xlsx).is_file():
        print(f"[ERROR] Label file not found: {label_xlsx}")
        sys.exit(1)

    labels = load_labels(label_xlsx)
    print(f"  Labels loaded: {len(labels)} subjects")
    print(f"  Stage distribution:\n{labels['stages'].value_counts().to_string()}")

    # ---- load MemTrax (optional if path not configured) ----
    if memtrax_xlsx and Path(memtrax_xlsx).is_file():
        print(f"  MemTrax  : {memtrax_xlsx}")
        memtrax = load_memtrax(memtrax_xlsx)
        print(f"  MemTrax loaded: {len(memtrax)} subjects, {len(memtrax.columns)} features")
        # Merge
        merged = labels.join(memtrax, how="inner")
        print(f"  After merge: {len(merged)} subjects with both labels + MemTrax")
    else:
        print(f"  MemTrax  : [not configured or file not found]")
        if memtrax_xlsx:
            print(f"    (looked for: {memtrax_xlsx})")
        merged = labels
        print(f"  Using labels only: {len(merged)} subjects")

    # ---- save ----
    csv_path = data_out / "merged.csv"
    merged.to_csv(str(csv_path))
    print(f"\nMerged data saved to: {csv_path}")
    print(f"Columns: {list(merged.columns)}")
    print(f"Subjects: {len(merged)}")


if __name__ == "__main__":
    main()
