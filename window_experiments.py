#!/usr/bin/env python3
"""Leakage-safe subject-level experiments on window embeddings.

The extractor produces many windows per subject, but this script aggregates
them *before* classification and creates all CV splits at subject level.
Scaling, PCA, model selection and probability estimation stay inside each
training fold.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import extract_subject_id, load_config
from window_utils import (
    WindowEmbeddings,
    aggregate_windows,
    available_window_sizes,
    load_window_embeddings,
)


@dataclass
class NestedCVResult:
    probabilities: np.ndarray
    fold: np.ndarray
    best_params: List[dict]


def _safe_n_splits(y: np.ndarray, requested: int, context: str) -> int:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    nonzero = counts[counts > 0]
    if len(nonzero) < 2:
        raise ValueError(f"{context} contains only one class")
    n_splits = min(int(requested), int(nonzero.min()))
    if n_splits < 2:
        raise ValueError(
            f"{context} needs at least 2 samples in each class; counts={counts.tolist()}"
        )
    return n_splits


def make_cv(y: np.ndarray, requested: int, random_state: int, context: str):
    return StratifiedKFold(
        n_splits=_safe_n_splits(y, requested, context),
        shuffle=True,
        random_state=random_state,
    )


def load_window_directory(path: Path) -> Dict[str, WindowEmbeddings]:
    """Load ``HDXXX.npz`` files from one modality."""
    if not path.is_dir():
        raise FileNotFoundError(f"window embedding directory not found: {path}")
    records: Dict[str, WindowEmbeddings] = {}
    for file_path in sorted(path.glob("*.npz")):
        subject_id = extract_subject_id(file_path.stem)
        if subject_id:
            records[subject_id] = load_window_embeddings(str(file_path))
    if not records:
        raise ValueError(f"no subject .npz files found in {path}")
    return records


def load_labels(cfg: dict) -> pd.Series:
    """Load and normalize the configured binary target."""
    csv_path = Path(cfg["paths"]["output_dir"]) / "data" / "merged.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"labels not found: {csv_path}")
    labels = pd.read_csv(csv_path, index_col=0)["stages"]
    normalized_index = []
    for value in labels.index:
        subject_id = extract_subject_id(str(value))
        normalized_index.append(subject_id or str(value).strip().upper())
    labels.index = normalized_index
    if labels.index.has_duplicates:
        duplicates = labels.index[labels.index.duplicated()].unique().tolist()
        raise ValueError(f"duplicate subject labels found: {duplicates[:10]}")
    return labels


def normalize_label(value) -> str:
    """Normalize Excel/CSV values such as 1, 1.0 and ' 1 ' consistently."""
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    try:
        numeric = float(text)
        if numeric.is_integer():
            return str(int(numeric))
    except ValueError:
        pass
    return text


def align_subjects(
    cfg: dict,
    labels: pd.Series,
    audio_records: Dict[str, WindowEmbeddings],
    video_records: Dict[str, WindowEmbeddings],
) -> Tuple[List[str], np.ndarray]:
    """Return subjects present in both modalities and their binary labels."""
    positive = {
        normalize_label(value) for value in cfg["fusion"]["labels"]["positive"]
    }
    negative = {
        normalize_label(value) for value in cfg["fusion"]["labels"]["negative"]
    }
    valid = positive | negative
    subject_ids = sorted(set(audio_records) & set(video_records) & set(labels.index))
    subject_ids = [
        sid for sid in subject_ids if normalize_label(labels.loc[sid]) in valid
    ]
    if not subject_ids:
        raise ValueError("no subjects have audio, video and a configured label")
    y = np.asarray(
        [int(normalize_label(labels.loc[sid]) in positive) for sid in subject_ids]
    )
    _safe_n_splits(y, 2, "aligned data")
    return subject_ids, y


def feature_matrix(
    records: Dict[str, WindowEmbeddings],
    subject_ids: Sequence[str],
    window_sec: float,
    aggregation: str,
    min_valid_ratio: float,
) -> np.ndarray:
    """Aggregate each subject independently into one row."""
    return np.stack(
        [
            aggregate_windows(
                records[sid],
                window_sec=window_sec,
                method=aggregation,
                min_valid_ratio=min_valid_ratio,
            )
            for sid in subject_ids
        ]
    )


def make_search(
    classifier_name: str,
    random_state: int,
    pca_variance: float,
    inner_cv,
    n_jobs: int,
) -> GridSearchCV:
    """Build a preprocessing/model pipeline and its compact parameter grid."""
    if classifier_name == "logistic":
        classifier = LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=random_state
        )
        grid = {
            "pca": ["passthrough", PCA(n_components=pca_variance)],
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
        }
    elif classifier_name == "svm":
        classifier = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )
        grid = {
            "pca": ["passthrough", PCA(n_components=pca_variance)],
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale", 0.001, 0.01],
        }
    elif classifier_name == "random_forest":
        classifier = RandomForestClassifier(
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        )
        grid = {
            "pca": ["passthrough", PCA(n_components=pca_variance)],
            "classifier__n_estimators": [300],
            "classifier__max_depth": [None, 6],
            "classifier__min_samples_leaf": [1, 3],
        }
    else:
        raise ValueError(f"unknown classifier: {classifier_name}")

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("pca", "passthrough"),
            ("classifier", classifier),
        ]
    )
    return GridSearchCV(
        pipeline,
        param_grid=grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )


def nested_cv_predict(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    inner_folds: int,
    pca_variance: float,
    random_state: int,
    n_jobs: int,
) -> NestedCVResult:
    """Return one honest out-of-fold probability per subject."""
    probabilities = np.full(len(y), np.nan, dtype=float)
    fold_assignment = np.full(len(y), -1, dtype=int)
    best_params: List[dict] = []

    for fold_index, (train_index, test_index) in enumerate(outer_splits):
        inner_cv = make_cv(
            y[train_index],
            requested=inner_folds,
            random_state=random_state + fold_index + 1,
            context=f"outer fold {fold_index} training data",
        )
        search = make_search(
            classifier_name,
            random_state=random_state + fold_index,
            pca_variance=pca_variance,
            inner_cv=inner_cv,
            n_jobs=n_jobs,
        )
        search.fit(X[train_index], y[train_index])
        probabilities[test_index] = search.predict_proba(X[test_index])[:, 1]
        fold_assignment[test_index] = fold_index
        best_params.append(search.best_params_)

    if not np.isfinite(probabilities).all() or np.any(fold_assignment < 0):
        raise RuntimeError("nested CV failed to predict every subject exactly once")
    return NestedCVResult(probabilities, fold_assignment, best_params)


def classification_metrics(y: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "auc_roc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "pr_baseline": float(np.mean(y)),
        "accuracy": accuracy_score(y, predictions),
        "balanced_accuracy": balanced_accuracy_score(y, predictions),
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "brier": brier_score_loss(y, probabilities),
    }


def bootstrap_interval(
    y: np.ndarray,
    probabilities: np.ndarray,
    metric,
    iterations: int,
    random_state: int,
) -> Tuple[float, float]:
    """Subject bootstrap interval; invalid one-class resamples are skipped."""
    rng = np.random.default_rng(random_state)
    values = []
    for _ in range(iterations):
        index = rng.integers(0, len(y), size=len(y))
        if len(np.unique(y[index])) < 2:
            continue
        values.append(metric(y[index], probabilities[index]))
    if len(values) < max(20, iterations // 10):
        return np.nan, np.nan
    return tuple(np.percentile(values, [2.5, 97.5]))


def summarize_result(
    y: np.ndarray,
    probabilities: np.ndarray,
    bootstrap_iterations: int,
    random_state: int,
) -> dict:
    metrics = classification_metrics(y, probabilities)
    auc_low, auc_high = bootstrap_interval(
        y,
        probabilities,
        roc_auc_score,
        iterations=bootstrap_iterations,
        random_state=random_state,
    )
    pr_low, pr_high = bootstrap_interval(
        y,
        probabilities,
        average_precision_score,
        iterations=bootstrap_iterations,
        random_state=random_state + 1,
    )
    metrics.update(
        {
            "auc_ci_low": auc_low,
            "auc_ci_high": auc_high,
            "pr_ci_low": pr_low,
            "pr_ci_high": pr_high,
        }
    )
    return metrics


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def nested_late_fusion_predict(
    X_audio: np.ndarray,
    X_video: np.ndarray,
    y: np.ndarray,
    audio_classifier: str,
    video_classifier: str,
    outer_splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    inner_folds: int,
    pca_variance: float,
    random_state: int,
    n_jobs: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Learn late-fusion weights using OOF base predictions inside each outer fold."""
    audio_prob = np.full(len(y), np.nan)
    video_prob = np.full(len(y), np.nan)
    fused_prob = np.full(len(y), np.nan)

    for fold_index, (train_index, test_index) in enumerate(outer_splits):
        inner_cv = make_cv(
            y[train_index],
            requested=inner_folds,
            random_state=random_state + 100 + fold_index,
            context=f"fusion outer fold {fold_index}",
        )
        train_base_probabilities = []
        test_base_probabilities = []
        for X, classifier_name in (
            (X_audio, audio_classifier),
            (X_video, video_classifier),
        ):
            # Fit the outer-training model used for the untouched outer test
            # fold. Its hyperparameters are selected only within outer train.
            outer_search = make_search(
                classifier_name,
                random_state=random_state + fold_index,
                pca_variance=pca_variance,
                inner_cv=inner_cv,
                n_jobs=n_jobs,
            )
            outer_search.fit(X[train_index], y[train_index])
            test_prob = outer_search.predict_proba(X[test_index])[:, 1]

            # Build strictly cross-fitted meta-training scores. Parameter
            # selection for each score happens inside that score's training
            # subset, so the held-out subject's label cannot influence it.
            X_outer_train = X[train_index]
            y_outer_train = y[train_index]
            oof = np.full(len(train_index), np.nan)
            for meta_fold, (meta_train, meta_valid) in enumerate(
                inner_cv.split(X_outer_train, y_outer_train)
            ):
                innermost_cv = make_cv(
                    y_outer_train[meta_train],
                    requested=inner_folds,
                    random_state=random_state + 1000 + fold_index * 10 + meta_fold,
                    context=(
                        f"fusion outer fold {fold_index}, "
                        f"meta fold {meta_fold} training data"
                    ),
                )
                meta_search = make_search(
                    classifier_name,
                    random_state=random_state + 1000 + fold_index * 10 + meta_fold,
                    pca_variance=pca_variance,
                    inner_cv=innermost_cv,
                    n_jobs=n_jobs,
                )
                meta_search.fit(
                    X_outer_train[meta_train], y_outer_train[meta_train]
                )
                oof[meta_valid] = meta_search.predict_proba(
                    X_outer_train[meta_valid]
                )[:, 1]
            if not np.isfinite(oof).all():
                raise RuntimeError("late-fusion meta predictions are incomplete")
            train_base_probabilities.append(oof)
            test_base_probabilities.append(test_prob)

        meta_train = np.column_stack(
            [_logit(value) for value in train_base_probabilities]
        )
        meta_test = np.column_stack(
            [_logit(value) for value in test_base_probabilities]
        )
        stacker = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state + fold_index,
        )
        stacker.fit(meta_train, y[train_index])

        audio_prob[test_index] = test_base_probabilities[0]
        video_prob[test_index] = test_base_probabilities[1]
        fused_prob[test_index] = stacker.predict_proba(meta_test)[:, 1]

    for name, values in (
        ("audio", audio_prob),
        ("video", video_prob),
        ("fusion", fused_prob),
    ):
        if not np.isfinite(values).all():
            raise RuntimeError(f"{name} fusion predictions are incomplete")
    return audio_prob, video_prob, fused_prob


def _nearest_available(requested: float, available: Sequence[float]) -> float:
    return min(available, key=lambda value: abs(value - requested))


def main():
    parser = argparse.ArgumentParser(
        description="Subject-level nested CV for audio/video window embeddings"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--audio-window-dir", default=None)
    parser.add_argument("--video-window-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--classifiers", default=None)
    parser.add_argument("--aggregations", default=None)
    parser.add_argument("--outer-folds", type=int, default=None)
    parser.add_argument("--inner-folds", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--fusion-audio-window", type=float, default=None)
    parser.add_argument("--fusion-video-window", type=float, default=None)
    parser.add_argument("--fusion-aggregation", default=None)
    parser.add_argument("--fusion-audio-classifier", default=None)
    parser.add_argument("--fusion-video-classifier", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment_cfg = cfg.get("window_experiments", {})
    random_state = int(experiment_cfg.get("random_state", 42))
    outer_folds = args.outer_folds or int(experiment_cfg.get("outer_folds", 5))
    inner_folds = args.inner_folds or int(experiment_cfg.get("inner_folds", 4))
    bootstrap_iterations = args.bootstrap or int(
        experiment_cfg.get("bootstrap_iterations", 2000)
    )
    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = int(experiment_cfg.get("n_jobs", -1))
    pca_variance = float(
        experiment_cfg.get("pca_variance", cfg["baseline"].get("pca_variance", 0.95))
    )
    classifiers = (
        args.classifiers.split(",")
        if args.classifiers
        else list(experiment_cfg.get("classifiers", ["logistic", "random_forest"]))
    )
    aggregations = (
        args.aggregations.split(",")
        if args.aggregations
        else list(experiment_cfg.get("aggregations", ["mean", "mean_std"]))
    )
    min_audio_ratio = float(experiment_cfg.get("min_audio_speech_ratio", 0.0))
    min_video_ratio = float(experiment_cfg.get("min_video_valid_ratio", 0.0))

    output_dir = Path(cfg["paths"]["output_dir"])
    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else output_dir / "results" / "windowed"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    audio_window_dir = (
        Path(args.audio_window_dir)
        if args.audio_window_dir
        else output_dir / "emb_windows" / "audio"
    )
    video_window_dir = (
        Path(args.video_window_dir)
        if args.video_window_dir
        else output_dir / "emb_windows" / "video"
    )
    audio_records = load_window_directory(audio_window_dir)
    video_records = load_window_directory(video_window_dir)
    labels = load_labels(cfg)
    subject_ids, y = align_subjects(cfg, labels, audio_records, video_records)

    audio_sizes = available_window_sizes(audio_records[sid] for sid in subject_ids)
    video_sizes = available_window_sizes(video_records[sid] for sid in subject_ids)
    if not audio_sizes or not video_sizes:
        raise ValueError("subjects do not share a common window size within each modality")

    outer_cv = make_cv(y, outer_folds, random_state, "full dataset")
    outer_splits = list(outer_cv.split(np.zeros(len(y)), y))

    print("=== Windowed Subject-Level Experiments ===")
    print(f"  Subjects        : {len(y)} (negative={(y == 0).sum()}, positive={(y == 1).sum()})")
    print(f"  Audio windows   : {audio_sizes}")
    print(f"  Video windows   : {video_sizes}")
    print(f"  Outer folds     : {len(outer_splits)}; inner requested={inner_folds}")
    print(f"  PR baseline     : {y.mean():.4f}")

    matrices: Dict[Tuple[str, float, str], np.ndarray] = {}
    result_rows, prediction_frames = [], []
    modality_records = {"audio": audio_records, "video": video_records}
    modality_sizes = {"audio": audio_sizes, "video": video_sizes}
    modality_valid = {"audio": min_audio_ratio, "video": min_video_ratio}

    for modality in ("audio", "video"):
        for window_sec in modality_sizes[modality]:
            for aggregation in aggregations:
                key = (modality, window_sec, aggregation)
                X = feature_matrix(
                    modality_records[modality],
                    subject_ids,
                    window_sec,
                    aggregation,
                    modality_valid[modality],
                )
                matrices[key] = X
                for classifier_name in classifiers:
                    print(
                        f"  {modality:5s} {window_sec:>5g}s "
                        f"{aggregation:>8s} {classifier_name:>13s}",
                        flush=True,
                    )
                    nested = nested_cv_predict(
                        X,
                        y,
                        classifier_name=classifier_name,
                        outer_splits=outer_splits,
                        inner_folds=inner_folds,
                        pca_variance=pca_variance,
                        random_state=random_state,
                        n_jobs=n_jobs,
                    )
                    metrics = summarize_result(
                        y,
                        nested.probabilities,
                        bootstrap_iterations=bootstrap_iterations,
                        random_state=random_state,
                    )
                    result_rows.append(
                        {
                            "strategy": "unimodal",
                            "modality": modality,
                            "window_sec": window_sec,
                            "aggregation": aggregation,
                            "classifier": classifier_name,
                            "feature_dim": X.shape[1],
                            **metrics,
                            "best_params_by_fold": json.dumps(
                                nested.best_params, default=str
                            ),
                        }
                    )
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "subject_id": subject_ids,
                                "label": y,
                                "fold": nested.fold,
                                "probability": nested.probabilities,
                                "strategy": "unimodal",
                                "modality": modality,
                                "window_sec": window_sec,
                                "aggregation": aggregation,
                                "classifier": classifier_name,
                            }
                        )
                    )

    requested_audio = args.fusion_audio_window
    if requested_audio is None:
        requested_audio = float(experiment_cfg.get("fusion_audio_window_sec", 8))
    requested_video = args.fusion_video_window
    if requested_video is None:
        requested_video = float(experiment_cfg.get("fusion_video_window_sec", 4))
    fusion_audio_window = _nearest_available(requested_audio, audio_sizes)
    fusion_video_window = _nearest_available(requested_video, video_sizes)
    fusion_aggregation = (
        args.fusion_aggregation
        or experiment_cfg.get("fusion_aggregation", "mean_std")
    )
    audio_classifier = (
        args.fusion_audio_classifier
        or experiment_cfg.get("fusion_audio_classifier", "random_forest")
    )
    video_classifier = (
        args.fusion_video_classifier
        or experiment_cfg.get("fusion_video_classifier", "logistic")
    )
    for modality, window_sec, records, min_ratio in (
        ("audio", fusion_audio_window, audio_records, min_audio_ratio),
        ("video", fusion_video_window, video_records, min_video_ratio),
    ):
        key = (modality, window_sec, fusion_aggregation)
        if key not in matrices:
            matrices[key] = feature_matrix(
                records,
                subject_ids,
                window_sec,
                fusion_aggregation,
                min_ratio,
            )
    X_audio = matrices[("audio", fusion_audio_window, fusion_aggregation)]
    X_video = matrices[("video", fusion_video_window, fusion_aggregation)]

    # Fair early-fusion comparison on the same predeclared window settings.
    X_early = np.concatenate([X_audio, X_video], axis=1)
    for classifier_name in classifiers:
        nested = nested_cv_predict(
            X_early,
            y,
            classifier_name=classifier_name,
            outer_splits=outer_splits,
            inner_folds=inner_folds,
            pca_variance=pca_variance,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        metrics = summarize_result(
            y,
            nested.probabilities,
            bootstrap_iterations=bootstrap_iterations,
            random_state=random_state,
        )
        result_rows.append(
            {
                "strategy": "early_concat",
                "modality": "audio+video",
                "window_sec": f"a{fusion_audio_window:g}_v{fusion_video_window:g}",
                "aggregation": fusion_aggregation,
                "classifier": classifier_name,
                "feature_dim": X_early.shape[1],
                **metrics,
                "best_params_by_fold": json.dumps(nested.best_params, default=str),
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "subject_id": subject_ids,
                    "label": y,
                    "fold": nested.fold,
                    "probability": nested.probabilities,
                    "strategy": "early_concat",
                    "modality": "audio+video",
                    "window_sec": f"a{fusion_audio_window:g}_v{fusion_video_window:g}",
                    "aggregation": fusion_aggregation,
                    "classifier": classifier_name,
                }
            )
        )

    print(
        "  learned late fusion: "
        f"audio={fusion_audio_window:g}s/{audio_classifier}, "
        f"video={fusion_video_window:g}s/{video_classifier}, "
        f"aggregation={fusion_aggregation}"
    )
    audio_prob, video_prob, fused_prob = nested_late_fusion_predict(
        X_audio,
        X_video,
        y,
        audio_classifier=audio_classifier,
        video_classifier=video_classifier,
        outer_splits=outer_splits,
        inner_folds=inner_folds,
        pca_variance=pca_variance,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    mean_prob = (audio_prob + video_prob) / 2.0
    fusion_predictions = {
        "audio_fusion_base": audio_prob,
        "video_fusion_base": video_prob,
        "mean_probability": mean_prob,
        "learned_stacking": fused_prob,
    }
    fusion_fold = np.full(len(y), -1, dtype=int)
    for fold_index, (_, test_index) in enumerate(outer_splits):
        fusion_fold[test_index] = fold_index
    for strategy, probabilities in fusion_predictions.items():
        strategy_modality = (
            "audio"
            if strategy == "audio_fusion_base"
            else "video"
            if strategy == "video_fusion_base"
            else "audio+video"
        )
        strategy_classifier = (
            f"{audio_classifier}+{video_classifier}"
            if strategy in {"mean_probability", "learned_stacking"}
            else audio_classifier
            if strategy.startswith("audio")
            else video_classifier
        )
        metrics = summarize_result(
            y,
            probabilities,
            bootstrap_iterations=bootstrap_iterations,
            random_state=random_state,
        )
        result_rows.append(
            {
                "strategy": strategy,
                "modality": strategy_modality,
                "window_sec": f"a{fusion_audio_window:g}_v{fusion_video_window:g}",
                "aggregation": fusion_aggregation,
                "classifier": strategy_classifier,
                "feature_dim": 2 if strategy == "learned_stacking" else np.nan,
                **metrics,
                "best_params_by_fold": "",
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "subject_id": subject_ids,
                    "label": y,
                    "fold": fusion_fold,
                    "probability": probabilities,
                    "strategy": strategy,
                    "modality": strategy_modality,
                    "window_sec": f"a{fusion_audio_window:g}_v{fusion_video_window:g}",
                    "aggregation": fusion_aggregation,
                    "classifier": strategy_classifier,
                }
            )
        )

    results = pd.DataFrame(result_rows).sort_values("auc_roc", ascending=False)
    results["audio_window_dir"] = str(audio_window_dir)
    results["video_window_dir"] = str(video_window_dir)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    results_path = results_dir / "metrics.csv"
    predictions_path = results_dir / "subject_predictions.csv"
    results.to_csv(results_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    display_columns = [
        "strategy",
        "modality",
        "window_sec",
        "aggregation",
        "classifier",
        "auc_roc",
        "auc_ci_low",
        "auc_ci_high",
        "pr_auc",
        "pr_baseline",
        "balanced_accuracy",
        "brier",
    ]
    print("\nTop results:")
    print(results[display_columns].head(20).to_string(index=False))
    print(f"\nSaved: {results_path}")
    print(f"Saved: {predictions_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise
