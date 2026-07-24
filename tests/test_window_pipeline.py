import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils import resolve_model_path
from video_discovery import discover_video_sources
from window_experiments import (
    classification_metrics,
    nested_late_fusion_predict,
    nested_cv_predict,
)
from window_utils import (
    WindowEmbeddings,
    aggregate_windows,
    evenly_spaced_subset,
    load_window_embeddings,
    make_sliding_windows,
    save_window_embeddings,
)

try:
    import torch
    from extract_video import pool_video_hidden
except ModuleNotFoundError:
    torch = None
    pool_video_hidden = None


class WindowUtilityTests(unittest.TestCase):
    def test_nested_video_collection_discovery(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            first = base / "A录像无量表" / "batch" / "HD001"
            second = base / "HD241-HD272" / "HD241"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "4.mp4").touch()
            (first / "4（1）.mp4").touch()
            (first / "4(1).mp4").touch()
            (second / "4.mp4").touch()
            sources = discover_video_sources(
                base, ["A录像无量表", "HD241-HD272"]
            )
            self.assertEqual(
                [(item.subject_id, item.video_path.name) for item in sources],
                [("HD001", "4.mp4"), ("HD241", "4.mp4")],
            )

    def test_first_numbered_4_copy_is_used_when_exact_file_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            folder = base / "one" / "HD001"
            folder.mkdir(parents=True)
            (folder / "4（2）.mp4").touch()
            (folder / "4（1）.mp4").touch()
            sources = discover_video_sources(base, ["one"])
            self.assertEqual(
                [(source.subject_id, source.video_path.name) for source in sources],
                [("HD001", "4（1）.mp4")],
            )

    def test_subject_without_any_4_recording_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            missing = base / "one" / "HD001"
            valid = base / "one" / "HD002"
            missing.mkdir(parents=True)
            valid.mkdir(parents=True)
            (missing / "recording.mp4").touch()
            (valid / "4.mp4").touch()
            sources = discover_video_sources(base, ["one"])
            self.assertEqual(
                [source.subject_id for source in sources], ["HD002"]
            )

    def test_model_resolver_reuses_cache_without_download(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cached = Path(temp_dir) / "OpenGVLab" / "VideoMAE2"
            cached.mkdir(parents=True)
            resolved = resolve_model_path(
                None,
                "OpenGVLab/VideoMAE2",
                cache_dir=temp_dir,
                allow_download=False,
            )
            self.assertEqual(Path(resolved), cached.resolve())

    def test_model_resolver_fails_closed_when_download_is_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                resolve_model_path(
                    None,
                    "OpenGVLab/VideoMAE2",
                    cache_dir=temp_dir,
                    allow_download=False,
                )

    def test_even_subset_preserves_endpoints_and_limit(self):
        items = list(range(101))
        selected = evenly_spaced_subset(items, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 100)
        self.assertEqual(selected, sorted(set(selected)))

    def test_sliding_windows_include_tail_without_duplicates(self):
        windows = make_sliding_windows(10.0, 4.0, overlap=0.5)
        self.assertEqual(windows, [(0.0, 4.0), (2.0, 6.0), (4.0, 8.0), (6.0, 10.0)])

        short = make_sliding_windows(3.0, 4.0, overlap=0.5)
        self.assertEqual(short, [(0.0, 3.0)])

    def test_round_trip_and_aggregation(self):
        record = WindowEmbeddings(
            embeddings=np.asarray([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]),
            start_sec=np.asarray([0.0, 2.0, 0.0]),
            end_sec=np.asarray([4.0, 6.0, 8.0]),
            window_sec=np.asarray([4.0, 4.0, 8.0]),
            valid_ratio=np.asarray([1.0, 0.8, 1.0]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = save_window_embeddings(temp_dir, "HD001", record)
            loaded = load_window_embeddings(str(path))
            np.testing.assert_allclose(loaded.embeddings, record.embeddings)
            np.testing.assert_allclose(
                aggregate_windows(loaded, 4.0, method="mean"),
                np.asarray([2.0, 3.0]),
            )
            np.testing.assert_allclose(
                aggregate_windows(loaded, 4.0, method="mean_std"),
                np.asarray([2.0, 3.0, 1.0, 1.0]),
            )

    @unittest.skipIf(torch is None, "PyTorch is not installed in the local environment")
    def test_videomae_pool_uses_all_tokens(self):
        hidden = torch.tensor([[[1.0, 10.0], [3.0, 20.0], [5.0, 30.0]]])
        pooled = pool_video_hidden(hidden, "mean")
        torch.testing.assert_close(pooled, torch.tensor([[3.0, 20.0]]))
        self.assertFalse(torch.equal(pooled, hidden[:, 0, :]))


class SyntheticNestedCVTests(unittest.TestCase):
    def test_nested_cv_produces_one_oof_prediction_per_subject(self):
        rng = np.random.default_rng(7)
        n = 40
        y = np.asarray([0] * 20 + [1] * 20)
        X = rng.normal(size=(n, 12))
        X[:, :3] += y[:, None] * 1.5

        from window_experiments import make_cv

        outer = list(make_cv(y, 4, 42, "test").split(X, y))
        result = nested_cv_predict(
            X,
            y,
            classifier_name="logistic",
            outer_splits=outer,
            inner_folds=3,
            pca_variance=0.95,
            random_state=42,
            n_jobs=1,
        )
        self.assertTrue(np.isfinite(result.probabilities).all())
        self.assertTrue((result.fold >= 0).all())
        self.assertEqual(len(result.best_params), 4)
        metrics = classification_metrics(y, result.probabilities)
        self.assertGreater(metrics["auc_roc"], 0.8)

    def test_learned_late_fusion_is_fully_out_of_fold(self):
        rng = np.random.default_rng(11)
        n = 30
        y = np.asarray([0] * 15 + [1] * 15)
        audio = rng.normal(size=(n, 8))
        video = rng.normal(size=(n, 8))
        audio[:, :2] += y[:, None] * 1.2
        video[:, 2:4] += y[:, None] * 0.6

        from window_experiments import make_cv

        outer = list(make_cv(y, 3, 42, "fusion test").split(audio, y))
        audio_prob, video_prob, fused_prob = nested_late_fusion_predict(
            audio,
            video,
            y,
            audio_classifier="logistic",
            video_classifier="logistic",
            outer_splits=outer,
            inner_folds=2,
            pca_variance=0.95,
            random_state=42,
            n_jobs=1,
        )
        self.assertTrue(np.isfinite(audio_prob).all())
        self.assertTrue(np.isfinite(video_prob).all())
        self.assertTrue(np.isfinite(fused_prob).all())
        self.assertTrue(((0.0 <= fused_prob) & (fused_prob <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
