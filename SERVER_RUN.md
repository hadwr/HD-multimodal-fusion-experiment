# Server execution guide

The data and model weights are intentionally not required for the local tests.
Run the following commands after syncing this repository to the server.

## 1. Environment smoke test

```bash
python -m py_compile \
  utils.py window_utils.py videomaev2_native.py extract_audio.py extract_video.py \
  baseline_experiments.py fusion_baseline.py window_experiments.py

python -m unittest discover -s tests -v
```

## 2. Check model compatibility before a full run

Use one GPU and keep the configured local model paths in `config.yaml`.
Both extractors are local-only by default and will not contact ModelScope.
If the server snapshot is stored elsewhere, pass
`--model-path /actual/local/model/directory`. Only use
`--allow-model-download` when a new download is intentionally required.

```bash
python extract_video.py --mode windows --window-sizes 4 --device cuda:0 --limit 2
python extract_audio.py --mode windows --window-sizes 8 --device cuda:0 --limit 2
```

The scripts print the loaded architecture and hidden size. Treat any
`newly initialized weights`, incompatible-key warning, invalid FPS, or decode
failure as an error. A subject extraction failure makes the process exit with
status 2 after completing the remaining subjects.

Video discovery scans the configured collection folders recursively:
`A录像无量表`, `A录像+运动量表+TFC`, `HD241-HD272`, `HD272-HD309`, and
`视屏编号整理（HD178开始）`. Add `--include-hc` to scan `HC_video`.
For each subject, the first available take is used in this order:
`4.mp4`, `4（1）.mp4`/`4(1).mp4`, `4（2）.mp4`/`4(2).mp4`, and so on.
Remaining copies are ignored. Only a subject without any matching `4` take is
logged as `[Skip]`.
Use `python extract_video.py --list-videos` to verify all mappings without
loading the model or decoding videos.

## 3. Full corrected extraction

```bash
python extract_video.py \
  --mode both \
  --window-sizes 2,4,8 \
  --overlap 0.5 \
  --pool mean \
  --device cuda:0 \
  --amp

python extract_audio.py \
  --mode both \
  --window-sizes 4,8,16 \
  --overlap 0.5 \
  --pool mean \
  --layer -1 \
  --min-speech-ratio 0.2 \
  --device cuda:0 \
  --amp
```

Outputs:

- Corrected global embeddings: `output/emb/{audio,video}/HDXXX.npy`
- Window embeddings: `output/emb_windows/{audio,video}/HDXXX.npz`

Each NPZ contains `embeddings`, `start_sec`, `end_sec`, `window_sec`, and
`valid_ratio`. For audio, `valid_ratio` is the energy-based active-speech
fraction. For video, it is the fraction of the requested temporal window
covered by the recording.

If whole-recording audio causes a CUDA OOM, use `--mode windows`. Windowed
experiments do not require the global embedding.

## 4. Leakage-safe window experiments

First run a relatively cheap validation:

```bash
python window_experiments.py \
  --classifiers logistic \
  --aggregations mean \
  --outer-folds 3 \
  --inner-folds 2 \
  --bootstrap 200 \
  --n-jobs 4
```

Then run the configured experiment:

```bash
python window_experiments.py
```

Primary outputs:

- `output/results/windowed/metrics.csv`
- `output/results/windowed/subject_predictions.csv`

All rows are subject-level out-of-fold estimates. The script reports the
positive prevalence as the no-skill PR-AUC baseline and includes subject
bootstrap confidence intervals.

## 5. Recommended ablations

Run different speech layers into separate output directories:

```bash
python extract_audio.py --mode windows --layer 6 --window-sizes 4,8,16 \
  --window-output-dir output/emb_windows/audio_layer6
python extract_audio.py --mode windows --layer 9 --window-sizes 4,8,16 \
  --window-output-dir output/emb_windows/audio_layer9

python window_experiments.py \
  --audio-window-dir output/emb_windows/audio_layer6 \
  --results-dir output/results/windowed_audio_layer6
```

Do not mix NPZ files extracted from different backbones, pooling modes, or
layers in the same directory. Keep one directory per extraction configuration.
