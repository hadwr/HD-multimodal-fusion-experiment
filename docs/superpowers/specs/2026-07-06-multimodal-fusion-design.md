# Multimodal Fusion Experiment — Design Spec

**Date:** 2026-07-06
**Status:** approved

## Overview

Extract embeddings from three modalities (video, audio, tabular MemTrax data), visualize their distribution via t-SNE/UMAP, then run multi-modal fusion classification and regression experiments.

- **Phase 1:** Unsupervised exploration — embedding extraction + visualization
- **Phase 2:** Supervised — pre vs stage-1/2 classification, HD/HC binary classification, regression

## Data Sources

| Modality | Path | Format |
|----------|------|--------|
| Video | `/NAS/public2_data/huangda/preprocessed_audio/HDXXX/4.mp4` | Preprocessed video, one per subject |
| Audio | `/public/labdata/huangda/HD/Data/dataset/HD_final/HDacoustic/HDXXX.wav` | Preprocessed audio, one per subject |
| MemTrax | TBD xlsx path | One row per subject, columns: accuracy, correct count, reaction time, etc. |
| Labels | `/public/labdata/huangda/HD/Data/dataset/20250610.xlsx` | `sample_id` column (HDXXX), `stages` column (pre/1/2) |

HC (healthy control) data lives in a separate folder (no scales available), classified by folder.

## Project Structure

```
HD-multimodal-fusion-experiment/
├── config.yaml              # All paths, model names, hyperparams
├── requirements.txt
├── extract_audio.py         # wav2vec2 → audio embeddings
├── extract_video.py         # VideoMAE → video embeddings
├── load_tabular.py          # Load MemTrax + labels → merged CSV
├── visualize.py             # t-SNE / UMAP plots
├── fusion_baseline.py       # Multi-modal fusion + classification/regression
└── utils.py                 # Shared utilities (path resolution, ID matching, etc.)
```

## Data Flow

```
config.yaml
    │
    ├─→ extract_audio.py  ──→ output/emb/audio/HDXXX.npy
    ├─→ extract_video.py  ──→ output/emb/video/HDXXX.npy
    ├─→ load_tabular.py   ──→ output/data/merged.csv   (index=HDXXX, cols=features+labels)
    │
    └─→ visualize.py      ←── merge embeddings + labels → t-SNE/UMAP PNGs
    └─→ fusion_baseline.py ←── same → classification/regression metrics
```

## Module Details

### extract_audio.py
- Load wav2vec2 via ModelScope → `transformers.Wav2Vec2Model.from_pretrained(local_path)`
- Resample audio to 16kHz if needed
- Extract last hidden state, mean-pool over time dimension
- Save as `output/emb/audio/HDXXX.npy`

### extract_video.py
- Load VideoMAE-base via ModelScope → `transformers.VideoMAEModel.from_pretrained(local_path)`
- Uniformly sample 16 frames, resize to 224×224
- Extract CLS token as embedding (fallback: mean pool all patches)
- Save as `output/emb/video/HDXXX.npy`

### load_tabular.py
- Read MemTrax xlsx and label xlsx
- Match by `sample_id` ↔ HDXXX extracted from filenames
- Output `output/data/merged.csv` with index=HDXXX, feature columns + `stage` label column

### visualize.py
- Load all embeddings + labels
- Run sklearn t-SNE and/or umap-learn
- Generate plots: single-modality + fused (concat), colored by stage
- Save PNGs to `output/figures/`

### fusion_baseline.py
- Load merged embeddings + labels
- Fusion methods: simple concatenation, basic attention
- Classifier: MLP on fused features
- Tasks: pre vs (1,2) classification; HD/HC binary classification; regression on scale scores
- Output metrics: accuracy, F1, RMSE

## config.yaml Key Fields

```yaml
paths:
  audio_dir: /public/labdata/huangda/HD/Data/dataset/HD_final/HDacoustic
  video_dir: /NAS/public2_data/huangda/preprocessed_audio
  memtrax_xlsx: null              # TBD
  label_xlsx: /public/labdata/huangda/HD/Data/dataset/20250610.xlsx
  output_dir: ./output
  hc_audio_dir: null              # TBD
  hc_video_dir: null              # TBD

models:
  audio:
    hf_name: facebook/wav2vec2-base
    ms_name: iic/wav2vec2-base
  video:
    hf_name: MCG-NJU/videomae-base
    ms_name: iic/videomae-base
  frame_count: 16
  sample_rate: 16000

visualization:
  method: umap                    # umap | tsne | both
  perplexity: 30
  n_neighbors: 15

fusion:
  methods: [concat]               # concat first, attention later
  test_size: 0.2
  random_state: 42
```

## Key Design Decisions

1. **ModelScope proxy**: Use `modelscope` SDK `snapshot_download()` for model weights, then pass local path to `transformers`
2. **Pooling**: Audio → mean pool over time; Video → CLS token (native VideoMAE support)
3. **Frame sampling**: 16 frames uniformly sampled, 224×224, following VideoMAE standard
4. **ID matching**: Extract HDXXX from filenames, join with xlsx `sample_id`
5. **Single-modality + fused visuals**: Each modality gets its own t-SNE/UMAP plot plus a fused one
6. **YAGNI**: No data augmentation, no hyperparameter search, no complex attention/transformer fusion — get the baseline working first

## Out of Scope (Phase 1)

- Data augmentation
- Hyperparameter tuning
- Complex fusion architectures (cross-attention, transformer fusion)
- HC scale data (none available)
- Real-time inference or serving
