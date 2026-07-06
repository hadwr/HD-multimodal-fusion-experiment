#!/usr/bin/env python3
"""
Extract audio embeddings using wav2vec2 (via ModelScope → transformers).

For each ``HDXXX.wav`` file in the audio directory, this script:
1. Loads & resamples audio to 16 kHz
2. Runs wav2vec2-base, takes the last hidden state
3. Mean-pools over the time dimension
4. Saves the resulting vector as ``output/emb/audio/HDXXX.npy``

Usage::

    python extract_audio.py                    # use config.yaml defaults
    python extract_audio.py --config my.yaml   # custom config
    python extract_audio.py --device cpu       # force CPU
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

from utils import (
    load_config,
    resolve_model_path,
    extract_subject_id,
    save_embedding,
)


def load_wav2vec2_local(local_path: str, device: str = "cuda"):
    """Load wav2vec2 from a local ModelScope-downloaded directory."""
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    processor = Wav2Vec2Processor.from_pretrained(local_path)
    model = Wav2Vec2Model.from_pretrained(local_path)
    model = model.to(device)
    model.eval()
    return model, processor


@torch.no_grad()
def extract_audio_embedding(
    audio_path: str,
    model,
    processor,
    device: str = "cuda",
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Load an audio file, run wav2vec2, return the mean-pooled embedding.

    Parameters
    ----------
    audio_path : str
        Path to a .wav file.
    model : Wav2Vec2Model
    processor : Wav2Vec2Processor
    device : str
    target_sr : int
        Target sample rate (wav2vec2 expects 16 kHz).

    Returns
    -------
    embedding : np.ndarray  shape (hidden_dim,)
    """
    # ---- load & resample ----
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # mono

    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    # ---- processor expects 1-D array ----
    audio_np = waveform.squeeze(0).numpy()
    inputs = processor(audio_np, sampling_rate=target_sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ---- forward ----
    outputs = model(**inputs)
    last_hidden = outputs.last_hidden_state  # (1, T, hidden_dim)

    # ---- mean pool over time ----
    embedding = last_hidden.squeeze(0).mean(dim=0)  # (hidden_dim,)
    return embedding.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Extract wav2vec2 audio embeddings")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--device", default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--model-path", default=None,
                        help="Pre-downloaded model directory (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg.get("device", "cuda")

    audio_dir = cfg["paths"]["audio_dir"]
    output_dir = Path(cfg["paths"]["output_dir"])
    emb_out = output_dir / "emb" / "audio"

    # ---- resolve model path (local > ModelScope) ----
    model_local = args.model_path or cfg["models"]["audio"].get("local_path")
    ms_name = cfg["models"]["audio"]["ms_name"]
    target_sr = cfg["models"]["audio"]["sample_rate"]

    print(f"=== Audio Embedding Extraction ===")
    print(f"  Audio dir : {audio_dir}")
    print(f"  Model     : {ms_name}  (via ModelScope)")
    print(f"  Device    : {device}")
    print(f"  Output    : {emb_out}")

    if not Path(audio_dir).is_dir():
        print(f"[ERROR] Audio directory not found: {audio_dir}")
        print("[HINT] Are you running on the server with the data mounted?")
        sys.exit(1)

    local_path = resolve_model_path(model_local, ms_name)
    model, processor = load_wav2vec2_local(local_path, device=device)

    # ---- iterate over .wav files ----
    wav_files = sorted(Path(audio_dir).glob("*.wav"))
    if not wav_files:
        print(f"[ERROR] No .wav files found in {audio_dir}")
        sys.exit(1)

    print(f"  Subjects  : {len(wav_files)} .wav files")

    for wav_path in tqdm(wav_files, desc="Extracting audio"):
        sid = extract_subject_id(wav_path.stem)
        if sid is None:
            print(f"  [Skip] Cannot parse subject ID from: {wav_path.name}")
            continue

        try:
            emb = extract_audio_embedding(
                str(wav_path), model, processor, device=device, target_sr=target_sr
            )
            save_embedding(str(emb_out), sid, emb)
        except Exception as e:
            print(f"  [Error] {sid}: {e}")
            continue

    print(f"Done. Embeddings saved to {emb_out}")


if __name__ == "__main__":
    main()
