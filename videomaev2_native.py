"""Minimal VideoMAE V2 encoder for OpenGVLab's original ``mae-b`` checkpoint.

The implementation follows the MIT-licensed OpenGVLab/VideoMAEv2 encoder, but
keeps only the components needed for feature extraction.  It lets this project
reuse the already downloaded raw checkpoint bundle, which is not a Hugging
Face ``from_pretrained`` directory.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbed(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_frames: int,
        tubelet_size: int,
        embed_dim: int,
    ):
        super().__init__()
        self.image_size = image_size
        self.proj = nn.Conv3d(
            3,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )
        self.num_patches = (
            (num_frames // tubelet_size)
            * (image_size // patch_size)
            * (image_size // patch_size)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 5:
            raise ValueError(f"expected video tensor (B,C,T,H,W), got {values.shape}")
        if values.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} frames, "
                f"got {tuple(values.shape[-2:])}"
            )
        return self.proj(values).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # Original VideoMAE V2 uses separate Q/V biases and a zero K bias.
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = values.shape
        qkv_bias = torch.cat(
            (self.q_bias, torch.zeros_like(self.v_bias), self.v_bias)
        )
        qkv = F.linear(values, self.qkv.weight, qkv_bias)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, -1)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        values = (attention @ value).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(values)


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(values)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, layer_norm_eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = Mlp(dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values + self.attn(self.norm1(values))
        return values + self.mlp(self.norm2(values))


def _sinusoid_table(num_positions: int, dim: int) -> torch.Tensor:
    positions = np.arange(num_positions, dtype=np.float64)[:, None]
    dimensions = np.arange(dim, dtype=np.float64)[None, :]
    angles = positions / np.power(10000.0, 2 * np.floor(dimensions / 2) / dim)
    table = np.empty((num_positions, dim), dtype=np.float32)
    table[:, 0::2] = np.sin(angles[:, 0::2])
    table[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.from_numpy(table).unsqueeze(0)


class NativeVideoMAEv2Encoder(nn.Module):
    """Unmasked VideoMAE V2 encoder returning normalized patch tokens."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            image_size, patch_size, num_frames, tubelet_size, hidden_size
        )
        self.register_buffer(
            "pos_embed",
            _sinusoid_table(self.patch_embed.num_patches, hidden_size),
            persistent=True,
        )
        self.blocks = nn.ModuleList(
            [Block(hidden_size, num_heads, layer_norm_eps) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.hidden_size = hidden_size
        self.num_frames = num_frames
        self._hd_pixel_layout = "bcthw"
        self._hd_backend = "videomaev2_native"

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        values = self.patch_embed(pixel_values)
        if values.shape[1] != self.pos_embed.shape[1]:
            raise ValueError(
                f"checkpoint expects {self.num_frames} frames and "
                f"{self.pos_embed.shape[1]} tokens, got {values.shape[1]} tokens"
            )
        values = values + self.pos_embed.to(values.dtype)
        for block in self.blocks:
            values = block(values)
        return self.norm(values)


def _unwrap_state_dict(checkpoint) -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    for key in ("model", "module", "state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            checkpoint = candidate
            break
    return {
        str(key): value
        for key, value in checkpoint.items()
        if isinstance(value, torch.Tensor)
    }


def _map_encoder_key(key: str) -> str:
    for prefix in ("module.", "model.", "backbone."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    if key.startswith("encoder."):
        key = key[len("encoder.") :]
    if key.startswith("fc_norm."):
        key = "norm." + key[len("fc_norm.") :]
    return key


def load_native_videomaev2_base(
    checkpoint_path: str,
    device: str,
    num_frames: int = 16,
    image_size: int = 224,
) -> Tuple[NativeVideoMAEv2Encoder, dict]:
    """Load a raw OpenGVLab VideoMAE V2 base checkpoint with strict coverage."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"VideoMAE V2 checkpoint not found: {path}")

    model = NativeVideoMAEv2Encoder(
        image_size=image_size,
        num_frames=num_frames,
    )
    try:
        raw = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only support
        raw = torch.load(str(path), map_location="cpu")
    source = _unwrap_state_dict(raw)
    target = model.state_dict()
    mapped = {}
    shape_mismatches = []
    for source_key, value in source.items():
        key = _map_encoder_key(source_key)
        if key in target:
            if target[key].shape == value.shape:
                mapped[key] = value
            else:
                shape_mismatches.append(
                    f"{source_key}: {tuple(value.shape)} != {tuple(target[key].shape)}"
                )

    parameter_sizes = dict(model.named_parameters())
    total_parameter_count = sum(value.numel() for value in parameter_sizes.values())
    loaded_parameter_count = sum(
        parameter_sizes[key].numel()
        for key in mapped
        if key in parameter_sizes
    )
    coverage = loaded_parameter_count / max(total_parameter_count, 1)
    if coverage < 0.90:
        example_keys = list(source)[:12]
        raise RuntimeError(
            f"Checkpoint is not compatible with the VideoMAE V2 base encoder "
            f"(parameter coverage={coverage:.1%}). "
            f"Example checkpoint keys={example_keys}; "
            f"shape mismatches={shape_mismatches[:5]}"
        )

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    model.to(device).eval()
    report = {
        "backend": model._hd_backend,
        "checkpoint": str(path),
        "parameter_coverage": coverage,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "hidden_size": model.hidden_size,
    }
    return model, report
