"""Extract per-timebin features from Phase 1 MAE pretrained encoder.

The MAE encoder learned channel-level representations from Phase 1 pretraining.
We use it as a frozen feature extractor: slide 8-bin windows across each session,
pool encoder outputs per center timebin to get (n_bins, 128) feature maps.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class PreNormEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block (matches pillar2_mae_model.py)."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_padding_mask: Tensor | None = None) -> Tensor:
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class MAEEncoder(nn.Module):
    """Standalone MAE encoder for feature extraction.

    Loads encoder weights from a Phase 1 MAE checkpoint and extracts
    per-timebin features by pooling channel representations.
    """

    def __init__(
        self,
        n_channels: int = 96,
        context_bins: int = 8,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_ff: int = 512,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.context_bins = context_bins
        self.d_model = d_model
        # Match MAE's center index: context_bins // 2 - 1 for even, // 2 for odd
        self.center_idx = context_bins // 2 - 1 if context_bins % 2 == 0 else context_bins // 2

        # Channel embedding (matches pillar2_mae_model.ChannelEmbedding)
        self.scalar_proj = nn.Linear(1, d_model)
        self.channel_embed = nn.Embedding(n_channels, d_model)
        self.time_embed = nn.Embedding(context_bins, d_model)

        # Encoder blocks + norm
        self.encoder_blocks = nn.ModuleList([
            PreNormEncoderBlock(d_model, n_heads, dim_ff, dropout=0.0)
            for _ in range(n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "MAEEncoder":
        """Load encoder weights from a full MAE checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt["model_state"]

        model = cls()  # defaults match Phase 1 MAE architecture

        # Map MAE state dict keys to our simplified encoder
        my_state = {}
        for k, v in state.items():
            if not hasattr(v, "shape"):
                continue
            if k.startswith("channel_embedding.") and "dropout" not in k:
                my_state[k.replace("channel_embedding.", "")] = v
            elif k.startswith("encoder_blocks.") or k.startswith("encoder_norm."):
                my_state[k] = v

        model.load_state_dict(my_state)
        model.train(False)
        return model.to(device)

    @torch.no_grad()
    def forward(self, sbp_window: Tensor, active_mask: Tensor) -> Tensor:
        """Extract center-timebin features from an 8-bin SBP window.

        Args:
            sbp_window: (B, T, C) raw or z-scored SBP
            active_mask: (B, C) or (C,) boolean, True=active channel

        Returns:
            (B, d_model) pooled features for the center timebin
        """
        B, T, C = sbp_window.shape
        device = sbp_window.device

        if active_mask.dim() == 1:
            active_mask = active_mask.unsqueeze(0).expand(B, -1)

        # Build flat token sequences: (B, T*C)
        ch_ids = torch.arange(C, device=device).repeat(T).unsqueeze(0).expand(B, -1)
        t_ids = torch.arange(T, device=device).repeat_interleave(C).unsqueeze(0).expand(B, -1)
        values = sbp_window.reshape(B, T * C)

        # Embed: scalar projection + channel + time embeddings
        x = self.scalar_proj(values.unsqueeze(-1)) + self.channel_embed(ch_ids) + self.time_embed(t_ids)

        # Padding mask for inactive channels (repeated across timebins)
        active_flat = active_mask.unsqueeze(1).expand(-1, T, -1).reshape(B, T * C)
        pad_mask = ~active_flat
        x = x * active_flat.unsqueeze(-1).float()

        # Run through encoder
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=pad_mask)
        x = self.encoder_norm(x)

        # Reshape to (B, T, C, d_model), extract center timebin, pool over channels
        x = x.reshape(B, T, C, self.d_model)
        center_x = x[:, self.center_idx]  # (B, C, d_model)
        w = active_mask.unsqueeze(-1).float()  # (B, C, 1)
        pooled = (center_x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)  # (B, d_model)
        return pooled


def extract_session_features(
    encoder: MAEEncoder,
    sbp: np.ndarray,
    active_mask: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract MAE features for an entire session.

    Slides 8-bin windows with stride=1, extracts center-timebin features.

    Args:
        encoder: Frozen MAEEncoder
        sbp: (n_bins, 96) SBP array (raw, not z-scored -- matching MAE training)
        active_mask: (96,) boolean
        device: torch device
        batch_size: number of windows per forward pass

    Returns:
        (n_bins, d_model) feature array
    """
    n_bins = sbp.shape[0]
    ctx = encoder.context_bins
    center = encoder.center_idx

    # Pad edges so every bin can be a center
    pad_before = center
    pad_after = ctx - center - 1
    padded = np.pad(sbp, ((pad_before, pad_after), (0, 0)), mode="edge")

    features = np.zeros((n_bins, encoder.d_model), dtype=np.float32)
    mask_t = torch.from_numpy(active_mask).to(device)

    for start in range(0, n_bins, batch_size):
        end = min(start + batch_size, n_bins)
        windows = np.stack([padded[i : i + ctx] for i in range(start, end)])

        windows_t = torch.from_numpy(windows).to(device)
        mask_b = mask_t.unsqueeze(0).expand(end - start, -1)

        feat = encoder(windows_t, mask_b)  # (B, d_model)
        features[start:end] = feat.cpu().numpy()

    return features
