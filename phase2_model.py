"""Model architectures for Phase 2 kinematic decoding.

Three architectures:
  1. TransformerDecoder — time-series transformer treating each bin as a token
  2. POYODecoder — Perceiver IO inspired by POYO-1 (Azabou et al. 2023)
     with channel-level tokenization, latent compression, and output querying
  3. GRUDecoder — bidirectional GRU for ensemble diversity
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ============================================================================
# Shared components
# ============================================================================

class RotaryPositionEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) for temporal attention."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        assert d_model % 2 == 0
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Compute cos/sin tables for given position indices.

        Args:
            positions: (B, L) float timestamps or integer positions

        Returns:
            cos, sin each of shape (B, L, d_model)
        """
        freqs = torch.einsum("bl,d->bld", positions.float(), self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to input tensor x of shape (..., d)."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional RoPE and key padding mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        q_rope: tuple[Tensor, Tensor] | None = None,
        k_rope: tuple[Tensor, Tensor] | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        q = self.q_proj(query).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        if q_rope is not None:
            cos_q, sin_q = q_rope
            cos_q = cos_q[:, :Lq, :self.head_dim].unsqueeze(1)
            sin_q = sin_q[:, :Lq, :self.head_dim].unsqueeze(1)
            q = apply_rope(q, cos_q, sin_q)
        if k_rope is not None:
            cos_k, sin_k = k_rope
            cos_k = cos_k[:, :Lk, :self.head_dim].unsqueeze(1)
            sin_k = sin_k[:, :Lk, :self.head_dim].unsqueeze(1)
            k = apply_rope(k, cos_k, sin_k)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk) True = masked position
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (self-attention or cross-attention)."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        context: Tensor | None = None,
        x_rope: tuple[Tensor, Tensor] | None = None,
        ctx_rope: tuple[Tensor, Tensor] | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        kv = context if context is not None else x
        kv_rope = ctx_rope if ctx_rope is not None else x_rope

        normed = self.norm1(x)
        normed_kv = self.norm1(kv) if context is not None else normed

        x = x + self.attn(normed, normed_kv, normed_kv, q_rope=x_rope, k_rope=kv_rope, key_padding_mask=key_padding_mask)
        x = x + self.ff(self.norm2(x))
        return x


# ============================================================================
# Model A: Time-Series Transformer Decoder
# ============================================================================

class TransformerDecoder(nn.Module):
    """Per-timestep Transformer decoder for kinematic prediction.

    Each time bin's 96-channel SBP vector is projected to d_model, processed
    through a stack of Transformer encoder layers, then decoded per-timestep.
    """

    def __init__(
        self,
        n_channels: int = 96,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        dim_ff: int = 512,
        dropout: float = 0.1,
        max_context: int = 256,
        n_outputs: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Embedding(max_context, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout) for _ in range(n_layers)
        ])

        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_outputs)

    def forward(self, sbp: Tensor, **kwargs) -> Tensor:
        """
        Args:
            sbp: (B, T, 96) z-scored SBP

        Returns:
            (B, T, 2) predicted positions
        """
        B, T, _ = sbp.shape
        pos_ids = torch.arange(T, device=sbp.device).unsqueeze(0).expand(B, -1)

        x = self.input_proj(sbp) + self.pos_embed(pos_ids)
        x = self.input_drop(self.input_norm(x))

        for layer in self.layers:
            x = layer(x)

        x = self.head_norm(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Model B: POYO-Perceiver Decoder
# ============================================================================

class POYODecoder(nn.Module):
    """POYO-inspired Perceiver IO for kinematic decoding.

    Architecture (adapted from Azabou et al. 2023 for binned SBP data):
      1. Input tokenization: each (time_bin, active_channel) -> token
         via scalar_proj(sbp_value) + channel_embed(ch_id)
      2. Cross-attention: compress variable-length input tokens into N
         fixed latent tokens
      3. Self-attention: L layers on the latent sequence with RoPE
      4. Output query: T output tokens (one per output time bin) cross-attend
         to latent space to produce per-timestep predictions
      5. MLP head: latent -> 2 position outputs
    """

    def __init__(
        self,
        n_channels: int = 96,
        d_model: int = 128,
        n_latents: int = 64,
        n_self_attn_layers: int = 6,
        n_heads: int = 8,
        dim_ff: int = 512,
        dropout: float = 0.1,
        max_context: int = 256,
        n_outputs: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_latents = n_latents
        self.n_channels = n_channels

        # Input tokenization
        self.scalar_proj = nn.Linear(1, d_model)
        self.channel_embed = nn.Embedding(n_channels, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Latent tokens
        self.latent_tokens = nn.Parameter(torch.randn(1, n_latents, d_model) * 0.02)

        # Perceiver cross-attention: input -> latent
        self.input_cross_attn = TransformerBlock(d_model, n_heads, dim_ff, dropout)

        # Self-attention on latent space
        self.self_attn_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dim_ff, dropout)
            for _ in range(n_self_attn_layers)
        ])

        # Output query tokens (one per context time bin, will be expanded)
        self.output_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.output_pos_embed = nn.Embedding(max_context, d_model)

        # Output cross-attention: latent -> output
        self.output_cross_attn = TransformerBlock(d_model, n_heads, dim_ff, dropout)

        # RoPE for temporal encoding
        self.rope = RotaryPositionEncoding(d_model // n_heads, max_len=max_context * n_channels)

        # Prediction head
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, n_outputs),
        )

    def _tokenize_input(
        self, sbp: Tensor, active_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert (B, T, 96) SBP into variable-length channel-level tokens.

        Returns:
            tokens: (B, max_tokens, d_model)
            time_ids: (B, max_tokens) time bin index per token
            padding_mask: (B, max_tokens) True = padding
        """
        B, T, C = sbp.shape
        device = sbp.device

        # Precompute all tokens densely, then mask
        # (B, T, C) -> (B, T*C)
        flat_sbp = sbp.reshape(B, T * C)
        # Channel IDs: (T*C,) tiled
        ch_ids = torch.arange(C, device=device).unsqueeze(0).expand(T, -1).reshape(-1)
        ch_ids = ch_ids.unsqueeze(0).expand(B, -1)
        # Time IDs
        t_ids = torch.arange(T, device=device).unsqueeze(1).expand(-1, C).reshape(-1)
        t_ids = t_ids.unsqueeze(0).expand(B, -1)

        # Active mask expanded: (B, T*C)
        # active_mask is (B, 96) — expand over time
        active_expanded = active_mask.unsqueeze(1).expand(-1, T, -1).reshape(B, T * C)

        # Embed all tokens
        tok = self.scalar_proj(flat_sbp.unsqueeze(-1)) + self.channel_embed(ch_ids)
        tok = self.input_norm(tok)

        # Mask out inactive channel tokens
        padding_mask = ~active_expanded  # True = pad

        return tok, t_ids.float(), padding_mask

    def forward(self, sbp: Tensor, active_mask: Tensor | None = None, **kwargs) -> Tensor:
        """
        Args:
            sbp: (B, T, 96) z-scored SBP
            active_mask: (B, 96) boolean mask, True = active channel

        Returns:
            (B, T, 2) predicted positions
        """
        B, T, C = sbp.shape
        device = sbp.device

        if active_mask is None:
            active_mask = (sbp.abs().sum(dim=1) > 0)  # (B, 96)

        # 1. Tokenize input
        input_tokens, time_ids, padding_mask = self._tokenize_input(sbp, active_mask)

        # 2. RoPE for input tokens
        input_rope = self.rope(time_ids)

        # 3. Latent tokens with evenly-spaced timestamps
        latents = self.latent_tokens.expand(B, -1, -1)
        latent_times = torch.linspace(0, T - 1, self.n_latents, device=device)
        latent_times = latent_times.unsqueeze(0).expand(B, -1)
        latent_rope = self.rope(latent_times)

        # 4. Cross-attention: compress inputs -> latents
        latents = self.input_cross_attn(
            latents, context=input_tokens,
            x_rope=latent_rope, ctx_rope=input_rope,
            key_padding_mask=padding_mask,
        )

        # 5. Self-attention on latent sequence
        for layer in self.self_attn_layers:
            latents = layer(latents, x_rope=latent_rope)

        # 6. Output queries: one per time bin
        out_pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        output_tokens = self.output_query.expand(B, T, -1) + self.output_pos_embed(out_pos_ids)
        out_times = torch.arange(T, device=device, dtype=torch.float).unsqueeze(0).expand(B, -1)
        out_rope = self.rope(out_times)

        # 7. Cross-attention: latents -> output
        output = self.output_cross_attn(
            output_tokens, context=latents,
            x_rope=out_rope, ctx_rope=latent_rope,
        )

        # 8. Prediction head
        output = self.head_norm(output)
        return self.head(output)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Model C: Bidirectional GRU Decoder
# ============================================================================

class GRUDecoder(nn.Module):
    """Bidirectional GRU decoder for kinematic prediction.

    Different inductive bias from transformers: recurrent state propagation
    captures temporal dynamics naturally, while bidirectionality allows
    attending to both past and future within the window.
    """

    def __init__(
        self,
        n_channels: int = 96,
        d_model: int = 128,
        n_layers: int = 3,
        dropout: float = 0.2,
        n_outputs: int = 2,
    ):
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_outputs),
        )

    def forward(self, sbp: Tensor, **kwargs) -> Tensor:
        """
        Args:
            sbp: (B, T, 96) z-scored SBP

        Returns:
            (B, T, 2) predicted positions
        """
        x = self.input_proj(sbp)
        x, _ = self.gru(x)
        return self.head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Factory
# ============================================================================

def build_model(config) -> nn.Module:
    """Build model based on config.model_type."""
    if config.model_type == "transformer":
        return TransformerDecoder(
            n_channels=config.n_channels,
            d_model=config.tf_d_model,
            n_layers=config.tf_n_layers,
            n_heads=config.tf_n_heads,
            dim_ff=config.tf_dim_ff,
            dropout=config.tf_dropout,
            max_context=config.context_bins * 2,
            n_outputs=config.n_position_outputs,
        )
    elif config.model_type == "poyo":
        return POYODecoder(
            n_channels=config.n_channels,
            d_model=config.poyo_d_model,
            n_latents=config.poyo_n_latents,
            n_self_attn_layers=config.poyo_n_self_attn_layers,
            n_heads=config.poyo_n_heads,
            dim_ff=config.poyo_dim_ff,
            dropout=config.poyo_dropout,
            max_context=config.context_bins * 2,
            n_outputs=config.n_position_outputs,
        )
    elif config.model_type == "gru":
        return GRUDecoder(
            n_channels=config.n_channels,
            d_model=config.gru_d_model,
            n_layers=config.gru_n_layers,
            dropout=config.gru_dropout,
            n_outputs=config.n_position_outputs,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")
