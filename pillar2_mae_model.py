"""PyTorch masked autoencoder for spatiotemporal SBP reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass
class MAEForwardOutput:
    """Structured output of a masked autoencoder forward pass."""

    pred_values: Tensor
    target_values: Tensor
    masked_channel_idx: Tensor
    masked_padding_mask: Tensor
    visible_padding_mask: Tensor


class ChannelEmbedding(nn.Module):
    """Project scalar SBP values and add learned channel/time embeddings.

    This implementation intentionally uses learned per-channel embeddings instead of a fabricated
    Utah-array spatial grid, because channel-to-electrode coordinates are not available in the
    anonymized competition data.
    """

    def __init__(self, n_channels: int, context_bins: int, d_model: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.context_bins = context_bins
        self.d_model = d_model
        self.scalar_proj = nn.Linear(1, d_model)
        self.channel_embed = nn.Embedding(n_channels, d_model)
        self.time_embed = nn.Embedding(context_bins, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, channel_ids: Tensor, time_ids: Tensor) -> Tensor:
        """Embed scalar token values with learned channel/time identities.

        Args:
            values: Tensor `(B, L)` or `(B, L, 1)`
            channel_ids: Tensor `(B, L)`
            time_ids: Tensor `(B, L)`

        Returns:
            Tensor `(B, L, d_model)`
        """
        if values.dim() == 2:
            values = values.unsqueeze(-1)
        if values.dim() != 3:
            raise ValueError(f"values must have shape (B, L) or (B, L, 1), got {tuple(values.shape)}")
        x = self.scalar_proj(values)
        x = x + self.channel_embed(channel_ids) + self.time_embed(time_ids)
        return self.dropout(x)


class SessionEmbedding(nn.Module):
    """Learned session embeddings with interpolation for unseen days."""

    def __init__(self, session_ids: Sequence[str], session_days: Sequence[int], d_model: int) -> None:
        super().__init__()
        if len(session_ids) != len(session_days):
            raise ValueError("session_ids and session_days length mismatch")
        if len(session_ids) == 0:
            raise ValueError("SessionEmbedding requires at least one training session")
        self.d_model = d_model
        self.session_ids = [str(s) for s in session_ids]
        self.session_to_idx = {sid: i for i, sid in enumerate(self.session_ids)}
        if len(self.session_to_idx) != len(self.session_ids):
            raise ValueError("Duplicate session IDs in SessionEmbedding")

        days = torch.tensor([int(d) for d in session_days], dtype=torch.float32)
        sorted_order = torch.argsort(days)
        self.register_buffer("session_days", days, persistent=True)
        self.register_buffer("sorted_days", days[sorted_order], persistent=True)
        self.register_buffer("sorted_order", sorted_order.to(torch.long), persistent=True)
        self.embedding = nn.Embedding(len(self.session_ids), d_model)

    def forward_ids(self, session_ids: Sequence[str], device: torch.device | None = None) -> Tensor:
        """Lookup embeddings for known training session IDs."""
        idx = []
        for sid in session_ids:
            if sid not in self.session_to_idx:
                raise KeyError(f"Unknown training session ID for SessionEmbedding: {sid}")
            idx.append(self.session_to_idx[sid])
        idx_t = torch.tensor(idx, dtype=torch.long, device=device if device is not None else self.embedding.weight.device)
        return self.embedding(idx_t)

    def interpolate_for_day(self, day: int | float) -> Tensor:
        """Interpolate an embedding for an unseen session day.

        Clamps to the nearest training day outside the observed range.
        """
        device = self.embedding.weight.device
        d = float(day)
        sorted_days = self.sorted_days.to(device)
        sorted_idx = self.sorted_order.to(device)
        weights = self.embedding(sorted_idx)

        if d <= float(sorted_days[0].item()):
            return weights[0]
        if d >= float(sorted_days[-1].item()):
            return weights[-1]

        pos = int(torch.searchsorted(sorted_days, torch.tensor([d], device=device), right=False).item())
        hi = max(1, min(pos, sorted_days.numel() - 1))
        lo = hi - 1
        d_lo = float(sorted_days[lo].item())
        d_hi = float(sorted_days[hi].item())
        if math.isclose(d_lo, d_hi):
            return weights[lo]
        alpha = (d - d_lo) / (d_hi - d_lo)
        return (1.0 - alpha) * weights[lo] + alpha * weights[hi]

    def forward_days(self, days: Tensor) -> Tensor:
        """Vectorized interpolation for a batch of session days.

        Args:
            days: Tensor `(B,)`

        Returns:
            Tensor `(B, d_model)`
        """
        if days.dim() != 1:
            raise ValueError(f"days must be 1D, got {tuple(days.shape)}")
        embs = [self.interpolate_for_day(float(d.item())) for d in days]
        return torch.stack(embs, dim=0)


class PreNormEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block."""

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


class PreNormDecoderBlock(nn.Module):
    """Pre-norm transformer decoder block with self- and cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        h = self.norm1(x)
        self_out, _ = self.self_attn(h, h, h, key_padding_mask=tgt_key_padding_mask, need_weights=False)
        x = x + self.drop1(self_out)
        h = self.norm2(x)
        cross_out, _ = self.cross_attn(
            h,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop2(cross_out)
        x = x + self.drop3(self.ffn(self.norm3(x)))
        return x


class MaskedAutoencoder(nn.Module):
    """Masked autoencoder over spatiotemporal SBP windows.

    Encoder receives only visible channel tokens across the temporal context window.
    Decoder predicts the center-bin masked channels.
    """

    def __init__(
        self,
        *,
        n_channels: int,
        context_bins: int,
        d_model: int,
        n_encoder_layers: int,
        n_decoder_layers: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        train_session_ids: Sequence[str],
        train_session_days: Sequence[int],
    ) -> None:
        super().__init__()
        if n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if context_bins <= 0:
            raise ValueError("context_bins must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_channels = n_channels
        self.context_bins = context_bins
        self.d_model = d_model
        self.target_time_index = context_bins // 2 if context_bins % 2 == 1 else (context_bins // 2 - 1)

        self.channel_embedding = ChannelEmbedding(n_channels=n_channels, context_bins=context_bins, d_model=d_model, dropout=dropout)
        self.session_embedding = SessionEmbedding(train_session_ids, train_session_days, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        self.encoder_blocks = nn.ModuleList(
            [PreNormEncoderBlock(d_model=d_model, n_heads=n_heads, dim_ff=dim_ff, dropout=dropout) for _ in range(n_encoder_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        decoder_ff = max(dim_ff // 2, d_model * 2)
        self.decoder_blocks = nn.ModuleList(
            [PreNormDecoderBlock(d_model=d_model, n_heads=n_heads, dim_ff=decoder_ff, dropout=dropout) for _ in range(n_decoder_layers)]
        )
        self.decoder_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, 1)

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _resolve_session_vectors(
        self,
        batch_size: int,
        session_ids: Sequence[str] | None,
        session_days: Tensor | None,
        device: torch.device,
    ) -> Tensor:
        if session_ids is not None:
            if len(session_ids) != batch_size:
                raise ValueError("session_ids length must match batch size")
            return self.session_embedding.forward_ids(session_ids, device=device)
        if session_days is not None:
            if session_days.dim() != 1 or session_days.shape[0] != batch_size:
                raise ValueError("session_days must have shape (B,)")
            return self.session_embedding.forward_days(session_days.to(device=device, dtype=torch.float32))
        raise ValueError("Either session_ids or session_days must be provided")

    def _build_visible_tokens(self, sbp_window: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Pack visible tokens into padded batched sequences.

        Returns:
            values, channel_ids, time_ids, padding_mask
            where padding_mask=True indicates padded positions.
        """
        bsz, t_bins, n_channels = sbp_window.shape
        device = sbp_window.device
        values_list: list[Tensor] = []
        ch_list: list[Tensor] = []
        time_list: list[Tensor] = []
        lengths: list[int] = []

        for b in range(bsz):
            visible = ~mask[b]
            visible_channels = torch.nonzero(visible, as_tuple=False).squeeze(1)
            if visible_channels.numel() == 0:
                raise ValueError("All channels masked for a batch item; encoder requires visible tokens")
            vals_bt = []
            ch_bt = []
            time_bt = []
            for t in range(t_bins):
                vals_bt.append(sbp_window[b, t, visible_channels])
                ch_bt.append(visible_channels)
                time_bt.append(torch.full_like(visible_channels, fill_value=t))
            vals = torch.cat(vals_bt, dim=0)
            chs = torch.cat(ch_bt, dim=0)
            tis = torch.cat(time_bt, dim=0)
            values_list.append(vals)
            ch_list.append(chs)
            time_list.append(tis)
            lengths.append(int(vals.numel()))

        max_len = max(lengths)
        values = sbp_window.new_zeros((bsz, max_len))
        channel_ids = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        time_ids = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        padding_mask = torch.ones((bsz, max_len), dtype=torch.bool, device=device)
        for b in range(bsz):
            l = lengths[b]
            values[b, :l] = values_list[b]
            channel_ids[b, :l] = ch_list[b]
            time_ids[b, :l] = time_list[b]
            padding_mask[b, :l] = False
        return values, channel_ids, time_ids, padding_mask

    def _build_decoder_queries(self, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Build decoder query token identities for center-bin masked channels."""
        bsz, n_channels = mask.shape
        device = mask.device
        masked_ids_list = [torch.nonzero(mask[b], as_tuple=False).squeeze(1) for b in range(bsz)]
        lengths = [int(ids.numel()) for ids in masked_ids_list]
        if any(l == 0 for l in lengths):
            raise ValueError("No masked channels for at least one batch item")
        max_len = max(lengths)

        channel_ids = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
        time_ids = torch.full((bsz, max_len), fill_value=self.target_time_index, dtype=torch.long, device=device)
        padding_mask = torch.ones((bsz, max_len), dtype=torch.bool, device=device)
        for b, ids in enumerate(masked_ids_list):
            l = ids.numel()
            channel_ids[b, :l] = ids
            padding_mask[b, :l] = False
        return channel_ids, time_ids, padding_mask

    def forward(
        self,
        sbp_window: Tensor,
        mask: Tensor,
        session_ids: Sequence[str] | None = None,
        session_days: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Run a forward pass.

        Args:
            sbp_window: Tensor `(B, T, 96)`.
            mask: Boolean Tensor `(B, 96)` with True for masked channels.
            session_ids: Training session IDs for known-session embedding lookup.
            session_days: Session day values for interpolation (used for unseen test sessions).

        Returns:
            Dict containing predictions, targets and masks.
        """
        if sbp_window.dim() != 3:
            raise ValueError(f"sbp_window must have shape (B, T, C), got {tuple(sbp_window.shape)}")
        if mask.dim() != 2:
            raise ValueError(f"mask must have shape (B, C), got {tuple(mask.shape)}")
        bsz, t_bins, n_channels = sbp_window.shape
        if t_bins != self.context_bins:
            raise ValueError(f"Expected context_bins={self.context_bins}, got {t_bins}")
        if n_channels != self.n_channels or mask.shape != (bsz, self.n_channels):
            raise ValueError("Input channel dimensions do not match model")

        device = sbp_window.device
        session_vec = self._resolve_session_vectors(bsz, session_ids, session_days, device=device)  # (B, D)

        vis_values, vis_ch, vis_time, vis_pad = self._build_visible_tokens(sbp_window, mask)
        enc_x = self.channel_embedding(vis_values, vis_ch, vis_time)
        enc_x = enc_x + session_vec[:, None, :]
        for block in self.encoder_blocks:
            enc_x = block(enc_x, key_padding_mask=vis_pad)
        memory = self.encoder_norm(enc_x)

        dec_ch, dec_time, dec_pad = self._build_decoder_queries(mask)
        # Decoder queries are learned mask tokens with identity embeddings added.
        dec_values = torch.zeros((bsz, dec_ch.shape[1]), dtype=sbp_window.dtype, device=device)
        dec_x = self.channel_embedding(dec_values, dec_ch, dec_time)
        dec_x = dec_x + self.mask_token.expand(bsz, dec_ch.shape[1], -1) + session_vec[:, None, :]
        for block in self.decoder_blocks:
            dec_x = block(dec_x, memory, tgt_key_padding_mask=dec_pad, memory_key_padding_mask=vis_pad)
        dec_x = self.decoder_norm(dec_x)
        pred_values = self.output_head(dec_x).squeeze(-1)

        center_vals = sbp_window[:, self.target_time_index, :]
        target_values = torch.gather(center_vals, dim=1, index=dec_ch)

        return {
            "pred_values": pred_values,
            "target_values": target_values,
            "masked_channel_idx": dec_ch,
            "masked_padding_mask": dec_pad,
            "visible_padding_mask": vis_pad,
        }

    @staticmethod
    def compute_loss(
        predictions: Tensor,
        targets: Tensor,
        masked_idx: Tensor,
        channel_vars: Tensor | None = None,
        padding_mask: Tensor | None = None,
        eps: float = 1e-6,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute inverse-channel-variance weighted MSE loss.

        Args:
            predictions: Tensor `(B, M)`
            targets: Tensor `(B, M)`
            masked_idx: Tensor `(B, M)` channel indices for each prediction.
            channel_vars: Optional Tensor `(C,)` or `(B, C)` of variances.
            padding_mask: Optional boolean Tensor `(B, M)` where True = padding.
            eps: Variance floor.

        Returns:
            Tuple `(loss, metrics)` where metrics contains scalar summaries.
        """
        if predictions.shape != targets.shape or predictions.shape != masked_idx.shape:
            raise ValueError("predictions, targets, and masked_idx must have the same shape")
        sq_err = (predictions - targets) ** 2

        if channel_vars is None:
            weights = torch.ones_like(sq_err)
        else:
            if channel_vars.dim() == 1:
                gathered_vars = channel_vars.to(predictions.device)[masked_idx]
            elif channel_vars.dim() == 2:
                gathered_vars = torch.gather(channel_vars.to(predictions.device), 1, masked_idx)
            else:
                raise ValueError("channel_vars must be shape (C,) or (B, C)")
            weights = 1.0 / torch.clamp(gathered_vars, min=eps)

        if padding_mask is not None:
            valid = (~padding_mask).to(dtype=predictions.dtype)
            sq_err = sq_err * valid
            weights = weights * valid

        weighted_num = (sq_err * weights).sum()
        weighted_den = torch.clamp(weights.sum(), min=eps)
        loss = weighted_num / weighted_den

        if padding_mask is not None:
            valid_count = int((~padding_mask).sum().item())
        else:
            valid_count = int(predictions.numel())
        metrics = {
            "loss": float(loss.detach().item()),
            "mse": float(sq_err.sum().detach().item() / max(valid_count, 1)),
            "valid_tokens": float(valid_count),
        }
        return loss, metrics


__all__ = [
    "ChannelEmbedding",
    "MaskedAutoencoder",
    "SessionEmbedding",
    "MAEForwardOutput",
]
