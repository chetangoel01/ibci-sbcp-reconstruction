"""Neural decoders for Phase 2 kinematic prediction."""

from __future__ import annotations

import torch
from torch import nn

from pillar2_mae_model import ChannelEmbedding, PreNormEncoderBlock, SessionEmbedding


class GRUKinematicsDecoder(nn.Module):
    """Bidirectional GRU over SBP windows with a center-bin prediction head."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
        target_index: int,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.target_index = target_index
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.dim() != 3:
            raise ValueError(f"Expected input shape (B, T, F), got {tuple(inputs.shape)}")
        x = self.input_proj(self.input_norm(inputs))
        seq, _ = self.gru(x)
        center = seq[:, self.target_index, :]
        return self.head(center)


class TransformerKinematicsDecoder(nn.Module):
    """Phase 2 transformer decoder built from the Phase 1 encoder components."""

    def __init__(
        self,
        *,
        n_channels: int,
        context_bins: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        dim_ff: int,
        dropout: float,
        output_dim: int,
        train_session_ids: list[str],
        train_session_days: list[int],
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_channels = n_channels
        self.context_bins = context_bins
        self.d_model = d_model

        self.channel_embedding = ChannelEmbedding(
            n_channels=n_channels,
            context_bins=context_bins,
            d_model=d_model,
            dropout=dropout,
        )
        self.session_embedding = SessionEmbedding(train_session_ids, train_session_days, d_model)
        self.encoder_blocks = nn.ModuleList(
            [PreNormEncoderBlock(d_model=d_model, n_heads=n_heads, dim_ff=dim_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.encoder_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, output_dim),
        )

    def _build_visible_tokens(
        self,
        sbp_window: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, t_bins, n_channels = sbp_window.shape
        device = sbp_window.device
        values_list: list[torch.Tensor] = []
        ch_list: list[torch.Tensor] = []
        time_list: list[torch.Tensor] = []
        lengths: list[int] = []

        for b in range(bsz):
            visible_channels = torch.nonzero(active_mask[b], as_tuple=False).squeeze(1)
            if visible_channels.numel() == 0:
                raise ValueError("A batch item has zero active channels")
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

    def forward(
        self,
        sbp_window: torch.Tensor,
        active_mask: torch.Tensor,
        session_days: torch.Tensor,
    ) -> torch.Tensor:
        if sbp_window.dim() != 3:
            raise ValueError(f"Expected sbp_window shape (B, T, C), got {tuple(sbp_window.shape)}")
        if active_mask.dim() != 2:
            raise ValueError(f"Expected active_mask shape (B, C), got {tuple(active_mask.shape)}")
        if session_days.dim() != 1:
            raise ValueError(f"Expected session_days shape (B,), got {tuple(session_days.shape)}")
        bsz, t_bins, n_channels = sbp_window.shape
        if t_bins != self.context_bins or n_channels != self.n_channels or active_mask.shape != (bsz, self.n_channels):
            raise ValueError("Transformer input dimensions do not match model config")

        vis_values, vis_ch, vis_time, vis_pad = self._build_visible_tokens(sbp_window, active_mask)
        x = self.channel_embedding(vis_values, vis_ch, vis_time)
        session_vec = self.session_embedding.forward_days(session_days.to(device=sbp_window.device, dtype=torch.float32))
        x = x + session_vec[:, None, :]
        for block in self.encoder_blocks:
            x = block(x, key_padding_mask=vis_pad)
        x = self.encoder_norm(x)

        valid = (~vis_pad).to(dtype=x.dtype)
        pooled = (x * valid.unsqueeze(-1)).sum(dim=1) / torch.clamp(valid.sum(dim=1, keepdim=True), min=1.0)
        return self.head(pooled)


__all__ = ["GRUKinematicsDecoder", "TransformerKinematicsDecoder"]
