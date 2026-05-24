"""Student world model.

The public interface is intentionally small: evaluation calls
``forward(obs_norm, act_norm, hidden)`` and expects a normalized state delta.
This model keeps that contract while using a stronger recurrent residual
dynamics predictor for long open-loop rollouts.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 2,
        use_gru: bool = False,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        in_dim = obs_dim + act_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(max(1, int(num_layers) - 1))])
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        self.linear_head = nn.Linear(in_dim, obs_dim)
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self._init_heads()

    def _init_heads(self) -> None:
        nn.init.normal_(self.linear_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.linear_head.bias)
        final = self.residual_head[-1]
        if isinstance(final, nn.Linear):
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final.bias)

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru:
            return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        x = torch.cat([obs_norm, act_norm], dim=-1)
        feat = self.encoder(x)
        for block in self.blocks:
            feat = block(feat)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
        raw_delta = self.linear_head(x) + self.residual_head(feat)
        delta = self.delta_limit * torch.tanh(raw_delta / self.delta_limit)
        return delta, hidden
