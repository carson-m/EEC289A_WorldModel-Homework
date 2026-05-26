"""Student world model (Powerful + Lightweight Hybrid)."""

from __future__ import annotations
import torch
from torch import nn

class ResBlock(nn.Module):
    """Single Residual Block for lightweight capacity."""
    def __init__(self, dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layer(x)

class StudentWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 1,
        hidden_dim: int = 256,
        num_layers: int = 2,  # Only 2 GRU needed!
        use_gru: bool = True,
        delta_limit: float = 3.0,
    ):
        super().__init__()
        self.use_gru = bool(use_gru)
        self.delta_limit = float(delta_limit)
        in_dim = obs_dim + act_dim

        # 1. Lightweight Encoder (1 Linear + 1 ResBlock)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            ResBlock(hidden_dim)
        )

        # 2. The Memory Core
        self.gru = nn.GRUCell(hidden_dim, hidden_dim) if self.use_gru else None
        
        # POWER TRICK: Orthogonal Initialization stops exploding/vanishing gradients in long rollouts
        if self.gru is not None:
            nn.init.orthogonal_(self.gru.weight_hh)
            nn.init.orthogonal_(self.gru.weight_ih)

        # 3. Lightweight Decoder (1 ResBlock + 1 Linear)
        self.decoder = nn.Sequential(
            ResBlock(hidden_dim),
            nn.Linear(hidden_dim, obs_dim)
        )
        # Zero init for stability
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

        # 4. POWER TRICK: Linear Skip Connection
        # Solves local 1-step physics instantly without clogging the GRU
        self.skip = nn.Linear(in_dim, obs_dim)
        nn.init.zeros_(self.skip.weight)
        nn.init.zeros_(self.skip.bias)

    def initial_hidden(self, batch_size: int, device: torch.device):
        if not self.use_gru: return None
        return torch.zeros(batch_size, self.gru.hidden_size, device=device)

    def forward(self, obs_norm: torch.Tensor, act_norm: torch.Tensor, hidden=None):
        # Noise Injection to prevent overfitting to exact states
        if self.training:
            obs_norm = obs_norm + torch.randn_like(obs_norm) * 0.002

        x = torch.cat([obs_norm, act_norm], dim=-1)

        # Deep Processing
        feat = self.encoder(x)
        if self.gru is not None:
            if hidden is None:
                hidden = self.initial_hidden(obs_norm.shape[0], obs_norm.device)
            hidden = self.gru(feat, hidden)
            feat = hidden
            
        delta_pred = self.decoder(feat)
        
        # Add the linear skip directly to the final prediction
        skip_pred = self.skip(x)

        return delta_pred + skip_pred, hidden