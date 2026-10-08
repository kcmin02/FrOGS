"""FiLM conditioning on c = (dmu, T) (§A.8): per-channel, spatially uniform scale and shift."""
import torch
import torch.nn as nn


class FiLM(nn.Module):
    """gamma(c) * h + beta(c); the last layer is zero-initialized so FiLM starts as the identity."""

    def __init__(self, cond_dim: int, num_channels: int, spatial_dims: int, hidden_dim: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.spatial_dims = spatial_dims
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * num_channels),
        )
        with torch.no_grad():
            self.net[-1].weight.zero_()
            self.net[-1].bias.zero_()

    def forward(self, h, c):
        gamma_raw, beta = self.net(c).chunk(2, dim=-1)
        shape = (h.shape[0], self.num_channels) + (1,) * self.spatial_dims
        return (1.0 + gamma_raw).view(shape) * h + beta.view(shape)
