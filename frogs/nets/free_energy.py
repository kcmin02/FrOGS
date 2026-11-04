"""Auxiliary free-energy network F_phi(t, c) (§A.2).

Only its time derivative enters the training loss; it never enters the log-weights.
"""
import torch
from torch.func import jacfwd, vmap


class FreeEnergyNet(torch.nn.Module):
    """MLP on [t, c_normalized] -> R."""

    def __init__(self, hidden_sizes, cond_low, cond_high, cond_dim: int = 2):
        super().__init__()
        layers, width = [], 1 + cond_dim
        for size in hidden_sizes:
            layers += [torch.nn.Linear(width, size), torch.nn.SiLU()]
            width = size
        layers.append(torch.nn.Linear(width, 1))
        self.net = torch.nn.Sequential(*layers)
        lo = torch.as_tensor(cond_low, dtype=torch.get_default_dtype())
        hi = torch.as_tensor(cond_high, dtype=torch.get_default_dtype())
        self.register_buffer("_cond_mid", 0.5 * (lo + hi))
        self.register_buffer("_cond_half", 0.5 * (hi - lo))

    def _single(self, t, c):
        t = t.unsqueeze(0) if t.dim() == 0 else t
        x = torch.cat([t, (c - self._cond_mid) / self._cond_half], dim=0)
        return self.net(x).squeeze()

    def forward(self, t, c):
        return vmap(self._single, in_dims=(0, 0), out_dims=(0,))(t, c)

    def gradt(self, t, c):
        """d F / d t at fixed c, shape (B,)."""
        return vmap(jacfwd(self._single, 0), in_dims=(0, 0), out_dims=(0,),
                    randomness="different")(t, c)
