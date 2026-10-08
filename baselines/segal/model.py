"""SEGAL autoregressive network of Damewood et al. (2022), inference only.

Re-implements the masked-linear network of github.com/learningmatter-mit/Segal (MIT License) so that
the released weights load unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {"sigmoid": torch.sigmoid, "tanh": torch.tanh}


class SegalAR(nn.Module):
    """P_AR(sigma | T, field) over N binary sites; every layer also sees (T/1000, 10 field).

    The autoregressive masks are baked into the released weights (site i depends on sites < i).
    """

    def __init__(self, n_sites: int, num_layers: int, activation: str):
        super().__init__()
        self.n_sites = n_sites
        self.activation = ACTIVATIONS[activation]
        self.layers = nn.ModuleList(nn.Linear(2 * n_sites + 2, 2 * n_sites, bias=False)
                                    for _ in range(num_layers))

    @classmethod
    def from_checkpoint(cls, path: str, num_layers: int, activation: str) -> SegalAR:
        state = torch.load(path, map_location="cpu", weights_only=True)
        names = ["shared_layer"] + [f"shared_layer_{k}" for k in range(2, num_layers + 1)]
        model = cls(state["Model.shared_layer.weight"].shape[0] // 2, num_layers, activation)
        model.layers.load_state_dict({f"{i}.weight": state[f"Model.{name}.weight"]
                                      for i, name in enumerate(names)})
        return model

    @staticmethod
    def _cond(T, field):
        return torch.cat([T.view(-1, 1) / 1000.0, field.view(-1, 1) * 10.0], dim=1)

    def _logits(self, occ, cond):
        """(B, N, 2) one-hot occupations -> (B, N, 2) logits."""
        h = torch.cat([cond, occ.reshape(len(occ), -1)], dim=1)
        for k, layer in enumerate(self.layers):
            if k:
                h = torch.cat([cond, self.activation(h)], dim=1)
            h = layer(h)
        return h.view(-1, self.n_sites, 2)

    def log_prob(self, sigma, T, field):
        """log P_AR for sigma (B, N) in {-1, +1}."""
        occ = F.one_hot(((sigma + 1) * 0.5).long(), num_classes=2).float()
        log_p = F.log_softmax(self._logits(occ, self._cond(T, field)), dim=2)
        return (log_p * occ).sum(dim=2).sum(dim=1)

    @torch.no_grad()
    def sample(self, T, field):
        """One configuration per entry of T and field, (B,) -> (B, N) in {-1, +1}."""
        cond = self._cond(T, field)
        occ = torch.zeros(len(cond), self.n_sites, 2, device=cond.device)
        for i in range(self.n_sites):
            probs = F.softmax(self._logits(occ, cond)[:, i, :], dim=1)
            occ[:, i, :] = F.one_hot(torch.multinomial(probs, 1)[:, 0], num_classes=2).float()
        return occ[:, :, 1] * 2.0 - 1.0
