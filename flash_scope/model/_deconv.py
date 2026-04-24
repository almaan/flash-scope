from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.negative_binomial import NegativeBinomial


class FlashScopeModel(nn.Module):
    def __init__(
        self,
        r: np.ndarray,
        p: np.ndarray,
        n_spots: int,
        n_types: int,
        n_genes: int,
    ):
        super().__init__()
        self.n_spots = n_spots
        self.n_types = n_types
        self.n_genes = n_genes

        self.w = nn.Parameter(torch.ones(n_spots, n_types) / n_types)
        self.nu = nn.Parameter(torch.ones(1, n_genes))
        self.eta = nn.Parameter(torch.zeros(1, n_genes))
        self.alpha = nn.Parameter(torch.ones(1))

        self.register_buffer("r_sc", torch.tensor(r.T.astype(np.float32)).contiguous())
        p_arr = np.asarray(p, dtype=np.float32).ravel()
        self.register_buffer("p_sc", torch.tensor(p_arr).unsqueeze(0).contiguous())

        self._softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta = self._softplus(self.nu)
        v = self._softplus(self.w[idx])
        eps = self._softplus(self.eta)
        gamma = self._softplus(self.alpha)
        r_sp = beta * torch.mm(v, self.r_sc) + gamma * eps
        p_sp = self.p_sc
        return r_sp, p_sp

    def loss(self, x: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return -NegativeBinomial(total_count=r, probs=p).log_prob(x).sum()

    @torch.no_grad()
    def get_proportions(self) -> np.ndarray:
        props = self._softplus(self.w)
        props = props / props.sum(dim=1, keepdim=True)
        return props.detach().cpu().numpy()
