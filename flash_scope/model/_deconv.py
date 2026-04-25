from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlashScopeModel(nn.Module):
    """Negative binomial mixture model for spatial deconvolution.

    Learns per-spot mixing weights over cell types by maximizing the
    NB log-likelihood of observed spatial counts given reference parameters.
    Noise is modelled as an additional (K+1)-th type with per-spot proportion.

    Parameters
    ----------
    r : ndarray, shape (n_genes, n_types)
        NB rate parameters from the reference.
    logits : ndarray, shape (n_genes,)
        NB logits (log-odds of success probability) from the reference.
    n_spots : int
        Number of spatial spots.
    n_types : int
        Number of cell types.
    n_genes : int
        Number of genes.
    init_w : ndarray or None, shape (n_spots, n_types + 1)
        Initial mixing weights (pre-softplus). If None, random normal init.
    """

    def __init__(
        self,
        r: np.ndarray,
        logits: np.ndarray,
        n_spots: int,
        n_types: int,
        n_genes: int,
        init_w: np.ndarray | None = None,
    ):
        super().__init__()
        self.n_spots = n_spots
        self.n_types = n_types
        self.n_genes = n_genes

        if init_w is not None:
            self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))
        else:
            self.w = nn.Parameter(torch.zeros(n_spots, n_types + 1))
            nn.init.normal_(self.w, mean=0.0, std=1.0)

        self.beta = nn.Parameter(torch.zeros(1, n_genes))
        nn.init.normal_(self.beta, mean=0.0, std=0.1)

        self.eta = nn.Parameter(torch.zeros(1, n_genes))
        nn.init.normal_(self.eta, mean=0.0, std=1.0)

        # r_sc: (K, G) — reference rates per type
        self.register_buffer("r_sc", torch.tensor(r.T.astype(np.float32)).contiguous())
        o_arr = np.asarray(logits, dtype=np.float32).ravel()
        self.register_buffer("o_sc", torch.tensor(o_arr).unsqueeze(0).contiguous())

        self._softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-spot NB rate for a batch of spots."""
        beta = self._softplus(self.beta)  # (1, G)
        eps = self._softplus(self.eta)    # (1, G)
        v = self._softplus(self.w[idx])   # (batch, K+1)

        # r_hat: (K+1, G) — concatenate scaled reference rates with noise
        r_hat = torch.cat([beta * self.r_sc, eps], dim=0)

        # r_sp: (batch, G)
        r_sp = torch.mm(v, r_hat)
        return r_sp

    def loss(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        lgamma_x1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sum negative NB log-likelihood plus noise prior."""
        r = torch.clamp(r, min=1e-6, max=1e6)
        if lgamma_x1 is None:
            lgamma_x1 = torch.lgamma(x + 1)
        log_prob = (
            torch.lgamma(x + r)
            - torch.lgamma(r)
            - lgamma_x1
            + r * F.logsigmoid(-self.o_sc)
            + x * F.logsigmoid(self.o_sc)
        )
        nll = -log_prob.sum()
        noise_prior = -0.5 * torch.sum(self.eta ** 2)
        return nll - noise_prior

    @torch.no_grad()
    def get_proportions(self) -> np.ndarray:
        """Return normalized cell type proportions, shape ``(n_spots, n_types)``."""
        props = self._softplus(self.w)
        props = props[:, :self.n_types]
        props = props / props.sum(dim=1, keepdim=True)
        return props.detach().cpu().numpy()
