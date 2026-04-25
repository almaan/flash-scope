from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.cluster import MiniBatchKMeans

from flash_scope._utils import to_dense_array


def _inverse_softplus(x: np.ndarray) -> np.ndarray:
    return np.log(np.expm1(np.clip(x, 1e-6, 20.0)))


def nnls_init(
    X_sp: np.ndarray,
    R: np.ndarray,
    damping: float = 0.5,
) -> np.ndarray:
    """Compute initial mixing weights via Tikhonov-regularized NNLS.

    Solves a damped non-negative least squares problem per spot using the
    reference NB parameters as a basis matrix. Returns pre-softplus weights.

    Parameters
    ----------
    X_sp : ndarray, shape (n_spots, n_genes)
        Spatial expression matrix.
    R : ndarray, shape (n_genes, n_types)
        Reference NB dispersion parameters (used as basis).
    damping : float
        Tikhonov regularization strength. Higher values produce smoother
        (less peaked) initializations.

    Returns
    -------
    ndarray, shape (n_spots, n_types)
        Pre-softplus mixing weights.
    """
    X = to_dense_array(X_sp).astype(np.float64)
    basis = np.asarray(R, dtype=np.float64)
    n_spots = X.shape[0]
    n_types = basis.shape[1]

    basis_aug = np.vstack([basis, np.sqrt(damping) * np.eye(n_types)])
    pad = np.zeros(n_types, dtype=np.float64)

    W = np.empty((n_spots, n_types), dtype=np.float32)
    for i in range(n_spots):
        x_aug = np.concatenate([X[i], pad])
        w_i, _ = nnls(basis_aug, x_aug)
        W[i] = w_i

    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    W = W / row_sums

    W = np.clip(W, 1e-3, 1.0)
    W = W / W.sum(axis=1, keepdims=True)

    W_pre = _inverse_softplus(W).astype(np.float32)
    # Append noise column (K+1) initialized near zero
    noise_col = np.zeros((n_spots, 1), dtype=np.float32)
    return np.hstack([W_pre, noise_col])


def coarse_init(
    X_sp: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
    n_clusters: int = 100,
    coarse_epochs: int = 100,
    device: str = "auto",
    verbose: bool = False,
) -> np.ndarray:
    """Compute initial mixing weights via coarse-to-fine dual fit.

    Clusters spatial spots with MiniBatchKMeans, sums expression per cluster,
    fits a small NB mixture model on the clusters, then broadcasts the
    fitted proportions back to individual spots.

    Parameters
    ----------
    X_sp : ndarray, shape (n_spots, n_genes)
        Spatial expression matrix.
    R : ndarray, shape (n_genes, n_types)
        Reference NB dispersion parameters.
    P : ndarray, shape (n_genes,) or (n_genes, 1)
        Reference NB success probabilities.
    n_clusters : int
        Number of clusters (clamped to n_spots if larger).
    coarse_epochs : int
        Training epochs for the coarse model.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    verbose : bool
        Print progress messages.

    Returns
    -------
    ndarray, shape (n_spots, n_types)
        Pre-softplus mixing weights.
    """
    from flash_scope.model._deconv import FlashScopeModel
    from flash_scope.model._trainer import fit

    X = to_dense_array(X_sp).astype(np.float32)
    n_spots, n_genes = X.shape
    n_types = R.shape[1]
    k = min(n_clusters, n_spots)

    if verbose:
        print(f"[flash-scope] coarse init: clustering {n_spots} spots into {k} clusters")

    kmeans = MiniBatchKMeans(n_clusters=k, random_state=0, batch_size=min(1024, n_spots))
    labels = kmeans.fit_predict(X)

    X_coarse = np.zeros((k, n_genes), dtype=np.float32)
    for c in range(k):
        X_coarse[c] = X[labels == c].sum(axis=0)

    ad_coarse = ad.AnnData(
        X=X_coarse,
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]),
    )

    if verbose:
        print(f"[flash-scope] coarse init: fitting {k}-spot model for {coarse_epochs} epochs")

    coarse_model = FlashScopeModel(
        r=np.asarray(R, dtype=np.float32),
        logits=np.asarray(P, dtype=np.float32),
        n_spots=k,
        n_types=n_types,
        n_genes=n_genes,
    )
    coarse_model = fit(coarse_model, ad_coarse, epochs=coarse_epochs, device=device)

    coarse_props = coarse_model.get_proportions()  # (k, n_types) — noise column already stripped
    props = coarse_props[labels]

    props_pre = _inverse_softplus(props).astype(np.float32)
    # Append noise column (K+1) initialized near zero
    noise_col = np.zeros((n_spots, 1), dtype=np.float32)
    return np.hstack([props_pre, noise_col])
