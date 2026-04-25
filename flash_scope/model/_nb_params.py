from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

from flash_scope._utils import to_dense_array

EPS = 1e-8


def estimate_nb_params(
    adata: ad.AnnData,
    label_col: str,
    backend: str = "numpy",
    return_labels: bool = False,
    winsorize_pct: float = 5.0,
    shrinkage: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Estimate per-gene, per-cell-type negative binomial parameters.

    Uses method-of-moments on the reference scRNA-seq data, with optional
    winsorization and empirical Bayes shrinkage of dispersion.

    Parameters
    ----------
    adata : AnnData
        Reference single-cell data with cell type labels.
    label_col : str
        Column in ``adata.obs`` with cell type labels.
    backend : str
        ``"numpy"`` (CPU) or ``"torch"`` (GPU-accelerated).
    return_labels : bool
        If True, also return the sorted unique label array.
    winsorize_pct : float
        Percentile for winsorizing per-type expression (0 disables).
    shrinkage : bool
        Apply empirical Bayes shrinkage to dispersion estimates.

    Returns
    -------
    R : DataFrame, shape (n_genes, n_types)
        NB rate parameters (raw, not normalized).
    logits : DataFrame, shape (n_genes, 1)
        NB logits (log-odds of success probability).
    labels : ndarray (only if ``return_labels=True``)
        Sorted unique cell type labels.
    """
    X = to_dense_array(adata.X)
    labels = adata.obs[label_col].values.astype(str)
    uni_labels = np.unique(labels)

    if backend == "numpy":
        R, P_arr = _estimate_numpy(X, labels, uni_labels, winsorize_pct=winsorize_pct, shrinkage=shrinkage)
    elif backend == "torch":
        R, P_arr = _estimate_torch(X, labels, uni_labels, winsorize_pct=winsorize_pct, shrinkage=shrinkage)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'numpy' or 'torch'.")

    R_df = pd.DataFrame(R, index=adata.var_names, columns=uni_labels)
    P_df = pd.DataFrame(P_arr, index=adata.var_names)

    if return_labels:
        return R_df, P_df, uni_labels
    return R_df, P_df


def _winsorize(X: np.ndarray, label_idx: np.ndarray, n_types: int, pct: float) -> np.ndarray:
    X = X.copy()
    lo = pct / 100.0
    hi = 1.0 - lo
    for k in range(n_types):
        mask = label_idx == k
        sub = X[mask]
        lower = np.quantile(sub, lo, axis=0)
        upper = np.quantile(sub, hi, axis=0)
        X[mask] = np.clip(sub, lower, upper)
    return X


def _shrink_r(r: np.ndarray, counts: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    mu = r.mean(axis=0, keepdims=True)
    tau2 = r.var(axis=0, keepdims=True)
    tau2 = np.maximum(tau2, EPS)
    sigma2 = var / (counts[:, None] + EPS)
    w = tau2 / (tau2 + sigma2)
    return mu + w * (r - mu)


def _estimate_numpy(
    X: np.ndarray,
    labels: np.ndarray,
    uni_labels: np.ndarray,
    winsorize_pct: float = 0.0,
    shrinkage: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells, n_genes = X.shape
    n_types = len(uni_labels)

    label_idx = np.zeros(n_cells, dtype=np.intp)
    for k, lab in enumerate(uni_labels):
        label_idx[labels == lab] = k

    # Library size normalization: equalize sequencing depth across cells
    lib_size = X.sum(axis=1, keepdims=True).astype(np.float64)
    lib_size = np.maximum(lib_size, 1.0)
    median_lib = np.median(lib_size)
    X_norm = X * (median_lib / lib_size)

    Xw = X_norm
    if winsorize_pct > 0:
        Xw = _winsorize(X_norm.astype(np.float64), label_idx, n_types, winsorize_pct)

    counts = np.bincount(label_idx, minlength=n_types).astype(np.float64)

    indicator = sp.csr_matrix(
        (np.ones(n_cells, dtype=np.float64), (label_idx, np.arange(n_cells))),
        shape=(n_types, n_cells),
    )
    Xw64 = Xw.astype(np.float64)
    sums = indicator @ Xw64
    sq_sums = indicator @ (Xw64 ** 2)

    mean = sums / counts[:, None] + EPS
    var = (sq_sums / counts[:, None] - (sums / counts[:, None]) ** 2)
    var = var * counts[:, None] / (counts[:, None] - 1 + EPS) + EPS

    a = var
    b = mean
    c = -mean

    p = np.clip((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), EPS, 1 - EPS)
    r = mean * p / (1 - p)

    if shrinkage:
        r = _shrink_r(r, counts, mean, var)
        r = np.maximum(r, EPS)
        p = np.clip(r / (r + mean), EPS, 1 - EPS)

    R = r.T

    P_weighted = np.sum(p * r, axis=0) / (np.sum(r, axis=0) + EPS)
    P_weighted = np.clip(P_weighted, EPS, 1 - EPS)
    logits = np.log(P_weighted / (1 - P_weighted)).astype(np.float64)

    return R, logits


def _estimate_torch(
    X: np.ndarray,
    labels: np.ndarray,
    uni_labels: np.ndarray,
    winsorize_pct: float = 0.0,
    shrinkage: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_cells, n_genes = X.shape
    n_types = len(uni_labels)

    label_idx_np = np.zeros(n_cells, dtype=np.intp)
    for k, lab in enumerate(uni_labels):
        label_idx_np[labels == lab] = k

    # Library size normalization: equalize sequencing depth across cells
    lib_size = X.sum(axis=1, keepdims=True).astype(np.float64)
    lib_size = np.maximum(lib_size, 1.0)
    median_lib = np.median(lib_size)
    X_norm = X * (median_lib / lib_size)

    Xw = X_norm
    if winsorize_pct > 0:
        Xw = _winsorize(X_norm.astype(np.float64), label_idx_np, n_types, winsorize_pct)

    X_t = torch.tensor(Xw, dtype=torch.float64, device=device)

    label_idx = torch.tensor(label_idx_np, dtype=torch.long, device=device)

    counts = torch.bincount(label_idx, minlength=n_types).to(torch.float64)

    sums = torch.zeros(n_types, n_genes, dtype=torch.float64, device=device)
    sums.index_add_(0, label_idx, X_t)

    sq_sums = torch.zeros(n_types, n_genes, dtype=torch.float64, device=device)
    sq_sums.index_add_(0, label_idx, X_t ** 2)

    mean = sums / counts[:, None] + EPS
    var = (sq_sums / counts[:, None] - (sums / counts[:, None]) ** 2)
    var = var * counts[:, None] / (counts[:, None] - 1 + EPS) + EPS

    a = var
    b = mean
    c = -mean

    p = torch.clamp((-b + torch.sqrt(b ** 2 - 4 * a * c)) / (2 * a), EPS, 1 - EPS)
    r = mean * p / (1 - p)

    if shrinkage:
        mu = r.mean(dim=0, keepdim=True)
        tau2 = r.var(dim=0, keepdim=True)
        tau2 = torch.clamp(tau2, min=EPS)
        sigma2 = var / (counts[:, None] + EPS)
        w = tau2 / (tau2 + sigma2)
        r = mu + w * (r - mu)
        r = torch.clamp(r, min=EPS)
        p = torch.clamp(r / (r + mean), EPS, 1 - EPS)

    col_sums = sums.clone()
    col_sums[col_sums == 0] = 1.0
    R = r.T

    P_weighted = torch.sum(p * r, dim=0) / (torch.sum(r, dim=0) + EPS)
    P_weighted = torch.clamp(P_weighted, EPS, 1 - EPS)
    logits = torch.log(P_weighted / (1 - P_weighted))

    return R.cpu().numpy(), logits.cpu().numpy()
