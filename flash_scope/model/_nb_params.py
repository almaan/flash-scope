from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

from flash_scope._utils import to_dense_array

EPS = 1e-8


def estimate_nb_params(
    adata: ad.AnnData,
    label_col: str,
    backend: str = "numpy",
    return_labels: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    X = to_dense_array(adata.X)
    labels = adata.obs[label_col].values.astype(str)
    uni_labels = np.unique(labels)

    if backend == "numpy":
        R, P_arr = _estimate_numpy(X, labels, uni_labels)
    elif backend == "torch":
        R, P_arr = _estimate_torch(X, labels, uni_labels)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'numpy' or 'torch'.")

    R_df = pd.DataFrame(R, index=adata.var_names, columns=uni_labels)
    P_df = pd.DataFrame(P_arr, index=adata.var_names)

    if return_labels:
        return R_df, P_df, uni_labels
    return R_df, P_df


def _estimate_numpy(
    X: np.ndarray,
    labels: np.ndarray,
    uni_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_cells, n_genes = X.shape
    n_types = len(uni_labels)

    label_idx = np.zeros(n_cells, dtype=np.intp)
    for k, lab in enumerate(uni_labels):
        label_idx[labels == lab] = k

    counts = np.bincount(label_idx, minlength=n_types).astype(np.float64)

    sums = np.zeros((n_types, n_genes), dtype=np.float64)
    np.add.at(sums, label_idx, X)

    sq_sums = np.zeros((n_types, n_genes), dtype=np.float64)
    np.add.at(sq_sums, label_idx, X.astype(np.float64) ** 2)

    mean = sums / counts[:, None] + EPS
    var = (sq_sums / counts[:, None] - (sums / counts[:, None]) ** 2)
    var = var * counts[:, None] / (counts[:, None] - 1 + EPS) + EPS

    a = var
    b = mean
    c = -mean

    p = np.clip((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a), EPS, 1 - EPS)
    r = mean * p / (1 - p)

    col_sums = sums.copy()
    col_sums[col_sums == 0] = 1.0
    R = (r / col_sums).T

    P_weighted = np.sum(p * r, axis=0) / (np.sum(r, axis=0) + EPS)
    P_weighted = P_weighted.astype(np.float64)

    return R, P_weighted


def _estimate_torch(
    X: np.ndarray,
    labels: np.ndarray,
    uni_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_cells, n_genes = X.shape
    n_types = len(uni_labels)

    X_t = torch.tensor(X, dtype=torch.float64, device=device)

    label_idx = torch.zeros(n_cells, dtype=torch.long, device=device)
    for k, lab in enumerate(uni_labels):
        label_idx[labels == lab] = k

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

    col_sums = sums.clone()
    col_sums[col_sums == 0] = 1.0
    R = (r / col_sums).T

    P_weighted = torch.sum(p * r, dim=0) / (torch.sum(r, dim=0) + EPS)

    return R.cpu().numpy(), P_weighted.cpu().numpy()
