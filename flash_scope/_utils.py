from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def to_dense_array(X) -> np.ndarray:
    if isinstance(X, np.ndarray):
        return X
    if sp.issparse(X):
        return X.toarray()
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    return np.asarray(X)
