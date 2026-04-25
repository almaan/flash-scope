from __future__ import annotations

from pathlib import Path

import anndata as ad


def read_h5ad(path: str | Path) -> ad.AnnData:
    """Read an h5ad file and return an AnnData object."""
    return ad.read_h5ad(Path(path))
