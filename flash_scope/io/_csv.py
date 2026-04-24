from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd

from flash_scope.io._parquet import _df_to_anndata


def read_csv(
    path: str | Path,
    gene_columns: list[str] | None = None,
    obs_columns: list[str] | None = None,
    **kwargs,
) -> ad.AnnData:
    df = pd.read_csv(Path(path), **kwargs)
    return _df_to_anndata(df, gene_columns=gene_columns, obs_columns=obs_columns)
