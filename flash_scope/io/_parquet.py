from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def read_parquet(
    path: str | Path,
    gene_columns: list[str] | None = None,
    obs_columns: list[str] | None = None,
) -> ad.AnnData:
    df = pd.read_parquet(Path(path))
    return _df_to_anndata(df, gene_columns=gene_columns, obs_columns=obs_columns)


def _df_to_anndata(
    df: pd.DataFrame,
    gene_columns: list[str] | None = None,
    obs_columns: list[str] | None = None,
) -> ad.AnnData:
    if obs_columns is None:
        obs_columns = []

    obs_df = df[obs_columns].reset_index(drop=True) if obs_columns else pd.DataFrame(index=range(len(df)))

    if gene_columns is not None:
        expr_cols = gene_columns
    else:
        remaining = [c for c in df.columns if c not in obs_columns]
        expr_cols = [c for c in remaining if pd.api.types.is_numeric_dtype(df[c])]

    X = df[expr_cols].values.astype(np.float32)
    var_df = pd.DataFrame(index=expr_cols)
    obs_df.index = [str(i) for i in range(len(obs_df))]

    return ad.AnnData(X=X, obs=obs_df, var=var_df)
