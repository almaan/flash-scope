from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc

from flash_scope._utils import to_dense_array


def filter_by_label(adata: ad.AnnData, label_col: str, min_count: int = 20) -> ad.AnnData:
    vc = adata.obs[label_col].value_counts()
    keep = vc.index[vc >= min_count].values
    mask = np.isin(adata.obs[label_col].values, keep)
    return adata[mask].copy()


def filter_genes(adata: ad.AnnData, min_counts: int = 100) -> ad.AnnData:
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_counts=min_counts)
    return adata


def intersect_vars(
    ad_sc: ad.AnnData,
    ad_sp: ad.AnnData,
    gene_list: list[str] | None = None,
) -> tuple[ad.AnnData, ad.AnnData]:
    inter = ad_sc.var_names.intersection(ad_sp.var_names)
    if gene_list is not None:
        inter = pd.Index(gene_list).intersection(inter)
    return ad_sc[:, inter].copy(), ad_sp[:, inter].copy()


def densify(adata: ad.AnnData) -> ad.AnnData:
    adata = adata.copy()
    adata.X = to_dense_array(adata.X)
    return adata


def preprocess(
    ad_sc: ad.AnnData,
    ad_sp: ad.AnnData,
    label_col: str,
    gene_list: list[str] | None = None,
    min_label_member: int = 20,
    min_sc_counts: int = 100,
    min_sp_counts: int = 100,
) -> tuple[ad.AnnData, ad.AnnData]:
    ad_sc = ad_sc.copy()
    ad_sp = ad_sp.copy()
    ad_sc.var_names_make_unique()
    ad_sp.var_names_make_unique()
    ad_sc = filter_by_label(ad_sc, label_col, min_count=min_label_member)
    ad_sc = filter_genes(ad_sc, min_counts=min_sc_counts)
    ad_sp = filter_genes(ad_sp, min_counts=min_sp_counts)
    ad_sc, ad_sp = intersect_vars(ad_sc, ad_sp, gene_list=gene_list)
    ad_sc = densify(ad_sc)
    ad_sp = densify(ad_sp)
    return ad_sc, ad_sp
