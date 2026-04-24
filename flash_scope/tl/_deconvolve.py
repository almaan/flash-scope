from __future__ import annotations

import anndata as ad
import pandas as pd

from flash_scope.pp import preprocess
from flash_scope.model import estimate_nb_params, FlashScopeModel, fit


def deconvolve(
    ad_sc: ad.AnnData,
    ad_sp: ad.AnnData | list[ad.AnnData],
    label_col: str,
    batch_size: int = 1024,
    epochs: int = 500,
    lr: float = 1e-3,
    device: str = "auto",
    gene_list: list[str] | None = None,
    min_label_member: int = 20,
    min_sc_counts: int = 100,
    min_sp_counts: int = 100,
    use_compile: bool = True,
    **kwargs,
) -> pd.DataFrame:
    if isinstance(ad_sp, list):
        _ad_sp = ad.concat(ad_sp, join="outer")
    else:
        _ad_sp = ad_sp.copy()

    obs_names = _ad_sp.obs_names.copy()

    _ad_sc, _ad_sp = preprocess(
        ad_sc.copy(), _ad_sp,
        label_col=label_col,
        gene_list=gene_list,
        min_label_member=min_label_member,
        min_sc_counts=min_sc_counts,
        min_sp_counts=min_sp_counts,
    )

    R, P, labels = estimate_nb_params(_ad_sc, label_col=label_col, return_labels=True)

    model = FlashScopeModel(
        r=R.values,
        p=P.values,
        n_spots=_ad_sp.n_obs,
        n_types=len(labels),
        n_genes=_ad_sp.n_vars,
    )

    model = fit(
        model, _ad_sp,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        use_compile=use_compile,
    )

    props = model.get_proportions()
    return pd.DataFrame(props, index=obs_names, columns=labels)
