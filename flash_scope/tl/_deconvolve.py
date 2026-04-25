from __future__ import annotations

import anndata as ad
import pandas as pd

from flash_scope.pp import preprocess
from flash_scope.model import estimate_nb_params, FlashScopeModel, fit, nnls_init, coarse_init
from flash_scope._utils import to_dense_array


def deconvolve(
    ad_sc: ad.AnnData,
    ad_sp: ad.AnnData | list[ad.AnnData],
    label_col: str,
    batch_size: int = 1024,
    epochs: int = 5000,
    lr: float = 0.01,
    device: str = "auto",
    gene_list: list[str] | None = None,
    min_label_member: int = 20,
    min_sc_counts: int = 100,
    min_sp_counts: int = 100,
    use_compile: bool = True,
    warm_start: bool = True,
    warm_start_damping: float = 0.5,
    winsorize_pct: float = 0.0,
    shrinkage: bool = False,
    coarse_fit: bool = False,
    n_clusters: int = 100,
    coarse_epochs: int = 100,
    grad_clip: float | None = None,
    tol: float = 1e-4,
    patience: int = 50,
    l1_w: float = 0.0,
    verbose: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """End-to-end spatial transcriptomics deconvolution.

    Preprocesses data, estimates NB parameters from the reference, fits a
    negative binomial mixture model, and returns per-spot cell type proportions.

    Parameters
    ----------
    ad_sc : AnnData
        Single-cell reference with cell type labels in ``obs[label_col]``.
    ad_sp : AnnData or list[AnnData]
        Spatial data. If a list, datasets are concatenated (outer join).
    label_col : str
        Column in ``ad_sc.obs`` containing cell type labels.
    batch_size : int
        Mini-batch size for model training.
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate for Adam.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    gene_list : list[str] or None
        Restrict analysis to these genes.
    min_label_member : int
        Drop cell types with fewer than this many cells.
    min_sc_counts : int
        Min total counts to keep a gene in the reference.
    min_sp_counts : int
        Min total counts to keep a gene in the spatial data.
    use_compile : bool
        Apply ``torch.compile`` on CUDA.
    warm_start : bool
        Initialize mixing weights via damped NNLS.
    warm_start_damping : float
        Tikhonov damping for NNLS warm-start.
    winsorize_pct : float
        Percentile for winsorizing expression before NB param estimation.
    shrinkage : bool
        Apply empirical Bayes shrinkage to NB dispersion estimates.
    coarse_fit : bool
        Initialize via coarse-to-fine dual fit (clusters spots first).
    n_clusters : int
        Number of clusters for coarse fit.
    coarse_epochs : int
        Epochs for the coarse model.
    grad_clip : float or None
        Max gradient norm. ``None`` disables clipping.
    tol : float
        Relative improvement threshold for early stopping.
    patience : int
        Epochs without improvement before early stopping. 0 disables.
    l1_w : float
        L1 penalty on mixing weights (encourages sparsity). 0 disables.
    verbose : bool
        Print progress messages.

    Returns
    -------
    DataFrame
        Cell type proportions, shape ``(n_spots, n_types)``, indexed by
        spot names with cell type columns.
    """
    if isinstance(ad_sp, list):
        _ad_sp = ad.concat(ad_sp, join="outer")
    else:
        _ad_sp = ad_sp.copy()

    obs_names = _ad_sp.obs_names.copy()

    if verbose:
        print(f"[flash-scope] preprocessing ({_ad_sp.n_obs} spots, {ad_sc.n_obs} cells)")

    _ad_sc, _ad_sp = preprocess(
        ad_sc.copy(), _ad_sp,
        label_col=label_col,
        gene_list=gene_list,
        min_label_member=min_label_member,
        min_sc_counts=min_sc_counts,
        min_sp_counts=min_sp_counts,
    )

    if verbose:
        print(f"[flash-scope] after filtering: {_ad_sp.n_vars} genes, "
              f"{_ad_sc.obs[label_col].nunique()} cell types")

    R, P, labels = estimate_nb_params(
        _ad_sc, label_col=label_col, return_labels=True,
        winsorize_pct=winsorize_pct, shrinkage=shrinkage,
    )

    if verbose:
        print(f"[flash-scope] NB params estimated for {len(labels)} types x {R.shape[0]} genes")

    init_w = None
    if coarse_fit:
        init_w = coarse_init(
            to_dense_array(_ad_sp.X), R.values, P.values,
            n_clusters=n_clusters, coarse_epochs=coarse_epochs,
            device=device, verbose=verbose,
        )
    elif warm_start:
        if verbose:
            print("[flash-scope] computing NNLS warm-start for mixing weights")
        init_w = nnls_init(to_dense_array(_ad_sp.X), R.values, damping=warm_start_damping)

    model = FlashScopeModel(
        r=R.values,
        logits=P.values,
        n_spots=_ad_sp.n_obs,
        n_types=len(labels),
        n_genes=_ad_sp.n_vars,
        init_w=init_w,
    )

    model = fit(
        model, _ad_sp,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        use_compile=use_compile,
        verbose=verbose,
        grad_clip=grad_clip,
        tol=tol,
        patience=patience,
        l1_w=l1_w,
    )

    props = model.get_proportions()
    return pd.DataFrame(props, index=obs_names, columns=labels)
