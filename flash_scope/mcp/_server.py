from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pandas as pd
from anndata import AnnData

from fastmcp import FastMCP

import flash_scope.io as fsio
from flash_scope.pp import preprocess
from flash_scope.model import estimate_nb_params, FlashScopeModel, fit as fit_model


@dataclass
class ServerState:
    ad_sc: AnnData | None = None
    ad_sp: AnnData | None = None
    model: FlashScopeModel | None = None
    proportions: pd.DataFrame | None = None
    labels: list[str] | None = None


_IO_DISPATCH = {
    "h5ad": fsio.read_h5ad,
    "parquet": fsio.read_parquet,
    "csv": fsio.read_csv,
}


def _load_reference(state: ServerState, path: str, format: str = "h5ad") -> str:
    reader = _IO_DISPATCH.get(format)
    if reader is None:
        return f"Unknown format: {format}. Supported: {list(_IO_DISPATCH)}"
    state.ad_sc = reader(path)
    n_obs, n_vars = state.ad_sc.shape
    cols = list(state.ad_sc.obs.columns)
    return f"Loaded reference: {n_obs} cells, {n_vars} genes. Columns: {cols}"


def _load_spatial(state: ServerState, path: str, format: str = "h5ad") -> str:
    reader = _IO_DISPATCH.get(format)
    if reader is None:
        return f"Unknown format: {format}. Supported: {list(_IO_DISPATCH)}"
    state.ad_sp = reader(path)
    n_obs, n_vars = state.ad_sp.shape
    cols = list(state.ad_sp.obs.columns)
    return f"Loaded spatial: {n_obs} spots, {n_vars} genes. Columns: {cols}"


def _preprocess(
    state: ServerState,
    label_col: str,
    min_label_member: int = 20,
    min_sc_counts: int = 100,
    min_sp_counts: int = 100,
) -> str:
    if state.ad_sc is None or state.ad_sp is None:
        return "Error: load reference and spatial data first."
    state.ad_sc, state.ad_sp = preprocess(
        state.ad_sc, state.ad_sp,
        label_col=label_col,
        min_label_member=min_label_member,
        min_sc_counts=min_sc_counts,
        min_sp_counts=min_sp_counts,
    )
    return (f"Preprocessed: {state.ad_sc.n_obs} cells, {state.ad_sp.n_obs} spots, "
            f"{state.ad_sc.n_vars} genes remaining.")


def _fit(
    state: ServerState,
    label_col: str,
    epochs: int = 500,
    batch_size: int = 1024,
) -> str:
    if state.ad_sc is None or state.ad_sp is None:
        return "Error: preprocess data first."

    R, P, labels = estimate_nb_params(state.ad_sc, label_col=label_col, return_labels=True)
    state.labels = list(labels)

    model = FlashScopeModel(
        r=R.values, p=P.values,
        n_spots=state.ad_sp.n_obs,
        n_types=len(labels),
        n_genes=state.ad_sp.n_vars,
    )

    t0 = time.time()
    state.model = fit_model(model, state.ad_sp, epochs=epochs, batch_size=batch_size)
    elapsed = time.time() - t0

    props = state.model.get_proportions()
    state.proportions = pd.DataFrame(props, index=state.ad_sp.obs_names, columns=labels)

    return f"Fit complete in {elapsed:.1f}s. {len(labels)} cell types, {state.ad_sp.n_obs} spots."


def _get_proportions(state: ServerState, top_n: int = 5) -> str:
    if state.proportions is None:
        return "Error: run fit first."

    results = []
    for spot_id, row in state.proportions.iterrows():
        top = row.nlargest(top_n)
        results.append({
            "spot": str(spot_id),
            "proportions": {k: round(float(v), 4) for k, v in top.items()},
        })
    return json.dumps(results)


def create_server() -> FastMCP:
    mcp = FastMCP("flash-scope")
    state = ServerState()

    @mcp.tool()
    def load_reference(path: str, format: str = "h5ad") -> str:
        """Load reference scRNA-seq data. Formats: h5ad, parquet, csv."""
        return _load_reference(state, path, format)

    @mcp.tool()
    def load_spatial(path: str, format: str = "h5ad") -> str:
        """Load spatial transcriptomics data. Formats: h5ad, parquet, csv."""
        return _load_spatial(state, path, format)

    @mcp.tool()
    def preprocess_data(
        label_col: str,
        min_label_member: int = 20,
        min_sc_counts: int = 100,
        min_sp_counts: int = 100,
    ) -> str:
        """Preprocess loaded reference and spatial data. Filters genes and intersects variables."""
        return _preprocess(state, label_col, min_label_member, min_sc_counts, min_sp_counts)

    @mcp.tool()
    def fit_model_tool(label_col: str, epochs: int = 500, batch_size: int = 1024) -> str:
        """Run NB deconvolution on preprocessed data. Returns training summary."""
        return _fit(state, label_col, epochs, batch_size)

    @mcp.tool()
    def get_proportions(top_n: int = 5) -> str:
        """Get cell type proportions per spot. Returns JSON with top_n types per spot."""
        return _get_proportions(state, top_n)

    return mcp


def serve():
    mcp = create_server()
    mcp.run()
