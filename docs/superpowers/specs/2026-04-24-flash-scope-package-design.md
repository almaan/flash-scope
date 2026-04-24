# Flash-Scope Package Design Spec

## Overview

Convert monolithic spatial transcriptomics deconvolution script into a modular Python package called `flash-scope`. Uses negative binomial mixture model to estimate cell type proportions from spatial transcriptomics data given a single-cell RNA-seq reference.

## Decisions

- **Core data structure**: AnnData (no custom wrapper)
- **Trainer**: Pure PyTorch (no Lightning dependency)
- **NB estimation**: NumPy default + optional PyTorch GPU backend
- **Packaging**: pyproject.toml + pip, optional extras `[mcp]`, `[cuda]`
- **MCP**: FastMCP server exposing LLM agent tools
- **API style**: Scanpy-like namespaces (`fs.io`, `fs.pp`, `fs.tl`, `fs.model`)

## Package Structure

```
flash_scope/
├── __init__.py            # top-level: import io, pp, tl, model; expose __version__
├── io/
│   ├── __init__.py        # re-export read_h5ad, read_parquet, read_csv
│   ├── _anndata.py        # read_h5ad(path) → AnnData
│   ├── _parquet.py        # read_parquet(path, gene_columns, obs_columns) → AnnData
│   └── _csv.py            # read_csv(path, gene_columns, obs_columns, **kwargs) → AnnData
├── pp/
│   ├── __init__.py        # re-export preprocessing functions
│   └── _preprocess.py     # filter_by_label, filter_genes, intersect_vars, densify, preprocess
├── model/
│   ├── __init__.py        # re-export FlashScopeModel, estimate_nb_params, fit
│   ├── _nb_params.py      # estimate_nb_params(adata, label_col, backend="numpy")
│   ├── _deconv.py         # FlashScopeModel(nn.Module)
│   └── _trainer.py        # fit(model, adata, epochs, batch_size, lr, device, compile)
├── tl/
│   ├── __init__.py        # re-export deconvolve
│   └── _deconvolve.py     # deconvolve(ad_sc, ad_sp, label_col, ...) → DataFrame
├── mcp/
│   ├── __init__.py        # re-export serve
│   ├── __main__.py        # python -m flash_scope.mcp entry point
│   └── _server.py         # FastMCP server: load_reference, load_spatial, preprocess, fit, get_proportions
└── _utils.py              # resolve_device, to_dense_array
```

## Module Specifications

### `fs.io` — I/O Adapters

All adapters produce AnnData. Thin wrappers that read source format and construct AnnData.

#### `read_h5ad(path: str | Path) → AnnData`
Thin wrapper around `anndata.read_h5ad`. Exists for API consistency.

#### `read_parquet(path: str | Path, gene_columns: list[str] | None = None, obs_columns: list[str] | None = None) → AnnData`
Read parquet via pandas. `gene_columns` specifies expression columns (default: all numeric). `obs_columns` specifies metadata columns. Constructs AnnData from split DataFrame.

#### `read_csv(path: str | Path, gene_columns: list[str] | None = None, obs_columns: list[str] | None = None, **kwargs) → AnnData`
Same logic as parquet, reads CSV. Pass-through `**kwargs` to `pd.read_csv`.

### `fs.pp` — Preprocessing

All functions return copies, never mutate input.

#### `preprocess(ad_sc, ad_sp, label_col, gene_list=None, min_label_member=20, min_sc_counts=100, min_sp_counts=100) → tuple[AnnData, AnnData]`
Orchestrator. Calls filter_by_label → filter_genes (both) → intersect_vars → densify (both). Returns filtered, intersected, dense copies.

#### `filter_by_label(adata, label_col, min_count=20) → AnnData`
Remove observations whose label has fewer than `min_count` members.

#### `filter_genes(adata, min_counts=100) → AnnData`
Filter genes by minimum total counts. Uses scanpy's `sc.pp.filter_genes`.

#### `intersect_vars(ad_sc, ad_sp, gene_list=None) → tuple[AnnData, AnnData]`
Intersect var_names between datasets. Optional gene_list further restricts intersection.

#### `densify(adata) → AnnData`
Ensure `.X` is dense ndarray (handles sparse matrices).

### `fs.model` — Model & Training

#### `estimate_nb_params(adata, label_col, backend="numpy", return_labels=False) → tuple[DataFrame, DataFrame] | tuple[DataFrame, DataFrame, ndarray]`

Estimate negative binomial parameters (R, P) from reference scRNA-seq data.

**NumPy backend** (default):
- Vectorized: no Python loop over cell types.
- Use advanced indexing to compute grouped mean/variance across all types simultaneously.
- Build label index array, compute per-group statistics via `np.add.at` or equivalent broadcasting.

**PyTorch backend** (`backend="torch"`):
- Same math, tensor operations on GPU.
- For large references (100k+ cells, many types).

**Math** (per cell type k, per gene g):
```
mean_gk = E[X_gk]
var_gk = Var[X_gk]
p_gk = (-var + sqrt(var^2 + 4*var*mean)) / (2*var)   # clipped to [1e-8, 1-1e-8]
r_gk = mean * p / (1 - p)
R_gk = r_gk / sum(X_gk)   # normalized by total counts
P_g = weighted average of p across types (weighted by R)
```

Returns:
- R: DataFrame (genes × types) — normalized r parameters
- P: DataFrame (genes,) — gene-level p parameter
- labels: ndarray (optional) — unique label names

#### `FlashScopeModel(nn.Module)`

Pure PyTorch module. Architecture:

**Parameters (learnable):**
- `w`: (n_spots, n_types) — initialized to 1/n_types. Proportion logits.
- `nu`: (1, n_genes) — initialized to ones. Gene-level scaling.
- `eta`: (1, n_genes) — initialized to zeros. Background expression.
- `alpha`: (1,) — initialized to ones. Background scaling.

**Buffers (fixed):**
- `r_sc`: (n_types, n_genes) — from estimate_nb_params
- `p_sc`: (1, n_genes) — gene-level p (weighted average across types, broadcast in forward)

**Forward pass:**
```
beta = softplus(nu)                     # 1 × G
v = softplus(w[idx])                    # batch × K
eps = softplus(eta)                     # 1 × G
gamma = softplus(alpha)                 # scalar
r_sp = beta * (v @ r_sc) + gamma * eps  # batch × G
p_sp = p_sc                             # 1 × G (broadcast to batch)
```

**Loss:** Negative log-likelihood of NB distribution: `-NB(total_count=r, probs=p).log_prob(x).sum()`

**`get_proportions() → ndarray`**: Softplus(w) normalized per spot. Returns (n_spots, n_types).

#### `fit(model, adata, epochs=500, batch_size=1024, lr=1e-3, device="auto", use_compile=True) → FlashScopeModel`

Pure PyTorch training loop:

1. Build Dataset/DataLoader from adata (pin_memory=True if GPU).
2. `device = resolve_device(device)`.
3. `torch.compile(model)` if `use_compile=True` and device is CUDA.
4. `torch.amp.autocast(device_type)` for mixed precision on GPU.
5. Adam optimizer, lr=1e-3.
6. Training loop: forward → loss → backward → step.
7. Return fitted model (moved to CPU).

### `fs.tl` — Tools

#### `deconvolve(ad_sc, ad_sp, label_col, batch_size=1024, epochs=500, lr=1e-3, device="auto", gene_list=None, min_label_member=20, min_sc_counts=100, min_sp_counts=100, **kwargs) → DataFrame`

High-level orchestrator:
1. If `ad_sp` is list, concatenate with `ad.concat(join="outer")`.
2. `pp.preprocess(ad_sc, ad_sp, label_col, ...)`.
3. `model.estimate_nb_params(ad_sc, label_col, return_labels=True)`.
4. Construct `FlashScopeModel`.
5. `model.fit(model, ad_sp, ...)`.
6. `model.get_proportions()`.
7. Return DataFrame (spots × cell types), indexed by `ad_sp.obs_names`, columns = labels.

### `fs.mcp` — MCP Server

FastMCP server for LLM agent interaction.

**State**: `ServerState` dataclass holding:
- `ad_sc: AnnData | None`
- `ad_sp: AnnData | None`
- `model: FlashScopeModel | None`
- `proportions: DataFrame | None`

**Tools:**

1. **`load_reference(path: str, format: str = "h5ad") → str`**
   Load reference scRNA-seq data. Dispatches to `fs.io` based on format.
   Returns summary string (n_obs, n_vars, available columns).

2. **`load_spatial(path: str, format: str = "h5ad") → str`**
   Load spatial data. Same dispatch pattern.
   Returns summary string.

3. **`preprocess(label_col: str, min_label_member: int = 20, min_sc_counts: int = 100, min_sp_counts: int = 100) → str`**
   Preprocess loaded data. Stores result in state.
   Returns summary of filtering (genes remaining, cells remaining).

4. **`fit(label_col: str, epochs: int = 500, batch_size: int = 1024) → str`**
   Run deconvolution on preprocessed data.
   Returns training summary (final loss, time elapsed).

5. **`get_proportions(top_n: int = 5) → str`**
   Get cell type proportions from fitted model.
   Returns JSON: per-spot top_n cell types with proportions.

**Entry points:**
- `python -m flash_scope.mcp` — starts server via `__main__.py`
- `fs.mcp.serve()` — programmatic start

### `_utils.py`

#### `resolve_device(device: str = "auto") → torch.device`
Returns CUDA device if available and requested, else CPU.

#### `to_dense_array(X) → ndarray`
Convert sparse matrix or tensor to dense ndarray.

## Performance Optimizations

1. **`torch.compile()`**: Applied to model by default on CUDA. JIT-fuses operations.
2. **Mixed precision**: `torch.amp.autocast` for float16 on GPU. ~2x speedup.
3. **Vectorized NB params**: No Python loop over cell types. Grouped computation via advanced indexing/broadcasting.
4. **DataLoader tuning**: `pin_memory=True`, `num_workers=0` (data fits in memory), persistent workers off.
5. **Contiguous tensors**: Ensure `.contiguous()` before matmul operations.
6. **Optional GPU NB estimation**: `backend="torch"` for large references.

## Packaging

**pyproject.toml**:
```toml
[project]
name = "flash-scope"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "anndata>=0.10",
    "numpy>=1.24",
    "pandas>=2.0",
    "torch>=2.0",
    "scanpy>=1.9",
    "scipy>=1.10",
]

[project.optional-dependencies]
mcp = ["fastmcp>=0.1"]
dev = ["pytest>=7.0", "pytest-cov"]
```

## Testing Strategy

- **Unit tests per module**: io, pp, model, tl, mcp.
- **Synthetic fixtures**: 10 genes, 5 cell types, 50 spots. Deterministic seed.
- **Integration test**: Full pipeline read → deconvolve → proportions sum ≈ 1 per spot.
- **MCP tests**: Tool call → response shape/content validation.
- **Performance regression**: Training 100 epochs on synthetic data completes in <5s.

## API Usage Examples

```python
import flash_scope as fs

# Quick one-liner
props = fs.tl.deconvolve(ad_sc, ad_sp, label_col="cell_type")

# Step-by-step with control
ad_sc, ad_sp = fs.pp.preprocess(ad_sc, ad_sp, label_col="cell_type")
R, P, labels = fs.model.estimate_nb_params(ad_sc, label_col="cell_type")
model = fs.model.FlashScopeModel(R.values, P.values, n_spots=ad_sp.n_obs, n_types=len(labels), n_genes=ad_sp.n_vars)
model = fs.model.fit(model, ad_sp, epochs=500, device="auto")
props = model.get_proportions()

# From parquet
ad_sp = fs.io.read_parquet("spatial.parquet", obs_columns=["x", "y", "batch"])

# MCP server
fs.mcp.serve()
```
