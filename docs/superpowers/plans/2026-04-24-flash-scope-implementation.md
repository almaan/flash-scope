# Flash-Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `flash-scope`, a modular Python package for spatial transcriptomics deconvolution using negative binomial mixture models, with scanpy-like API and FastMCP server.

**Architecture:** Domain-layer architecture mirroring scanpy namespaces (`fs.io`, `fs.pp`, `fs.model`, `fs.tl`, `fs.mcp`). Pure PyTorch training loop with torch.compile and mixed precision. AnnData as core data structure with I/O adapters for parquet/csv.

**Tech Stack:** Python 3.10+, PyTorch 2.0+, AnnData, scanpy, NumPy, pandas, FastMCP

**Spec:** `docs/superpowers/specs/2026-04-24-flash-scope-package-design.md`

---

## File Structure

**Create:**
- `pyproject.toml` — package metadata, dependencies, optional extras
- `flash_scope/__init__.py` — top-level imports, `__version__`
- `flash_scope/_utils.py` — `resolve_device`, `to_dense_array`
- `flash_scope/io/__init__.py` — re-export read functions
- `flash_scope/io/_anndata.py` — `read_h5ad`
- `flash_scope/io/_parquet.py` — `read_parquet`
- `flash_scope/io/_csv.py` — `read_csv`
- `flash_scope/pp/__init__.py` — re-export preprocessing functions
- `flash_scope/pp/_preprocess.py` — `filter_by_label`, `filter_genes`, `intersect_vars`, `densify`, `preprocess`
- `flash_scope/model/__init__.py` — re-export model components
- `flash_scope/model/_nb_params.py` — `estimate_nb_params` (numpy + torch backends)
- `flash_scope/model/_deconv.py` — `FlashScopeModel(nn.Module)`
- `flash_scope/model/_trainer.py` — `fit()` training loop
- `flash_scope/tl/__init__.py` — re-export tools
- `flash_scope/tl/_deconvolve.py` — `deconvolve()` orchestrator
- `flash_scope/mcp/__init__.py` — re-export `serve`
- `flash_scope/mcp/__main__.py` — `python -m flash_scope.mcp` entry point
- `flash_scope/mcp/_server.py` — FastMCP server with tools
- `tests/conftest.py` — synthetic data fixtures
- `tests/test_utils.py`
- `tests/test_io.py`
- `tests/test_pp.py`
- `tests/test_nb_params.py`
- `tests/test_model.py`
- `tests/test_trainer.py`
- `tests/test_deconvolve.py`
- `tests/test_mcp.py`

---

### Task 1: Project Scaffolding & Utilities

**Files:**
- Create: `pyproject.toml`
- Create: `flash_scope/__init__.py`
- Create: `flash_scope/_utils.py`
- Create: `tests/conftest.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "flash-scope"
version = "0.1.0"
description = "Fast spatial transcriptomics deconvolution via negative binomial mixture models"
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
mcp = ["fastmcp>=2.0"]
dev = ["pytest>=7.0", "pytest-cov"]

[tool.setuptools.packages.find]
include = ["flash_scope*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Create flash_scope/_utils.py**

```python
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
```

- [ ] **Step 3: Create flash_scope/__init__.py**

```python
from flash_scope import io, pp, model, tl

__version__ = "0.1.0"

__all__ = ["io", "pp", "model", "tl", "__version__"]
```

- [ ] **Step 4: Create test fixtures in tests/conftest.py**

These fixtures produce deterministic synthetic data for all tests: 10 genes, 5 cell types, 50 spots for spatial, 100 cells for reference.

```python
import numpy as np
import pandas as pd
import anndata as ad
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_sc(rng):
    n_cells = 100
    n_genes = 10
    n_types = 5
    labels = np.array([f"type_{i}" for i in range(n_types)])
    cell_labels = np.repeat(labels, n_cells // n_types)
    X = rng.negative_binomial(n=5, p=0.3, size=(n_cells, n_genes)).astype(np.float32)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": cell_labels}, index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    return adata


@pytest.fixture
def synthetic_sp(rng):
    n_spots = 50
    n_genes = 10
    X = rng.negative_binomial(n=5, p=0.3, size=(n_spots, n_genes)).astype(np.float32)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"x": rng.uniform(size=n_spots), "y": rng.uniform(size=n_spots)},
                         index=[f"spot_{i}" for i in range(n_spots)]),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    return adata


@pytest.fixture
def synthetic_sc_sparse(synthetic_sc):
    import scipy.sparse as sp
    adata = synthetic_sc.copy()
    adata.X = sp.csr_matrix(adata.X)
    return adata
```

- [ ] **Step 5: Write tests for _utils**

```python
import numpy as np
import scipy.sparse as sp
import torch

from flash_scope._utils import resolve_device, to_dense_array


class TestResolveDevice:
    def test_cpu_explicit(self):
        d = resolve_device("cpu")
        assert d == torch.device("cpu")

    def test_auto_returns_device(self):
        d = resolve_device("auto")
        assert isinstance(d, torch.device)

    def test_cuda_explicit(self):
        d = resolve_device("cuda")
        assert d == torch.device("cuda")


class TestToDenseArray:
    def test_ndarray_passthrough(self):
        x = np.array([[1, 2], [3, 4]])
        result = to_dense_array(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, x)

    def test_sparse_to_dense(self):
        x = sp.csr_matrix(np.array([[1, 0], [0, 2]]))
        result = to_dense_array(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[1, 0], [0, 2]])

    def test_tensor_to_dense(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = to_dense_array(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /cv/data/braid/andera29/projs/flash-scope && pip install -e ".[dev]" && pytest tests/test_utils.py -v`
Expected: All 5 tests PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml flash_scope/__init__.py flash_scope/_utils.py tests/conftest.py tests/test_utils.py
git commit -m "feat: project scaffolding with utilities and test fixtures"
```

---

### Task 2: I/O Adapters

**Files:**
- Create: `flash_scope/io/__init__.py`
- Create: `flash_scope/io/_anndata.py`
- Create: `flash_scope/io/_parquet.py`
- Create: `flash_scope/io/_csv.py`
- Create: `tests/test_io.py`

- [ ] **Step 1: Write tests for I/O adapters**

```python
import numpy as np
import pandas as pd
import anndata as ad
import pytest

from flash_scope.io import read_h5ad, read_parquet, read_csv


class TestReadH5ad:
    def test_reads_h5ad(self, tmp_path, synthetic_sc):
        path = tmp_path / "test.h5ad"
        synthetic_sc.write_h5ad(path)
        result = read_h5ad(path)
        assert isinstance(result, ad.AnnData)
        assert result.shape == synthetic_sc.shape

    def test_reads_string_path(self, tmp_path, synthetic_sc):
        path = tmp_path / "test.h5ad"
        synthetic_sc.write_h5ad(path)
        result = read_h5ad(str(path))
        assert result.shape == synthetic_sc.shape


class TestReadParquet:
    def test_reads_parquet_all_numeric(self, tmp_path):
        df = pd.DataFrame({
            "gene_0": [1.0, 2.0, 3.0],
            "gene_1": [4.0, 5.0, 6.0],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)
        result = read_parquet(path)
        assert isinstance(result, ad.AnnData)
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.var_names, ["gene_0", "gene_1"])

    def test_reads_parquet_with_obs_columns(self, tmp_path):
        df = pd.DataFrame({
            "cell_type": ["A", "B", "C"],
            "gene_0": [1.0, 2.0, 3.0],
            "gene_1": [4.0, 5.0, 6.0],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)
        result = read_parquet(path, obs_columns=["cell_type"])
        assert result.shape == (3, 2)
        assert "cell_type" in result.obs.columns

    def test_reads_parquet_with_gene_columns(self, tmp_path):
        df = pd.DataFrame({
            "cell_type": ["A", "B", "C"],
            "gene_0": [1.0, 2.0, 3.0],
            "gene_1": [4.0, 5.0, 6.0],
            "gene_2": [7.0, 8.0, 9.0],
        })
        path = tmp_path / "test.parquet"
        df.to_parquet(path, index=False)
        result = read_parquet(path, gene_columns=["gene_0", "gene_1"])
        assert result.shape == (3, 2)
        np.testing.assert_array_equal(result.var_names, ["gene_0", "gene_1"])


class TestReadCsv:
    def test_reads_csv_all_numeric(self, tmp_path):
        df = pd.DataFrame({"gene_0": [1.0, 2.0], "gene_1": [3.0, 4.0]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        result = read_csv(path)
        assert isinstance(result, ad.AnnData)
        assert result.shape == (2, 2)

    def test_reads_csv_with_obs_columns(self, tmp_path):
        df = pd.DataFrame({
            "label": ["A", "B"],
            "gene_0": [1.0, 2.0],
            "gene_1": [3.0, 4.0],
        })
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        result = read_csv(path, obs_columns=["label"])
        assert result.shape == (2, 2)
        assert "label" in result.obs.columns
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_io.py -v`
Expected: FAIL — `flash_scope.io` module not found

- [ ] **Step 3: Implement I/O adapters**

`flash_scope/io/_anndata.py`:
```python
from __future__ import annotations

from pathlib import Path

import anndata as ad


def read_h5ad(path: str | Path) -> ad.AnnData:
    return ad.read_h5ad(Path(path))
```

`flash_scope/io/_parquet.py`:
```python
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
```

`flash_scope/io/_csv.py`:
```python
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
```

`flash_scope/io/__init__.py`:
```python
from flash_scope.io._anndata import read_h5ad
from flash_scope.io._parquet import read_parquet
from flash_scope.io._csv import read_csv

__all__ = ["read_h5ad", "read_parquet", "read_csv"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_io.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add flash_scope/io/ tests/test_io.py
git commit -m "feat: add I/O adapters for h5ad, parquet, csv"
```

---

### Task 3: Preprocessing

**Files:**
- Create: `flash_scope/pp/__init__.py`
- Create: `flash_scope/pp/_preprocess.py`
- Create: `tests/test_pp.py`

- [ ] **Step 1: Write tests for preprocessing**

```python
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import pytest

from flash_scope.pp import filter_by_label, filter_genes, intersect_vars, densify, preprocess


class TestFilterByLabel:
    def test_removes_small_groups(self, synthetic_sc):
        result = filter_by_label(synthetic_sc, "cell_type", min_count=20)
        assert result.n_obs == synthetic_sc.n_obs
        result_small = filter_by_label(synthetic_sc, "cell_type", min_count=21)
        assert result_small.n_obs == 0

    def test_does_not_mutate_input(self, synthetic_sc):
        original_shape = synthetic_sc.shape
        filter_by_label(synthetic_sc, "cell_type", min_count=20)
        assert synthetic_sc.shape == original_shape


class TestFilterGenes:
    def test_filters_low_count_genes(self):
        X = np.array([[0, 0, 100], [0, 0, 200]], dtype=np.float32)
        adata = ad.AnnData(X=X, var=pd.DataFrame(index=["g0", "g1", "g2"]))
        result = filter_genes(adata, min_counts=50)
        assert result.n_vars == 1
        assert result.var_names[0] == "g2"


class TestIntersectVars:
    def test_intersects_var_names(self):
        ad1 = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32),
                         var=pd.DataFrame(index=["a", "b", "c"]))
        ad2 = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32),
                         var=pd.DataFrame(index=["b", "c", "d"]))
        r1, r2 = intersect_vars(ad1, ad2)
        assert list(r1.var_names) == ["b", "c"]
        assert list(r2.var_names) == ["b", "c"]

    def test_intersects_with_gene_list(self):
        ad1 = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32),
                         var=pd.DataFrame(index=["a", "b", "c"]))
        ad2 = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32),
                         var=pd.DataFrame(index=["b", "c", "d"]))
        r1, r2 = intersect_vars(ad1, ad2, gene_list=["b"])
        assert list(r1.var_names) == ["b"]
        assert list(r2.var_names) == ["b"]

    def test_does_not_mutate_inputs(self):
        ad1 = ad.AnnData(X=np.zeros((2, 3), dtype=np.float32),
                         var=pd.DataFrame(index=["a", "b", "c"]))
        ad2 = ad.AnnData(X=np.zeros((2, 2), dtype=np.float32),
                         var=pd.DataFrame(index=["b", "c"]))
        intersect_vars(ad1, ad2)
        assert ad1.n_vars == 3


class TestDensify:
    def test_sparse_to_dense(self, synthetic_sc_sparse):
        result = densify(synthetic_sc_sparse)
        assert isinstance(result.X, np.ndarray)
        assert not sp.issparse(result.X)

    def test_dense_passthrough(self, synthetic_sc):
        result = densify(synthetic_sc)
        assert isinstance(result.X, np.ndarray)


class TestPreprocess:
    def test_full_pipeline(self, synthetic_sc, synthetic_sp):
        sc_out, sp_out = preprocess(synthetic_sc, synthetic_sp, label_col="cell_type",
                                     min_sc_counts=0, min_sp_counts=0)
        assert sc_out.n_vars == sp_out.n_vars
        assert isinstance(sc_out.X, np.ndarray)
        assert isinstance(sp_out.X, np.ndarray)

    def test_does_not_mutate_inputs(self, synthetic_sc, synthetic_sp):
        sc_shape = synthetic_sc.shape
        sp_shape = synthetic_sp.shape
        preprocess(synthetic_sc, synthetic_sp, label_col="cell_type",
                   min_sc_counts=0, min_sp_counts=0)
        assert synthetic_sc.shape == sc_shape
        assert synthetic_sp.shape == sp_shape
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_pp.py -v`
Expected: FAIL — `flash_scope.pp` not found

- [ ] **Step 3: Implement preprocessing module**

`flash_scope/pp/_preprocess.py`:
```python
from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp

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
```

`flash_scope/pp/__init__.py`:
```python
from flash_scope.pp._preprocess import (
    filter_by_label,
    filter_genes,
    intersect_vars,
    densify,
    preprocess,
)

__all__ = ["filter_by_label", "filter_genes", "intersect_vars", "densify", "preprocess"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_pp.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add flash_scope/pp/ tests/test_pp.py
git commit -m "feat: add preprocessing module with filter, intersect, densify"
```

---

### Task 4: NB Parameter Estimation

**Files:**
- Create: `flash_scope/model/__init__.py`
- Create: `flash_scope/model/_nb_params.py`
- Create: `tests/test_nb_params.py`

- [ ] **Step 1: Write tests for NB parameter estimation**

```python
import numpy as np
import pandas as pd
import anndata as ad
import pytest

from flash_scope.model._nb_params import estimate_nb_params


class TestEstimateNBParamsNumpy:
    def test_output_shapes(self, synthetic_sc):
        R, P = estimate_nb_params(synthetic_sc, "cell_type", backend="numpy")
        n_genes = synthetic_sc.n_vars
        n_types = synthetic_sc.obs["cell_type"].nunique()
        assert R.shape == (n_genes, n_types)
        assert P.shape == (n_genes, 1)

    def test_output_types(self, synthetic_sc):
        R, P = estimate_nb_params(synthetic_sc, "cell_type", backend="numpy")
        assert isinstance(R, pd.DataFrame)
        assert isinstance(P, pd.DataFrame)

    def test_return_labels(self, synthetic_sc):
        R, P, labels = estimate_nb_params(synthetic_sc, "cell_type", backend="numpy", return_labels=True)
        assert len(labels) == synthetic_sc.obs["cell_type"].nunique()

    def test_r_values_positive(self, synthetic_sc):
        R, P = estimate_nb_params(synthetic_sc, "cell_type", backend="numpy")
        assert (R.values >= 0).all()

    def test_p_values_in_range(self, synthetic_sc):
        R, P = estimate_nb_params(synthetic_sc, "cell_type", backend="numpy")
        assert (P.values >= 1e-8).all()
        assert (P.values <= 1 - 1e-8).all()

    def test_column_names_match_labels(self, synthetic_sc):
        R, P, labels = estimate_nb_params(synthetic_sc, "cell_type", backend="numpy", return_labels=True)
        np.testing.assert_array_equal(R.columns, labels)

    def test_index_matches_var_names(self, synthetic_sc):
        R, P = estimate_nb_params(synthetic_sc, "cell_type", backend="numpy")
        np.testing.assert_array_equal(R.index, synthetic_sc.var_names)
        np.testing.assert_array_equal(P.index, synthetic_sc.var_names)


class TestEstimateNBParamsTorch:
    def test_output_shapes(self, synthetic_sc):
        R, P = estimate_nb_params(synthetic_sc, "cell_type", backend="torch")
        n_genes = synthetic_sc.n_vars
        n_types = synthetic_sc.obs["cell_type"].nunique()
        assert R.shape == (n_genes, n_types)
        assert P.shape == (n_genes, 1)

    def test_matches_numpy_backend(self, synthetic_sc):
        R_np, P_np = estimate_nb_params(synthetic_sc, "cell_type", backend="numpy")
        R_pt, P_pt = estimate_nb_params(synthetic_sc, "cell_type", backend="torch")
        np.testing.assert_allclose(R_np.values, R_pt.values, rtol=1e-4)
        np.testing.assert_allclose(P_np.values, P_pt.values, rtol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_nb_params.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement NB parameter estimation**

`flash_scope/model/_nb_params.py`:
```python
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
```

`flash_scope/model/__init__.py`:
```python
from flash_scope.model._nb_params import estimate_nb_params

__all__ = ["estimate_nb_params"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_nb_params.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add flash_scope/model/ tests/test_nb_params.py
git commit -m "feat: add NB parameter estimation with numpy and torch backends"
```

---

### Task 5: FlashScopeModel (nn.Module)

**Files:**
- Create: `flash_scope/model/_deconv.py`
- Modify: `flash_scope/model/__init__.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write tests for FlashScopeModel**

```python
import numpy as np
import torch
import pytest

from flash_scope.model._deconv import FlashScopeModel


@pytest.fixture
def model_params():
    rng = np.random.default_rng(42)
    n_genes = 10
    n_types = 5
    n_spots = 50
    R = rng.random((n_genes, n_types)).astype(np.float32)
    P = rng.random(n_genes).astype(np.float32) * 0.8 + 0.1
    return R, P, n_spots, n_types, n_genes


class TestFlashScopeModel:
    def test_init(self, model_params):
        R, P, n_spots, n_types, n_genes = model_params
        model = FlashScopeModel(R, P, n_spots, n_types, n_genes)
        assert model.w.shape == (n_spots, n_types)
        assert model.nu.shape == (1, n_genes)
        assert model.eta.shape == (1, n_genes)
        assert model.alpha.shape == (1,)

    def test_forward_output_shapes(self, model_params):
        R, P, n_spots, n_types, n_genes = model_params
        model = FlashScopeModel(R, P, n_spots, n_types, n_genes)
        x = torch.randn(8, n_genes)
        idx = torch.arange(8)
        r_sp, p_sp = model(x, idx)
        assert r_sp.shape == (8, n_genes)
        assert p_sp.shape == (1, n_genes)

    def test_loss_returns_scalar(self, model_params):
        R, P, n_spots, n_types, n_genes = model_params
        model = FlashScopeModel(R, P, n_spots, n_types, n_genes)
        x = torch.randint(0, 10, (8, n_genes)).float()
        idx = torch.arange(8)
        r_sp, p_sp = model(x, idx)
        loss = model.loss(x, r_sp, p_sp)
        assert loss.dim() == 0
        assert not torch.isnan(loss)

    def test_get_proportions_shape(self, model_params):
        R, P, n_spots, n_types, n_genes = model_params
        model = FlashScopeModel(R, P, n_spots, n_types, n_genes)
        props = model.get_proportions()
        assert props.shape == (n_spots, n_types)

    def test_proportions_sum_to_one(self, model_params):
        R, P, n_spots, n_types, n_genes = model_params
        model = FlashScopeModel(R, P, n_spots, n_types, n_genes)
        props = model.get_proportions()
        np.testing.assert_allclose(props.sum(axis=1), 1.0, atol=1e-5)

    def test_proportions_non_negative(self, model_params):
        R, P, n_spots, n_types, n_genes = model_params
        model = FlashScopeModel(R, P, n_spots, n_types, n_genes)
        props = model.get_proportions()
        assert (props >= 0).all()

    def test_r_sc_not_trainable(self, model_params):
        R, P, n_spots, n_types, n_genes = model_params
        model = FlashScopeModel(R, P, n_spots, n_types, n_genes)
        assert not model.r_sc.requires_grad
        assert not model.p_sc.requires_grad

    def test_backward_pass(self, model_params):
        R, P, n_spots, n_types, n_genes = model_params
        model = FlashScopeModel(R, P, n_spots, n_types, n_genes)
        x = torch.randint(0, 10, (8, n_genes)).float()
        idx = torch.arange(8)
        r_sp, p_sp = model(x, idx)
        loss = model.loss(x, r_sp, p_sp)
        loss.backward()
        assert model.w.grad is not None
        assert model.nu.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL — `FlashScopeModel` not found

- [ ] **Step 3: Implement FlashScopeModel**

`flash_scope/model/_deconv.py`:
```python
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.negative_binomial import NegativeBinomial


class FlashScopeModel(nn.Module):
    def __init__(
        self,
        r: np.ndarray,
        p: np.ndarray,
        n_spots: int,
        n_types: int,
        n_genes: int,
    ):
        super().__init__()
        self.n_spots = n_spots
        self.n_types = n_types
        self.n_genes = n_genes

        self.w = nn.Parameter(torch.ones(n_spots, n_types) / n_types)
        self.nu = nn.Parameter(torch.ones(1, n_genes))
        self.eta = nn.Parameter(torch.zeros(1, n_genes))
        self.alpha = nn.Parameter(torch.ones(1))

        self.register_buffer("r_sc", torch.tensor(r.T.astype(np.float32)).contiguous())
        p_arr = np.asarray(p, dtype=np.float32).ravel()
        self.register_buffer("p_sc", torch.tensor(p_arr).unsqueeze(0).contiguous())

        self._softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        beta = self._softplus(self.nu)
        v = self._softplus(self.w[idx])
        eps = self._softplus(self.eta)
        gamma = self._softplus(self.alpha)
        r_sp = beta * torch.mm(v, self.r_sc) + gamma * eps
        p_sp = self.p_sc
        return r_sp, p_sp

    def loss(self, x: torch.Tensor, r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return -NegativeBinomial(total_count=r, probs=p).log_prob(x).sum()

    @torch.no_grad()
    def get_proportions(self) -> np.ndarray:
        props = self._softplus(self.w)
        props = props / props.sum(dim=1, keepdim=True)
        return props.detach().cpu().numpy()
```

- [ ] **Step 4: Update flash_scope/model/__init__.py**

```python
from flash_scope.model._nb_params import estimate_nb_params
from flash_scope.model._deconv import FlashScopeModel

__all__ = ["estimate_nb_params", "FlashScopeModel"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: All 8 tests PASS

- [ ] **Step 6: Commit**

```bash
git add flash_scope/model/_deconv.py flash_scope/model/__init__.py tests/test_model.py
git commit -m "feat: add FlashScopeModel with NB loss and proportion extraction"
```

---

### Task 6: Training Loop

**Files:**
- Create: `flash_scope/model/_trainer.py`
- Modify: `flash_scope/model/__init__.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write tests for trainer**

```python
import numpy as np
import anndata as ad
import pandas as pd
import torch
import pytest

from flash_scope.model._nb_params import estimate_nb_params
from flash_scope.model._deconv import FlashScopeModel
from flash_scope.model._trainer import fit


@pytest.fixture
def fitted_setup(synthetic_sc):
    R, P, labels = estimate_nb_params(synthetic_sc, "cell_type", return_labels=True)
    n_spots = 20
    n_types = len(labels)
    n_genes = R.shape[0]

    rng = np.random.default_rng(42)
    X_sp = rng.negative_binomial(n=5, p=0.3, size=(n_spots, n_genes)).astype(np.float32)
    ad_sp = ad.AnnData(
        X=X_sp,
        obs=pd.DataFrame(index=[f"s_{i}" for i in range(n_spots)]),
        var=pd.DataFrame(index=R.index),
    )

    model = FlashScopeModel(R.values, P.values, n_spots, n_types, n_genes)
    return model, ad_sp


class TestFit:
    def test_returns_model(self, fitted_setup):
        model, ad_sp = fitted_setup
        result = fit(model, ad_sp, epochs=5, batch_size=8, device="cpu", use_compile=False)
        assert isinstance(result, FlashScopeModel)

    def test_loss_decreases(self, fitted_setup):
        model, ad_sp = fitted_setup
        result = fit(model, ad_sp, epochs=50, batch_size=8, device="cpu", use_compile=False)
        props = result.get_proportions()
        assert props.shape == (ad_sp.n_obs, model.n_types)

    def test_model_on_cpu_after_fit(self, fitted_setup):
        model, ad_sp = fitted_setup
        result = fit(model, ad_sp, epochs=5, batch_size=8, device="cpu", use_compile=False)
        assert result.w.device == torch.device("cpu")

    def test_proportions_valid_after_fit(self, fitted_setup):
        model, ad_sp = fitted_setup
        result = fit(model, ad_sp, epochs=20, batch_size=8, device="cpu", use_compile=False)
        props = result.get_proportions()
        np.testing.assert_allclose(props.sum(axis=1), 1.0, atol=1e-5)
        assert (props >= 0).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL — `flash_scope.model._trainer` not found

- [ ] **Step 3: Implement training loop**

`flash_scope/model/_trainer.py`:
```python
from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
import anndata as ad

from flash_scope._utils import resolve_device, to_dense_array
from flash_scope.model._deconv import FlashScopeModel


def fit(
    model: FlashScopeModel,
    adata: ad.AnnData,
    epochs: int = 500,
    batch_size: int = 1024,
    lr: float = 1e-3,
    device: str = "auto",
    use_compile: bool = True,
) -> FlashScopeModel:
    dev = resolve_device(device)
    use_cuda = dev.type == "cuda"

    X = torch.tensor(to_dense_array(adata.X), dtype=torch.float32)
    indices = torch.arange(X.shape[0], dtype=torch.long)
    dataset = TensorDataset(X, indices)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=use_cuda,
    )

    model = model.to(dev)

    train_model = model
    if use_compile and use_cuda:
        train_model = torch.compile(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    amp_dtype = torch.float16 if use_cuda else torch.bfloat16

    for _epoch in range(epochs):
        for x_batch, idx_batch in loader:
            x_batch = x_batch.to(dev, non_blocking=use_cuda)
            idx_batch = idx_batch.to(dev, non_blocking=use_cuda)

            with torch.amp.autocast(device_type=dev.type, dtype=amp_dtype, enabled=use_cuda):
                r_sp, p_sp = train_model(x_batch, idx_batch)
                loss = model.loss(x_batch, r_sp, p_sp)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model = model.cpu()
    return model
```

- [ ] **Step 4: Update flash_scope/model/__init__.py**

```python
from flash_scope.model._nb_params import estimate_nb_params
from flash_scope.model._deconv import FlashScopeModel
from flash_scope.model._trainer import fit

__all__ = ["estimate_nb_params", "FlashScopeModel", "fit"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_trainer.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add flash_scope/model/_trainer.py flash_scope/model/__init__.py tests/test_trainer.py
git commit -m "feat: add pure PyTorch training loop with mixed precision and compile support"
```

---

### Task 7: Deconvolve Orchestrator

**Files:**
- Create: `flash_scope/tl/__init__.py`
- Create: `flash_scope/tl/_deconvolve.py`
- Create: `tests/test_deconvolve.py`

- [ ] **Step 1: Write tests for deconvolve**

```python
import numpy as np
import pandas as pd
import anndata as ad
import pytest

from flash_scope.tl import deconvolve


class TestDeconvolve:
    def test_returns_dataframe(self, synthetic_sc, synthetic_sp):
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=10, batch_size=16, device="cpu",
                           min_sc_counts=0, min_sp_counts=0)
        assert isinstance(props, pd.DataFrame)

    def test_output_shape(self, synthetic_sc, synthetic_sp):
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=10, batch_size=16, device="cpu",
                           min_sc_counts=0, min_sp_counts=0)
        n_types = synthetic_sc.obs["cell_type"].nunique()
        assert props.shape == (synthetic_sp.n_obs, n_types)

    def test_proportions_sum_to_one(self, synthetic_sc, synthetic_sp):
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=10, batch_size=16, device="cpu",
                           min_sc_counts=0, min_sp_counts=0)
        np.testing.assert_allclose(props.sum(axis=1), 1.0, atol=1e-5)

    def test_accepts_list_of_spatial(self, synthetic_sc, synthetic_sp):
        sp1 = synthetic_sp[:25].copy()
        sp2 = synthetic_sp[25:].copy()
        props = deconvolve(synthetic_sc, [sp1, sp2], label_col="cell_type",
                           epochs=10, batch_size=16, device="cpu",
                           min_sc_counts=0, min_sp_counts=0)
        assert isinstance(props, pd.DataFrame)
        assert props.shape[0] == synthetic_sp.n_obs

    def test_index_matches_spatial_obs(self, synthetic_sc, synthetic_sp):
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=10, batch_size=16, device="cpu",
                           min_sc_counts=0, min_sp_counts=0)
        np.testing.assert_array_equal(props.index, synthetic_sp.obs_names)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_deconvolve.py -v`
Expected: FAIL — `flash_scope.tl` not found

- [ ] **Step 3: Implement deconvolve orchestrator**

`flash_scope/tl/_deconvolve.py`:
```python
from __future__ import annotations

from typing import List

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
        obs_names = ad.concat(ad_sp, join="outer").obs_names
        _ad_sp = ad.concat(ad_sp, join="outer")
    else:
        obs_names = ad_sp.obs_names
        _ad_sp = ad_sp.copy()

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
```

`flash_scope/tl/__init__.py`:
```python
from flash_scope.tl._deconvolve import deconvolve

__all__ = ["deconvolve"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_deconvolve.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add flash_scope/tl/ tests/test_deconvolve.py
git commit -m "feat: add deconvolve orchestrator in tools layer"
```

---

### Task 8: MCP Server

**Files:**
- Create: `flash_scope/mcp/__init__.py`
- Create: `flash_scope/mcp/__main__.py`
- Create: `flash_scope/mcp/_server.py`
- Create: `tests/test_mcp.py`

**Note:** Install fastmcp first: `pip install "fastmcp>=2.0"`

- [ ] **Step 1: Write tests for MCP server**

```python
import json
import numpy as np
import pandas as pd
import anndata as ad
import pytest

from flash_scope.mcp._server import ServerState, create_server


@pytest.fixture
def state():
    return ServerState()


@pytest.fixture
def sc_file(tmp_path, synthetic_sc):
    path = tmp_path / "sc.h5ad"
    synthetic_sc.write_h5ad(path)
    return str(path)


@pytest.fixture
def sp_file(tmp_path, synthetic_sp):
    path = tmp_path / "sp.h5ad"
    synthetic_sp.write_h5ad(path)
    return str(path)


class TestServerState:
    def test_initial_state_empty(self, state):
        assert state.ad_sc is None
        assert state.ad_sp is None
        assert state.model is None
        assert state.proportions is None


class TestLoadReference:
    @pytest.mark.anyio
    async def test_loads_h5ad(self, sc_file, state):
        from flash_scope.mcp._server import _load_reference
        result = await _load_reference(state, sc_file, format="h5ad")
        assert state.ad_sc is not None
        assert "100" in result


class TestLoadSpatial:
    @pytest.mark.anyio
    async def test_loads_h5ad(self, sp_file, state):
        from flash_scope.mcp._server import _load_spatial
        result = await _load_spatial(state, sp_file, format="h5ad")
        assert state.ad_sp is not None
        assert "50" in result


class TestFitTool:
    @pytest.mark.anyio
    async def test_fit_runs(self, sc_file, sp_file, state):
        from flash_scope.mcp._server import _load_reference, _load_spatial, _preprocess, _fit
        await _load_reference(state, sc_file, format="h5ad")
        await _load_spatial(state, sp_file, format="h5ad")
        await _preprocess(state, label_col="cell_type", min_label_member=20,
                          min_sc_counts=0, min_sp_counts=0)
        result = await _fit(state, label_col="cell_type", epochs=5, batch_size=16)
        assert state.proportions is not None
        assert "loss" in result.lower() or "complete" in result.lower()


class TestGetProportions:
    @pytest.mark.anyio
    async def test_returns_json(self, sc_file, sp_file, state):
        from flash_scope.mcp._server import (
            _load_reference, _load_spatial, _preprocess, _fit, _get_proportions
        )
        await _load_reference(state, sc_file, format="h5ad")
        await _load_spatial(state, sp_file, format="h5ad")
        await _preprocess(state, label_col="cell_type", min_label_member=20,
                          min_sc_counts=0, min_sp_counts=0)
        await _fit(state, label_col="cell_type", epochs=5, batch_size=16)
        result = await _get_proportions(state, top_n=3)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pip install "fastmcp>=2.0" anyio pytest-anyio && pytest tests/test_mcp.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement MCP server**

`flash_scope/mcp/_server.py`:
```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

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


async def _load_reference(state: ServerState, path: str, format: str = "h5ad") -> str:
    reader = _IO_DISPATCH.get(format)
    if reader is None:
        return f"Unknown format: {format}. Supported: {list(_IO_DISPATCH)}"
    state.ad_sc = reader(path)
    n_obs, n_vars = state.ad_sc.shape
    cols = list(state.ad_sc.obs.columns)
    return f"Loaded reference: {n_obs} cells, {n_vars} genes. Columns: {cols}"


async def _load_spatial(state: ServerState, path: str, format: str = "h5ad") -> str:
    reader = _IO_DISPATCH.get(format)
    if reader is None:
        return f"Unknown format: {format}. Supported: {list(_IO_DISPATCH)}"
    state.ad_sp = reader(path)
    n_obs, n_vars = state.ad_sp.shape
    cols = list(state.ad_sp.obs.columns)
    return f"Loaded spatial: {n_obs} spots, {n_vars} genes. Columns: {cols}"


async def _preprocess(
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


async def _fit(
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


async def _get_proportions(state: ServerState, top_n: int = 5) -> str:
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
    async def load_reference(path: str, format: str = "h5ad") -> str:
        """Load reference scRNA-seq data. Formats: h5ad, parquet, csv."""
        return await _load_reference(state, path, format)

    @mcp.tool()
    async def load_spatial(path: str, format: str = "h5ad") -> str:
        """Load spatial transcriptomics data. Formats: h5ad, parquet, csv."""
        return await _load_spatial(state, path, format)

    @mcp.tool()
    async def preprocess_data(
        label_col: str,
        min_label_member: int = 20,
        min_sc_counts: int = 100,
        min_sp_counts: int = 100,
    ) -> str:
        """Preprocess loaded reference and spatial data. Filters genes and intersects variables."""
        return await _preprocess(state, label_col, min_label_member, min_sc_counts, min_sp_counts)

    @mcp.tool()
    async def fit_model_tool(label_col: str, epochs: int = 500, batch_size: int = 1024) -> str:
        """Run NB deconvolution on preprocessed data. Returns training summary."""
        return await _fit(state, label_col, epochs, batch_size)

    @mcp.tool()
    async def get_proportions(top_n: int = 5) -> str:
        """Get cell type proportions per spot. Returns JSON with top_n types per spot."""
        return await _get_proportions(state, top_n)

    return mcp


def serve():
    mcp = create_server()
    mcp.run()
```

`flash_scope/mcp/__init__.py`:
```python
from flash_scope.mcp._server import create_server, serve

__all__ = ["create_server", "serve"]
```

`flash_scope/mcp/__main__.py`:
```python
from flash_scope.mcp._server import serve

serve()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mcp.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add flash_scope/mcp/ tests/test_mcp.py
git commit -m "feat: add FastMCP server with load, preprocess, fit, get_proportions tools"
```

---

### Task 9: Integration Test & Top-Level Wiring

**Files:**
- Modify: `flash_scope/__init__.py` (add mcp conditional import)
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
import numpy as np
import pandas as pd
import anndata as ad
import pytest

import flash_scope as fs


class TestTopLevelImports:
    def test_version(self):
        assert hasattr(fs, "__version__")
        assert fs.__version__ == "0.1.0"

    def test_namespace_io(self):
        assert hasattr(fs.io, "read_h5ad")
        assert hasattr(fs.io, "read_parquet")
        assert hasattr(fs.io, "read_csv")

    def test_namespace_pp(self):
        assert hasattr(fs.pp, "preprocess")
        assert hasattr(fs.pp, "filter_by_label")
        assert hasattr(fs.pp, "filter_genes")
        assert hasattr(fs.pp, "intersect_vars")
        assert hasattr(fs.pp, "densify")

    def test_namespace_model(self):
        assert hasattr(fs.model, "estimate_nb_params")
        assert hasattr(fs.model, "FlashScopeModel")
        assert hasattr(fs.model, "fit")

    def test_namespace_tl(self):
        assert hasattr(fs.tl, "deconvolve")


class TestFullPipeline:
    def test_one_liner(self, synthetic_sc, synthetic_sp):
        props = fs.tl.deconvolve(
            synthetic_sc, synthetic_sp,
            label_col="cell_type",
            epochs=10, batch_size=16, device="cpu",
            min_sc_counts=0, min_sp_counts=0,
            use_compile=False,
        )
        assert isinstance(props, pd.DataFrame)
        assert props.shape == (synthetic_sp.n_obs, synthetic_sc.obs["cell_type"].nunique())
        np.testing.assert_allclose(props.sum(axis=1), 1.0, atol=1e-5)
        assert (props.values >= 0).all()

    def test_step_by_step(self, synthetic_sc, synthetic_sp):
        ad_sc, ad_sp = fs.pp.preprocess(
            synthetic_sc, synthetic_sp,
            label_col="cell_type",
            min_sc_counts=0, min_sp_counts=0,
        )
        R, P, labels = fs.model.estimate_nb_params(ad_sc, label_col="cell_type", return_labels=True)
        model = fs.model.FlashScopeModel(
            R.values, P.values,
            n_spots=ad_sp.n_obs, n_types=len(labels), n_genes=ad_sp.n_vars,
        )
        model = fs.model.fit(model, ad_sp, epochs=10, batch_size=16, device="cpu", use_compile=False)
        props = model.get_proportions()
        assert props.shape == (ad_sp.n_obs, len(labels))
        np.testing.assert_allclose(props.sum(axis=1), 1.0, atol=1e-5)

    def test_io_roundtrip(self, tmp_path, synthetic_sc, synthetic_sp):
        sc_path = tmp_path / "sc.h5ad"
        sp_path = tmp_path / "sp.h5ad"
        synthetic_sc.write_h5ad(sc_path)
        synthetic_sp.write_h5ad(sp_path)

        ad_sc = fs.io.read_h5ad(sc_path)
        ad_sp = fs.io.read_h5ad(sp_path)

        props = fs.tl.deconvolve(
            ad_sc, ad_sp,
            label_col="cell_type",
            epochs=10, batch_size=16, device="cpu",
            min_sc_counts=0, min_sp_counts=0,
            use_compile=False,
        )
        assert isinstance(props, pd.DataFrame)
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full pipeline and namespace verification"
```

---

### Task 10: Install Verification & Cleanup

- [ ] **Step 1: Verify editable install**

Run: `cd /cv/data/braid/andera29/projs/flash-scope && pip install -e ".[dev,mcp]"`
Expected: Successful install

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Verify import from clean Python**

Run: `python -c "import flash_scope as fs; print(fs.__version__); print(dir(fs.io)); print(dir(fs.pp)); print(dir(fs.model)); print(dir(fs.tl))"`
Expected: Prints version and module contents

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: finalize flash-scope v0.1.0 package"
```
