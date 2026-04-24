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
