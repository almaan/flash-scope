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
