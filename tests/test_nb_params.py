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

    def test_logits_are_finite(self, synthetic_sc):
        R, logits = estimate_nb_params(synthetic_sc, "cell_type", backend="numpy")
        assert np.all(np.isfinite(logits.values))

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
        np.testing.assert_allclose(R_np.values, R_pt.values, rtol=5e-2)
        np.testing.assert_allclose(P_np.values, P_pt.values, rtol=5e-2)
