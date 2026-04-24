import numpy as np
import pandas as pd
import anndata as ad
import pytest

from flash_scope.tl import deconvolve


class TestDeconvolve:
    def test_returns_dataframe(self, synthetic_sc, synthetic_sp):
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=10, batch_size=16, device="cpu",
                           min_sc_counts=0, min_sp_counts=0, use_compile=False)
        assert isinstance(props, pd.DataFrame)

    def test_output_shape(self, synthetic_sc, synthetic_sp):
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=10, batch_size=16, device="cpu",
                           min_sc_counts=0, min_sp_counts=0, use_compile=False)
        n_types = synthetic_sc.obs["cell_type"].nunique()
        assert props.shape == (synthetic_sp.n_obs, n_types)

    def test_proportions_sum_to_one(self, synthetic_sc, synthetic_sp):
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=10, batch_size=16, device="cpu",
                           min_sc_counts=0, min_sp_counts=0, use_compile=False)
        np.testing.assert_allclose(props.sum(axis=1), 1.0, atol=1e-5)

    def test_accepts_list_of_spatial(self, synthetic_sc, synthetic_sp):
        sp1 = synthetic_sp[:25].copy()
        sp2 = synthetic_sp[25:].copy()
        props = deconvolve(synthetic_sc, [sp1, sp2], label_col="cell_type",
                           epochs=10, batch_size=16, device="cpu",
                           min_sc_counts=0, min_sp_counts=0, use_compile=False)
        assert isinstance(props, pd.DataFrame)
        assert props.shape[0] == synthetic_sp.n_obs

    def test_index_matches_spatial_obs(self, synthetic_sc, synthetic_sp):
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=10, batch_size=16, device="cpu",
                           min_sc_counts=0, min_sp_counts=0, use_compile=False)
        np.testing.assert_array_equal(props.index, synthetic_sp.obs_names)
