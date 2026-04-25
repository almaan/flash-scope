import numpy as np
import torch
import pytest

from flash_scope.model._init import nnls_init, coarse_init, _inverse_softplus
from flash_scope.model._deconv import FlashScopeModel


@pytest.fixture
def nnls_data():
    rng = np.random.default_rng(42)
    n_spots = 50
    n_genes = 10
    n_types = 5
    R = rng.random((n_genes, n_types)).astype(np.float32) + 0.1
    X = rng.negative_binomial(n=5, p=0.3, size=(n_spots, n_genes)).astype(np.float32)
    return X, R, n_spots, n_types, n_genes


class TestNNLSInit:
    def test_output_shape(self, nnls_data):
        X, R, n_spots, n_types, _ = nnls_data
        W = nnls_init(X, R)
        assert W.shape == (n_spots, n_types + 1)

    def test_output_finite(self, nnls_data):
        X, R, n_spots, _, _ = nnls_data
        W = nnls_init(X, R)
        assert np.all(np.isfinite(W))

    def test_softplus_proportions_valid(self, nnls_data):
        X, R, n_spots, _, _ = nnls_data
        W = nnls_init(X, R)
        sp = np.log1p(np.exp(W))
        assert (sp > 0).all()
        row_sums = sp.sum(axis=1)
        np.testing.assert_allclose(row_sums / row_sums, 1.0, atol=1e-5)

    def test_proportions_sum_to_one_after_normalize(self, nnls_data):
        X, R, n_spots, n_types, n_genes = nnls_data
        W = nnls_init(X, R)
        model = FlashScopeModel(R, R[:, 0], n_spots, n_types, n_genes, init_w=W)
        props = model.get_proportions()
        np.testing.assert_allclose(props.sum(axis=1), 1.0, atol=1e-5)

    def test_model_init_w_shape(self, nnls_data):
        X, R, n_spots, n_types, n_genes = nnls_data
        W = nnls_init(X, R)
        model = FlashScopeModel(R, R[:, 0], n_spots, n_types, n_genes, init_w=W)
        assert model.w.shape == (n_spots, n_types + 1)

    def test_model_default_init_shape(self, nnls_data):
        _, R, n_spots, n_types, n_genes = nnls_data
        model = FlashScopeModel(R, R[:, 0], n_spots, n_types, n_genes)
        assert model.w.shape == (n_spots, n_types + 1)


class TestInverseSoftplus:
    def test_roundtrip(self):
        x = np.array([0.1, 0.5, 1.0, 5.0], dtype=np.float32)
        recovered = np.log1p(np.exp(_inverse_softplus(x)))
        np.testing.assert_allclose(recovered, x, atol=1e-4)


class TestCoarseInit:
    def test_output_shape(self, nnls_data):
        X, R, n_spots, n_types, n_genes = nnls_data
        P = R[:, 0:1]
        W = coarse_init(X, R, P, n_clusters=10, coarse_epochs=5)
        assert W.shape == (n_spots, n_types + 1)

    def test_output_finite(self, nnls_data):
        X, R, n_spots, _, _ = nnls_data
        P = R[:, 0:1]
        W = coarse_init(X, R, P, n_clusters=10, coarse_epochs=5)
        assert np.all(np.isfinite(W))

    def test_proportions_valid(self, nnls_data):
        X, R, n_spots, n_types, n_genes = nnls_data
        P = R[:, 0:1]
        W = coarse_init(X, R, P, n_clusters=10, coarse_epochs=5)
        model = FlashScopeModel(R, P, n_spots, n_types, n_genes, init_w=W)
        props = model.get_proportions()
        assert (props >= 0).all()
        np.testing.assert_allclose(props.sum(axis=1), 1.0, atol=1e-5)

    def test_n_clusters_exceeds_n_spots(self, nnls_data):
        X, R, n_spots, n_types, _ = nnls_data
        P = R[:, 0:1]
        W = coarse_init(X, R, P, n_clusters=n_spots + 100, coarse_epochs=5)
        assert W.shape == (n_spots, n_types + 1)


class TestDeconvolveWarmStart:
    def test_warm_start_runs(self, synthetic_sc, synthetic_sp):
        from flash_scope.tl import deconvolve
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=5, warm_start=True)
        assert props.shape[0] == synthetic_sp.n_obs

    def test_no_warm_start_runs(self, synthetic_sc, synthetic_sp):
        from flash_scope.tl import deconvolve
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=5, warm_start=False)
        assert props.shape[0] == synthetic_sp.n_obs

    def test_coarse_fit_runs(self, synthetic_sc, synthetic_sp):
        from flash_scope.tl import deconvolve
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=5, coarse_fit=True, n_clusters=10, coarse_epochs=5)
        assert props.shape[0] == synthetic_sp.n_obs

    def test_coarse_fit_overrides_warm_start(self, synthetic_sc, synthetic_sp):
        from flash_scope.tl import deconvolve
        props = deconvolve(synthetic_sc, synthetic_sp, label_col="cell_type",
                           epochs=5, coarse_fit=True, warm_start=True,
                           n_clusters=10, coarse_epochs=5)
        assert props.shape[0] == synthetic_sp.n_obs
