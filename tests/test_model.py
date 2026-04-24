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
