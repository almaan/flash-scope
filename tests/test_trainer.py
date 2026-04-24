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
