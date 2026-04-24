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
