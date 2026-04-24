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
    def test_loads_h5ad(self, sc_file, state):
        from flash_scope.mcp._server import _load_reference
        result = _load_reference(state, sc_file, format="h5ad")
        assert state.ad_sc is not None
        assert "100" in result


class TestLoadSpatial:
    def test_loads_h5ad(self, sp_file, state):
        from flash_scope.mcp._server import _load_spatial
        result = _load_spatial(state, sp_file, format="h5ad")
        assert state.ad_sp is not None
        assert "50" in result


class TestFitTool:
    def test_fit_runs(self, sc_file, sp_file, state):
        from flash_scope.mcp._server import _load_reference, _load_spatial, _preprocess, _fit
        _load_reference(state, sc_file, format="h5ad")
        _load_spatial(state, sp_file, format="h5ad")
        _preprocess(state, label_col="cell_type", min_label_member=20,
                    min_sc_counts=0, min_sp_counts=0)
        result = _fit(state, label_col="cell_type", epochs=5, batch_size=16)
        assert state.proportions is not None
        assert "complete" in result.lower()


class TestGetProportions:
    def test_returns_json(self, sc_file, sp_file, state):
        from flash_scope.mcp._server import (
            _load_reference, _load_spatial, _preprocess, _fit, _get_proportions
        )
        _load_reference(state, sc_file, format="h5ad")
        _load_spatial(state, sp_file, format="h5ad")
        _preprocess(state, label_col="cell_type", min_label_member=20,
                    min_sc_counts=0, min_sp_counts=0)
        _fit(state, label_col="cell_type", epochs=5, batch_size=16)
        result = _get_proportions(state, top_n=3)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0
