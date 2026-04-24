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
