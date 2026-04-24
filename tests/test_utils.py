import numpy as np
import scipy.sparse as sp
import torch

from flash_scope._utils import resolve_device, to_dense_array


class TestResolveDevice:
    def test_cpu_explicit(self):
        d = resolve_device("cpu")
        assert d == torch.device("cpu")

    def test_auto_returns_device(self):
        d = resolve_device("auto")
        assert isinstance(d, torch.device)

    def test_cuda_explicit(self):
        d = resolve_device("cuda")
        assert d == torch.device("cuda")


class TestToDenseArray:
    def test_ndarray_passthrough(self):
        x = np.array([[1, 2], [3, 4]])
        result = to_dense_array(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, x)

    def test_sparse_to_dense(self):
        x = sp.csr_matrix(np.array([[1, 0], [0, 2]]))
        result = to_dense_array(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[1, 0], [0, 2]])

    def test_tensor_to_dense(self):
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = to_dense_array(x)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [[1.0, 2.0], [3.0, 4.0]])
