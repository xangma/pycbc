"""Guard the private search bridge without changing public CPU dispatch."""

import os

import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import Array
from pycbc.types.backend import wrap_backend_array
from pycbc.types.torch_compat import cpu_compatible, cpu_context

torch = pytest.importorskip("torch")


def test_bridge_restores_locked_scheme_and_threads_after_exception():
    with scheme.TorchScheme("cpu", num_threads=1) as context:
        environment = os.environ.get("OMP_NUM_THREADS")
        threads = torch.get_num_threads()
        original = Array(np.arange(8, dtype=np.float32))
        assert cpu_compatible(original)
        copied = original.numpy().copy()
        with pytest.raises(ValueError, match="deliberate"):
            with cpu_context():
                assert isinstance(scheme.mgr.state, scheme.CPUScheme)
                assert scheme.mgr._lock
                host = Array(copied)
                assert isinstance(host.data, np.ndarray)
                host[:] = 0
                raise ValueError("deliberate")
        assert scheme.mgr.state is context
        assert scheme.mgr._lock
        assert os.environ.get("OMP_NUM_THREADS") == environment
        assert torch.get_num_threads() == threads
        np.testing.assert_array_equal(original.numpy(), np.arange(8))


def test_bridge_requires_torch_and_preserves_unlocked_default(monkeypatch):
    with pytest.raises(RuntimeError, match="require a Torch"):
        with cpu_context():
            pass
    context = scheme.TorchScheme("cpu")
    monkeypatch.setattr(scheme.mgr, "state", context)
    monkeypatch.setattr(scheme.mgr, "_lock", False)
    with cpu_context():
        assert scheme.mgr._lock
    assert scheme.mgr.state is context
    assert not scheme.mgr._lock


def test_bridge_rejects_autograd_and_tensor_subclasses():
    class UserTensor(torch.Tensor):
        pass

    with scheme.TorchScheme("cpu"):
        def eligible(tensor):
            return cpu_compatible(wrap_backend_array(tensor))

        assert eligible(torch.ones(4))
        assert not eligible(torch.ones(4, dtype=torch.float64))
        assert not eligible(torch.ones(4, requires_grad=True))
        assert not eligible(torch.ones(4).as_subclass(UserTensor))
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(torch.ones(4), torch.ones(4))
            assert not eligible(dual)

