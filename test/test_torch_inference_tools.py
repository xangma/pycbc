"""Focused tests for lazy Torch inference helper dispatch."""

import os
import pathlib
import subprocess
import sys

import numpy
import pytest


def test_numpy_subset_uses_the_model_generator(monkeypatch):
    from pycbc.inference.models import tools

    def reject_global_choice(*args, **kwargs):
        raise AssertionError("subset sampling must use the model generator")

    monkeypatch.setattr(numpy.random, "choice", reject_global_choice)
    expected = numpy.random.default_rng(27).choice(100, size=7, replace=False)
    choice, host_choice = tools._random_permutation(
        numpy.arange(100), 7, numpy.random.default_rng(27)
    )
    numpy.testing.assert_array_equal(choice, expected)
    assert host_choice is choice


def test_numpy_helpers_do_not_import_torch_backend():
    code = r"""
import importlib.util
import os
import pathlib
import sys
import types
import numpy

root = pathlib.Path(os.environ["PYCBC_TEST_ROOT"])
packages = {
    "pycbc": root / "pycbc",
    "pycbc.inference": root / "pycbc/inference",
    "pycbc.inference.models": root / "pycbc/inference/models",
    "pycbc.types": root / "pycbc/types",
}
for name, path in packages.items():
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
distributions = types.ModuleType("pycbc.distributions")
distributions.JointDistribution = object
sys.modules["pycbc.distributions"] = distributions
detector = types.ModuleType("pycbc.detector")
detector.Detector = object
sys.modules["pycbc.detector"] = detector

backend_spec = importlib.util.spec_from_file_location(
    "pycbc.types.backend", root / "pycbc/types/backend.py"
)
backend = importlib.util.module_from_spec(backend_spec)
sys.modules[backend_spec.name] = backend
backend_spec.loader.exec_module(backend)
tools_spec = importlib.util.spec_from_file_location(
    "pycbc.inference.models.tools",
    root / "pycbc/inference/models/tools.py",
)
tools = importlib.util.module_from_spec(tools_spec)
sys.modules[tools_spec.name] = tools
tools_spec.loader.exec_module(tools)

torch_backend = "pycbc.inference.models.tools_torch"
assert torch_backend not in sys.modules
hd, hh = tools._fused_inner_hd_hh(
    numpy.array([1.0 + 2.0j, 3.0 - 1.0j]),
    numpy.array([2.0 - 1.0j, 0.5 + 4.0j]),
)
assert numpy.isfinite(hd)
assert numpy.isfinite(hh)
assert tools._selected_values(numpy.arange(4), [1, 3]).tolist() == [1, 3]
assert torch_backend not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYCBC_TEST_ROOT"] = str(
        pathlib.Path(__file__).resolve().parents[1]
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_torch_dispatch_matches_numpy_and_preserves_gradients():
    torch = pytest.importorskip("torch")
    from pycbc.inference.models import tools

    h_numpy = numpy.array([1.0 + 2.0j, 3.0 - 1.0j])
    d_numpy = numpy.array([2.0 - 1.0j, 0.5 + 4.0j])
    h = torch.tensor(h_numpy, dtype=torch.complex128, requires_grad=True)
    d = torch.tensor(d_numpy, dtype=torch.complex128)

    got_hd, got_hh = tools._fused_inner_hd_hh(h, d)
    expected_hd, expected_hh = tools._fused_inner_hd_hh(h_numpy, d_numpy)
    numpy.testing.assert_allclose(got_hd.detach().numpy(), expected_hd)
    numpy.testing.assert_allclose(got_hh.detach().numpy(), expected_hh)
    (got_hd.real + got_hh).backward()
    assert h.grad is not None

    loglr = numpy.array([-1.0, 0.5, 1.5, -0.25])
    numpy.random.seed(91)
    got_indices = tools.draw_sample(
        torch.tensor(loglr, dtype=torch.float64), size=16
    )
    numpy.random.seed(91)
    expected_indices = tools.draw_sample(loglr, size=16)
    numpy.testing.assert_array_equal(got_indices, expected_indices)

    sh = numpy.array([1.5 + 0.25j, 0.5 - 1.0j, 2.0 + 0.5j])
    hh = numpy.array([0.75, 1.25, 2.5])
    expected = tools.marginalize_likelihood(sh, hh, phase=True)
    actual = tools.marginalize_likelihood(
        torch.tensor(sh, dtype=torch.complex128),
        torch.tensor(hh, dtype=torch.float64),
        phase=True,
    )
    assert actual == pytest.approx(expected)


def test_public_backend_protocol_and_frequency_lookup():
    torch = pytest.importorskip("torch")
    from pycbc.inference.models import tools

    class PublicTorchValue:
        backend = "torch"

        def __init__(self, value):
            self.backend_array = value

    left = PublicTorchValue(torch.tensor([1.0, 2.0], dtype=torch.float64))
    right = PublicTorchValue(torch.tensor([3.0, 4.0], dtype=torch.float64))
    assert tools._inner(left, right).item() == pytest.approx(11.0)

    frequencies = torch.arange(16, dtype=torch.float64) * 0.25
    assert tools._last_index_at_or_below(frequencies, 1.1) == 4
    with pytest.raises(IndexError, match="no values"):
        tools._last_index_at_or_below(frequencies, -0.1)
