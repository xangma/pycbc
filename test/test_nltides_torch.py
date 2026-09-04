import numpy as np
import pytest

from pycbc import scheme
from pycbc.waveform import get_fd_waveform
from pycbc.waveform import nltides
from pycbc.waveform import nltides_torch


_NLTIDE_PARAMS = dict(
    mass1=1.4,
    mass2=1.3,
    delta_f=1.0,
    f_lower=20.0,
    f_final=256.0,
    f_ref=20.0,
    distance=100.0,
    f0=50.0,
    amplitude=1.0e-10,
    n=1.0,
)


@pytest.fixture
def preserve_scheme():
    """Restore the process-wide PyCBC scheme singleton after a test."""
    old_scheme = scheme.mgr.state
    old_single = scheme.Scheme._single
    try:
        yield
    finally:
        scheme.mgr.state = old_scheme
        scheme.Scheme._single = old_single


def _activate_scheme(state):
    scheme.Scheme._single = None
    scheme.mgr.state = state


def _reject_numpy_backend():
    class RejectNumpy:
        def __getattr__(self, name):
            raise AssertionError(
                f"nonlinear-tide Torch correction used numpy.{name}"
            )

    return RejectNumpy()


def test_nltides_numpy_phase_is_finite():
    frequencies = np.arange(513) * 0.5
    phase = nltides.nltides_fourier_phase_difference(
        frequencies, 0.5, 50.0, 1.0e-10, 1.0, 1.4, 1.3
    )

    assert phase.shape == frequencies.shape
    assert phase.dtype == np.float64
    assert np.isfinite(phase).all()
    assert np.all(np.diff(phase) > 0.0)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_nltides_torch_phase_matches_numpy(dtype, monkeypatch):
    torch = pytest.importorskip("torch")
    torch_dtype = getattr(torch, dtype)
    frequencies = np.arange(513) * 0.5
    expected = nltides.nltides_fourier_phase_difference(
        frequencies, 0.5, 50.0, 1.0e-10, 1.0, 1.4, 1.3
    )
    torch_frequencies = torch.arange(513, dtype=torch_dtype) * 0.5

    monkeypatch.setattr(nltides, "numpy", _reject_numpy_backend())
    actual = nltides.nltides_fourier_phase_difference(
        torch_frequencies, 0.5, 50.0, 1.0e-10, 1.0, 1.4, 1.3
    )

    assert actual.device.type == "cpu"
    assert actual.dtype == torch_dtype
    tolerances = (
        dict(rtol=2.0e-6, atol=1.0e-9)
        if torch_dtype == torch.float32
        else dict(rtol=1.0e-13, atol=1.0e-15)
    )
    torch.testing.assert_close(
        actual,
        torch.as_tensor(expected, dtype=torch_dtype),
        **tolerances,
    )


def test_nltides_torch_phase_preserves_autograd():
    torch = pytest.importorskip("torch")
    frequencies = (
        torch.arange(513, dtype=torch.float64) * 0.5
    ).requires_grad_()

    phase = nltides.nltides_fourier_phase_difference(
        frequencies, 0.5, 50.0, 1.0e-10, 1.0, 1.4, 1.3
    )
    phase.sum().backward()

    assert frequencies.grad is not None
    assert torch.isfinite(frequencies.grad).all()


@pytest.mark.parametrize("device", ("cpu", "cuda", "mps"))
def test_taylorf2nl_correction_stays_on_torch_device(
    device, monkeypatch, preserve_scheme
):
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")

    from pycbc.types.array_torch import TorchArrayData

    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")
    _activate_scheme(scheme.TorchScheme(device))

    base_params = {
        key: value
        for key, value in _NLTIDE_PARAMS.items()
        if key not in ("f0", "amplitude", "n")
    }
    base = get_fd_waveform(approximant="TaylorF2", **base_params)
    base_arrays = tuple(
        series._data.tensor.detach().cpu().numpy().copy() for series in base
    )
    frequencies = np.arange(len(base[0])) * base[0].delta_f
    phase = nltides.nltides_fourier_phase_difference(
        frequencies,
        base[0].delta_f,
        _NLTIDE_PARAMS["f0"],
        _NLTIDE_PARAMS["amplitude"],
        _NLTIDE_PARAMS["n"],
        _NLTIDE_PARAMS["mass1"],
        _NLTIDE_PARAMS["mass2"],
    )
    correction = np.exp(-1.0j * phase)
    reference_arrays = tuple(
        values * correction.astype(values.dtype) for values in base_arrays
    )

    phase_inputs = []
    phase_function = nltides_torch.nltides_fourier_phase_difference

    def recording_phase(frequencies, *args):
        phase_inputs.append(frequencies)
        return phase_function(frequencies, *args)

    def reject_host_transfer(*_args, **_kwargs):
        raise AssertionError("nonlinear-tide correction copied data to host")

    with monkeypatch.context() as patch:
        patch.setattr(nltides, "numpy", _reject_numpy_backend())
        patch.setattr(
            nltides_torch,
            "nltides_fourier_phase_difference",
            recording_phase,
        )
        patch.setattr(TorchArrayData, "numpy", reject_host_transfer)
        actual = get_fd_waveform(
            approximant="TaylorF2NL", **_NLTIDE_PARAMS
        )

    assert len(phase_inputs) == 1
    assert isinstance(phase_inputs[0], torch.Tensor)
    assert phase_inputs[0].device.type == device
    assert all(series._data.tensor.device.type == device for series in actual)

    tolerance = (
        2.0e-6
        if actual[0]._data.tensor.dtype == torch.complex64
        else 1.0e-12
    )
    for expected, series in zip(reference_arrays, actual):
        values = series._data.tensor.detach().cpu().numpy()
        np.testing.assert_array_equal(values == 0.0, expected == 0.0)
        nonzero = np.abs(expected) > 0.0
        # The physical amplitudes underflow when a complex64 norm squares
        # them, so accumulate this diagnostic in complex128.
        values_nonzero = values[nonzero].astype(np.complex128)
        expected_nonzero = expected[nonzero].astype(np.complex128)
        relative_error = np.linalg.norm(
            values_nonzero - expected_nonzero
        ) / np.linalg.norm(expected_nonzero)
        assert relative_error < tolerance
