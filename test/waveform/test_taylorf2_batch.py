"""Contract tests for the explicit Torch TaylorF2 batch API."""

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.types import FrequencySeries  # noqa: E402
from pycbc.waveform import get_fd_waveform, get_fd_waveform_batch  # noqa: E402


def _params(**updates):
    params = {
        "mass1": 1.4,
        "mass2": 1.3,
        "spin1z": 0.02,
        "spin2z": -0.01,
        "delta_f": 1.0,
        "f_lower": 20.0,
        "f_final": 128.0,
        "distance": 100.0,
        "inclination": 0.4,
        "coa_phase": 0.2,
    }
    params.update(updates)
    return params


@pytest.fixture(autouse=True)
def preserve_scheme():
    """Restore PyCBC's process-wide scheme state after every test."""
    old_state = _scheme.mgr.state
    old_lock = _scheme.mgr._lock
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr._lock = False
        _scheme.mgr.state = old_state
        _scheme.mgr._lock = old_lock
        _scheme.Scheme._single = old_single


def _activate(scheme_type, *args):
    _scheme.mgr._lock = False
    _scheme.Scheme._single = None
    state = scheme_type(*args)
    _scheme.mgr.state = state
    return state


def _metadata_array(values):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def _assert_padding_is_exact(result):
    first_bins = _metadata_array(result.first_bins)
    end_bins = _metadata_array(result.end_bins)
    for row, (first_bin, end_bin) in enumerate(zip(first_bins, end_bins)):
        first_bin = int(first_bin)
        end_bin = int(end_bin)
        for polarization in (result.hplus, result.hcross):
            assert torch.equal(
                polarization[row, :first_bin],
                torch.zeros_like(polarization[row, :first_bin]),
            )
            assert torch.equal(
                polarization[row, end_bin:],
                torch.zeros_like(polarization[row, end_bin:]),
            )
            assert torch.count_nonzero(
                polarization[row, first_bin:end_bin]
            ) == end_bin - first_bin


def test_scalar_parameters_broadcast_and_result_contract():
    _activate(_scheme.TorchScheme, "cpu")
    result = get_fd_waveform_batch(
        "TaylorF2",
        **_params(
            mass1=np.array([1.4, 1.5, 1.6]),
            mass2=torch.tensor([1.3, 1.4, 1.2], dtype=torch.float64),
            spin1z=[0.02],
            spin2z=-0.01,
            distance=100.0,
        ),
    )

    hplus, hcross = result
    assert hplus is result.hplus
    assert hcross is result.hcross
    assert hplus.shape == hcross.shape == (3, 129)
    assert hplus.dtype == hcross.dtype == torch.complex128
    assert hplus.device.type == hcross.device.type == "cpu"
    assert result.delta_f == pytest.approx(1.0)
    assert float(result.epoch) == pytest.approx(-1.0)
    np.testing.assert_array_equal(result.first_bins, [20, 20, 20])
    np.testing.assert_array_equal(result.end_bins, [129, 129, 129])
    _assert_padding_is_exact(result)

    singleton = get_fd_waveform_batch("TaylorF2", **_params())
    assert singleton.hplus.shape == singleton.hcross.shape == (1, 129)
    np.testing.assert_array_equal(singleton.first_bins, [20])
    np.testing.assert_array_equal(singleton.end_bins, [129])


@pytest.mark.parametrize(
    "updates",
    [
        {
            "mass1": [1.4, 1.5],
            "mass2": [1.3, 1.4, 1.2],
        },
        {"mass1": np.array([], dtype=np.float64)},
        {"mass1": np.array([[1.4], [1.5]])},
        {"mass1": [1.4, 1.5, 1.6], "f_lower": [20.0, 25.0]},
        {"mass1": [1.4, 1.5], "delta_f": [1.0, 1.0]},
    ],
    ids=(
        "mismatched-physical-parameter-lengths",
        "empty-vector",
        "matrix-parameter",
        "mismatched-cutoff-length",
        "vector-frequency-spacing",
    ),
)
def test_batch_parameter_shapes_are_validated(updates):
    _activate(_scheme.TorchScheme, "cpu")
    with pytest.raises(ValueError):
        get_fd_waveform_batch("TaylorF2", **_params(**updates))


def test_per_row_cutoffs_and_exact_zero_padding():
    delta_f = 0.5
    f_lower = torch.tensor([20.25, 30.0, 40.75], dtype=torch.float64)
    f_final = torch.tensor([80.0, 96.25, 72.75], dtype=torch.float64)
    expected_first = torch.ceil(f_lower / delta_f).to(torch.int64)
    expected_end = torch.floor(f_final / delta_f).to(torch.int64) + 1

    _activate(_scheme.TorchScheme, "cpu")
    result = get_fd_waveform_batch(
        "TaylorF2",
        **_params(
            mass1=[1.4, 1.5, 1.6],
            mass2=[1.3, 1.4, 1.2],
            delta_f=delta_f,
            f_lower=f_lower,
            f_final=f_final,
        ),
    )

    assert result.hplus.shape == result.hcross.shape == (
        3,
        int(expected_end.max()),
    )
    np.testing.assert_array_equal(result.first_bins, expected_first.numpy())
    np.testing.assert_array_equal(result.end_bins, expected_end.numpy())
    _assert_padding_is_exact(result)


def test_each_batch_row_matches_scalar_taylorf2(monkeypatch):
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")
    mass1 = [1.4, 1.8, 2.1]
    mass2 = [1.3, 1.2, 1.1]
    spin1z = [0.02, -0.1, 0.2]
    spin2z = [-0.01, 0.05, -0.15]
    distance = [100.0, 150.0, 230.0]
    inclination = [0.2, 0.7, 1.1]
    coa_phase = [0.0, 0.3, 0.8]
    f_lower = [20.0, 24.5, 31.0]
    f_final = [128.0, 96.0, 80.0]

    _activate(_scheme.TorchScheme, "cpu")
    result = get_fd_waveform_batch(
        "TaylorF2",
        **_params(
            mass1=mass1,
            mass2=mass2,
            spin1z=spin1z,
            spin2z=spin2z,
            distance=distance,
            inclination=inclination,
            coa_phase=coa_phase,
            f_lower=f_lower,
            f_final=f_final,
        ),
    )

    for row in range(len(mass1)):
        scalar_hplus, scalar_hcross = get_fd_waveform(
            approximant="TaylorF2",
            **_params(
                mass1=mass1[row],
                mass2=mass2[row],
                spin1z=spin1z[row],
                spin2z=spin2z[row],
                distance=distance[row],
                inclination=inclination[row],
                coa_phase=coa_phase[row],
                f_lower=f_lower[row],
                f_final=f_final[row],
            ),
        )
        assert isinstance(scalar_hplus, FrequencySeries)
        assert isinstance(scalar_hcross, FrequencySeries)
        assert len(scalar_hplus) == int(_metadata_array(result.end_bins)[row])
        assert scalar_hplus.delta_f == pytest.approx(result.delta_f)
        assert float(scalar_hplus.epoch) == pytest.approx(float(result.epoch))
        torch.testing.assert_close(
            result.hplus[row, : len(scalar_hplus)],
            scalar_hplus._data.tensor,
            rtol=2.0e-10,
            atol=0.0,
        )
        torch.testing.assert_close(
            result.hcross[row, : len(scalar_hcross)],
            scalar_hcross._data.tensor,
            rtol=2.0e-10,
            atol=0.0,
        )

    _assert_padding_is_exact(result)


def test_advanced_batch_rows_match_scalar_taylorf2(monkeypatch):
    """Exercise shared PN orders and vector-valued TaylorF2 extensions."""
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")
    vector_params = {
        "mass1": [1.4, 2.0],
        "mass2": [1.3, 1.6],
        "spin1z": [0.02, 0.1],
        "spin2z": [-0.01, -0.04],
        "distance": [100.0, 150.0],
        "inclination": [0.8, 0.3],
        "coa_phase": [0.2, 0.7],
        "f_ref": [30.0, 25.0],
        "f_lower": [20.0, 22.0],
        "f_final": 0.0,
        "lambda1": [800.0, 300.0],
        "lambda2": [700.0, 100.0],
        "dquad_mon1": [0.0, 2.2],
        "dquad_mon2": [0.0, 1.5],
        "dchi3": [0.0, 0.02],
        "dchi6l": [-0.01, 0.0],
        "long_asc_nodes": [0.37, 0.11],
        "delta_f": 0.5,
        "tidal_order": 15,
    }
    vector_names = {
        name
        for name, value in vector_params.items()
        if isinstance(value, list)
    }

    _activate(_scheme.TorchScheme, "cpu")
    result = get_fd_waveform_batch("TaylorF2", **vector_params)

    for row in range(2):
        scalar_params = {
            name: value[row] if name in vector_names else value
            for name, value in vector_params.items()
        }
        scalar_hplus, scalar_hcross = get_fd_waveform(
            approximant="TaylorF2",
            **scalar_params,
        )
        end_bin = int(_metadata_array(result.end_bins)[row])
        assert len(scalar_hplus) == len(scalar_hcross) == end_bin
        torch.testing.assert_close(
            result.hplus[row, :end_bin],
            scalar_hplus._data.tensor,
            rtol=2.0e-10,
            atol=0.0,
        )
        torch.testing.assert_close(
            result.hcross[row, :end_bin],
            scalar_hcross._data.tensor,
            rtol=2.0e-10,
            atol=0.0,
        )

    _assert_padding_is_exact(result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_batch_matches_cpu():
    params = _params(
        mass1=[1.4, 1.8, 2.1],
        mass2=[1.3, 1.2, 1.1],
        spin1z=[0.02, -0.1, 0.2],
        spin2z=[-0.01, 0.05, -0.15],
        f_lower=[20.0, 24.5, 31.0],
        f_final=[128.0, 96.0, 80.0],
    )

    _activate(_scheme.TorchScheme, "cpu")
    cpu_result = get_fd_waveform_batch("TaylorF2", **params)
    cpu_hplus = cpu_result.hplus.clone()
    cpu_hcross = cpu_result.hcross.clone()

    _activate(_scheme.TorchScheme, "cuda")
    cuda_result = get_fd_waveform_batch("TaylorF2", **params)

    assert cuda_result.hplus.device.type == "cuda"
    assert cuda_result.hcross.device.type == "cuda"
    torch.testing.assert_close(
        cuda_result.hplus.cpu(), cpu_hplus, rtol=2.0e-10, atol=0.0
    )
    torch.testing.assert_close(
        cuda_result.hcross.cpu(), cpu_hcross, rtol=2.0e-10, atol=0.0
    )
    np.testing.assert_array_equal(
        cuda_result.first_bins.cpu(), cpu_result.first_bins
    )
    np.testing.assert_array_equal(
        cuda_result.end_bins.cpu(), cpu_result.end_bins
    )
    _assert_padding_is_exact(cuda_result)


def test_scalar_public_api_remains_a_frequency_series(monkeypatch):
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")
    _activate(_scheme.TorchScheme, "cpu")
    hplus, hcross = get_fd_waveform(
        approximant="TaylorF2",
        **_params(),
    )
    assert isinstance(hplus, FrequencySeries)
    assert isinstance(hcross, FrequencySeries)
    assert hplus.ndim == hcross.ndim == 1


def test_unsupported_approximant_and_scheme_are_explicit_errors():
    _activate(_scheme.TorchScheme, "cpu")
    with pytest.raises(ValueError, match="supports only TaylorF2"):
        get_fd_waveform_batch("NotRegistered", **_params())

    _activate(_scheme.CPUScheme)
    with pytest.raises(RuntimeError, match="Torch"):
        get_fd_waveform_batch("TaylorF2", **_params())


def test_distance_gradient_matches_inverse_distance_scaling():
    distance = torch.tensor(
        [80.0, 130.0, 250.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    _activate(_scheme.TorchScheme, "cpu")
    result = get_fd_waveform_batch(
        "TaylorF2",
        **_params(
            mass1=[1.4, 1.5, 1.6],
            mass2=[1.3, 1.4, 1.2],
            distance=distance,
        ),
    )

    power = (
        result.hplus.abs().square() + result.hcross.abs().square()
    ).sum(dim=-1)
    power.sum().backward()

    assert distance.grad is not None
    assert torch.isfinite(distance.grad).all()
    assert torch.count_nonzero(distance.grad) == distance.numel()
    expected = -2.0 * power.detach() / distance.detach()
    torch.testing.assert_close(
        distance.grad,
        expected,
        rtol=2.0e-10,
        atol=0.0,
    )


def test_mass_spin_and_phase_gradients_pass_gradcheck():
    mass1 = torch.tensor(
        [1.4, 1.8], dtype=torch.float64, requires_grad=True
    )
    spin1z = torch.tensor(
        [0.02, -0.1], dtype=torch.float64, requires_grad=True
    )
    coa_phase = torch.tensor(
        [0.2, 0.7], dtype=torch.float64, requires_grad=True
    )

    _activate(_scheme.TorchScheme, "cpu")

    def scaled_loss(mass, spin, phase):
        result = get_fd_waveform_batch(
            "TaylorF2",
            **_params(
                mass1=mass,
                mass2=[1.3, 1.2],
                spin1z=spin,
                spin2z=[-0.01, 0.05],
                coa_phase=phase,
            ),
        )
        loss = (
            result.hplus.real
            + 0.2 * result.hplus.imag
            + 0.3 * result.hcross.real
            - 0.1 * result.hcross.imag
        ).sum()
        return loss * 1.0e22

    assert torch.autograd.gradcheck(
        scaled_loss,
        (mass1, spin1z, coa_phase),
        eps=1.0e-6,
        atol=2.0e-4,
        rtol=2.0e-4,
        fast_mode=True,
    )
    scaled_loss(mass1, spin1z, coa_phase).backward()
    for parameter in (mass1, spin1z, coa_phase):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) == parameter.numel()


def test_phasing_tensor_subclass_preserves_values_and_gradients():
    from pycbc.waveform.taylorf2_torch import taylorf2_aligned_phasing

    class PhysicalTensor(torch.Tensor):
        pass

    mass = torch.tensor([1.4, 1.8], dtype=torch.float64, requires_grad=True)
    spin = torch.tensor([0.02, -0.1], dtype=torch.float64, requires_grad=True)
    reference = taylorf2_aligned_phasing(mass, 1.2, spin, 0.05)
    actual = taylorf2_aligned_phasing(
        mass.as_subclass(PhysicalTensor), 1.2,
        spin.as_subclass(PhysicalTensor), 0.05,
    )
    for name in ("v", "vlogv", "vlogvsq"):
        torch.testing.assert_close(
            getattr(actual, name), getattr(reference, name)
        )
    actual_grads = torch.autograd.grad(actual.v.sum(), (mass, spin))
    expected_grads = torch.autograd.grad(reference.v.sum(), (mass, spin))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert torch.isfinite(actual_grad).all()
        assert torch.count_nonzero(actual_grad) == actual_grad.numel()
        torch.testing.assert_close(actual_grad, expected_grad)
