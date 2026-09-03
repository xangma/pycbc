import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import lal_compat as _lal  # noqa: E402
from pycbc import scheme as _scheme  # noqa: E402
from pycbc.types import Array  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomxphm_torch import (  # noqa: E402
    _wigner_columns,
    imrphenomxphm_native_supported,
    imrphenomxphm_sequence_native_supported,
)


_MSA_FLAGS = dict(
    phenom_x_prec_version=223,
    phenom_xp_convention=1,
    phenom_xp_final_spin_mod=0,
)
_MSA_FINAL_SPIN_FLAGS = dict(_MSA_FLAGS, phenom_xp_final_spin_mod=3)
_MSA_ALIAS_FLAGS = dict(
    _MSA_FLAGS,
    phenom_x_prec_version=300,
    phenom_xp_final_spin_mod=4,
)
_NATIVE_MODELS = (
    {},
    _MSA_FLAGS,
    _MSA_FINAL_SPIN_FLAGS,
    _MSA_ALIAS_FLAGS,
)

_SEQUENCE_PARAMS = dict(
    mass1=12.0,
    mass2=35.0,
    spin1x=0.15,
    spin1y=-0.25,
    spin1z=0.4,
    spin2x=0.05,
    spin2y=0.2,
    spin2z=-0.3,
    distance=320.0,
    inclination=1.1,
    coa_phase=0.0,
    long_asc_nodes=-0.4,
    f_ref=0.0,
)
_SAMPLE_POINTS = [17.3, 22.0, 50.0, 150.0, 400.0, 850.0, 1000.0, 1500.0]
_NATIVE_FLAG_ENVS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_IMRPHENOMXPHM_NATIVE",
)


def _old_lalsimulation_reference(module):
    try:
        version = tuple(
            int(part)
            for part in module.__version__.split("+", 1)[0].split(".")[:3]
        )
    except (AttributeError, TypeError, ValueError):
        version = ()
    return bool(version) and version <= (5, 3, 1)


def _clear_native_flags(monkeypatch):
    """Remove every native flag so the registry default applies."""
    for name in _NATIVE_FLAG_ENVS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def preserve_scheme():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _activate_scheme(scheme):
    _scheme.Scheme._single = None
    _scheme.mgr.state = scheme


def _relative_error(actual, expected):
    nonzero = np.abs(expected) > 0.0
    assert nonzero.any()
    return np.linalg.norm(actual[nonzero] - expected[nonzero]) / np.linalg.norm(
        expected[nonzero]
    )


@pytest.mark.parametrize("mode", [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)])
def test_imrphenomxphm_wigner_columns_are_orthonormal(mode):
    beta = torch.tensor([0.1, 0.7, 1.4], dtype=torch.float64)
    positive, negative = _wigner_columns(
        *mode,
        torch.cos(beta / 2.0),
        torch.sin(beta / 2.0),
    )
    ell, _ = mode

    assert len(positive) == 2 * ell + 1
    assert len(negative) == 2 * ell + 1
    positive = torch.stack(positive)
    negative = torch.stack(negative)
    torch.testing.assert_close(
        torch.sum(positive * positive, dim=0),
        torch.ones_like(beta),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        torch.sum(negative * negative, dim=0),
        torch.ones_like(beta),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    torch.testing.assert_close(
        torch.sum(positive * negative, dim=0),
        torch.zeros_like(beta),
        rtol=0.0,
        atol=2.0e-14,
    )


@pytest.mark.parametrize(
    "model_flags",
    _NATIVE_MODELS,
    ids=("default", "msa-final-spin-0", "msa-final-spin-3", "msa-v300-alias"),
)
def test_imrphenomxphm_sequence_matches_lalsimulation(
    model_flags,
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **model_flags,
        **_SEQUENCE_PARAMS,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXPHM sequence called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **model_flags,
        **_SEQUENCE_PARAMS,
    )

    for result in actual:
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
    if (
        model_flags == _MSA_FLAGS
        and _old_lalsimulation_reference(lalsimulation)
    ):
        pytest.skip(
            "installed lalsimulation predates the IMRPhenomXPHM "
            "MSA final-spin-0 sequence reference used by the native port"
        )

    for expected, result in zip(reference_arrays, actual):
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < 5.0e-5


def test_imrphenomxphm_regular_grid_matches_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=40.0,
        mass2=20.0,
        spin1x=0.2,
        spin1y=0.1,
        spin1z=0.3,
        spin2x=-0.1,
        spin2y=0.05,
        spin2z=-0.2,
        distance=500.0,
        inclination=0.7,
        coa_phase=1.2,
        long_asc_nodes=0.3,
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
        f_ref=30.0,
        phase_order=2.5,
        amplitude_order="3",
        spin_order=4.5,
        tidal_order=0,
        eccentricity_order=4,
    )
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXPHM", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXPHM called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomXPHM", **params)

    for expected, expected_array, result in zip(
        reference,
        reference_arrays,
        actual,
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected_array == 0.0)
        assert _relative_error(result_array, expected_array) < 7.0e-4


@pytest.mark.parametrize(
    "model_flags",
    _NATIVE_MODELS,
    ids=("default", "msa-final-spin-0", "msa-final-spin-3", "msa-v300-alias"),
)
def test_imrphenomxphm_phase_plan_is_request_local_and_bitwise(
    model_flags,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxas_torch as xas_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    params = dict(
        **_SEQUENCE_PARAMS,
        **model_flags,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_INTRINSIC_CACHE", "0")

    prepare_calls = 0
    original_prepare = xas_torch._prepare_phase_plan

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(xas_torch, "_prepare_phase_plan", counted_prepare)
    monkeypatch.setenv("PYCBC_IMRPHENOMX_PHASE_PLAN", "0")
    eager = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert prepare_calls == 0

    monkeypatch.setenv("PYCBC_IMRPHENOMX_PHASE_PLAN", "1")
    planned = xphm_torch.imrphenomxphm_fd_torch(**params)

    assert prepare_calls == 1
    for eager_series, planned_series in zip(eager, planned):
        assert len(eager_series) == len(planned_series)
        assert eager_series.delta_f == planned_series.delta_f
        assert float(eager_series.epoch) == float(planned_series.epoch)
        assert torch.equal(
            eager_series._data.tensor,
            planned_series._data.tensor,
        )


def test_imrphenomxphm_mode32_amp_plan_is_request_local_and_bitwise(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxhm_mode32_torch as mode32_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    params = dict(
        **_SEQUENCE_PARAMS,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_INTRINSIC_CACHE", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMX_PHASE_PLAN", "1")
    monkeypatch.setenv(mode32_torch._DERIVATIVE_REGION_SPECIALIZATION_ENV, "0")
    monkeypatch.setenv(mode32_torch._ANALYTIC_PHASE_DERIVATIVES_ENV, "0")

    prepare_calls = 0
    original_prepare = mode32_torch._prepare_mergerringdown_amp_plan

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        mode32_torch,
        "_prepare_mergerringdown_amp_plan",
        counted_prepare,
    )
    monkeypatch.setenv(mode32_torch._AMP_PLAN_ENV, "0")
    eager = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert prepare_calls == 0

    monkeypatch.setenv(mode32_torch._AMP_PLAN_ENV, "1")
    planned = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert prepare_calls == 1

    for eager_series, planned_series in zip(eager, planned):
        assert len(eager_series) == len(planned_series)
        assert eager_series.delta_f == planned_series.delta_f
        assert float(eager_series.epoch) == float(planned_series.epoch)
        assert torch.equal(eager_series._data.tensor, planned_series._data.tensor)


def test_mode32_amp_plan_switch_is_strict_and_defaults_on(monkeypatch):
    import pycbc.waveform.imrphenomxhm_mode32_torch as mode32_torch

    monkeypatch.delenv(mode32_torch._AMP_PLAN_ENV, raising=False)
    assert mode32_torch._amp_plan_enabled()
    monkeypatch.setenv(mode32_torch._AMP_PLAN_ENV, "0")
    assert not mode32_torch._amp_plan_enabled()
    monkeypatch.setenv(mode32_torch._AMP_PLAN_ENV, "maybe")
    with pytest.raises(ValueError, match=mode32_torch._AMP_PLAN_ENV):
        mode32_torch._amp_plan_enabled()


def test_imrphenomxphm_scalar_region_dispatch_is_bitwise_and_reduces_grad_calls(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxas_torch as xas_torch
    import pycbc.waveform.imrphenomxhm_mode32_torch as mode32_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    params = dict(
        **_SEQUENCE_PARAMS,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setenv(xphm_torch._INTRINSIC_CACHE_ENV, "0")
    monkeypatch.setenv(xas_torch._PHASE_PLAN_ENV, "1")
    monkeypatch.setenv(mode32_torch._AMP_PLAN_ENV, "1")
    monkeypatch.setenv(xas_torch._EXACT_SCALAR_DERIVATIVES_ENV, "0")
    monkeypatch.setenv(xas_torch._EXACT_SCALAR_AMP_DERIVATIVES_ENV, "0")
    monkeypatch.setenv(mode32_torch._ANALYTIC_PHASE_DERIVATIVES_ENV, "0")
    monkeypatch.setenv(mode32_torch._DERIVATIVE_REGION_SPECIALIZATION_ENV, "0")
    monkeypatch.setenv(mode32_torch._SCRIPTED_ANALYTIC_PHASE_TAIL_ENV, "0")

    original_grad = torch.autograd.grad
    grad_calls = 0

    def counted_grad(*args, **kwargs):
        nonlocal grad_calls
        grad_calls += 1
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    monkeypatch.setenv(xas_torch._SCALAR_REGION_DISPATCH_ENV, "0")
    dense = xphm_torch.imrphenomxphm_fd_torch(**params)
    dense_grad_calls = grad_calls

    grad_calls = 0
    monkeypatch.setenv(xas_torch._SCALAR_REGION_DISPATCH_ENV, "1")
    dispatched = xphm_torch.imrphenomxphm_fd_torch(**params)
    dispatched_grad_calls = grad_calls

    assert dense_grad_calls == 30
    assert dispatched_grad_calls == 18
    for dense_series, dispatched_series in zip(dense, dispatched):
        assert len(dense_series) == len(dispatched_series)
        assert dense_series.delta_f == dispatched_series.delta_f
        assert float(dense_series.epoch) == float(dispatched_series.epoch)
        assert torch.equal(
            dense_series._data.tensor.contiguous().view(torch.int64),
            dispatched_series._data.tensor.contiguous().view(torch.int64),
        )


def test_imrphenomxphm_phase_anchor_cache_is_request_local_and_bitwise(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxas_torch as xas_torch
    import pycbc.waveform.imrphenomxhm_mode21_torch as mode21_torch
    import pycbc.waveform.imrphenomxhm_mode32_torch as mode32_torch
    import pycbc.waveform.imrphenomxhm_mode33_torch as mode33_torch
    import pycbc.waveform.imrphenomxhm_mode44_torch as mode44_torch
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    if not xhm_torch._plain_request_runtime_supported():
        pytest.skip("carrier-phase cache is unavailable on this Torch runtime")

    params = dict(
        **_SEQUENCE_PARAMS,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    monkeypatch.setenv(xphm_torch._INTRINSIC_CACHE_ENV, "0")
    monkeypatch.setenv(xas_torch._PHASE_PLAN_ENV, "1")
    monkeypatch.setenv(xas_torch._SCALAR_REGION_DISPATCH_ENV, "1")
    monkeypatch.setenv(mode32_torch._AMP_PLAN_ENV, "1")
    monkeypatch.setenv(xhm_torch._BATCHED_TINY_SOLVES_ENV, "0")
    monkeypatch.setenv(
        "PYCBC_IMRPHENOMXHM_FIXED_SCHEMA_AMPLITUDE_TRIPLET",
        "0",
    )
    monkeypatch.setenv(xas_torch._EXACT_SCALAR_DERIVATIVES_ENV, "0")
    monkeypatch.setenv(xas_torch._EXACT_SCALAR_AMP_DERIVATIVES_ENV, "0")

    counts = {"phase": 0, "derivative": 0, "grad": 0}
    original_phase = xas_torch.Phase
    original_derivative = xas_torch.PhaseDerivative
    original_grad = torch.autograd.grad

    def counted_phase(*args, **kwargs):
        counts["phase"] += 1
        return original_phase(*args, **kwargs)

    def counted_derivative(*args, **kwargs):
        counts["derivative"] += 1
        return original_derivative(*args, **kwargs)

    def counted_grad(*args, **kwargs):
        counts["grad"] += 1
        return original_grad(*args, **kwargs)

    for module in (mode21_torch, mode32_torch, mode33_torch, mode44_torch):
        monkeypatch.setattr(module, "Phase", counted_phase)
        monkeypatch.setattr(module, "PhaseDerivative", counted_derivative)
    monkeypatch.setattr(torch.autograd, "grad", counted_grad)

    monkeypatch.setenv(xhm_torch._PHASE_ANCHOR_CACHE_ENV, "0")
    eager = xphm_torch.imrphenomxphm_fd_torch(**params)
    eager_counts = dict(counts)

    counts.update(phase=0, derivative=0, grad=0)
    monkeypatch.setenv(xhm_torch._PHASE_ANCHOR_CACHE_ENV, "1")
    cached = xphm_torch.imrphenomxphm_fd_torch(**params)
    cached_counts = dict(counts)

    counts.update(phase=0, derivative=0, grad=0)
    cached_again = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert counts == cached_counts
    assert eager_counts["phase"] - cached_counts["phase"] == 5
    assert eager_counts["derivative"] - cached_counts["derivative"] == 3
    assert eager_counts["grad"] - cached_counts["grad"] == 3

    for eager_series, cached_series, repeated_series in zip(
        eager,
        cached,
        cached_again,
    ):
        eager_bits = eager_series._data.tensor.contiguous().view(torch.int64)
        assert torch.equal(
            eager_bits,
            cached_series._data.tensor.contiguous().view(torch.int64),
        )
        assert torch.equal(
            eager_bits,
            repeated_series._data.tensor.contiguous().view(torch.int64),
        )


@pytest.mark.parametrize("model", ("XHM", "XPHM"))
@pytest.mark.parametrize("sequence", (False, True), ids=("fd", "sequence"))
@pytest.mark.parametrize(
    "equal_mass",
    (False, True),
    ids=("unequal_mass", "equal_mass"),
)
def test_phase_anchor_cache_eager_exact_matrix_with_tiny_solves_off(
    model,
    sequence,
    equal_mass,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    if not xhm_torch._plain_request_runtime_supported():
        pytest.skip("carrier-phase cache is unavailable on this Torch runtime")

    monkeypatch.setenv(xhm_torch._BATCHED_TINY_SOLVES_ENV, "0")
    monkeypatch.setenv(
        "PYCBC_IMRPHENOMXHM_FIXED_SCHEMA_AMPLITUDE_TRIPLET",
        "0",
    )
    params = dict(
        approximant=f"IMRPhenom{model}",
        mass1=35.0 if equal_mass else 12.0,
        mass2=35.0,
        spin1z=0.2,
        spin2z=-0.1,
        distance=500.0,
        inclination=1.1,
        coa_phase=0.3,
        f_ref=25.0,
        mode_array=[(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)],
    )
    if model == "XPHM":
        params.update(
            spin1x=0.1,
            spin1y=-0.2,
            spin2x=0.05,
            spin2y=0.1,
        )
        generator = (
            xphm_torch.imrphenomxphm_fd_sequence_torch
            if sequence
            else xphm_torch.imrphenomxphm_fd_torch
        )
    else:
        generator = (
            xhm_torch.imrphenomxhm_fd_sequence_torch
            if sequence
            else xhm_torch.imrphenomxhm_fd_torch
        )
    if sequence:
        params["sample_points"] = [17.3, 22.0, 50.0, 150.0, 400.0]
    else:
        params.update(delta_f=1.0, f_lower=20.0, f_final=256.0)

    created_anchors = []
    original_anchor_type = xhm_torch._CarrierPhaseAnchors

    class RecordingAnchors(original_anchor_type):
        def __init__(self):
            super().__init__()
            created_anchors.append(self)

    monkeypatch.setattr(xhm_torch, "_CarrierPhaseAnchors", RecordingAnchors)
    monkeypatch.setenv(xhm_torch._PHASE_ANCHOR_CACHE_ENV, "0")
    reference = generator(**params)
    assert created_anchors == []

    monkeypatch.setenv(xhm_torch._PHASE_ANCHOR_CACHE_ENV, "1")
    actual = generator(**params)
    assert len(created_anchors) == 1

    for expected, result in zip(reference, actual):
        expected_tensor = expected._data.tensor
        result_tensor = result._data.tensor
        assert torch.equal(
            expected_tensor.contiguous().view(torch.int64),
            result_tensor.contiguous().view(torch.int64),
        )
        if not sequence:
            assert result.delta_f == expected.delta_f
            assert float(result.epoch) == float(expected.epoch)


def test_imrphenomxphm_twist_reuse_switch_is_strict_and_defaults_off(
    monkeypatch,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    monkeypatch.delenv(xphm_torch._TWIST_REUSE_ENV, raising=False)
    assert not xphm_torch._twist_reuse_enabled()
    monkeypatch.setenv(xphm_torch._TWIST_REUSE_ENV, "maybe")
    with pytest.raises(ValueError, match=xphm_torch._TWIST_REUSE_ENV):
        xphm_torch._twist_reuse_enabled()


def test_imrphenomxphm_stacked_twist_switch_is_strict_and_defaults_off(
    monkeypatch,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    monkeypatch.delenv(xphm_torch._STACKED_TWIST_ENV, raising=False)
    assert not xphm_torch._stacked_twist_enabled()
    monkeypatch.setenv(xphm_torch._STACKED_TWIST_ENV, "maybe")
    with pytest.raises(ValueError, match=xphm_torch._STACKED_TWIST_ENV):
        xphm_torch._stacked_twist_enabled()


def test_imrphenomxphm_bulk_twist_exponentials_switch_is_strict_and_off(
    monkeypatch,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    monkeypatch.delenv(
        xphm_torch._BULK_TWIST_EXPONENTIALS_ENV,
        raising=False,
    )
    assert not xphm_torch._bulk_twist_exponentials_enabled()
    monkeypatch.setenv(xphm_torch._BULK_TWIST_EXPONENTIALS_ENV, "maybe")
    with pytest.raises(
        ValueError,
        match=xphm_torch._BULK_TWIST_EXPONENTIALS_ENV,
    ):
        xphm_torch._bulk_twist_exponentials_enabled()


def test_imrphenomxphm_twist_exponential_recurrence_switch_is_strict_and_off(
    monkeypatch,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    monkeypatch.delenv(
        xphm_torch._TWIST_EXPONENTIAL_RECURRENCE_ENV,
        raising=False,
    )
    assert not xphm_torch._twist_exponential_recurrence_enabled()
    monkeypatch.setenv(
        xphm_torch._TWIST_EXPONENTIAL_RECURRENCE_ENV,
        "maybe",
    )
    with pytest.raises(
        ValueError,
        match=xphm_torch._TWIST_EXPONENTIAL_RECURRENCE_ENV,
    ):
        xphm_torch._twist_exponential_recurrence_enabled()


@pytest.mark.parametrize("real_dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("ell", (1, 2, 3, 4))
def test_imrphenomxphm_twist_exponential_recurrence_matches_lal_arithmetic(
    ell,
    real_dtype,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    alpha = torch.linspace(-17.25, 11.75, 1009, dtype=real_dtype)
    alpha[504] = -0.0
    packed_ell, negative_rows, positive_rows = (
        xphm_torch._twist_exponential_recurrence(alpha, ell)
    )
    assert packed_ell == ell

    exp_i_alpha = torch.exp(1j * alpha)
    exp_mi_alpha = 1.0 / exp_i_alpha
    expected_positive = [torch.ones_like(exp_i_alpha), exp_i_alpha]
    expected_negative = [torch.ones_like(exp_i_alpha), exp_mi_alpha]
    for _ in range(2, ell + 1):
        expected_positive.append(exp_i_alpha * expected_positive[-1])
        expected_negative.append(exp_mi_alpha * expected_negative[-1])
    expected_rows = (
        tuple(reversed(expected_negative[1:]))
        + (expected_positive[0],)
        + tuple(expected_positive[1:])
    )

    for actual, expected in zip(positive_rows, expected_rows):
        assert torch.equal(actual, expected)
    for actual, expected in zip(negative_rows, reversed(expected_rows)):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("real_dtype", "word_dtype"),
    (
        (torch.float32, torch.int32),
        (torch.float64, torch.int64),
    ),
    ids=("float32", "float64"),
)
@pytest.mark.parametrize("ell", (1, 2, 3, 4))
def test_imrphenomxphm_bulk_twist_exponential_rows_are_bitwise(
    ell,
    real_dtype,
    word_dtype,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    alpha = torch.linspace(-17.25, 11.75, 1009, dtype=real_dtype)
    alpha[504] = -0.0
    packed = xphm_torch._bulk_twist_exponentials(alpha, ell)
    packed_ell, negative_rows, positive_rows = packed
    assert packed_ell == ell
    for row, emm in enumerate(range(-ell, ell + 1)):
        scalar_negative = torch.exp(-1j * emm * alpha)
        scalar_positive = torch.exp(1j * emm * alpha)
        assert torch.equal(
            scalar_negative.contiguous().view(word_dtype),
            negative_rows[row].contiguous().view(word_dtype),
        )
        assert torch.equal(
            scalar_positive.contiguous().view(word_dtype),
            positive_rows[row].contiguous().view(word_dtype),
        )


@pytest.mark.parametrize(
    ("complex_dtype", "word_dtype"),
    (
        (torch.complex64, torch.int32),
        (torch.complex128, torch.int64),
    ),
    ids=("complex64", "complex128"),
)
def test_imrphenomxphm_stacked_twist_reduction_is_bitwise(
    complex_dtype,
    word_dtype,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    real_dtype = (
        torch.float32 if complex_dtype == torch.complex64 else torch.float64
    )
    values = torch.linspace(-3.25, 2.75, 9 * 1009, dtype=real_dtype).reshape(
        9,
        1009,
    )
    terms = torch.complex(values, torch.sin(values * 1.7)).to(complex_dtype)
    scalar = torch.zeros_like(terms[0])
    for row in terms:
        scalar += row

    stacked = xphm_torch._ordered_stacked_twist_sum(terms)
    assert torch.equal(
        scalar.contiguous().view(word_dtype),
        stacked.contiguous().view(word_dtype),
    )


@pytest.mark.parametrize(
    ("real_dtype", "complex_dtype", "word_dtype"),
    (
        (torch.float32, torch.complex64, torch.int32),
        (torch.float64, torch.complex128, torch.int64),
    ),
    ids=("float32", "float64"),
)
def test_imrphenomxphm_stacked_twist_modes_are_bitwise(
    real_dtype,
    complex_dtype,
    word_dtype,
    preserve_scheme,
):
    from dataclasses import replace

    import pycbc.waveform.imrphenomxp_torch as xp_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = xphm_torch._xp_params(
        {
            **_SEQUENCE_PARAMS,
            "approximant": "IMRPhenomXPHM",
            "f_lower": 20.0,
        }
    )
    inputs = replace(
        xp_torch._validated_inputs(params),
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    model = xp_torch._build_model(inputs)
    frequencies = torch.linspace(20.0, 512.0, 1009, dtype=real_dtype)
    samples = torch.complex(
        torch.sin(frequencies * 0.013),
        torch.cos(frequencies * 0.017),
    ).to(complex_dtype)
    packed_harmonics = xphm_torch._packed_twist_harmonics(
        model,
        xphm_torch._COPRECESSING_MODES,
        frequencies.device,
    )
    assert all(type(value) is torch.Tensor for value in packed_harmonics.values())

    for ell, mprime in xphm_torch._COPRECESSING_MODES:
        angles = xphm_torch._mode_angles(model, frequencies, mprime)
        packed_exponentials = xphm_torch._packed_twist_exponentials(
            angles[0],
            ell,
        )
        assert all(
            type(value) is torch.Tensor for value in packed_exponentials[1:]
        )
        scalar = xphm_torch._twist_mode(
            model,
            frequencies,
            samples,
            ell,
            mprime,
            mode_angles=angles,
        )
        stacked = xphm_torch._stacked_twist_mode(
            model,
            frequencies,
            samples,
            ell,
            mprime,
            angles,
            packed_harmonics[ell],
            packed_exponentials,
        )
        assert stacked is not None
        for scalar_polarization, stacked_polarization in zip(scalar, stacked):
            assert torch.equal(
                scalar_polarization.contiguous().view(word_dtype),
                stacked_polarization.contiguous().view(word_dtype),
            )


def test_imrphenomxphm_stacked_twist_guards_and_scalar_fallback(
    preserve_scheme,
):
    from dataclasses import replace

    import pycbc.waveform.imrphenomxp_torch as xp_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = xphm_torch._xp_params(
        {
            **_SEQUENCE_PARAMS,
            "approximant": "IMRPhenomXPHM",
            "f_lower": 20.0,
        }
    )
    model = xp_torch._build_model(xp_torch._validated_inputs(params))
    frequencies = torch.linspace(20.0, 512.0, 257, dtype=torch.float64)
    samples = torch.complex(
        torch.sin(frequencies * 0.013),
        torch.cos(frequencies * 0.017),
    )
    active_modes = {(2, 2): samples}
    assert (
        xphm_torch._stacked_twist_request_device(
            model,
            frequencies,
            active_modes,
        )
        == frequencies.device
    )

    tensor_subclass = type("StackedTwistTensor", (torch.Tensor,), {})
    assert xphm_torch._stacked_twist_request_device(
        model,
        frequencies.as_subclass(tensor_subclass),
        active_modes,
    ) is None
    assert xphm_torch._stacked_twist_request_device(
        model,
        frequencies,
        {(2, 2): samples.to_sparse()},
    ) is None
    assert xphm_torch._stacked_twist_request_device(
        model,
        frequencies,
        {(2, 2): torch.conj(samples)},
    ) is None
    assert xphm_torch._stacked_twist_request_device(
        model,
        frequencies,
        {(2, 2): torch._neg_view(samples)},
    ) is None
    assert xphm_torch._stacked_twist_request_device(
        model,
        frequencies,
        {(2, 2): samples.detach().requires_grad_()},
    ) is None
    bad_model = replace(
        model,
        harmonics=(
            model.harmonics[0].as_subclass(tensor_subclass),
            *model.harmonics[1:],
        ),
    )
    assert xphm_torch._stacked_twist_request_device(
        bad_model,
        frequencies,
        active_modes,
    ) is None
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(
            frequencies,
            torch.ones_like(frequencies),
        )
        assert xphm_torch._stacked_twist_request_device(
            model,
            dual,
            active_modes,
        ) is None

    angles = xphm_torch._mode_angles(model, frequencies, 2)
    packed_harmonics = xphm_torch._packed_twist_harmonics(
        model,
        ((2, 2),),
        frequencies.device,
    )[2]
    packed_exponentials = xphm_torch._packed_twist_exponentials(
        angles[0],
        2,
    )
    scalar = xphm_torch._twist_mode(
        model,
        frequencies,
        samples,
        2,
        2,
        mode_angles=angles,
    )
    fallback = xphm_torch._twist_mode(
        model,
        frequencies,
        samples,
        2,
        2,
        mode_angles=angles,
        stacked_twist=True,
        packed_harmonics=packed_harmonics.as_subclass(tensor_subclass),
        packed_exponentials=packed_exponentials,
    )
    for scalar_polarization, fallback_polarization in zip(scalar, fallback):
        assert torch.equal(
            scalar_polarization.contiguous().view(torch.int64),
            fallback_polarization.contiguous().view(torch.int64),
        )


@pytest.mark.parametrize(
    "case_params",
    (
        {},
        {
            **_MSA_FLAGS,
            "mass1": 40.0,
            "mass2": 20.0,
            "spin1x": -0.3,
            "spin1y": 0.1,
            "spin1z": 0.6,
            "spin2x": 0.2,
            "spin2y": -0.1,
            "spin2z": 0.2,
        },
        {
            **_MSA_FINAL_SPIN_FLAGS,
            "mass1": 55.0,
            "mass2": 9.0,
            "spin1x": 0.05,
            "spin1y": 0.35,
            "spin1z": -0.2,
            "spin2x": -0.1,
            "spin2y": 0.08,
            "spin2z": 0.5,
        },
        {
            **_MSA_ALIAS_FLAGS,
            "mass1": 30.0,
            "mass2": 30.0,
            "spin1x": 0.2,
            "spin1y": -0.15,
            "spin1z": 0.3,
            "spin2x": -0.25,
            "spin2y": 0.1,
            "spin2z": -0.35,
        },
    ),
    ids=("default", "final-spin-0", "final-spin-3", "v300-final-spin-4"),
)
def test_imrphenomxphm_stacked_twist_full_waveform_is_bitwise(
    case_params,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = {
        **_SEQUENCE_PARAMS,
        **case_params,
        "approximant": "IMRPhenomXPHM",
        "delta_f": 0.5,
        "f_lower": 20.0,
        "f_final": 512.0,
    }
    monkeypatch.setenv(xphm_torch._INTRINSIC_CACHE_ENV, "0")
    calls = 0
    original_stacked = xphm_torch._stacked_twist_mode

    def counted_stacked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_stacked(*args, **kwargs)

    monkeypatch.setattr(xphm_torch, "_stacked_twist_mode", counted_stacked)
    monkeypatch.setenv(xphm_torch._STACKED_TWIST_ENV, "0")
    scalar = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == 0

    monkeypatch.setenv(xphm_torch._STACKED_TWIST_ENV, "1")
    stacked = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == len(xphm_torch._COPRECESSING_MODES)
    for scalar_series, stacked_series in zip(scalar, stacked):
        assert torch.equal(
            scalar_series._data.tensor.contiguous().view(torch.int64),
            stacked_series._data.tensor.contiguous().view(torch.int64),
        )


def test_imrphenomxphm_stacked_twist_shares_bulk_exponential_pack(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = {
        **_SEQUENCE_PARAMS,
        "approximant": "IMRPhenomXPHM",
        "delta_f": 0.5,
        "f_lower": 20.0,
        "f_final": 512.0,
    }
    monkeypatch.setenv(xphm_torch._INTRINSIC_CACHE_ENV, "0")
    monkeypatch.setenv(xphm_torch._BULK_TWIST_EXPONENTIALS_ENV, "1")
    calls = 0
    original_packed = xphm_torch._packed_twist_exponentials

    def counted_packed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_packed(*args, **kwargs)

    monkeypatch.setattr(
        xphm_torch,
        "_packed_twist_exponentials",
        counted_packed,
    )
    monkeypatch.setenv(xphm_torch._STACKED_TWIST_ENV, "0")
    scalar = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == len({mode[1] for mode in xphm_torch._COPRECESSING_MODES})

    calls = 0
    monkeypatch.setenv(xphm_torch._STACKED_TWIST_ENV, "1")
    stacked = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == len({mode[1] for mode in xphm_torch._COPRECESSING_MODES})
    for scalar_series, stacked_series in zip(scalar, stacked):
        assert torch.equal(
            scalar_series._data.tensor.contiguous().view(torch.int64),
            stacked_series._data.tensor.contiguous().view(torch.int64),
        )


def test_imrphenomxphm_stacked_twist_defers_to_cpu_twist_reuse(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = {
        **_SEQUENCE_PARAMS,
        "approximant": "IMRPhenomXPHM",
        "delta_f": 0.5,
        "f_lower": 20.0,
        "f_final": 512.0,
    }
    monkeypatch.setenv(xphm_torch._INTRINSIC_CACHE_ENV, "0")
    monkeypatch.setenv(xphm_torch._TWIST_REUSE_ENV, "1")
    monkeypatch.setenv(xphm_torch._BULK_TWIST_HARMONICS_ENV, "1")
    calls = 0
    original_stacked = xphm_torch._stacked_twist_mode

    def counted_stacked(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_stacked(*args, **kwargs)

    monkeypatch.setattr(
        xphm_torch,
        "_stacked_twist_mode",
        counted_stacked,
    )
    monkeypatch.setenv(xphm_torch._STACKED_TWIST_ENV, "0")
    reused = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == 0

    monkeypatch.setenv(xphm_torch._STACKED_TWIST_ENV, "1")
    stacked_gate = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == 0
    for reused_series, stacked_series in zip(reused, stacked_gate):
        assert torch.equal(
            reused_series._data.tensor.contiguous().view(torch.int64),
            stacked_series._data.tensor.contiguous().view(torch.int64),
        )


def test_imrphenomxphm_bulk_twist_exponentials_are_request_local_and_bitwise(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    params = dict(
        **_SEQUENCE_PARAMS,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setenv(xphm_torch._MODE_ANGLE_REUSE_ENV, "1")
    monkeypatch.setenv(xphm_torch._TWIST_REUSE_ENV, "1")
    monkeypatch.setenv(xphm_torch._BULK_TWIST_HARMONICS_ENV, "1")

    calls = 0
    original_bulk = xphm_torch._bulk_twist_exponentials

    def counted_bulk(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_bulk(*args, **kwargs)

    monkeypatch.setattr(xphm_torch, "_bulk_twist_exponentials", counted_bulk)
    monkeypatch.setenv(xphm_torch._BULK_TWIST_EXPONENTIALS_ENV, "0")
    scalar = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == 0

    monkeypatch.setenv(xphm_torch._BULK_TWIST_EXPONENTIALS_ENV, "1")
    packed = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == 4

    calls = 0
    packed_again = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == 4
    for scalar_series, packed_series, repeated_series in zip(
        scalar,
        packed,
        packed_again,
    ):
        scalar_bits = scalar_series._data.tensor.contiguous().view(torch.int64)
        assert torch.equal(
            scalar_bits,
            packed_series._data.tensor.contiguous().view(torch.int64),
        )
        assert torch.equal(
            scalar_bits,
            repeated_series._data.tensor.contiguous().view(torch.int64),
        )


def test_imrphenomxphm_bulk_twist_harmonics_switch_is_strict_and_defaults_off(
    monkeypatch,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    monkeypatch.delenv(xphm_torch._BULK_TWIST_HARMONICS_ENV, raising=False)
    assert not xphm_torch._bulk_twist_harmonics_enabled()
    monkeypatch.setenv(xphm_torch._BULK_TWIST_HARMONICS_ENV, "maybe")
    with pytest.raises(ValueError, match=xphm_torch._BULK_TWIST_HARMONICS_ENV):
        xphm_torch._bulk_twist_harmonics_enabled()


def test_imrphenomxphm_scripted_twist_harmonics_switch_is_strict_and_off(
    monkeypatch,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    monkeypatch.delenv(xphm_torch._SCRIPTED_TWIST_HARMONICS_ENV, raising=False)
    assert not xphm_torch._scripted_twist_harmonics_enabled()
    monkeypatch.setenv(xphm_torch._SCRIPTED_TWIST_HARMONICS_ENV, "maybe")
    with pytest.raises(ValueError, match=xphm_torch._SCRIPTED_TWIST_HARMONICS_ENV):
        xphm_torch._scripted_twist_harmonics_enabled()


def test_imrphenomxphm_cudagraph_twist_harmonics_switch_is_strict_and_off(
    monkeypatch,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    monkeypatch.delenv(xphm_torch._CUDAGRAPH_TWIST_HARMONICS_ENV, raising=False)
    assert not xphm_torch._cudagraph_twist_harmonics_enabled()
    monkeypatch.setenv(xphm_torch._CUDAGRAPH_TWIST_HARMONICS_ENV, "maybe")
    with pytest.raises(
        ValueError,
        match=xphm_torch._CUDAGRAPH_TWIST_HARMONICS_ENV,
    ):
        xphm_torch._cudagraph_twist_harmonics_enabled()


def test_imrphenomxphm_bulk_twist_harmonics_fall_back_strictly(
    monkeypatch,
    preserve_scheme,
):
    from dataclasses import replace

    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = xphm_torch._xp_params(
        {
            **_SEQUENCE_PARAMS,
            "approximant": "IMRPhenomXPHM",
            "f_lower": 20.0,
        }
    )
    inputs = xphm_torch._validated_inputs(params)
    harmonics = xphm_torch._bulk_twist_harmonics(inputs)

    assert len(harmonics) == 21
    assert all(value.ndim == 0 for value in harmonics.values())
    assert all(value.dtype == torch.complex128 for value in harmonics.values())
    assert all(value.device.type == "cpu" for value in harmonics.values())

    monkeypatch.setenv(xphm_torch._SCRIPTED_TWIST_HARMONICS_ENV, "1")
    scripted = xphm_torch._bulk_twist_harmonics(inputs)
    for mode in harmonics:
        assert torch.equal(
            torch.view_as_real(harmonics[mode]).view(torch.int64),
            torch.view_as_real(scripted[mode]).view(torch.int64),
        )

    assert xphm_torch._bulk_twist_harmonics(
        replace(inputs, theta_jn=torch.tensor([inputs.theta_jn]))
    ) is None
    assert xphm_torch._bulk_twist_harmonics(
        replace(
            inputs,
            theta_jn=torch.tensor(inputs.theta_jn, dtype=torch.float32),
        )
    ) is None
    assert xphm_torch._bulk_twist_harmonics(
        replace(
            inputs,
            theta_jn=torch.tensor(inputs.theta_jn, requires_grad=True),
        )
    ) is None
    assert xphm_torch._bulk_twist_harmonics(
        replace(inputs, complex_dtype=torch.complex64)
    ) is None
    assert xphm_torch._bulk_twist_harmonics(
        replace(inputs, device=torch.device("meta"))
    ) is None
    theta_subclass = torch.tensor(
        inputs.theta_jn,
        dtype=torch.float64,
    ).as_subclass(type("ThetaTensor", (torch.Tensor,), {}))
    assert xphm_torch._bulk_twist_harmonics(
        replace(inputs, theta_jn=theta_subclass)
    ) is None
    assert xphm_torch._bulk_twist_harmonics(
        replace(
            inputs,
            theta_jn=torch._neg_view(
                torch.tensor(-inputs.theta_jn, dtype=torch.float64)
            ),
        )
    ) is None

    with torch.autograd.forward_ad.dual_level():
        theta = torch.tensor(inputs.theta_jn, dtype=torch.float64)
        dual = torch.autograd.forward_ad.make_dual(theta, torch.ones_like(theta))
        assert xphm_torch._bulk_twist_harmonics(
            replace(inputs, theta_jn=dual)
        ) is None


def test_imrphenomxphm_bulk_twist_harmonics_are_request_local_and_bitwise(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxp_torch as xp_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    params = dict(
        **_SEQUENCE_PARAMS,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setenv(xphm_torch._TWIST_REUSE_ENV, "1")

    calls = {"scalar": 0, "bulk": 0, "scripted": 0}
    original_xp_harmonic = xp_torch.spin_weighted_spherical_harmonic
    original_xphm_harmonic = xphm_torch.spin_weighted_spherical_harmonic
    original_bulk = xphm_torch.spin_minus_two_spherical_harmonics_phi_zero
    original_scripted = (
        xphm_torch.scripted_spin_minus_two_spherical_harmonics_phi_zero
    )

    def counted_xp_harmonic(*args, **kwargs):
        calls["scalar"] += 1
        return original_xp_harmonic(*args, **kwargs)

    def counted_xphm_harmonic(*args, **kwargs):
        calls["scalar"] += 1
        return original_xphm_harmonic(*args, **kwargs)

    def counted_bulk(*args, **kwargs):
        calls["bulk"] += 1
        return original_bulk(*args, **kwargs)

    def counted_scripted(*args, **kwargs):
        calls["scripted"] += 1
        return original_scripted(*args, **kwargs)

    monkeypatch.setattr(
        xp_torch,
        "spin_weighted_spherical_harmonic",
        counted_xp_harmonic,
    )
    monkeypatch.setattr(
        xphm_torch,
        "spin_weighted_spherical_harmonic",
        counted_xphm_harmonic,
    )
    monkeypatch.setattr(
        xphm_torch,
        "spin_minus_two_spherical_harmonics_phi_zero",
        counted_bulk,
    )
    monkeypatch.setattr(
        xphm_torch,
        "scripted_spin_minus_two_spherical_harmonics_phi_zero",
        counted_scripted,
    )

    monkeypatch.setenv(xphm_torch._SCRIPTED_TWIST_HARMONICS_ENV, "0")
    monkeypatch.setenv(xphm_torch._BULK_TWIST_HARMONICS_ENV, "0")
    scalar = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == {"scalar": 21, "bulk": 0, "scripted": 0}

    calls.update(scalar=0, bulk=0, scripted=0)
    monkeypatch.setenv(xphm_torch._BULK_TWIST_HARMONICS_ENV, "1")
    bulk = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == {"scalar": 0, "bulk": 1, "scripted": 0}

    calls.update(scalar=0, bulk=0, scripted=0)
    repeated = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == {"scalar": 0, "bulk": 1, "scripted": 0}

    calls.update(scalar=0, bulk=0, scripted=0)
    monkeypatch.setenv(xphm_torch._SCRIPTED_TWIST_HARMONICS_ENV, "1")
    scripted = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == {"scalar": 0, "bulk": 0, "scripted": 1}

    calls.update(scalar=0, bulk=0, scripted=0)
    scripted_again = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert calls == {"scalar": 0, "bulk": 0, "scripted": 1}

    for (
        scalar_series,
        bulk_series,
        repeated_series,
        scripted_series,
        scripted_again_series,
    ) in zip(
        scalar,
        bulk,
        repeated,
        scripted,
        scripted_again,
    ):
        scalar_bits = scalar_series._data.tensor.contiguous().view(torch.int64)
        assert torch.equal(
            scalar_bits,
            bulk_series._data.tensor.contiguous().view(torch.int64),
        )
        assert torch.equal(
            scalar_bits,
            repeated_series._data.tensor.contiguous().view(torch.int64),
        )
        assert torch.equal(
            scalar_bits,
            scripted_series._data.tensor.contiguous().view(torch.int64),
        )
        assert torch.equal(
            scalar_bits,
            scripted_again_series._data.tensor.contiguous().view(torch.int64),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_imrphenomxphm_cudagraph_twist_harmonics_are_full_waveform_bitwise(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform._spherical_harmonics_torch as harmonics_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cuda"))
    monkeypatch.setenv(xphm_torch._TWIST_REUSE_ENV, "1")
    monkeypatch.setenv(xphm_torch._BULK_TWIST_HARMONICS_ENV, "1")
    monkeypatch.setenv(xphm_torch._SCRIPTED_TWIST_HARMONICS_ENV, "1")
    monkeypatch.setenv(xphm_torch._CUDAGRAPH_TWIST_HARMONICS_ENV, "0")
    monkeypatch.setattr(
        xphm_torch,
        "scripted_spin_minus_two_spherical_harmonics_phi_zero",
        lambda *args, **kwargs: pytest.fail(
            "the non-bitwise CUDA TorchScript path must stay disabled"
        ),
    )
    parameters = dict(
        **_SEQUENCE_PARAMS,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    eager = xphm_torch.imrphenomxphm_fd_torch(**parameters)

    monkeypatch.setattr(
        harmonics_torch,
        "_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO",
        {},
    )
    monkeypatch.setattr(
        harmonics_torch,
        "_CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_FAILURES",
        set(),
    )
    monkeypatch.setenv(xphm_torch._CUDAGRAPH_TWIST_HARMONICS_ENV, "1")
    graphed = xphm_torch.imrphenomxphm_fd_torch(**parameters)

    for eager_series, graphed_series in zip(eager, graphed):
        assert torch.equal(
            eager_series._data.tensor.contiguous().view(torch.int64),
            graphed_series._data.tensor.contiguous().view(torch.int64),
        )
    assert len(harmonics_torch._CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO) == 1
    assert not harmonics_torch._CUDAGRAPH_SPIN_MINUS_TWO_PHI_ZERO_FAILURES


@pytest.mark.parametrize(
    ("real_dtype", "complex_dtype", "word_dtype"),
    (
        (torch.float32, torch.complex64, torch.int32),
        (torch.float64, torch.complex128, torch.int64),
    ),
    ids=("float32", "float64"),
)
def test_imrphenomxphm_twist_reuse_is_bitwise_and_reduces_primitive_calls(
    real_dtype,
    complex_dtype,
    word_dtype,
    monkeypatch,
    preserve_scheme,
):
    from dataclasses import replace

    import pycbc.waveform.imrphenomxp_torch as xp_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = xphm_torch._xp_params(
        {
            **_SEQUENCE_PARAMS,
            "approximant": "IMRPhenomXPHM",
            "f_lower": 20.0,
        }
    )
    inputs = replace(
        xp_torch._validated_inputs(params),
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    model = xp_torch._build_model(inputs)
    frequencies = torch.linspace(20.0, 512.0, 257, dtype=real_dtype)
    samples = torch.complex(
        torch.sin(frequencies * 0.013),
        torch.cos(frequencies * 0.017),
    )
    mode_angles = {
        mprime: xphm_torch._mode_angles(model, frequencies, mprime)
        for mprime in (1, 2, 3, 4)
    }

    calls = {"harmonic": 0, "exp": 0}
    original_harmonic = xphm_torch.spin_weighted_spherical_harmonic
    original_exp = torch.exp

    def counted_harmonic(*args, **kwargs):
        calls["harmonic"] += 1
        return original_harmonic(*args, **kwargs)

    def counted_exp(*args, **kwargs):
        calls["exp"] += 1
        return original_exp(*args, **kwargs)

    monkeypatch.setattr(
        xphm_torch,
        "spin_weighted_spherical_harmonic",
        counted_harmonic,
    )
    monkeypatch.setattr(torch, "exp", counted_exp)
    eager = {
        mode: xphm_torch._twist_mode(
            model,
            frequencies,
            samples,
            *mode,
            mode_angles=mode_angles[mode[1]],
        )
        for mode in xphm_torch._COPRECESSING_MODES
    }
    assert calls == {"harmonic": 33, "exp": 71}

    calls.update(harmonic=0, exp=0)
    twist_harmonics = {
        (2, emm): harmonic
        for emm, harmonic in zip(range(-2, 3), model.harmonics)
    }
    reused = {
        mode: xphm_torch._twist_mode(
            model,
            frequencies,
            samples,
            *mode,
            mode_angles=mode_angles[mode[1]],
            twist_harmonics=twist_harmonics,
        )
        for mode in xphm_torch._COPRECESSING_MODES
    }
    assert calls == {"harmonic": 16, "exp": 43}

    for mode in eager:
        for eager_polarization, reused_polarization in zip(
            eager[mode], reused[mode]
        ):
            assert torch.equal(
                eager_polarization.contiguous().view(word_dtype),
                reused_polarization.contiguous().view(word_dtype),
            )


def test_imrphenomxphm_twist_reuse_falls_back_for_ad_or_mismatched_tensors(
    preserve_scheme,
):
    from dataclasses import replace

    import pycbc.waveform.imrphenomxp_torch as xp_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = xphm_torch._xp_params(
        {
            **_SEQUENCE_PARAMS,
            "approximant": "IMRPhenomXPHM",
            "f_lower": 20.0,
        }
    )
    model = xp_torch._build_model(xp_torch._validated_inputs(params))
    frequencies = torch.linspace(20.0, 512.0, 33, dtype=torch.float64)
    samples = torch.complex(
        torch.sin(frequencies * 0.013),
        torch.cos(frequencies * 0.017),
    )
    active_modes = {(2, 2): samples}

    assert xphm_torch._twist_reuse_supported(
        model,
        frequencies,
        active_modes,
    )
    assert not xphm_torch._twist_reuse_supported(
        model,
        frequencies.detach().requires_grad_(),
        active_modes,
    )
    assert not xphm_torch._twist_reuse_supported(
        model,
        frequencies,
        {(2, 2): samples.detach().requires_grad_()},
    )
    with torch.autograd.forward_ad.dual_level():
        dual_frequencies = torch.autograd.forward_ad.make_dual(
            frequencies,
            torch.ones_like(frequencies),
        )
        assert not xphm_torch._twist_reuse_supported(
            model,
            dual_frequencies,
            active_modes,
        )
    assert not xphm_torch._twist_reuse_supported(
        model,
        frequencies.to(torch.float32),
        active_modes,
    )
    assert not xphm_torch._twist_reuse_supported(
        model,
        frequencies,
        {(2, 2): samples.to(torch.complex64)},
    )
    bad_model = replace(
        model,
        harmonics=(model.harmonics[0][None], *model.harmonics[1:]),
    )
    assert not xphm_torch._twist_reuse_supported(
        bad_model,
        frequencies,
        active_modes,
    )
    if torch.cuda.is_available():
        assert not xphm_torch._twist_reuse_supported(
            model,
            frequencies.cuda(),
            active_modes,
        )


@pytest.mark.parametrize(
    "model_flags",
    _NATIVE_MODELS,
    ids=("default", "msa-final-spin-0", "msa-final-spin-3", "msa-v300-alias"),
)
def test_imrphenomxphm_twist_reuse_is_request_local_and_bitwise(
    model_flags,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    params = dict(
        **_SEQUENCE_PARAMS,
        **model_flags,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setenv(xphm_torch._INTRINSIC_CACHE_ENV, "0")

    harmonic_calls = 0
    original_harmonic = xphm_torch.spin_weighted_spherical_harmonic

    def counted_harmonic(*args, **kwargs):
        nonlocal harmonic_calls
        harmonic_calls += 1
        return original_harmonic(*args, **kwargs)

    monkeypatch.setattr(
        xphm_torch,
        "spin_weighted_spherical_harmonic",
        counted_harmonic,
    )
    monkeypatch.setenv(xphm_torch._TWIST_REUSE_ENV, "0")
    eager = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert harmonic_calls == 33

    harmonic_calls = 0
    monkeypatch.setenv(xphm_torch._TWIST_REUSE_ENV, "1")
    reused = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert harmonic_calls == 16

    harmonic_calls = 0
    reused_again = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert harmonic_calls == 16
    for eager_series, reused_series, repeated_series in zip(
        eager,
        reused,
        reused_again,
    ):
        eager_bits = eager_series._data.tensor.contiguous().view(torch.int64)
        assert torch.equal(
            eager_bits,
            reused_series._data.tensor.contiguous().view(torch.int64),
        )
        assert torch.equal(
            eager_bits,
            repeated_series._data.tensor.contiguous().view(torch.int64),
        )


@pytest.mark.parametrize(
    ("real_dtype", "complex_dtype"),
    ((torch.float32, torch.complex64), (torch.float64, torch.complex128)),
    ids=("float32", "float64"),
)
def test_imrphenomxphm_reused_mode_angles_are_bitwise(
    real_dtype,
    complex_dtype,
    preserve_scheme,
):
    from dataclasses import replace

    import pycbc.waveform.imrphenomxp_torch as xp_torch
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = xphm_torch._xp_params(
        {
            **_SEQUENCE_PARAMS,
            "approximant": "IMRPhenomXPHM",
            "f_lower": 20.0,
        }
    )
    inputs = xp_torch._validated_inputs(params)
    inputs = replace(
        inputs,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    model = xp_torch._build_model(inputs)
    frequencies = torch.linspace(20.0, 512.0, 257, dtype=real_dtype)
    samples = torch.complex(
        torch.sin(frequencies * 0.013),
        torch.cos(frequencies * 0.017),
    )

    eager = {
        mode: xphm_torch._twist_mode(model, frequencies, samples, *mode)
        for mode in ((2, 2), (3, 2))
    }
    shared_angles = xphm_torch._mode_angles(model, frequencies, 2)
    reused = {
        mode: xphm_torch._twist_mode(
            model,
            frequencies,
            samples,
            *mode,
            mode_angles=shared_angles,
        )
        for mode in ((2, 2), (3, 2))
    }

    for mode in eager:
        for eager_polarization, reused_polarization in zip(
            eager[mode], reused[mode]
        ):
            assert torch.equal(eager_polarization, reused_polarization)


def test_imrphenomxphm_mode_angle_reuse_is_request_local_and_bitwise(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    params = dict(
        **_SEQUENCE_PARAMS,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_INTRINSIC_CACHE", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMX_PHASE_PLAN", "1")

    angle_calls = 0
    original_msa_angles = xphm_torch.msa_angles

    def counted_msa_angles(*args, **kwargs):
        nonlocal angle_calls
        angle_calls += 1
        return original_msa_angles(*args, **kwargs)

    monkeypatch.setattr(xphm_torch, "msa_angles", counted_msa_angles)
    monkeypatch.setenv(xphm_torch._MODE_ANGLE_REUSE_ENV, "0")
    monkeypatch.setenv(xphm_torch._BULK_MODE_ANGLES_ENV, "0")
    eager = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert angle_calls == len(xphm_torch._COPRECESSING_MODES)

    angle_calls = 0
    monkeypatch.setenv(xphm_torch._MODE_ANGLE_REUSE_ENV, "1")
    monkeypatch.setenv(xphm_torch._BULK_MODE_ANGLES_ENV, "0")
    reused = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert angle_calls == len({mode[1] for mode in xphm_torch._COPRECESSING_MODES})

    for eager_series, reused_series in zip(eager, reused):
        assert len(eager_series) == len(reused_series)
        assert eager_series.delta_f == reused_series.delta_f
        assert float(eager_series.epoch) == float(reused_series.epoch)
        assert torch.equal(eager_series._data.tensor, reused_series._data.tensor)


@pytest.mark.parametrize(
    ("real_dtype", "complex_dtype"),
    (
        (torch.float32, torch.complex64),
        (torch.float64, torch.complex128),
    ),
)
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_imrphenomxphm_bulk_mode_angles_match_scalar_bitwise(
    real_dtype,
    complex_dtype,
    device,
    preserve_scheme,
):
    from dataclasses import replace

    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    params = dict(
        **_SEQUENCE_PARAMS,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    _activate_scheme(_scheme.TorchScheme(device))
    inputs = xphm_torch._validated_inputs(xphm_torch._xp_params(params))
    inputs = replace(
        inputs,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
    )
    model = xphm_torch._build_model(inputs)
    frequencies = (
        torch.arange(40, 1025, dtype=real_dtype, device=device) * 0.5
    )
    mprimes = (2, 1, 3, 4)

    scalar = {
        mprime: xphm_torch._mode_angles(model, frequencies, mprime)
        for mprime in mprimes
    }
    bulk = xphm_torch._bulk_mode_angles(model, frequencies, mprimes)

    for mprime in mprimes:
        for scalar_angle, bulk_angle in zip(scalar[mprime], bulk[mprime]):
            assert torch.equal(scalar_angle, bulk_angle)


def test_imrphenomxphm_bulk_mode_angles_are_request_local_and_bitwise(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    params = dict(
        **_SEQUENCE_PARAMS,
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_INTRINSIC_CACHE", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMX_PHASE_PLAN", "1")
    monkeypatch.setenv(xphm_torch._MODE_ANGLE_REUSE_ENV, "1")

    angle_calls = 0
    original_msa_angles = xphm_torch.msa_angles

    def counted_msa_angles(*args, **kwargs):
        nonlocal angle_calls
        angle_calls += 1
        return original_msa_angles(*args, **kwargs)

    monkeypatch.setattr(xphm_torch, "msa_angles", counted_msa_angles)
    monkeypatch.setenv(xphm_torch._BULK_MODE_ANGLES_ENV, "0")
    scalar = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert angle_calls == len(
        {mode[1] for mode in xphm_torch._COPRECESSING_MODES}
    )

    angle_calls = 0
    monkeypatch.setenv(xphm_torch._BULK_MODE_ANGLES_ENV, "1")
    # Production admits this lane only on its qualified CUDA contract.  Force
    # admission here so the CPU test continues to exercise the request-local
    # bulk routing and exact fallback-independent arithmetic; the dedicated
    # CUDA contract tests cover the production admission predicate.
    monkeypatch.setattr(
        xphm_torch,
        "_bulk_mode_angles_supported",
        lambda *_args, **_kwargs: True,
    )
    bulk = xphm_torch.imrphenomxphm_fd_torch(**params)
    assert angle_calls == 1

    for scalar_series, bulk_series in zip(scalar, bulk):
        assert len(scalar_series) == len(bulk_series)
        assert scalar_series.delta_f == bulk_series.delta_f
        assert float(scalar_series.epoch) == float(bulk_series.epoch)
        assert torch.equal(scalar_series._data.tensor, bulk_series._data.tensor)


def test_imrphenomxphm_mode_angle_reuse_switch_is_strict_and_defaults_off(
    monkeypatch,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    monkeypatch.delenv(xphm_torch._MODE_ANGLE_REUSE_ENV, raising=False)
    assert not xphm_torch._mode_angle_reuse_enabled()
    monkeypatch.setenv(xphm_torch._MODE_ANGLE_REUSE_ENV, "maybe")
    with pytest.raises(ValueError, match=xphm_torch._MODE_ANGLE_REUSE_ENV):
        xphm_torch._mode_angle_reuse_enabled()


def test_imrphenomxphm_bulk_mode_angles_switch_is_strict_and_defaults_off(
    monkeypatch,
):
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    monkeypatch.delenv(xphm_torch._BULK_MODE_ANGLES_ENV, raising=False)
    assert not xphm_torch._bulk_mode_angles_enabled()
    monkeypatch.setenv(xphm_torch._BULK_MODE_ANGLES_ENV, "maybe")
    with pytest.raises(ValueError, match=xphm_torch._BULK_MODE_ANGLES_ENV):
        xphm_torch._bulk_mode_angles_enabled()


@pytest.mark.parametrize(
    ("mode_array", "tolerance"),
    [
        ([(2, 2)], 5.0e-5),
        ([(2, 1)], 5.0e-5),
        ([(3, 3)], 5.0e-5),
        ([(3, 2)], 5.0e-4),
        ([(4, 4)], 5.0e-5),
        ([(3, 3), (4, 4)], 5.0e-5),
        ([(4, 4), (2, 1), (2, 1)], 5.0e-5),
    ],
    ids=(
        "22",
        "21",
        "33",
        "32",
        "44",
        "multi",
        "duplicate-reordered",
    ),
)
def test_imrphenomxphm_sequence_mode_subsets_match_lalsimulation(
    mode_array,
    tolerance,
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    params = dict(_SEQUENCE_PARAMS, mode_array=mode_array)
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXPHM sequence called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < tolerance


def test_imrphenomxphm_empty_mode_array_is_zero(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("zero-mode IMRPhenomXPHM called lalsimulation")

    if _lal.LAL_AVAILABLE:
        monkeypatch.setattr(
            waveform.lalsimulation,
            "SimInspiralChooseFDWaveformSequence",
            reject_lal,
        )
        monkeypatch.setattr(
            waveform.lalsimulation,
            "SimInspiralChooseFDWaveform",
            reject_lal,
        )
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")

    _activate_scheme(_scheme.TorchScheme("cpu"))
    sequence = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        mode_array=[],
        **_SEQUENCE_PARAMS,
    )
    for polarization in sequence:
        assert polarization._data.tensor.device.type == "cpu"
        assert polarization._data.tensor.dtype == torch.complex128
        np.testing.assert_array_equal(polarization.numpy(), 0.0)

    grid = get_fd_waveform(
        approximant="IMRPhenomXPHM",
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
        mode_array=[],
        **_SEQUENCE_PARAMS,
    )
    assert len(grid[0]) == 1025
    for series in grid:
        assert series._data.tensor.device.type == "cpu"
        assert series._data.tensor.dtype == torch.complex128
        np.testing.assert_array_equal(series.numpy(), 0.0)


def test_imrphenomxphm_regular_grid_mode_subset_matches_lalsimulation(
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=40.0,
        mass2=20.0,
        spin1x=0.2,
        spin1y=0.1,
        spin1z=0.3,
        spin2x=-0.1,
        spin2y=0.05,
        spin2z=-0.2,
        distance=500.0,
        inclination=0.7,
        coa_phase=1.2,
        long_asc_nodes=0.3,
        delta_f=0.5,
        f_lower=20.0,
        f_final=512.0,
        f_ref=30.0,
        mode_array=[(3, 3), (4, 4)],
    )
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXPHM", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXPHM called lalsimulation")

    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomXPHM", **params)

    for expected, expected_array, result in zip(
        reference,
        reference_arrays,
        actual,
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        # LAL's regular-grid multibanding may leave the requested upper
        # endpoint zero for a sparse mode set. The native path deliberately
        # performs full mode evaluation, so include that endpoint in the norm.
        relative_error = np.linalg.norm(result_array - expected_array)
        relative_error /= np.linalg.norm(expected_array)
        assert relative_error < 5.0e-3


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        (_MSA_FLAGS, True),
        (_MSA_FINAL_SPIN_FLAGS, True),
        (_MSA_ALIAS_FLAGS, True),
        ({"phenom_x_prec_version": 223}, True),
        (dict(_MSA_FLAGS, mode_array=[]), True),
        (dict(_MSA_FLAGS, mode_array=[(2, 2)]), True),
        (dict(_MSA_FLAGS, mode_array=[(4, 4), (2, 1), (2, 1)]), True),
        (dict(_MSA_FLAGS, mode_array=(3, 3)), False),
        ({"phenom_x_prec_version": 102}, False),
        (dict(_MSA_FLAGS, phenom_xp_convention=0), False),
        (dict(_MSA_FLAGS, phenom_xp_final_spin_mod=2), False),
        (dict(_MSA_FLAGS, phenom_xp_final_spin_mod=3.5), False),
        (dict(_MSA_FLAGS, mode_array=[(2, -1)]), False),
        (dict(_MSA_FLAGS, mode_array=[(3, 1)]), False),
        (dict(_MSA_FLAGS, mode_array=[(2.0, 2.0)]), False),
        (dict(_MSA_FLAGS, mode_array=["22"]), False),
        (dict(_MSA_FLAGS, mode_array=[(2, 2, 1)]), False),
        (dict(_MSA_FLAGS, lambda1=100.0), False),
        (dict(_MSA_FLAGS, dchi3=0.1), False),
        (dict(_MSA_FLAGS, eccentricity=0.1), False),
        (dict(_MSA_FLAGS, phase_order=2.5), True),
        (dict(_MSA_FLAGS, amplitude_order="3"), True),
        (dict(_MSA_FLAGS, spin_order=4.5), True),
        (dict(_MSA_FLAGS, tidal_order=0), True),
        (dict(_MSA_FLAGS, eccentricity_order=4), True),
        (dict(_MSA_FLAGS, eccentricity_order=4.0), False),
        (dict(_MSA_FLAGS, frame_axis=1), False),
        (dict(_MSA_FLAGS, numrel_data="waveform.h5"), False),
        ({"approximant": "IMRPhenomXP"}, False),
    ],
)
def test_imrphenomxphm_native_support_boundary(params, expected):
    full_params = {"approximant": "IMRPhenomXPHM", **params}
    assert imrphenomxphm_native_supported(full_params) is expected
    assert imrphenomxphm_sequence_native_supported(full_params) is expected


def test_imrphenomxphm_sequence_avoids_host_transfer(
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types.array_torch import TorchArrayData

    import pycbc.waveform.waveform as waveform

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXPHM sequence called lalsimulation")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomXPHM sequence transferred to NumPy")

    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array(_SAMPLE_POINTS)
    if _lal.LAL_AVAILABLE:
        monkeypatch.setattr(
            waveform.lalsimulation,
            "SimInspiralChooseFDWaveformSequence",
            reject_lal,
        )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        polarizations = get_fd_waveform_sequence(
            approximant="IMRPhenomXPHM",
            sample_points=sample_points,
            **_SEQUENCE_PARAMS,
        )

    for polarization in polarizations:
        assert isinstance(polarization._data.tensor, torch.Tensor)


def test_imrphenomxphm_unsupported_options_use_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch
    import pycbc.waveform.waveform as waveform

    params = {**_SEQUENCE_PARAMS, "mode_array": [(2, 2), (2, -1)]}
    # The cache switch is irrelevant on the LAL route and must not be parsed.
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_INTRINSIC_CACHE", "invalid")
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    lal_generator = waveform.lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported XPHM sequence reached Torch")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        xphm_torch,
        "imrphenomxphm_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **params,
    )

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1.0e-14, atol=0.0)


@pytest.mark.parametrize(
    "opt_out_flag",
    ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOMXPHM_NATIVE"),
)
def test_imrphenomxphm_native_opt_out_uses_lal(
    opt_out_flag,
    monkeypatch,
    preserve_scheme,
):
    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    params = dict(
        mass1=35.0,
        mass2=20.0,
        spin1z=0.2,
        spin2z=-0.1,
        distance=500.0,
        delta_f=1.0,
        f_lower=20.0,
    )
    import pycbc.waveform.imrphenomxphm_torch as xphm_torch

    def unexpected_native(**_params):
        raise AssertionError("opted-out IMRPhenomXPHM reached the Torch generator")

    # Opt out through one flag while every other native flag stays absent.
    monkeypatch.setenv(opt_out_flag, "0")
    for name in _NATIVE_FLAG_ENVS:
        if name != opt_out_flag:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(xphm_torch, "imrphenomxphm_fd_torch", unexpected_native)

    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXPHM", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    result = get_fd_waveform(approximant="IMRPhenomXPHM", **params)

    assert all(isinstance(series._data.tensor, torch.Tensor) for series in result)
    for expected, series in zip(reference_arrays, result):
        np.testing.assert_array_equal(series.numpy(), expected)


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomxphm_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    pytest.importorskip("lal")
    pytest.importorskip("lalsimulation")
    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **_SEQUENCE_PARAMS,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXPHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXPHM",
        sample_points=_SAMPLE_POINTS,
        **_SEQUENCE_PARAMS,
    )

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    tolerance = 1.0e-2 if device_name == "mps" else 5.0e-5
    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == device_name
        assert result._data.tensor.dtype == expected_dtype
        result_array = result.numpy()
        np.testing.assert_array_equal(result_array == 0.0, expected == 0.0)
        assert _relative_error(result_array, expected) < tolerance
