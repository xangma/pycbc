import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform import imrphenomx_utils_torch as xutils  # noqa: E402
from pycbc.waveform import imrphenomxas_torch as xas_torch  # noqa: E402
from pycbc.waveform._torch_jax import torch_context  # noqa: E402
from pycbc.waveform.imrphenomxas_torch import (  # noqa: E402
    imrphenomxas_native_supported,
    imrphenomxas_sequence_native_supported,
)


@pytest.fixture
def preserve_scheme():
    """Restore the process-wide PyCBC scheme singleton after a test."""

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.fixture
def one_torch_thread():
    """Use the exact single-thread execution proven for region pruning."""

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _raw_tensor_equal(left, right):
    """Compare tensor storage bytes, including the sign bit of zero."""

    return torch.equal(
        left.detach().contiguous().view(torch.uint8),
        right.detach().contiguous().view(torch.uint8),
    )


def _activate_scheme(scheme):
    _scheme.Scheme._single = None
    _scheme.mgr.state = scheme


def _clear_native_flags(monkeypatch):
    for name in (
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
        "PYCBC_IMRPHENOMXAS_NATIVE",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("device_name", ("cpu", "mps", "cuda"))
def test_phenomx_coefficient_table_masters_are_cached_but_public_copies_are_owned(
    device_name,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dtype = torch.float32 if device_name == "mps" else torch.float64
    xutils._clear_phenomx_coeff_table_cache()
    try:
        phase_master = xutils._get_phenomx_phase_coeff_table_cached_master(
            device=device_name,
            dtype=dtype,
        )
        amplitude_master = xutils._get_phenomx_amp_coeff_table_cached_master(
            device=device_name,
            dtype=dtype,
        )
        phase = xutils.get_phenomx_phase_coeff_table(
            device=device_name,
            dtype=dtype,
        )
        phase_copy = xutils.get_phenomx_phase_coeff_table(
            device=device_name,
            dtype=dtype,
        )
        amplitude = xutils.get_phenomx_amp_coeff_table(
            device=device_name,
            dtype=dtype,
        )
        amplitude_copy = xutils.get_phenomx_amp_coeff_table(
            device=device_name,
            dtype=dtype,
        )
        assert phase_master is (
            xutils._get_phenomx_phase_coeff_table_cached_master(
                device=device_name,
                dtype=dtype,
            )
        )
        assert amplitude_master is (
            xutils._get_phenomx_amp_coeff_table_cached_master(
                device=device_name,
                dtype=dtype,
            )
        )
        assert phase is not phase_copy and phase is not phase_master
        assert amplitude is not amplitude_copy
        assert amplitude is not amplitude_master
        assert phase.data_ptr() != phase_copy.data_ptr()
        assert phase.data_ptr() != phase_master.data_ptr()
        assert amplitude.data_ptr() != amplitude_copy.data_ptr()
        assert amplitude.data_ptr() != amplitude_master.data_ptr()
        assert _raw_tensor_equal(phase, phase_master)
        assert _raw_tensor_equal(phase_copy, phase_master)
        assert _raw_tensor_equal(amplitude, amplitude_master)
        assert _raw_tensor_equal(amplitude_copy, amplitude_master)
        assert phase.device.type == device_name
        assert amplitude.device.type == device_name
        assert phase.dtype == dtype
        assert amplitude.dtype == dtype

        if device_name == "cpu":
            phase_float32 = xutils.get_phenomx_phase_coeff_table(
                device=device_name,
                dtype=torch.float32,
            )
            assert phase_float32 is not phase
            assert phase_float32.dtype == torch.float32
    finally:
        xutils._clear_phenomx_coeff_table_cache()


def test_public_phenomx_coefficient_mutation_and_autograd_are_isolated():
    xutils._clear_phenomx_coeff_table_cache()
    phase_master = xutils._get_phenomx_phase_coeff_table_cached_master(
        device="cpu",
        dtype=torch.float64,
    )
    amp_master = xutils._get_phenomx_amp_coeff_table_cached_master(
        device="cpu",
        dtype=torch.float64,
    )
    phase_bytes = phase_master.contiguous().view(torch.uint8).clone()
    amp_bytes = amp_master.contiguous().view(torch.uint8).clone()

    phase = xutils.get_phenomx_phase_coeff_table(
        device="cpu",
        dtype=torch.float64,
    )
    amplitude = xutils.get_phenomx_amp_coeff_table(
        device="cpu",
        dtype=torch.float64,
    )
    phase.zero_()
    amplitude.requires_grad_()
    amplitude.sum().backward()

    assert torch.equal(phase_master.contiguous().view(torch.uint8), phase_bytes)
    assert torch.equal(amp_master.contiguous().view(torch.uint8), amp_bytes)
    assert amp_master.grad is None and not amp_master.requires_grad
    assert amplitude.grad is not None
    assert _raw_tensor_equal(
        xutils.get_phenomx_phase_coeff_table(
            device="cpu",
            dtype=torch.float64,
        ),
        phase_master,
    )


def test_phase_prepares_each_region_once(monkeypatch):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=torch.float64)
    frequencies = torch.linspace(20.0, 400.0, 100, dtype=torch.float64)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=torch.float64)
    prepare_names = (
        "_prepare_phase_fit_rows",
        "_prepare_inspiral_phase",
        "_prepare_intermediate_phase",
        "_prepare_mergerringdown_phase",
    )
    call_counts = dict.fromkeys(prepare_names, 0)

    def counted_prepare(name):
        original = getattr(xas_torch, name)

        def evaluate(*args, **kwargs):
            call_counts[name] += 1
            return original(*args, **kwargs)

        return evaluate

    for name in prepare_names:
        monkeypatch.setattr(xas_torch, name, counted_prepare(name))

    with torch_context(frequencies):
        xas_torch.Phase(
            frequencies,
            theta,
            phase_coeffs,
            chip=0.3,
        )

    assert call_counts == dict.fromkeys(prepare_names, 1)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_phase_fit_rows_match_scalar_fits_bitwise(dtype):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=dtype)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=dtype)

    with torch_context(theta):
        m1, m2, chi1, chi2 = theta
        m1_seconds = m1 * xas_torch.MTSUN
        m2_seconds = m2 * xas_torch.MTSUN
        total_mass = m1_seconds + m2_seconds
        eta = m1_seconds * m2_seconds / total_mass.square()
        delta = torch.sqrt(torch.clamp_min(1.0 - 4.0 * eta, 0.0))
        mm1 = 0.5 * (1.0 + delta)
        mm2 = 0.5 * (1.0 - delta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        inspiral_spin = (
            chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)
        ) / (1.0 - 76.0 * eta / 113.0)
        merger_spin = (
            mm1.square() * chi1 + mm2.square() * chi2
        ) / (mm1.square() + mm2.square())
        chia = chi1 - chi2

        scalar_rows = []
        for row in range(13):
            spin = inspiral_spin if row < 4 else merger_spin
            no_spin = xutils.nospin_CV(
                phase_coeffs[row, : xas_torch.eqspin_indx], eta
            )
            equal_spin = xutils.Eqspin_CV(
                phase_coeffs[
                    row,
                    xas_torch.eqspin_indx : xas_torch.uneqspin_indx,
                ],
                eta,
                spin,
            )
            unequal_spin = xutils.Uneqspin_CV(
                phase_coeffs[row, xas_torch.uneqspin_indx :],
                eta,
                spin,
                chia,
            )
            scalar_rows.append((no_spin + equal_spin) + unequal_spin)

        bulk_rows = xas_torch._prepare_phase_fit_rows(theta, phase_coeffs)

    assert torch.equal(bulk_rows, torch.stack(scalar_rows))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_amp_fit_rows_match_scalar_fits_bitwise(dtype):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=dtype)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=dtype)

    with torch_context(theta):
        m1, m2, chi1, chi2 = theta
        m1_seconds = m1 * xas_torch.MTSUN
        m2_seconds = m2 * xas_torch.MTSUN
        total_mass = m1_seconds + m2_seconds
        eta = m1_seconds * m2_seconds / total_mass.square()
        delta = torch.sqrt(torch.clamp_min(1.0 - 4.0 * eta, 0.0))
        mm1 = 0.5 * (1.0 + delta)
        mm2 = 0.5 * (1.0 - delta)
        chi_eff = mm1 * chi1 + mm2 * chi2
        inspiral_spin = (
            chi_eff - (38.0 / 113.0) * eta * (chi1 + chi2)
        ) / (1.0 - 76.0 * eta / 113.0)
        merger_spin = (
            mm1.square() * chi1 + mm2.square() * chi2
        ) / (mm1.square() + mm2.square())
        chia = chi1 - chi2

        scalar_rows = []
        for row in range(7):
            spin = inspiral_spin if row < 3 else merger_spin
            no_spin = xutils.Amp_Nospin_CV(
                amp_coeffs[row, : xas_torch.amp_eqspin_indx], eta
            )
            equal_spin = xutils.Amp_Eqspin_CV(
                amp_coeffs[
                    row,
                    xas_torch.amp_eqspin_indx : xas_torch.amp_uneqspin_indx,
                ],
                eta,
                spin,
            )
            unequal_spin = xutils.Amp_Uneqspin_CV(
                amp_coeffs[row, xas_torch.amp_uneqspin_indx :],
                eta,
                spin,
                chia,
            )
            scalar_rows.append((no_spin + equal_spin) + unequal_spin)

        bulk_rows = xas_torch._prepare_amp_fit_rows(theta, amp_coeffs)

    assert torch.equal(bulk_rows, torch.stack(scalar_rows))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_mergerringdown_amp_plan_matches_eager_value_and_derivatives_bitwise(dtype):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=dtype)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=dtype)

    with torch_context(theta):
        plan = xas_torch._prepare_mergerringdown_amp_plan(
            theta,
            amp_coeffs,
            chip=0.3,
            fit_rows=xas_torch._prepare_amp_fit_rows(theta, amp_coeffs),
        )
        frequencies = torch.linspace(0.04, 0.2, 33, dtype=dtype)
        eager, eager_peak = xas_torch.get_mergerringdown_Amp(
            frequencies,
            theta,
            amp_coeffs,
            chip=0.3,
        )
        planned, planned_peak = xas_torch.get_mergerringdown_Amp(
            frequencies,
            theta,
            amp_coeffs,
            chip=0.3,
            _amp_plan=plan,
        )

        def value_and_derivatives(amp_plan=None):
            frequency = torch.tensor(0.1, dtype=dtype, requires_grad=True)
            value = xas_torch.get_mergerringdown_Amp(
                frequency,
                theta,
                amp_coeffs,
                chip=0.3,
                _amp_plan=amp_plan,
            )[0]
            first = torch.autograd.grad(value, frequency, create_graph=True)[0]
            second = torch.autograd.grad(first, frequency)[0]
            return value.detach(), first.detach(), second.detach()

        eager_derivatives = value_and_derivatives()
        planned_derivatives = value_and_derivatives(plan)

    assert torch.equal(planned, eager)
    assert torch.equal(planned_peak, eager_peak)
    for planned_value, eager_value in zip(planned_derivatives, eager_derivatives):
        assert torch.equal(planned_value, eager_value)


def test_amp_prepares_fit_rows_once(monkeypatch):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=torch.float64)
    frequencies = torch.linspace(20.0, 400.0, 100, dtype=torch.float64)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=torch.float64)
    original = xas_torch._prepare_amp_fit_rows
    call_count = 0

    def counted_prepare(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(xas_torch, "_prepare_amp_fit_rows", counted_prepare)

    with torch_context(frequencies):
        xas_torch.Amp(frequencies, theta, amp_coeffs, chip=0.3)

    assert call_count == 1


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize(
    ("intrinsics", "chip", "final_spin"),
    (
        ((40.0, 20.0, 0.4, -0.2), 0.3, None),
        ((35.0, 35.0, 0.0, 0.0), 0.0, None),
        ((60.0, 8.0, -0.75, 0.6), 0.7, -0.2),
        ((12.0, 35.0, 0.85, -0.65), 0.2, 0.4),
    ),
)
def test_amp_plan_matches_fallback_raw_bytes(
    dtype,
    intrinsics,
    chip,
    final_spin,
    monkeypatch,
):
    theta = torch.tensor(intrinsics, dtype=dtype)
    frequencies = torch.linspace(15.0, 1024.0, 257, dtype=dtype)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=dtype)

    with torch_context(frequencies):
        monkeypatch.setenv(xas_torch._AMP_PLAN_ENV, "0")
        fallback = xas_torch.Amp(
            frequencies,
            theta,
            amp_coeffs,
            D=431.0,
            chip=chip,
            final_spin=final_spin,
        )
        monkeypatch.setenv(xas_torch._AMP_PLAN_ENV, "1")
        planned = xas_torch.Amp(
            frequencies,
            theta,
            amp_coeffs,
            D=431.0,
            chip=chip,
            final_spin=final_spin,
        )

    assert torch.equal(
        planned.contiguous().view(torch.uint8),
        fallback.contiguous().view(torch.uint8),
    )


def test_amp_plan_prepares_each_region_once(monkeypatch):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=torch.float64)
    frequencies = torch.linspace(20.0, 400.0, 100, dtype=torch.float64)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=torch.float64)
    prepare_names = (
        "_prepare_inspiral_amp",
        "_prepare_intermediate_amp",
        "_prepare_mergerringdown_amp_plan",
    )
    call_counts = dict.fromkeys(prepare_names, 0)

    def counted_prepare(name):
        original = getattr(xas_torch, name)

        def evaluate(*args, **kwargs):
            call_counts[name] += 1
            return original(*args, **kwargs)

        return evaluate

    for name in prepare_names:
        monkeypatch.setattr(xas_torch, name, counted_prepare(name))

    monkeypatch.setenv(xas_torch._AMP_PLAN_ENV, "1")
    with torch_context(frequencies):
        xas_torch.Amp(frequencies, theta, amp_coeffs, chip=0.3)

    assert call_counts == dict.fromkeys(prepare_names, 1)


def test_amp_plan_gate_off_uses_unchanged_fallback(monkeypatch):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=torch.float64)
    frequencies = torch.linspace(20.0, 400.0, 100, dtype=torch.float64)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=torch.float64)

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("disabled amplitude plan must use the fallback path")

    monkeypatch.setattr(xas_torch, "_prepare_amp_plan", unexpected_prepare)
    monkeypatch.setenv(xas_torch._AMP_PLAN_ENV, "0")
    with torch_context(frequencies):
        values = xas_torch.Amp(frequencies, theta, amp_coeffs, chip=0.3)

    assert torch.isfinite(values).all()


def test_amp_plan_gate_bypasses_autograd_inputs(monkeypatch):
    theta = torch.tensor(
        [40.0, 20.0, 0.4, -0.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    frequencies = torch.linspace(20.0, 300.0, 32, dtype=torch.float64)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=torch.float64)

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("request-local amplitude reuse must bypass autograd inputs")

    monkeypatch.setattr(xas_torch, "_prepare_amp_plan", unexpected_prepare)
    monkeypatch.setenv(xas_torch._AMP_PLAN_ENV, "1")
    with torch_context(frequencies):
        values = xas_torch.Amp(frequencies, theta, amp_coeffs, chip=0.3)
        gradient = torch.autograd.grad(values.sum(), theta)[0]

    assert torch.isfinite(gradient).all()


def test_amp_plan_gate_bypasses_forward_ad_inputs(monkeypatch):
    frequencies = torch.linspace(20.0, 300.0, 32, dtype=torch.float64)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=torch.float64)

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("request-local amplitude reuse must bypass forward AD")

    monkeypatch.setattr(xas_torch, "_prepare_amp_plan", unexpected_prepare)
    monkeypatch.setenv(xas_torch._AMP_PLAN_ENV, "1")
    with torch.autograd.forward_ad.dual_level():
        theta = torch.autograd.forward_ad.make_dual(
            torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=torch.float64),
            torch.tensor([0.2, -0.1, 0.3, -0.4], dtype=torch.float64),
        )
        with torch_context(frequencies):
            values = xas_torch.Amp(
                frequencies,
                theta,
                amp_coeffs,
                chip=0.3,
            )
        primal, tangent = torch.autograd.forward_ad.unpack_dual(values)

    assert torch.isfinite(primal).all()
    assert torch.isfinite(tangent).all()


def test_amp_plan_gate_bypasses_tensor_subclasses(monkeypatch):
    class TensorSubclass(torch.Tensor):
        pass

    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=torch.float64)
    frequencies = torch.linspace(
        20.0,
        300.0,
        32,
        dtype=torch.float64,
    ).as_subclass(TensorSubclass)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=torch.float64)

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("tensor subclasses must retain eager dispatch")

    monkeypatch.setattr(xas_torch, "_prepare_amp_plan", unexpected_prepare)
    monkeypatch.setenv(xas_torch._AMP_PLAN_ENV, "1")
    with torch_context(frequencies):
        values = xas_torch.Amp(frequencies, theta, amp_coeffs, chip=0.3)

    assert isinstance(values, TensorSubclass)
    assert torch.isfinite(values).all()


def test_amp_plan_guard_rejects_unsupported_tensor_semantics():
    frequencies = torch.linspace(20.0, 300.0, 32, dtype=torch.float64)
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=torch.float64)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=torch.float64)

    def supported(test_f, test_theta, test_coeffs, *extra):
        return xas_torch._amp_plan_inputs_supported(
            test_f,
            test_theta,
            test_coeffs,
            (test_f, test_theta, test_coeffs, 1.0, 0.3, None, *extra),
        )

    sparse_frequencies = torch.sparse_coo_tensor(
        torch.tensor([[0, 1]], dtype=torch.long),
        torch.tensor([20.0, 30.0], dtype=torch.float64),
        size=frequencies.shape,
    )
    assert not supported(sparse_frequencies, theta, amp_coeffs)
    assert not supported(torch._neg_view(frequencies), theta, amp_coeffs)
    assert not supported(
        frequencies.to(torch.float16),
        theta.to(torch.float16),
        amp_coeffs.to(torch.float16),
    )
    assert not supported(frequencies, theta, amp_coeffs.to(torch.float32))
    assert not supported(
        frequencies,
        theta,
        amp_coeffs,
        torch.tensor(0.2, dtype=torch.float32),
    )


def test_amp_plan_gate_bypasses_non_single_waveform_inputs(monkeypatch):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=torch.float64)
    frequencies = torch.linspace(20.0, 300.0, 32, dtype=torch.float64).unsqueeze(0)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=torch.float64)

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("request-local amplitude reuse is limited to one waveform")

    monkeypatch.setattr(xas_torch, "_prepare_amp_plan", unexpected_prepare)
    monkeypatch.setenv(xas_torch._AMP_PLAN_ENV, "1")
    with torch_context(frequencies):
        values = xas_torch.Amp(frequencies, theta, amp_coeffs, chip=0.3)

    assert values.shape == frequencies.shape
    assert torch.isfinite(values).all()


def test_amp_plan_switch_is_strict_and_defaults_off(monkeypatch):
    monkeypatch.delenv(xas_torch._AMP_PLAN_ENV, raising=False)
    assert not xas_torch._amp_plan_enabled()
    monkeypatch.setenv(xas_torch._AMP_PLAN_ENV, "maybe")
    with pytest.raises(ValueError, match=xas_torch._AMP_PLAN_ENV):
        xas_torch._amp_plan_enabled()


def test_bulk_fit_preparation_bypasses_differentiable_intrinsics(monkeypatch):
    frequencies = torch.linspace(20.0, 300.0, 32, dtype=torch.float64)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=torch.float64)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=torch.float64)

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("bulk fit preparation must not replace the autograd path")

    monkeypatch.setattr(xas_torch, "_prepare_phase_fit_rows", unexpected_prepare)
    monkeypatch.setattr(xas_torch, "_prepare_amp_fit_rows", unexpected_prepare)

    with torch_context(frequencies):
        for function, coeffs in (
            (xas_torch.Phase, phase_coeffs),
            (xas_torch.Amp, amp_coeffs),
        ):
            theta = torch.tensor(
                [40.0, 20.0, 0.4, -0.2],
                dtype=torch.float64,
                requires_grad=True,
            )
            values = function(frequencies, theta, coeffs, chip=0.3)
            weights = torch.linspace(0.2, 1.2, values.numel(), dtype=values.dtype)
            gradient = torch.autograd.grad((values * weights).sum(), theta)[0]
            assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_phase_plan_matches_eager_piecewise_phase_bitwise(dtype):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=dtype)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=dtype)

    with torch_context(theta):
        plan = xas_torch._prepare_phase_plan(theta, phase_coeffs, chip=0.3)
        frequencies = torch.stack(
            (
                0.5 * plan.f1_Ms,
                plan.f1_Ms,
                0.5 * (plan.f1_Ms + plan.f2_Ms),
                plan.f2_Ms,
                1.1 * plan.f2_Ms,
            )
        ) / plan.total_mass_seconds
        eager = xas_torch.Phase(
            frequencies,
            theta,
            phase_coeffs,
            chip=0.3,
        )
        planned = xas_torch.Phase(
            frequencies,
            theta,
            phase_coeffs,
            chip=0.3,
            _phase_plan=plan,
        )
        eager_derivative = xas_torch.PhaseDerivative(
            frequencies,
            theta,
            phase_coeffs,
            chip=0.3,
        )
        planned_derivative = xas_torch.PhaseDerivative(
            frequencies,
            theta,
            phase_coeffs,
            chip=0.3,
            _phase_plan=plan,
        )
        dimensionless = frequencies * plan.total_mass_seconds
        eager_inspiral = xas_torch.get_inspiral_phase(
            dimensionless,
            theta,
            phase_coeffs,
        )
        planned_inspiral = xas_torch.get_inspiral_phase(
            dimensionless,
            theta,
            phase_coeffs,
            _phase_plan=plan,
        )

    assert torch.equal(planned, eager)
    assert torch.equal(planned_derivative, eager_derivative)
    assert torch.equal(planned_inspiral, eager_inspiral)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_xas_phase_plan_gate_reuses_once_and_is_bitwise(
    dtype,
    monkeypatch,
):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=dtype)
    extrinsic = torch.tensor([500.0, 0.0, 0.3], dtype=dtype)
    frequencies = torch.linspace(20.0, 400.0, 100, dtype=dtype)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=dtype)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=dtype)

    monkeypatch.setenv(xas_torch._PHASE_PLAN_ENV, "0")
    with torch_context(frequencies):
        eager = xas_torch._gen_IMRPhenomXAS(
            frequencies,
            theta,
            extrinsic,
            phase_coeffs,
            amp_coeffs,
            30.0,
            chip=0.3,
        )

    original = xas_torch._prepare_phase_plan
    call_count = 0

    def counted_prepare(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(xas_torch, "_prepare_phase_plan", counted_prepare)
    monkeypatch.setenv(xas_torch._PHASE_PLAN_ENV, "1")
    with torch_context(frequencies):
        planned, plan = xas_torch._gen_IMRPhenomXAS(
            frequencies,
            theta,
            extrinsic,
            phase_coeffs,
            amp_coeffs,
            30.0,
            chip=0.3,
            return_phase_plan=True,
        )

    assert call_count == 1
    assert plan is not None
    assert torch.equal(planned, eager)


def test_phase_plan_gate_bypasses_differentiable_intrinsics(monkeypatch):
    theta = torch.tensor(
        [40.0, 20.0, 0.4, -0.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    extrinsic = torch.tensor([500.0, 0.0, 0.3], dtype=torch.float64)
    frequencies = torch.linspace(20.0, 300.0, 32, dtype=torch.float64)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=torch.float64)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=torch.float64)

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("request-local phase reuse must bypass autograd inputs")

    monkeypatch.setenv(xas_torch._PHASE_PLAN_ENV, "1")
    monkeypatch.setattr(xas_torch, "_prepare_phase_plan", unexpected_prepare)
    with torch_context(frequencies):
        waveform, plan = xas_torch._gen_IMRPhenomXAS(
            frequencies,
            theta,
            extrinsic,
            phase_coeffs,
            amp_coeffs,
            30.0,
            chip=0.3,
            return_phase_plan=True,
        )
        gradient = torch.autograd.grad(waveform.abs().sum(), theta)[0]

    assert plan is None
    assert torch.isfinite(gradient).all()


def test_phase_plan_switch_is_strict_and_defaults_off(monkeypatch):
    monkeypatch.delenv(xas_torch._PHASE_PLAN_ENV, raising=False)
    assert not xas_torch._phase_plan_enabled()
    monkeypatch.setenv(xas_torch._PHASE_PLAN_ENV, "maybe")
    with pytest.raises(ValueError, match=xas_torch._PHASE_PLAN_ENV):
        xas_torch._phase_plan_enabled()


def test_fixed_schema_phase_plan_switch_is_strict_and_defaults_off(
    monkeypatch,
):
    monkeypatch.delenv(
        xas_torch._FIXED_SCHEMA_PHASE_PLAN_ENV,
        raising=False,
    )
    assert not xas_torch._fixed_schema_phase_plan_enabled()
    monkeypatch.setenv(
        xas_torch._FIXED_SCHEMA_PHASE_PLAN_ENV,
        "maybe",
    )
    with pytest.raises(
        ValueError,
        match=xas_torch._FIXED_SCHEMA_PHASE_PLAN_ENV,
    ):
        xas_torch._fixed_schema_phase_plan_enabled()


def test_region_pruning_switch_is_strict_and_defaults_off(monkeypatch):
    monkeypatch.delenv(xas_torch._REGION_PRUNING_ENV, raising=False)
    assert not xas_torch._region_pruning_enabled()
    monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "maybe")
    with pytest.raises(ValueError, match=xas_torch._REGION_PRUNING_ENV):
        xas_torch._region_pruning_enabled()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize(
    "system",
    (
        (40.0, 20.0, 0.4, -0.2),
        (12.0, 35.0, -0.7, 0.6),
        (80.0, 8.0, 0.9, -0.8),
    ),
)
def test_region_pruning_is_request_local_and_bitwise(
    dtype,
    system,
    monkeypatch,
    one_torch_thread,
):
    theta = torch.tensor(system, dtype=dtype)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=dtype)
    amp_coeffs = xutils.PhenomX_amp_coeff_table.to(dtype=dtype)
    phase_pruned_calls = 0
    amp_pruned_calls = 0
    original_pruned_phase = xas_torch._evaluate_pruned_phase
    original_pruned_amp = xas_torch._evaluate_pruned_amp

    def counted_phase(*args, **kwargs):
        nonlocal phase_pruned_calls
        phase_pruned_calls += 1
        return original_pruned_phase(*args, **kwargs)

    def counted_amp(*args, **kwargs):
        nonlocal amp_pruned_calls
        amp_pruned_calls += 1
        return original_pruned_amp(*args, **kwargs)

    monkeypatch.setattr(xas_torch, "_evaluate_pruned_phase", counted_phase)
    monkeypatch.setattr(xas_torch, "_evaluate_pruned_amp", counted_amp)
    monkeypatch.setenv(xas_torch._AMP_PLAN_ENV, "1")

    with torch_context(theta):
        plan = xas_torch._prepare_phase_plan(theta, phase_coeffs, chip=0.3)
    unit_plan = plan._replace(
        total_mass_seconds=torch.ones_like(plan.total_mass_seconds)
    )

    for length, irregular in ((513, False), (985, True)):
        if irregular:
            increments = torch.linspace(0.5, 1.5, length, dtype=dtype)
            frequency = 0.001 + 0.298 * increments.cumsum(0) / increments.sum()
        else:
            frequency = torch.linspace(0.001, 0.299, length, dtype=dtype)

        monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "0")
        with torch_context(frequency):
            dense_phase = xas_torch._evaluate_phase(frequency, unit_plan)
        monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "1")
        with torch_context(frequency):
            pruned_phase = xas_torch._evaluate_phase(frequency, unit_plan)
        assert _raw_tensor_equal(pruned_phase, dense_phase)

        physical_frequency = torch.linspace(20.0, 512.0, length, dtype=dtype)
        monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "0")
        with torch_context(physical_frequency):
            dense_amp = xas_torch.Amp(
                physical_frequency,
                theta,
                amp_coeffs,
                D=320.0,
                chip=0.3,
            )
        monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "1")
        with torch_context(physical_frequency):
            pruned_amp = xas_torch.Amp(
                physical_frequency,
                theta,
                amp_coeffs,
                D=320.0,
                chip=0.3,
            )
        assert _raw_tensor_equal(pruned_amp, dense_amp)

    assert phase_pruned_calls == 2
    assert amp_pruned_calls == 2


def test_region_pruning_preserves_dense_boundary_blends(
    monkeypatch,
    one_torch_thread,
):
    dtype = torch.float64
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=dtype)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=dtype)
    with torch_context(theta):
        plan = xas_torch._prepare_phase_plan(theta, phase_coeffs, chip=0.3)
    plan = plan._replace(total_mass_seconds=torch.ones_like(plan.total_mass_seconds))

    def unexpected_pruned(*args, **kwargs):
        pytest.fail("an exact boundary sample must retain the legacy blend")

    monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "1")
    monkeypatch.setattr(xas_torch, "_evaluate_pruned_phase", unexpected_pruned)
    for boundary in (
        plan.f1_Ms,
        plan.f2_Ms,
        torch.as_tensor(xutils.fM_CUT, dtype=dtype),
    ):
        frequency = torch.linspace(0.001, 0.299, 512, dtype=dtype)
        frequency = torch.sort(torch.cat((frequency, boundary.reshape(1)))).values
        assert xas_torch._region_pruning_vector_supported(frequency)
        with torch_context(frequency):
            guarded = xas_torch._evaluate_phase(frequency, plan)
        monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "0")
        with torch_context(frequency):
            dense = xas_torch._evaluate_phase(frequency, plan)
        assert _raw_tensor_equal(guarded, dense)
        monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "1")


def test_region_pruning_uses_globally_aligned_spans(
    monkeypatch,
    one_torch_thread,
):
    dtype = torch.float64
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=dtype)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=dtype)
    with torch_context(theta):
        plan = xas_torch._prepare_phase_plan(theta, phase_coeffs, chip=0.3)
    plan = plan._replace(total_mass_seconds=torch.ones_like(plan.total_mass_seconds))
    frequency = torch.linspace(0.001, 0.1, 513, dtype=dtype)
    calls = []

    for name in (
        "_evaluate_inspiral_phase",
        "_evaluate_intermediate_phase",
        "_evaluate_mergerringdown_phase",
    ):
        original = getattr(xas_torch, name)

        def counted(value, *args, _original=original, **kwargs):
            calls.append((value.storage_offset(), value.numel()))
            return _original(value, *args, **kwargs)

        monkeypatch.setattr(xas_torch, name, counted)

    monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "1")
    with torch_context(frequency):
        xas_torch._evaluate_phase(frequency, plan)

    assert len(calls) == 3
    assert all(offset % xas_torch._REGION_PRUNING_ALIGNMENT == 0 for offset, _ in calls)
    assert all(length < frequency.numel() for _, length in calls)
    assert all(
        length % xas_torch._REGION_PRUNING_ALIGNMENT == 0
        or offset + length == frequency.numel()
        for offset, length in calls
    )


def test_region_pruning_guards_unsupported_tensor_semantics(
    monkeypatch,
    one_torch_thread,
):
    dtype = torch.float64
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=dtype)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=dtype)
    with torch_context(theta):
        plan = xas_torch._prepare_phase_plan(theta, phase_coeffs, chip=0.3)
    plan = plan._replace(total_mass_seconds=torch.ones_like(plan.total_mass_seconds))
    frequency = torch.linspace(0.001, 0.299, 513, dtype=dtype)
    first, second = plan.f1_Ms, plan.f2_Ms
    monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "1")

    def indices(source=frequency, dimensionless=frequency, candidate_plan=plan):
        return xas_torch._piecewise_region_indices(
            source,
            dimensionless,
            first,
            second,
            candidate_plan,
        )

    assert indices() is not None
    assert (
        indices(
            source=frequency[:511].clone(),
            dimensionless=frequency[:511].clone(),
        )
        is None
    )
    noncontiguous = torch.stack((frequency, frequency), dim=1)[:, 0]
    assert indices(source=noncontiguous, dimensionless=noncontiguous) is None
    offset_view = torch.cat((frequency[:1], frequency))[1:]
    assert indices(source=offset_view, dimensionless=offset_view) is None
    assert indices(source=frequency.reshape(1, -1), dimensionless=frequency) is None
    assert indices(source=frequency.to(torch.float16), dimensionless=frequency) is None
    assert indices(source=torch._neg_view(frequency), dimensionless=frequency) is None
    assert indices(source=frequency.to_sparse(), dimensionless=frequency) is None

    class TensorSubclass(torch.Tensor):
        pass

    subclass = frequency.as_subclass(TensorSubclass)
    assert indices(source=subclass, dimensionless=subclass) is None
    requiring_grad = frequency.detach().clone().requires_grad_()
    assert indices(source=requiring_grad, dimensionless=requiring_grad) is None
    unsorted = frequency.clone()
    unsorted[[40, 41]] = unsorted[[41, 40]]
    assert indices(source=unsorted, dimensionless=unsorted) is None
    for bad_value in (-0.0, 0.0, float("nan"), float("inf"), 0.31):
        invalid = frequency.clone()
        invalid[0] = bad_value
        if bad_value == 0.31:
            invalid = torch.sort(invalid).values.clone()
        assert indices(source=invalid, dimensionless=invalid) is None

    exact_boundary = frequency.clone()
    exact_boundary[20] = first
    exact_boundary = torch.sort(exact_boundary).values.clone()
    assert indices(source=exact_boundary, dimensionless=exact_boundary) is None
    graph_plan = plan._replace(alpha0=plan.alpha0.detach().requires_grad_())
    assert indices(candidate_plan=graph_plan) is None
    with torch.autograd.forward_ad.dual_level():
        dual = torch.autograd.forward_ad.make_dual(
            frequency.clone(),
            torch.ones_like(frequency),
        )
        assert indices(source=dual, dimensionless=dual) is None
    monkeypatch.setattr(xas_torch.torch, "get_num_threads", lambda: 2)
    assert indices() is None


def test_region_pruning_rejects_devices_before_reductions(
    monkeypatch,
    one_torch_thread,
):
    frequency = torch.empty(513, device="meta", dtype=torch.float64)
    boundary = torch.tensor(0.02, dtype=torch.float64)
    monkeypatch.setenv(xas_torch._REGION_PRUNING_ENV, "1")

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported-device fallback must not reduce or extract scalars")

    monkeypatch.setattr(xas_torch.torch, "all", unexpected)
    monkeypatch.setattr(xas_torch.torch, "searchsorted", unexpected)
    assert (
        xas_torch._piecewise_region_indices(
            frequency,
            frequency,
            boundary,
            2.0 * boundary,
            object(),
        )
        is None
    )


def test_scalar_region_dispatch_switch_is_strict_and_defaults_off(monkeypatch):
    monkeypatch.delenv(xas_torch._SCALAR_REGION_DISPATCH_ENV, raising=False)
    assert not xas_torch._scalar_region_dispatch_enabled()
    monkeypatch.setenv(xas_torch._SCALAR_REGION_DISPATCH_ENV, "maybe")
    with pytest.raises(ValueError, match=xas_torch._SCALAR_REGION_DISPATCH_ENV):
        xas_torch._scalar_region_dispatch_enabled()


@pytest.mark.parametrize(
    ("dtype", "integer_dtype"),
    (
        (torch.float32, torch.int32),
        (torch.float64, torch.int64),
    ),
)
def test_scalar_region_dispatch_is_request_local_and_bitwise(
    dtype,
    integer_dtype,
    monkeypatch,
):
    theta = torch.tensor([40.0, 20.0, 0.4, -0.2], dtype=dtype)
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=dtype)
    monkeypatch.setenv(xas_torch._SCALAR_REGION_DISPATCH_ENV, "1")

    with torch_context(theta):
        plan = xas_torch._prepare_phase_plan(theta, phase_coeffs, chip=0.3)

    assert plan.scalar_region_dispatch
    # The gate is captured once by the immutable request plan.
    monkeypatch.setenv(xas_torch._SCALAR_REGION_DISPATCH_ENV, "0")
    unit_plan = plan._replace(
        total_mass_seconds=torch.ones_like(plan.total_mass_seconds)
    )
    dense_plan = unit_plan._replace(scalar_region_dispatch=False)
    cutoff = torch.as_tensor(xutils.fM_CUT, dtype=theta.dtype)
    probes = (
        (0, 0.5 * unit_plan.f1_Ms),
        (None, unit_plan.f1_Ms),
        (1, 0.5 * (unit_plan.f1_Ms + unit_plan.f2_Ms)),
        (None, unit_plan.f2_Ms),
        (2, 0.5 * (unit_plan.f2_Ms + cutoff)),
        (None, cutoff),
    )

    with torch_context(theta):
        for expected_region, frequency in probes:
            selected = xas_torch._scalar_phase_region(frequency, unit_plan)
            if expected_region is None:
                assert selected is None
            else:
                assert selected is not None
                assert selected[0] == expected_region

            for evaluator in (
                xas_torch._evaluate_phase,
                xas_torch._evaluate_phase_derivative,
            ):
                dispatched = evaluator(frequency, unit_plan)
                dense = evaluator(frequency, dense_plan)
                assert torch.equal(
                    dispatched.view(integer_dtype),
                    dense.view(integer_dtype),
                )

    vector = torch.stack((probes[0][1], probes[2][1]))
    graph_frequency = probes[0][1].detach().clone().requires_grad_(True)
    assert xas_torch._scalar_phase_region(vector, unit_plan) is None
    assert xas_torch._scalar_phase_region(graph_frequency, unit_plan) is None
    with torch.autograd.forward_ad.dual_level():
        dual_frequency = torch.autograd.forward_ad.make_dual(
            probes[0][1],
            torch.ones_like(probes[0][1]),
        )
        assert xas_torch._scalar_phase_region(dual_frequency, unit_plan) is None
    assert xas_torch._scalar_phase_region(float(probes[0][1]), unit_plan) is None
    assert (
        xas_torch._scalar_phase_region(
            torch.empty((), device="meta"),
            unit_plan,
        )
        is None
    )


def test_scalar_region_dispatch_bypasses_graph_bearing_plan(monkeypatch):
    theta = torch.tensor(
        [40.0, 20.0, 0.4, -0.2],
        dtype=torch.float64,
        requires_grad=True,
    )
    phase_coeffs = xutils.PhenomX_phase_coeff_table.to(dtype=torch.float64)
    monkeypatch.setenv(xas_torch._SCALAR_REGION_DISPATCH_ENV, "1")

    with torch_context(theta):
        plan = xas_torch._prepare_phase_plan(theta, phase_coeffs, chip=0.3)

    assert not plan.scalar_region_dispatch


CASES = [
    dict(
        mass1=35.0,
        mass2=28.0,
        spin1z=0.2,
        spin2z=-0.1,
        delta_f=0.25,
        f_lower=20.0,
        f_ref=30.0,
        distance=500.0,
        inclination=0.4,
        coa_phase=1.1,
        long_asc_nodes=0.37,
    ),
    dict(
        mass1=10.0,
        mass2=8.0,
        spin1z=0.6,
        spin2z=0.3,
        delta_f=0.5,
        f_lower=15.0,
        f_ref=30.0,
        distance=300.0,
        inclination=1.2,
        coa_phase=0.3,
    ),
    dict(
        mass1=67.0,
        mass2=43.5,
        spin1z=0.9,
        spin2z=-0.17,
        delta_f=0.5,
        f_lower=19.0,
        f_ref=245.0,
        distance=407.0,
        inclination=0.8,
        coa_phase=0.6,
    ),
    dict(
        mass1=18.0,
        mass2=42.0,
        spin1z=-0.4,
        spin2z=0.7,
        delta_f=0.25,
        f_lower=17.3,
        f_final=133.3,
        f_ref=0.0,
        distance=700.0,
        inclination=0.8,
        coa_phase=0.6,
    ),
]

TIDAL_CASES = [
    dict(
        mass1=1.4,
        mass2=1.3,
        spin1z=0.03,
        spin2z=-0.02,
        lambda1=400.0,
        lambda2=700.0,
        delta_f=0.5,
        f_lower=20.0,
        f_final=2048.0,
        f_ref=30.0,
        distance=100.0,
        inclination=0.4,
    ),
    dict(
        mass1=1.2,
        mass2=1.6,
        spin1z=-0.04,
        spin2z=0.05,
        lambda1=800.0,
        lambda2=300.0,
        dquad_mon1=3.0,
        dquad_mon2=4.0,
        delta_f=0.25,
        f_lower=19.3,
        f_final=1024.0,
        f_ref=0.0,
        distance=130.0,
        inclination=0.8,
        long_asc_nodes=0.2,
    ),
    dict(
        mass1=1.7,
        mass2=1.1,
        spin1z=0.1,
        spin2z=-0.03,
        lambda1=0.0,
        lambda2=0.0,
        delta_f=0.5,
        f_lower=20.0,
        f_ref=30.0,
        distance=90.0,
        inclination=0.3,
    ),
]

SEQUENCE_CASES = [
    (
        CASES[0],
        [20.0, 23.5, 30.0, 45.0, 100.0, 250.0, 400.0, 10000.0],
    ),
    (
        CASES[3],
        [17.3, 400.0, 22.0, 150.0],
    ),
]

TIDAL_SEQUENCE_CASES = [
    (
        TIDAL_CASES[0],
        [20.0, 23.5, 30.0, 50.0, 100.0, 500.0, 1000.0, 2048.0, 10000.0],
    ),
    (
        TIDAL_CASES[1],
        [19.3, 1500.0, 30.0, 1024.0],
    ),
]


def _sequence_params(params):
    return {
        key: value
        for key, value in params.items()
        if key not in {"delta_f", "f_lower", "f_final"}
    }


@pytest.mark.parametrize("params", CASES)
def test_imrphenomxas_matches_lal(
    params, monkeypatch, preserve_scheme
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXAS", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform(approximant="IMRPhenomXAS", **params)

    for expected, expected_array, result in zip(
        reference, reference_arrays, actual
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128

        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array == 0.0,
            expected_array == 0.0,
        )
        nonzero = np.abs(expected_array) > 0.0
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < 1.0e-10


@pytest.mark.parametrize(
    "approximant",
    ["IMRPhenomXAS_NRTidalv2", "IMRPhenomXAS_NRTidalv3"],
)
@pytest.mark.parametrize("params", TIDAL_CASES)
def test_imrphenomxas_nrtidal_matches_lal(
    approximant, params, monkeypatch, preserve_scheme
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant=approximant, **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform(approximant=approximant, **params)

    for expected, expected_array, result in zip(
        reference, reference_arrays, actual
    ):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128

        result_array = result.numpy()
        relative_error = np.linalg.norm(
            result_array - expected_array
        ) / np.linalg.norm(expected_array)
        assert relative_error < 5.0e-6

        # XHM's public LAL route is multibanded, so interpolation residuals
        # dominate pointwise relative error deep in the Planck-taper tail.
        significant = np.abs(expected_array) > (
            1.0e-4 * np.max(np.abs(expected_array))
        )
        point_error = np.max(
            np.abs(result_array[significant] - expected_array[significant])
            / np.abs(expected_array[significant])
        )
        assert point_error < 1.0e-4


def test_imrphenomxas_public_dispatch_does_not_call_lal(
    monkeypatch, preserve_scheme
):
    params = {
        **CASES[0],
        "phase_order": 2.5,
        "amplitude_order": "3",
        "eccentricity_order": 4,
    }
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXAS", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomxas_torch as xas_mod
    import pycbc.waveform.waveform as waveform_mod

    native = xas_mod.imrphenomxas_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXAS called lalsimulation")

    monkeypatch.setattr(xas_mod, "imrphenomxas_fd_torch", recording_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform(approximant="IMRPhenomXAS", **params)

    assert calls == 1
    for expected, result in zip(reference_arrays, actual):
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=1.0e-10, atol=0.0
        )


@pytest.mark.parametrize(
    "approximant",
    ["IMRPhenomXAS_NRTidalv2", "IMRPhenomXAS_NRTidalv3"],
)
def test_imrphenomxas_nrtidal_dispatch_does_not_call_lal(
    approximant, monkeypatch, preserve_scheme
):
    params = {
        **TIDAL_CASES[0],
        "phase_order": 2.5,
        "amplitude_order": "3",
        "eccentricity_order": 4,
    }
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant=approximant, **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.waveform as waveform_mod

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError(f"native {approximant} called LAL")

    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform(approximant=approximant, **params)

    for expected, result in zip(reference_arrays, actual):
        relative_error = np.linalg.norm(
            result.numpy() - expected
        ) / np.linalg.norm(expected)
        assert relative_error < 5.0e-6


@pytest.mark.parametrize(
    "flag",
    ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOMXAS_NATIVE"),
)
def test_imrphenomxas_switch_fallback(flag, monkeypatch, preserve_scheme):
    params = CASES[0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXAS", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomxas_torch as xas_mod

    def unexpected_native(**_params):
        raise AssertionError(f"{flag}=0 did not disable native IMRPhenomXAS")

    monkeypatch.setattr(xas_mod, "imrphenomxas_fd_torch", unexpected_native)
    _clear_native_flags(monkeypatch)
    monkeypatch.setenv(flag, "0")
    _activate_scheme(_scheme.TorchScheme())
    fallback = get_fd_waveform(approximant="IMRPhenomXAS", **params)

    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-14, atol=0.0
        )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"spin1x": 0.1}, False),
        ({"lambda1": 100.0}, False),
        ({"eccentricity": 0.01}, False),
        ({"phase_order": 7}, True),
        ({"phase_order": 2.5}, True),
        ({"amplitude_order": "3"}, True),
        ({"eccentricity_order": 4}, True),
        ({"eccentricity_order": 4.0}, False),
        ({"phase_order": 1 << 31}, False),
        ({"spin_order": 4}, False),
        ({"spin_order": -1.5}, True),
        ({"tidal_order": 12}, False),
        ({"tidal_order": -1.0}, True),
        ({"dchi3": 0.1}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"frame_axis": 1}, False),
        ({"numrel_data": "data.h5"}, False),
        ({"approximant": "IMRPhenomXP"}, False),
    ],
)
def test_imrphenomxas_native_support_boundary(changes, expected):
    params = {"approximant": "IMRPhenomXAS", **changes}
    assert imrphenomxas_native_supported(params) is expected


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"lambda1": 400.0, "lambda2": 700.0}, True),
        ({"lambda1": 400.0, "dquad_mon1": 3.0}, True),
        ({"lambda1": -1.0}, False),
        ({"lambda1": float("nan")}, False),
        ({"dquad_mon1": -1.0}, False),
        ({"lambda_octu1": 10.0}, False),
        ({"mode_array": [(2, 2)]}, False),
    ],
)
@pytest.mark.parametrize(
    "approximant",
    ["IMRPhenomXAS_NRTidalv2", "IMRPhenomXAS_NRTidalv3"],
)
def test_imrphenomxas_nrtidal_native_support_boundary(
    approximant, changes, expected
):
    params = {"approximant": approximant, **changes}
    assert imrphenomxas_native_supported(params) is expected


@pytest.mark.parametrize(
    ("approximant", "extra", "last_nonzero"),
    [
        ("IMRPhenomXAS", {}, 4096),
        (
            "IMRPhenomXAS_NRTidalv2",
            {"lambda1": 400.0, "lambda2": 700.0},
            4095,
        ),
        (
            "IMRPhenomXAS_NRTidalv3",
            {"lambda1": 400.0, "lambda2": 700.0},
            4095,
        ),
    ],
)
def test_imrphenomxas_power_of_two_layout_boundary(
    approximant,
    extra,
    last_nonzero,
    monkeypatch,
    preserve_scheme,
):
    params = dict(
        mass1=1.4,
        mass2=1.3,
        spin1z=0.03,
        spin2z=-0.02,
        delta_f=0.25,
        f_lower=20.0,
        f_final=1024.1,
        f_ref=30.0,
        distance=100.0,
        **extra,
    )
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    hp, _ = get_fd_waveform(approximant=approximant, **params)

    assert len(hp) == 4097
    assert np.flatnonzero(hp.numpy())[-1] == last_nonzero


def test_imrphenomxas_unsupported_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    params = {**CASES[0], "dchi3": 0.1}
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXAS", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.imrphenomxas_torch as xas_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported IMRPhenomXAS parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(xas_mod, "imrphenomxas_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme())
    fallback = get_fd_waveform(approximant="IMRPhenomXAS", **params)

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-14, atol=0.0
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize(
    ("approximant", "params", "cpu_tolerance"),
    [
        ("IMRPhenomXAS", CASES[0], 1.0e-10),
        ("IMRPhenomXAS_NRTidalv2", TIDAL_CASES[0], 5.0e-6),
        ("IMRPhenomXAS_NRTidalv3", TIDAL_CASES[0], 5.0e-6),
    ],
)
def test_imrphenomxas_stays_on_requested_device(
    device_name,
    approximant,
    params,
    cpu_tolerance,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform(approximant=approximant, **params)
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform(approximant=approximant, **params)

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    actual_array = actual.numpy()
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 5.0e-3 if device_name == "mps" else cpu_tolerance
    assert relative_error < tolerance


@pytest.mark.parametrize(
    ("approximant", "params"),
    [
        ("IMRPhenomXAS", CASES[0]),
        ("IMRPhenomXAS_NRTidalv2", TIDAL_CASES[0]),
        ("IMRPhenomXAS_NRTidalv3", TIDAL_CASES[0]),
    ],
)
def test_imrphenomxas_native_avoids_host_transfer(
    approximant, params, monkeypatch, preserve_scheme
):
    from pycbc.types.array_torch import TorchArrayData

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomXAS transferred data to NumPy")

    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    with torch.no_grad():
        hp, hc = get_fd_waveform(approximant=approximant, **params)

    assert isinstance(hp._data.tensor, torch.Tensor)
    assert isinstance(hc._data.tensor, torch.Tensor)


@pytest.mark.parametrize(("params", "sample_points"), SEQUENCE_CASES)
def test_imrphenomxas_sequence_matches_lal(
    params, sample_points, monkeypatch, preserve_scheme
):
    params = {
        **_sequence_params(params),
        "phase_order": 2.5,
        "amplitude_order": "3",
        "eccentricity_order": 4,
    }
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array == 0.0,
            expected == 0.0,
        )
        nonzero = np.abs(expected) > 0.0
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 1.0e-10


@pytest.mark.parametrize(
    "approximant",
    ["IMRPhenomXAS_NRTidalv2", "IMRPhenomXAS_NRTidalv3"],
)
@pytest.mark.parametrize(("params", "sample_points"), TIDAL_SEQUENCE_CASES)
def test_imrphenomxas_nrtidal_sequence_matches_lal(
    approximant, params, sample_points, monkeypatch, preserve_scheme
):
    params = _sequence_params(params)
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        np.testing.assert_array_equal(
            result_array == 0.0,
            expected == 0.0,
        )
        nonzero = np.abs(expected) > 0.0
        relative_error = np.linalg.norm(
            result_array[nonzero] - expected[nonzero]
        ) / np.linalg.norm(expected[nonzero])
        assert relative_error < 1.0e-9


@pytest.mark.parametrize(
    ("approximant", "params", "sample_points", "tolerance"),
    [
        ("IMRPhenomXAS", CASES[0], SEQUENCE_CASES[0][1], 1.0e-10),
        (
            "IMRPhenomXAS_NRTidalv2",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
            1.0e-9,
        ),
        (
            "IMRPhenomXAS_NRTidalv3",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
            1.0e-9,
        ),
    ],
)
def test_imrphenomxas_sequence_public_dispatch_does_not_call_lal(
    approximant,
    params,
    sample_points,
    tolerance,
    monkeypatch,
    preserve_scheme,
):
    params = _sequence_params(params)
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.imrphenomxas_torch as xas_mod
    import pycbc.waveform.waveform as waveform_mod

    native = xas_mod.imrphenomxas_fd_sequence_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError(f"native {approximant} sequence called LAL")

    monkeypatch.setattr(
        xas_mod,
        "imrphenomxas_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme())
    actual = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    assert calls == 1
    for expected, result in zip(reference_arrays, actual):
        assert isinstance(result._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            result.numpy(), expected, rtol=tolerance, atol=0.0
        )


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, True),
        ({"approximant": "IMRPhenomXAS"}, True),
        ({"approximant": "IMRPhenomXAS_NRTidalv2"}, True),
        ({"approximant": "IMRPhenomXAS_NRTidalv3"}, True),
        (
            {"approximant": "IMRPhenomXAS_NRTidalv3", "lambda1": -1.0},
            False,
        ),
        ({"approximant": "IMRPhenomXP"}, False),
        ({"dchi3": 0.1}, False),
        ({"lambda1": 100.0}, False),
    ],
)
def test_imrphenomxas_sequence_native_support_boundary(changes, expected):
    assert imrphenomxas_sequence_native_supported(changes) is expected


def test_imrphenomxas_sequence_unsupported_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    params = {**_sequence_params(CASES[0]), "dchi3": 0.1}
    sample_points = SEQUENCE_CASES[0][1]
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.imrphenomxas_torch as xas_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported XAS sequence parameters reached Torch")

    lal_generator = (
        waveform_mod.lalsimulation.SimInspiralChooseFDWaveformSequence
    )
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        xas_mod,
        "imrphenomxas_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme())
    fallback = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-14, atol=0.0
        )


def test_imrphenomxas_sequence_lal_fallback_supports_mps(
    monkeypatch, preserve_scheme
):
    if not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")

    params = {**_sequence_params(CASES[0]), "dchi3": 0.1}
    sample_points = SEQUENCE_CASES[0][1]
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme("mps"))
    fallback = get_fd_waveform_sequence(
        approximant="IMRPhenomXAS",
        sample_points=sample_points,
        **params,
    )

    for expected, actual in zip(reference_arrays, fallback):
        assert actual._data.tensor.device.type == "mps"
        assert actual._data.tensor.dtype == torch.complex64
        np.testing.assert_allclose(
            actual.numpy(), expected, rtol=1.0e-6, atol=0.0
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
@pytest.mark.parametrize(
    ("approximant", "params", "sample_points", "cpu_tolerance"),
    [
        ("IMRPhenomXAS", CASES[0], SEQUENCE_CASES[0][1], 1.0e-10),
        (
            "IMRPhenomXAS_NRTidalv2",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
            1.0e-9,
        ),
        (
            "IMRPhenomXAS_NRTidalv3",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
            1.0e-9,
        ),
    ],
)
def test_imrphenomxas_sequence_stays_on_requested_device(
    device_name,
    approximant,
    params,
    sample_points,
    cpu_tolerance,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = _sequence_params(params)
    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform_sequence(
        approximant=approximant,
        sample_points=sample_points,
        **params,
    )

    expected_dtype = (
        torch.complex64 if device_name == "mps" else torch.complex128
    )
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    actual_array = actual.numpy()
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 5.0e-3 if device_name == "mps" else cpu_tolerance
    assert relative_error < tolerance


@pytest.mark.parametrize(
    ("approximant", "params", "sample_values"),
    [
        ("IMRPhenomXAS", CASES[0], SEQUENCE_CASES[0][1]),
        (
            "IMRPhenomXAS_NRTidalv2",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
        ),
        (
            "IMRPhenomXAS_NRTidalv3",
            TIDAL_CASES[0],
            TIDAL_SEQUENCE_CASES[0][1],
        ),
    ],
)
def test_imrphenomxas_sequence_native_avoids_host_transfer(
    approximant,
    params,
    sample_values,
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomXAS sequence transferred to NumPy")

    monkeypatch.setenv("PYCBC_IMRPHENOMXAS_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme())
    sample_points = Array(sample_values)
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        hp, hc = get_fd_waveform_sequence(
            approximant=approximant,
            sample_points=sample_points,
            **_sequence_params(params),
        )

    assert isinstance(hp._data.tensor, torch.Tensor)
    assert isinstance(hc._data.tensor, torch.Tensor)


@pytest.mark.parametrize("batch_size", (2, 4))
def test_imrphenomxas_fd_batch_execution(batch_size):
    """Verify batched IMRPhenomXAS execution for batch size B > 1."""
    from pycbc.waveform.imrphenomxas_torch import imrphenomxas_fd_batch

    m1 = torch.linspace(20.0, 35.0, batch_size, dtype=torch.float64)
    m2 = torch.linspace(10.0, 15.0, batch_size, dtype=torch.float64)
    s1z = torch.linspace(0.1, 0.4, batch_size, dtype=torch.float64)
    s2z = torch.linspace(-0.2, 0.3, batch_size, dtype=torch.float64)
    dist = torch.full((batch_size,), 100.0, dtype=torch.float64)
    coa_phase = torch.linspace(0.0, 0.5, batch_size, dtype=torch.float64)

    hp, hc = imrphenomxas_fd_batch(
        mass1=m1,
        mass2=m2,
        spin1z=s1z,
        spin2z=s2z,
        distance=dist,
        coa_phase=coa_phase,
        f_lower=20.0,
        delta_f=1.0,
    )
    assert hp.shape[0] == batch_size
    assert hc.shape[0] == batch_size
    assert torch.all(torch.isfinite(hp))
    assert torch.all(torch.isfinite(hc))


@pytest.mark.parametrize("batch_size", (2, 3))
def test_imrphenomxas_fd_batch_tidal_execution(batch_size):
    """Verify batched IMRPhenomXAS with tidal parameters for batch B > 1."""
    from pycbc.waveform.imrphenomxas_torch import imrphenomxas_fd_batch

    m1 = torch.linspace(1.4, 1.6, batch_size, dtype=torch.float64)
    m2 = torch.linspace(1.2, 1.3, batch_size, dtype=torch.float64)
    s1z = torch.linspace(0.01, 0.03, batch_size, dtype=torch.float64)
    s2z = torch.linspace(-0.02, 0.02, batch_size, dtype=torch.float64)
    l1 = torch.linspace(200.0, 500.0, batch_size, dtype=torch.float64)
    l2 = torch.linspace(300.0, 600.0, batch_size, dtype=torch.float64)
    dist = torch.full((batch_size,), 100.0, dtype=torch.float64)
    coa_phase = torch.full((batch_size,), 0.0, dtype=torch.float64)

    hp, hc = imrphenomxas_fd_batch(
        mass1=m1,
        mass2=m2,
        spin1z=s1z,
        spin2z=s2z,
        lambda1=l1,
        lambda2=l2,
        distance=dist,
        coa_phase=coa_phase,
        f_lower=20.0,
        delta_f=1.0,
    )
    assert hp.shape[0] == batch_size
    assert hc.shape[0] == batch_size
    assert torch.all(torch.isfinite(hp))
    assert torch.all(torch.isfinite(hc))
