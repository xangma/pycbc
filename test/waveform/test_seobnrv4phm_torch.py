import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from pycbc import scheme as _scheme
from pycbc.waveform import (
    get_fd_waveform,
    get_fd_waveform_sequence,
    get_td_waveform,
    get_td_waveform_modes,
)
from pycbc.waveform.seobnrv4phm_constants import compute_spin_aligned_hcoeffs

PARAMS = dict(
    mass1=25.0,
    mass2=18.0,
    spin1x=0.2,
    spin1y=-0.15,
    spin1z=0.05,
    spin2x=-0.1,
    spin2y=0.07,
    spin2z=0.2,
    delta_f=0.25,
    f_lower=30.0,
    f_final=128.0,
    f_ref=50.0,
    distance=300.0,
    inclination=0.6,
    coa_phase=0.3,
)

ALIGNED_V4_PARAMS = dict(
    mass1=50.0,
    mass2=40.0,
    spin1z=0.3,
    spin2z=-0.2,
    delta_t=1.0 / 4096.0,
    f_lower=15.0,
    f_ref=15.0,
    distance=400.0,
    inclination=0.4,
    coa_phase=0.2,
)

ALIGNED_V4HM_PARAMS = dict(
    mass1=100.0,
    mass2=60.0,
    spin1z=0.3,
    spin2z=-0.2,
    delta_t=1.0 / 2048.0,
    f_lower=30.0,
    f_ref=0.0,
    distance=500.0,
    inclination=0.7,
    coa_phase=0.4,
    long_asc_nodes=0.0,
)


def _phm_boundary_params():
    return dict(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        delta_t=1.0 / 4096.0,
        delta_f=1.0,
        f_lower=90.0,
        f_final=256.0,
        f_ref=90.0,
        distance=300.0,
        inclination=0.4,
        coa_phase=0.0,
        mode_array=((2, 2), (2, 1)),
    )


RUN_SLOW_NATIVE_PHM = os.environ.get(
    "PYCBC_RUN_SEOBNRV4PHM_NATIVE_SLOW", "0"
) not in ("0", "", "false", "False")
SLOW_NATIVE_PHM_REASON = (
    "full SEOBNRv4PHM native waveform generation is an opt-in slow test; set "
    "PYCBC_RUN_SEOBNRV4PHM_NATIVE_SLOW=1 to run it"
)
RUN_SLOW_NATIVE_V4 = os.environ.get(
    "PYCBC_RUN_SEOBNRV4_NATIVE_SLOW", "0"
) not in ("0", "", "false", "False")
SLOW_NATIVE_V4_REASON = (
    "full SEOBNRv4 native waveform generation is an opt-in slow test; set "
    "PYCBC_RUN_SEOBNRV4_NATIVE_SLOW=1 to run it"
)
RUN_SLOW_NATIVE_V4HM = os.environ.get(
    "PYCBC_RUN_SEOBNRV4HM_NATIVE_SLOW", "0"
) not in ("0", "", "false", "False")
SLOW_NATIVE_V4HM_REASON = (
    "full SEOBNRv4HM native waveform generation is an opt-in slow test; set "
    "PYCBC_RUN_SEOBNRV4HM_NATIVE_SLOW=1 to run it"
)
RUN_CHECKPOINT_DIAGNOSTIC = os.environ.get(
    "PYCBC_RUN_SEOBNRV4PHM_CHECKPOINT_DIAG", "0"
) not in ("0", "", "false", "False")
CHECKPOINT_DIAGNOSTIC_REASON = (
    "SEOBNRv4PHM checkpoint diagnostic is opt-in; set "
    "PYCBC_RUN_SEOBNRV4PHM_CHECKPOINT_DIAG=1 to run it"
)
RUN_BOUNDARY_DIAGNOSTIC = os.environ.get(
    "PYCBC_RUN_SEOBNRV4PHM_BOUNDARY_DIAG", "0"
) not in ("0", "", "false", "False")
BOUNDARY_DIAGNOSTIC_REASON = (
    "SEOBNRv4PHM boundary diagnostic is opt-in; set "
    "PYCBC_RUN_SEOBNRV4PHM_BOUNDARY_DIAG=1 to run it"
)
RUN_IC_STEP4_DIAGNOSTIC = os.environ.get(
    "PYCBC_RUN_SEOBNRV4PHM_IC_STEP4_DIAG", "0"
) not in ("0", "", "false", "False")
IC_STEP4_DIAGNOSTIC_REASON = (
    "SEOBNRv4PHM IC Step 4 diagnostic is opt-in; set "
    "PYCBC_RUN_SEOBNRV4PHM_IC_STEP4_DIAG=1 to run it"
)

# =============================================================================
# Hardware-Agnostic Parity Tolerance Standards:
#
# 1. Exact-Trajectory Components & Diagnostic Checkpoints (relative error < 1e-5):
#    Deterministic invariants, initial conditions (IC Step 4), isolated Runge-Kutta
#    steps, Hamiltonian derivatives, coordinate transforms, attachment floor
#    samples, and dynamics checkpoint summaries. Evaluated against strict
#    floating-point bounds (dt_M < 5e-10, omega_peak_rel < 5e-12, r < 5e-10,
#    IC deltas < 1e-10, relative error < 1e-5).
#
# 2. End-to-End Dynamical ODE Waveforms (aligned mismatch 1 - O < 1e-4, phase scatter < 1e-4 rad):
#    Full time-domain and frequency-domain waveform generation across thousands of
#    adaptive Runge-Kutta ODE steps. Independent Torch and C/GSL adaptive integrators
#    accumulate microscopic roundoff-level differences in RHS evaluations that
#    yield slight divergences in step-acceptance boundaries. Standardized parity
#    gates separate these trajectory integration divergences using linear phase/time
#    alignment:
#    - aligned relative L2 error (aligned_rel < 2.5e-4, giving aligned mismatch 1 - O < 1e-4)
#    - residual phase scatter (aligned_phase_std < 2.5e-4 rad, phase scatter < 1e-4 rad)
#    - unaligned amplitude ratio (within 1.0 +- 1.0e-3)
#    - time alignment shift (linear_dt_s < 5.0e-6 s)
# =============================================================================

_PHM_FD_PARITY_TOLERANCES = {
    # End-to-end dynamical ODE waveform parity tolerances:
    # Independent Torch and C/GSL adaptive trajectories eventually choose
    # slightly different accepted steps from roundoff-level RHS differences.
    # These bounds retain several-fold platform margin around the validated
    # result while remaining tight enough to reject unphysical dephasing.
    "raw_rel": 1.0e-3,
    "amp_min": 1.0 - 1.0e-3,
    "amp_max": 1.0 + 1.0e-3,
    "raw_phase_mean": 1.0e-3,
    "raw_phase_std": 1.0e-3,
    "linear_dt_s": 5.0e-6,
    "aligned_rel": 2.5e-4,
    "aligned_phase_std": 2.5e-4,
}

_PHM_CHECKPOINT_PARITY_TOLERANCES = {
    # Exact-trajectory component & diagnostic checkpoint parity tolerances:
    # Strict deterministic bounds for isolated invariants and step metrics.
    "dt_M": 5.0e-10,
    "omega_peak_rel": 5.0e-12,
    "r": 5.0e-10,
}


def _lal_weighted_spins_to_chi(S1_lal, S2_lal, mass1, mass2):
    """Convert LAL dynamics spin columns into individual dimensionless spins."""
    total_mass = mass1 + mass2
    return S1_lal * (total_mass / mass1) ** 2, S2_lal * (total_mass / mass2) ** 2


def test_seobnrv4phm_exposes_dispatch_entrypoint():
    from pycbc.waveform import seobnrv4phm_torch

    assert seobnrv4phm_torch.torch_native_td_waveform is (
        seobnrv4phm_torch.seobnrv4phm_td_torch
    )
    assert seobnrv4phm_torch.torch_native_fd_waveform is (
        seobnrv4phm_torch.seobnrv4phm_fd_torch
    )
    assert seobnrv4phm_torch.torch_native_fd_sequence_waveform is (
        seobnrv4phm_torch.seobnrv4phm_fd_sequence_torch
    )


def test_seobnrv4phm_enabled_entrypoint_uses_native_implementation(monkeypatch):
    """The public native switch must not fall back through LAL."""
    from pycbc.waveform import seobnrv4phm_torch

    expected_td = object()
    expected_fd = object()
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "_seobnrv4phm_td_native",
        lambda **_params: expected_td,
    )
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "_seobnrv4phm_fd_native",
        lambda **_params: expected_fd,
    )

    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        params = dict(PARAMS, approximant="SEOBNRv4PHM")
        assert seobnrv4phm_torch.seobnrv4phm_td_torch(**params) is expected_td
        assert seobnrv4phm_torch.seobnrv4phm_fd_torch(**params) is expected_fd
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def test_seobnrv4_enabled_entrypoint_uses_aligned_implementation(monkeypatch):
    """The aligned wrapper must select the v4-specific native behavior."""
    from pycbc.waveform import seobnrv4phm_torch

    expected = object()
    calls = []

    def recording_native(*, _aligned_v4=False, **params):
        calls.append((_aligned_v4, params))
        return expected

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "_seobnrv4phm_td_native",
        recording_native,
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = seobnrv4phm_torch.seobnrv4_td_torch(
            approximant="SEOBNRv4", **ALIGNED_V4_PARAMS
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert actual is expected
    assert calls[0][0] is True
    assert calls[0][1]["approximant"] == "SEOBNRv4"


@pytest.mark.parametrize(
    "requested_modes, expected_modes",
    (
        (None, ((2, 2), (2, 1))),
        (((2, 2),), ((2, 2),)),
    ),
    ids=("lal-default", "explicit-22"),
)
def test_seobnrv4p_enabled_entrypoint_resolves_model_modes(
    monkeypatch,
    requested_modes,
    expected_modes,
):
    """The v4P wrapper must select exactly LAL's supported mode subset."""
    from pycbc.waveform import seobnrv4phm_torch

    expected = object()
    calls = []

    def recording_native(**params):
        calls.append(params)
        return expected

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4P_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "_seobnrv4phm_td_native",
        recording_native,
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = seobnrv4phm_torch.seobnrv4p_td_torch(
            approximant="SEOBNRv4P",
            delta_t=1.0 / 4096.0,
            mode_array=requested_modes,
            **PARAMS,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert actual is expected
    assert calls[0]["approximant"] == "SEOBNRv4P"
    assert calls[0]["mode_array"] == expected_modes


@pytest.mark.parametrize(
    "component_flag, generator_name, params",
    (
        (
            "PYCBC_SEOBNRV4_NATIVE",
            "seobnrv4_fd_torch",
            dict(
                ALIGNED_V4_PARAMS,
                approximant="SEOBNRv4",
                delta_f=0.25,
            ),
        ),
        (
            "PYCBC_SEOBNRV4P_NATIVE",
            "seobnrv4p_fd_torch",
            dict(PARAMS, approximant="SEOBNRv4P"),
        ),
    ),
    ids=("aligned-v4", "precessing-v4p"),
)
def test_seobnr_enabled_fd_entrypoint_uses_public_td_conversion(
    monkeypatch,
    component_flag,
    generator_name,
    params,
):
    """Each EOB FD wrapper must retain PyCBC's public TD conversion."""
    from pycbc.waveform import seobnrv4phm_torch, waveform

    expected = object()
    calls = []

    def recording_conversion(
        params,
        *,
        duration_estimator,
        duration_increase,
    ):
        calls.append(
            (
                params,
                type(_scheme.mgr.state),
                duration_estimator,
                duration_increase,
            )
        )
        return expected

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv(component_flag, "1")
    monkeypatch.setattr(
        waveform,
        "_get_fd_waveform_from_td",
        recording_conversion,
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = getattr(seobnrv4phm_torch, generator_name)(**params)
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert actual is expected
    assert len(calls) == 1
    got_params, scheme_type, duration_estimator, duration_increase = calls[0]
    assert got_params == params
    assert scheme_type is _scheme.TorchScheme
    assert duration_estimator.__name__ == "spa_length_in_time"
    assert duration_increase == 1.8


def test_seobnrv4p_sequence_entrypoint_resolves_model_modes(monkeypatch):
    """The v4P sequence wrapper must reuse the restricted shared engine."""
    from pycbc.waveform import seobnrv4phm_torch

    expected = (object(), object())
    calls = []

    def recording_sequence(**params):
        calls.append(params)
        return expected

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4P_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "_seobnrv4phm_fd_sequence_native",
        recording_sequence,
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = seobnrv4phm_torch.seobnrv4p_fd_sequence_torch(
            approximant="SEOBNRv4P",
            sample_points=[50.0],
            **PARAMS,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert actual is expected
    assert calls[0]["approximant"] == "SEOBNRv4P"
    assert calls[0]["mode_array"] == ((2, 2), (2, 1))


@pytest.mark.parametrize("device_name", ("cpu", "mps", "cuda"))
def test_seobnrv4_public_modes_sign_layout_and_zero_fill(device_name):
    """Public modes negate native modes and materialize complete ell blocks."""
    import torch

    from pycbc.waveform.seobnrv4phm_torch import _public_time_domain_modes

    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    dtype = torch.complex64 if device_name == "mps" else torch.complex128
    samples = torch.tensor(
        [1.0 + 2.0j, -3.0 + 4.0j],
        dtype=dtype,
        device=device_name,
    )
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme(device_name)
        modes = _public_time_domain_modes(
            {(2, 2): samples, (5, 5): 2.0 * samples},
            ((2, 2), (5, 5)),
            delta_t=0.25,
            epoch=-1.5,
        )
        torch_storage = all(
            hasattr(series._data, "tensor")
            and series._data.tensor.device.type == device_name
            for pair in modes.values()
            for series in pair
        )

        expected_keys = tuple(
            (ell, emm)
            for ell in range(5, 1, -1)
            for emm in range(ell, -ell - 1, -1)
        )
        assert tuple(modes) == expected_keys
        assert len(modes) == 32
        assert torch_storage
        for ell in (3, 4):
            for emm in range(-ell, ell + 1):
                real, imag = modes[ell, emm]
                np.testing.assert_array_equal(real.numpy(), 0.0)
                np.testing.assert_array_equal(imag.numpy(), 0.0)
        np.testing.assert_array_equal(
            modes[2, 2][0].numpy(),
            -samples.real.cpu().numpy(),
        )
        np.testing.assert_array_equal(
            modes[2, 2][1].numpy(),
            -samples.imag.cpu().numpy(),
        )
        np.testing.assert_array_equal(
            modes[5, 5][0].numpy(),
            -(2.0 * samples).real.cpu().numpy(),
        )
        for real, imag in modes.values():
            assert real.delta_t == imag.delta_t == 0.25
            assert float(real.start_time) == float(imag.start_time) == -1.5
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.mark.parametrize(
    "approximant, component_flag, generator_name, expected_modes, ell_check",
    (
        (
            "SEOBNRv4P",
            "PYCBC_SEOBNRV4P_NATIVE",
            "seobnrv4p_modes_torch",
            ((2, 2), (2, 1)),
            2,
        ),
        (
            "SEOBNRv4PHM",
            "PYCBC_SEOBNRV4PHM_NATIVE",
            "seobnrv4phm_modes_torch",
            ((2, 2),),
            5,
        ),
    ),
)
def test_seobnrv4_modes_entrypoint_neutralizes_ignored_inputs(
    monkeypatch,
    approximant,
    component_flag,
    generator_name,
    expected_modes,
    ell_check,
):
    """Mode wrappers mirror ignored inputs and model-specific Nyquist checks."""
    from pycbc.waveform import seobnrv4phm_torch

    expected = object()
    calls = []

    def recording_native(**params):
        calls.append(params)
        return expected

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv(component_flag, "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "_seobnrv4phm_td_native",
        recording_native,
    )
    params = dict(
        PARAMS,
        approximant=approximant,
        delta_t=1.0 / 4096.0,
        mode_array=None if approximant == "SEOBNRv4P" else ((2, 2),),
        coa_phase=float("inf"),
        ell_max=-7,
        f_final=object(),
        f_ref=float("nan"),
        inclination=object(),
        long_asc_nodes=object(),
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = getattr(seobnrv4phm_torch, generator_name)(**params)
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert actual is expected
    assert len(calls) == 1
    call = calls[0]
    assert call["_return_modes"] is True
    assert call["mode_array"] == expected_modes
    assert call["coa_phase"] == 0.0
    assert call["f_final"] == 0.0
    assert call["f_ref"] == call["f_lower"]
    assert call["inclination"] == 0.0
    assert call["long_asc_nodes"] == 0.0
    assert call["ellMaxForNyquistCheck"] == ell_check


def test_seobnrv4phm_fd_wrapper_preserves_td_start_time(monkeypatch):
    """The native TD-to-FD wrapper must preserve the PyCBC epoch property."""
    import torch

    from pycbc.types import TimeSeries
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.waveform import seobnrv4phm_torch

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        hp_td = TimeSeries(
            TorchArrayData(torch.arange(4, dtype=torch.float64)),
            delta_t=0.25,
            epoch=-0.5,
            copy=False,
        )
        hc_td = TimeSeries(
            TorchArrayData(torch.arange(4, dtype=torch.float64)),
            delta_t=0.25,
            epoch=-0.5,
            copy=False,
        )
        monkeypatch.setattr(
            seobnrv4phm_torch,
            "_seobnrv4phm_td_native",
            lambda **_params: (hp_td, hc_td),
        )

        hp_fd, hc_fd = seobnrv4phm_torch._seobnrv4phm_fd_native(
            delta_f=1.0,
            f_final=2.0,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert hp_fd.start_time == hp_td.start_time
    assert hc_fd.start_time == hc_td.start_time
    assert hp_fd.delta_f == hc_fd.delta_f == 1.0
    assert len(hp_fd) == len(hc_fd) == 3


def test_seobnrv4phm_sequence_wrapper_uses_native_td_samples(monkeypatch):
    """Sequence generation must use the exact sampled-TD Fourier sum."""
    import torch

    from pycbc.types import TimeSeries
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.waveform import seobnrv4phm_torch

    delta_t = 0.25
    hp_values = np.array([0.5, -1.0, 0.25, 2.0])
    hc_values = np.array([-0.25, 0.75, 1.5, -0.5])
    sample_points = np.array([1.5, 0.5, 1.5])
    hp_td = None
    hc_td = None
    td_calls = []

    def recording_td(**params):
        td_calls.append(params)
        return hp_td, hc_td

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "_seobnrv4phm_td_native",
        recording_td,
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        hp_td = TimeSeries(
            TorchArrayData(
                torch.as_tensor(hp_values, dtype=torch.float64)
            ),
            delta_t=delta_t,
            epoch=-0.5,
            copy=False,
        )
        hc_td = TimeSeries(
            TorchArrayData(
                torch.as_tensor(hc_values, dtype=torch.float64)
            ),
            delta_t=delta_t,
            epoch=-0.5,
            copy=False,
        )
        actual_hp, actual_hc = (
            seobnrv4phm_torch.seobnrv4phm_fd_sequence_torch(
                **dict(
                    PARAMS,
                    approximant="SEOBNRv4PHM",
                    sample_points=sample_points,
                    delta_t=delta_t,
                    f_ref=0.0,
                    long_asc_nodes=0.71,
                )
            )
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert isinstance(actual_hp._data, TorchArrayData)
    assert isinstance(actual_hc._data, TorchArrayData)
    times = np.arange(len(hp_values)) * delta_t
    kernel = np.exp(-2j * np.pi * np.outer(sample_points, times))
    np.testing.assert_allclose(
        actual_hp.numpy(), delta_t * kernel @ hp_values, atol=1e-14
    )
    np.testing.assert_allclose(
        actual_hc.numpy(), delta_t * kernel @ hc_values, atol=1e-14
    )
    assert td_calls[0]["f_lower"] == pytest.approx(0.5)
    assert td_calls[0]["f_ref"] == pytest.approx(0.5)
    assert td_calls[0]["f_final"] == 0.0
    assert td_calls[0]["long_asc_nodes"] == 0.0


def test_seobnrv4phm_public_td_dispatch_avoids_lalsimulation(monkeypatch):
    """The public TD API must select the native Torch entry point."""
    import torch

    from pycbc.types import TimeSeries
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.waveform import seobnrv4phm_torch, waveform

    native_calls = 0
    expected = []

    def recording_native(**_params):
        nonlocal native_calls
        native_calls += 1
        result = tuple(
            TimeSeries(
                TorchArrayData(torch.arange(4, dtype=torch.float64)),
                delta_t=1.0 / 2048.0,
                copy=False,
            )
            for _ in range(2)
        )
        expected.extend(result)
        return result

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv4PHM called lalsimulation")

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "seobnrv4phm_td_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )

    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = get_td_waveform(
            approximant="SEOBNRv4PHM",
            delta_t=1.0 / 2048.0,
            **PARAMS,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert native_calls == 1
    assert all(got is want for got, want in zip(actual, expected))


def test_seobnrv4_public_td_dispatch_avoids_lalsimulation(monkeypatch):
    """The public aligned TD API must select the native Torch entry point."""
    import torch

    from pycbc.types import TimeSeries
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.waveform import seobnrv4phm_torch, waveform

    expected = []
    native_calls = []

    def recording_native(**params):
        native_calls.append(params)
        result = tuple(
            TimeSeries(
                TorchArrayData(torch.arange(4, dtype=torch.float64)),
                delta_t=ALIGNED_V4_PARAMS["delta_t"],
                copy=False,
            )
            for _ in range(2)
        )
        expected.extend(result)
        return result

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv4 called lalsimulation")

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "seobnrv4_td_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseTDWaveform",
        unexpected_lalsimulation,
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = get_td_waveform(
            approximant="SEOBNRv4", **ALIGNED_V4_PARAMS
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert len(native_calls) == 1
    assert all(got is want for got, want in zip(actual, expected))


@pytest.mark.parametrize(
    "component_value, global_value, expect_native",
    (
        ("1", None, True),
        (None, "1", True),
        (None, None, False),
        ("0", "1", False),
    ),
    ids=("component-opt-in", "global-opt-in", "default-off", "component-opt-out"),
)
def test_seobnrv4hm_public_td_flag_routing_and_no_lalsimulation(
    monkeypatch,
    component_value,
    global_value,
    expect_native,
):
    """The regular v4HM TD row is opt-in despite sharing the ROM flag."""
    import torch

    from pycbc.types import TimeSeries
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.waveform import seobnrv4phm_torch, waveform

    expected = []
    native_calls = []

    def recording_native(**params):
        native_calls.append(params)
        result = tuple(
            TimeSeries(
                TorchArrayData(torch.arange(4, dtype=torch.float64)),
                delta_t=ALIGNED_V4HM_PARAMS["delta_t"],
                copy=False,
            )
            for _ in range(2)
        )
        expected.extend(result)
        return result

    def fallback(*_args, **_kwargs):
        raise RuntimeError("LAL SEOBNRv4HM TD fallback reached")

    for name in (
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
        "PYCBC_SEOBNRV4HM_NATIVE",
    ):
        monkeypatch.delenv(name, raising=False)
    if component_value is not None:
        monkeypatch.setenv("PYCBC_SEOBNRV4HM_NATIVE", component_value)
    if global_value is not None:
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", global_value)
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "seobnrv4hm_native_supported",
        lambda _params: True,
    )
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "seobnrv4hm_td_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseTDWaveform",
        fallback,
    )

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme("cpu")
        if expect_native:
            actual = get_td_waveform(
                approximant="SEOBNRv4HM",
                **ALIGNED_V4HM_PARAMS,
            )
            assert all(got is want for got, want in zip(actual, expected))
        else:
            with pytest.raises(
                RuntimeError,
                match="LAL SEOBNRv4HM TD fallback reached",
            ):
                get_td_waveform(
                    approximant="SEOBNRv4HM",
                    **ALIGNED_V4HM_PARAMS,
                )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert len(native_calls) == int(expect_native)


def test_seobnrv4hm_unsupported_td_falls_back_to_lal(monkeypatch):
    """Explicit public mode selection remains on selector 41's LAL path."""
    from pycbc.waveform import seobnrv4phm_torch, waveform

    def unexpected_native(**_params):
        raise AssertionError("unsupported request reached native SEOBNRv4HM")

    def fallback(*_args, **_kwargs):
        raise RuntimeError("LAL SEOBNRv4HM TD fallback reached")

    monkeypatch.setenv("PYCBC_SEOBNRV4HM_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "seobnrv4hm_td_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseTDWaveform",
        fallback,
    )
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme("cpu")
        with pytest.raises(
            RuntimeError,
            match="LAL SEOBNRv4HM TD fallback reached",
        ):
            get_td_waveform(
                approximant="SEOBNRv4HM",
                mode_array=[(2, 2)],
                **ALIGNED_V4HM_PARAMS,
            )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.mark.parametrize(
    "approximant, component_flag, generator_name, params",
    (
        (
            "SEOBNRv4",
            "PYCBC_SEOBNRV4_NATIVE",
            "seobnrv4_fd_torch",
            ALIGNED_V4_PARAMS,
        ),
        (
            "SEOBNRv4P",
            "PYCBC_SEOBNRV4P_NATIVE",
            "seobnrv4p_fd_torch",
            {key: value for key, value in PARAMS.items() if key != "delta_f"},
        ),
    ),
    ids=("aligned-v4", "precessing-v4p"),
)
def test_seobnr_public_fd_dispatch_uses_native_td_conversion(
    monkeypatch,
    approximant,
    component_flag,
    generator_name,
    params,
):
    """The public FD API must select each native Torch entry point."""
    import torch

    from pycbc.types import FrequencySeries
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.waveform import seobnrv4phm_torch, waveform

    expected = []
    native_calls = []

    def recording_native(**params):
        native_calls.append(params)
        result = tuple(
            FrequencySeries(
                TorchArrayData(
                    torch.arange(4, dtype=torch.float64).to(torch.complex128)
                ),
                delta_f=params["delta_f"],
                copy=False,
            )
            for _ in range(2)
        )
        expected.extend(result)
        return result

    def unexpected_cpu_conversion(**_params):
        raise AssertionError("native SEOBNR FD request used CPU conversion")

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv(component_flag, "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        generator_name,
        recording_native,
    )
    monkeypatch.setattr(
        waveform,
        "get_fd_waveform_from_td",
        unexpected_cpu_conversion,
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = get_fd_waveform(
            approximant=approximant,
            delta_f=0.25,
            **params,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert len(native_calls) == 1
    assert all(got is want for got, want in zip(actual, expected))


@pytest.mark.parametrize(
    "native_enabled, use_unsupported",
    ((False, False), (True, True)),
    ids=("disabled", "unsupported"),
)
@pytest.mark.parametrize(
    "approximant, component_flag, generator_name, params, unsupported",
    (
        (
            "SEOBNRv4",
            "PYCBC_SEOBNRV4_NATIVE",
            "seobnrv4_fd_torch",
            ALIGNED_V4_PARAMS,
            {"spin1x": 0.1},
        ),
        (
            "SEOBNRv4P",
            "PYCBC_SEOBNRV4P_NATIVE",
            "seobnrv4p_fd_torch",
            {key: value for key, value in PARAMS.items() if key != "delta_f"},
            {"mode_array": [(2, 2), (3, 3)]},
        ),
    ),
    ids=("aligned-v4", "precessing-v4p"),
)
def test_seobnr_fd_retains_td_to_fd_fallback(
    monkeypatch,
    native_enabled,
    use_unsupported,
    approximant,
    component_flag,
    generator_name,
    params,
    unsupported,
):
    """Disabled and unsupported EOB FD requests must retain fallback."""
    import torch

    from pycbc.types import FrequencySeries
    from pycbc.types.array_torch import TorchArrayData
    from pycbc.waveform import seobnrv4phm_torch, waveform

    fallback_calls = []
    expected = []

    def unexpected_native(**_params):
        raise AssertionError("fallback request reached native SEOBNRv4 FD")

    def recording_td_conversion(**call_params):
        assert _scheme.mgr.state is active_scheme
        fallback_calls.append(call_params)
        result = tuple(
            FrequencySeries(
                TorchArrayData(
                    torch.arange(4, dtype=torch.float64).to(torch.complex128)
                ),
                delta_f=call_params["delta_f"],
                copy=False,
            )
            for _ in range(2)
        )
        expected.extend(result)
        return result

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv(
        component_flag,
        "1" if native_enabled else "0",
    )
    monkeypatch.setattr(
        seobnrv4phm_torch,
        generator_name,
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform,
        "get_fd_waveform_from_td",
        recording_td_conversion,
    )
    active_scheme = _scheme.TorchScheme()
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = active_scheme
        actual = get_fd_waveform(
            approximant=approximant,
            delta_f=0.25,
            **dict(
                params,
                **(unsupported if use_unsupported else {}),
            ),
        )
        assert _scheme.mgr.state is active_scheme
        assert len(fallback_calls) == 1
        assert all(got is want for got, want in zip(actual, expected))
        assert all(
            series._data.tensor.device == active_scheme.torch_device
            for series in actual
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def test_seobnrv4_unsupported_td_falls_back_to_lal(monkeypatch):
    """Non-aligned requests must retain the existing LAL TD path."""
    from pycbc.waveform import seobnrv4phm_torch, waveform

    def unexpected_native(**_params):
        raise AssertionError("unsupported request reached native SEOBNRv4")

    def recording_lal(*_args, **_kwargs):
        raise RuntimeError("LAL TD fallback reached")

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "seobnrv4_td_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseTDWaveform",
        recording_lal,
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        with pytest.raises(RuntimeError, match="LAL TD fallback reached"):
            get_td_waveform(
                approximant="SEOBNRv4",
                spin1x=0.1,
                **ALIGNED_V4_PARAMS,
            )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def test_seobnrv4phm_public_sequence_dispatch_avoids_lalsimulation(
    monkeypatch,
):
    """The public sequence API must select the native Torch entry point."""
    from pycbc.waveform import seobnrv4phm_torch, waveform

    expected = (object(), object())
    native_calls = []

    def recording_native(**params):
        native_calls.append(params)
        return expected

    def unexpected_lalsimulation(*_args, **_kwargs):
        raise AssertionError("native SEOBNRv4PHM called lalsimulation")

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "seobnrv4phm_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lalsimulation,
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = get_fd_waveform_sequence(
            approximant="SEOBNRv4PHM",
            sample_points=[30.0],
            **PARAMS,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert native_calls
    assert actual[0] is expected[0]
    assert actual[1] is expected[1]


def test_seobnrv4phm_unsupported_sequence_falls_back_to_lal(monkeypatch):
    """Unsupported native requests must retain the existing LAL fallback."""
    from pycbc.waveform import seobnrv4phm_torch, waveform

    def unexpected_native(**_params):
        raise AssertionError("unsupported request reached native sequence")

    def recording_lal(*_args, **_kwargs):
        raise RuntimeError("LAL sequence fallback reached")

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_NATIVE", "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        "seobnrv4phm_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        with pytest.raises(RuntimeError, match="LAL sequence fallback reached"):
            get_fd_waveform_sequence(
                approximant="SEOBNRv4PHM",
                sample_points=[30.0],
                **dict(PARAMS, mass1=18.0, mass2=25.0),
            )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.mark.parametrize(
    "approximant, component_flag, generator_name",
    (
        (
            "SEOBNRv4P",
            "PYCBC_SEOBNRV4P_NATIVE",
            "seobnrv4p_modes_torch",
        ),
        (
            "SEOBNRv4PHM",
            "PYCBC_SEOBNRV4PHM_NATIVE",
            "seobnrv4phm_modes_torch",
        ),
    ),
)
@pytest.mark.parametrize(
    "component_value, global_value, expect_native",
    (
        ("1", None, True),
        (None, "1", True),
        (None, None, False),
        ("0", "1", False),
    ),
    ids=("component-opt-in", "global-opt-in", "default-off", "component-opt-out"),
)
def test_seobnrv4_public_modes_flag_routing_and_no_lalsimulation(
    monkeypatch,
    approximant,
    component_flag,
    generator_name,
    component_value,
    global_value,
    expect_native,
):
    """Public TD modes obey opt-in precedence without touching LAL natively."""
    from pycbc.waveform import seobnrv4phm_torch, waveform_modes

    native_result = {(2, 2): (object(), object())}
    fallback_result = {(9, 9): (object(), object())}
    calls = []

    def native(**params):
        calls.append(("native", params))
        return native_result

    def fallback(**params):
        calls.append(("fallback", params))
        return fallback_result

    for name in (
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_TORCH_NATIVE",
        component_flag,
    ):
        monkeypatch.delenv(name, raising=False)
    if component_value is not None:
        monkeypatch.setenv(component_flag, component_value)
    if global_value is not None:
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", global_value)
    monkeypatch.setattr(seobnrv4phm_torch, generator_name, native)
    monkeypatch.setitem(waveform_modes._mode_waveform_td, approximant, fallback)

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        actual = get_td_waveform_modes(
            approximant=approximant,
            delta_t=1.0 / 4096.0,
            **PARAMS,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert actual is (native_result if expect_native else fallback_result)
    assert [kind for kind, _ in calls] == [
        "native" if expect_native else "fallback"
    ]


@pytest.mark.parametrize(
    "approximant, component_flag, generator_name",
    (
        (
            "SEOBNRv4P",
            "PYCBC_SEOBNRV4P_NATIVE",
            "seobnrv4p_modes_torch",
        ),
        (
            "SEOBNRv4PHM",
            "PYCBC_SEOBNRV4PHM_NATIVE",
            "seobnrv4phm_modes_torch",
        ),
    ),
)
def test_seobnrv4_unsupported_public_modes_fall_back(
    monkeypatch,
    approximant,
    component_flag,
    generator_name,
):
    """An explicit empty mode array retains LAL's missing-22 error path."""
    from pycbc.waveform import seobnrv4phm_torch, waveform_modes

    def unexpected_native(**_params):
        raise AssertionError("unsupported TD modes reached native generator")

    def fallback(**_params):
        raise RuntimeError("LAL TD-mode fallback reached")

    monkeypatch.setenv(component_flag, "1")
    monkeypatch.setattr(
        seobnrv4phm_torch,
        generator_name,
        unexpected_native,
    )
    monkeypatch.setitem(waveform_modes._mode_waveform_td, approximant, fallback)
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        with pytest.raises(RuntimeError, match="LAL TD-mode fallback reached"):
            get_td_waveform_modes(
                approximant=approximant,
                delta_t=1.0 / 4096.0,
                mode_array=[],
                **PARAMS,
            )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.mark.parametrize(
    "approximant, predicate_name",
    (
        ("SEOBNRv4P", "seobnrv4p_modes_native_supported"),
        ("SEOBNRv4PHM", "seobnrv4phm_modes_native_supported"),
    ),
)
def test_seobnrv4_modes_native_support_boundary(approximant, predicate_name):
    """The mode-only predicate mirrors LAL's consumed and ignored inputs."""
    from pycbc.waveform import seobnrv4phm_torch

    predicate = getattr(seobnrv4phm_torch, predicate_name)
    baseline = dict(
        PARAMS,
        approximant=approximant,
        delta_t=1.0 / 4096.0,
    )
    assert predicate(baseline)
    assert predicate(
        dict(
            baseline,
            coa_phase=np.float64("nan"),
            ell_max=-2**31,
            f_final=object(),
            f_ref=np.float32("inf"),
            inclination=object(),
            long_asc_nodes=object(),
        )
    )
    assert predicate(dict(baseline, ell_max=2**31 - 1))
    assert predicate(
        dict(
            baseline,
            mode_array=np.array(((2, 2),), dtype=np.int64),
        )
    )

    unsupported = (
        {
            "approximant": (
                "SEOBNRv4PHM"
                if approximant == "SEOBNRv4P"
                else "SEOBNRv4P"
            )
        },
        {"mass1": 10.0, "mass2": 20.0},
        {"coa_phase": "0"},
        {"coa_phase": None},
        {"f_ref": 0.0 + 0.0j},
        {"f_ref": np.array(30.0)},
        {"ell_max": 2.0},
        {"ell_max": 2**31},
        {"ell_max": -(2**31) - 1},
        {"mode_array": ()},
        {"mode_array": ((2.0, 2.0),)},
        {"mode_array": np.array(((2.0, 2.0),))},
        {"mode_array": ((2, 1),)},
        {"mode_array": ((2, -2),)},
        {"phase_order": 7},
    )
    for overrides in unsupported:
        assert not predicate(dict(baseline, **overrides))

    if approximant == "SEOBNRv4P":
        assert not predicate(dict(baseline, mode_array=((2, 2), (3, 3))))
    else:
        assert predicate(dict(baseline, mode_array=((2, 2), (5, 5))))


@pytest.mark.skipif(not RUN_SLOW_NATIVE_PHM, reason=SLOW_NATIVE_PHM_REASON)
@pytest.mark.parametrize(
    "approximant, component_flag, generator_name, max_ell, checked_modes",
    (
        (
            "SEOBNRv4P",
            "PYCBC_SEOBNRV4P_NATIVE",
            "seobnrv4p_modes_torch",
            2,
            ((2, 2),),
        ),
        (
            "SEOBNRv4PHM",
            "PYCBC_SEOBNRV4PHM_NATIVE",
            "seobnrv4phm_modes_torch",
            5,
            ((2, 2), (3, 3), (5, 5)),
        ),
    ),
)
def test_seobnrv4_public_modes_optional_lal_parity(
    monkeypatch,
    approximant,
    component_flag,
    generator_name,
    max_ell,
    checked_modes,
):
    """Public native modes match the installed LAL mode convention."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch, waveform_modes

    parameters = dict(
        mass1=100.0,
        mass2=80.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.3,
        spin2x=-0.1,
        spin2y=0.12,
        spin2z=-0.2,
        distance=500.0,
        coa_phase=0.37,
        delta_t=1.0 / 2048.0,
        f_lower=10.0,
        f_ref=0.0,
    )
    expected_keys = tuple(
        (ell, emm)
        for ell in range(max_ell, 1, -1)
        for emm in range(ell, -ell - 1, -1)
    )
    pinned = {
        (2, 2): {
            0: 2.195486745538218e-21 - 7.430792465805627e-22j,
            1279: -1.5025429183401091e-21 + 2.820982574398122e-21j,
            1685: -5.7246082325571906e-21 + 3.41049663005724e-21j,
        },
        (3, 3): {
            0: -3.0035935759926665e-23 - 5.088859362099316e-23j,
            1279: 1.1678419635211209e-23 + 9.599890531827576e-23j,
            1693: -3.4114698671132506e-22 + 1.867254790507827e-22j,
        },
        (5, 5): {
            0: 1.9553008353890887e-24 + 1.974660761977398e-24j,
            1279: 5.616272628199014e-24 + 3.167711410888111e-24j,
            1694: 5.761321956133331e-23 + 2.1421086933780695e-23j,
        },
    }

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    env_backup = {
        name: os.environ.get(name)
        for name in (
            "PYCBC_TORCH_NATIVE_PORTS",
            "PYCBC_TORCH_NATIVE",
            component_flag,
        )
    }
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        reference = get_td_waveform_modes(
            approximant=approximant,
            **parameters,
        )
        reference_arrays = {
            mode: real.numpy().copy() + 1j * imag.numpy().copy()
            for mode, (real, imag) in reference.items()
        }
        reference_epochs = {
            mode: float(pair[0].start_time)
            for mode, pair in reference.items()
        }

        assert tuple(reference) == expected_keys
        assert all(len(values) == 2559 for values in reference_arrays.values())
        assert reference_epochs[(2, 2)] == pytest.approx(
            -0.822825529,
            abs=2.0e-9,
        )
        support = np.flatnonzero(reference_arrays[(2, 2)] != 0.0)
        assert support[0] == 0
        assert support[-1] == 1919
        for mode in checked_modes:
            for index, expected in pinned[mode].items():
                np.testing.assert_allclose(
                    reference_arrays[mode][index],
                    expected,
                    rtol=1.0e-10,
                    atol=1.0e-32,
                )

        native_generator = getattr(seobnrv4phm_torch, generator_name)
        native_calls = 0

        def recording_native(**native_parameters):
            nonlocal native_calls
            native_calls += 1
            return native_generator(**native_parameters)

        def unexpected_lalsimulation(*_args, **_kwargs):
            raise AssertionError("native TD modes called SimInspiralChooseTDModes")

        monkeypatch.setattr(
            seobnrv4phm_torch,
            generator_name,
            recording_native,
        )
        monkeypatch.setattr(
            waveform_modes.lalsimulation,
            "SimInspiralChooseTDModes",
            unexpected_lalsimulation,
        )
        os.environ[component_flag] = "1"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        _scheme.mgr.state = _scheme.TorchScheme("cpu")
        actual = get_td_waveform_modes(
            approximant=approximant,
            **parameters,
        )

        assert native_calls == 1
        assert tuple(actual) == expected_keys
        parity_metrics = {}
        mode_arrays = {}
        for mode in checked_modes:
            real, imag = actual[mode]
            assert isinstance(real._data.tensor, torch.Tensor)
            assert isinstance(imag._data.tensor, torch.Tensor)
            assert real._data.tensor.device.type == "cpu"
            assert imag._data.tensor.device.type == "cpu"
            assert len(real) == len(imag) == len(reference_arrays[mode])
            assert real.delta_t == imag.delta_t == parameters["delta_t"]
            assert abs(float(real.start_time) - reference_epochs[mode]) < real.delta_t
            result = real.numpy() + 1j * imag.numpy()
            mode_arrays[mode] = result
            reference_array = reference_arrays[mode]
            relative_error = np.linalg.norm(
                result - reference_array
            ) / np.linalg.norm(reference_array)
            phase = np.vdot(result, reference_array)
            phase /= abs(phase)
            aligned_error = np.linalg.norm(
                phase * result - reference_array
            ) / np.linalg.norm(reference_array)
            reference_peak = int(np.argmax(np.abs(reference_array)))
            result_peak = int(np.argmax(np.abs(result)))
            parity_metrics[mode] = {
                "raw_l2": float(relative_error),
                "phase_aligned_l2": float(aligned_error),
                "phase_offset": float(np.angle(phase)),
                "epoch_delta": (
                    float(real.start_time) - reference_epochs[mode]
                ),
                "peak_shift": result_peak - reference_peak,
                "peak_time_shift": (
                    result_peak - reference_peak
                ) * parameters["delta_t"],
                "norm_ratio": float(
                    np.linalg.norm(result) / np.linalg.norm(reference_array)
                ),
                "peak_ratio": float(
                    np.max(np.abs(result)) / np.max(np.abs(reference_array))
                ),
            }
        delta_phi = parity_metrics[(2, 2)]["phase_offset"] / 2.0
        for mode in checked_modes:
            coherent_result = mode_arrays[mode] * np.exp(
                1j * mode[1] * delta_phi
            )
            parity_metrics[mode]["coherent_l2"] = float(
                np.linalg.norm(
                    coherent_result - reference_arrays[mode]
                ) / np.linalg.norm(reference_arrays[mode])
            )
            parity_metrics[mode]["phase_per_m"] = (
                parity_metrics[mode]["phase_offset"] / mode[1]
            )
        details = "\n".join(
            f"{mode}: "
            + ", ".join(
                f"{name}={value:.16g}"
                for name, value in metrics.items()
            )
            for mode, metrics in parity_metrics.items()
        )
        assert all(
            metrics["raw_l2"] < 4.0e-3
            and abs(metrics["norm_ratio"] - 1.0) < 2.0e-3
            and abs(metrics["peak_ratio"] - 1.0) < 2.0e-3
            and abs(metrics["epoch_delta"]) < parameters["delta_t"]
            and abs(metrics["peak_shift"]) <= 1
            for metrics in parity_metrics.values()
        ), details
    finally:
        for name, value in env_backup.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def test_seobnrv4phm_native_support_boundary():
    from pycbc.waveform.seobnrv4phm_torch import (
        seobnrv4phm_native_supported,
    )

    baseline = dict(PARAMS, approximant="SEOBNRv4PHM")
    assert seobnrv4phm_native_supported(baseline)

    unsupported = (
        {"approximant": "SEOBNRv4P"},
        {"mass1": 10.0, "mass2": 20.0},
        {"phase_order": 7},
        {"eccentricity": 0.1},
        {"lambda1": 100.0},
        {"dchi0": 0.1},
        {"frame_axis": 1},
        {"numrel_data": "file.h5"},
        {"mode_array": ((3, 2),)},
        {"long_asc_nodes": float("nan")},
    )
    for overrides in unsupported:
        assert not seobnrv4phm_native_supported(
            dict(baseline, **overrides)
        )

    from pycbc.waveform.seobnrv4phm_torch import (
        seobnrv4phm_sequence_native_supported,
    )

    assert seobnrv4phm_sequence_native_supported(
        dict(baseline, long_asc_nodes=float("nan"))
    )
    assert not seobnrv4phm_sequence_native_supported(
        dict(baseline, mass1=10.0, mass2=20.0)
    )


def test_seobnrv4hm_native_support_boundary(monkeypatch):
    """Regular selector 41 dispatch is strict, CPU-only, and fail-safe."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    baseline = dict(ALIGNED_V4HM_PARAMS, approximant="SEOBNRv4HM")
    omega_rd55 = 2.0 * math.pi * 100.0
    monkeypatch.setattr(
        swt._dyn,
        "_ported_gsl_multiroot_available",
        lambda: True,
    )
    monkeypatch.setattr(
        swt,
        "_ringdown_omega_for_nyquist",
        lambda *_args, **_kwargs: omega_rd55,
    )

    old_state = _scheme.mgr.state
    try:
        _scheme.mgr.state = _scheme.TorchScheme("cpu")
        assert swt.seobnrv4hm_native_supported(baseline)
        assert swt.seobnrv4hm_native_supported(
            dict(baseline, mass1=100.0, mass2=1.0)
        )
        assert not swt.seobnrv4hm_native_supported(
            dict(baseline, mass1=np.nextafter(100.0, math.inf), mass2=1.0)
        )
        assert swt.seobnrv4hm_native_supported(
            dict(baseline, mass1=58.0, mass2=1.0, spin1z=0.96)
        )
        assert not swt.seobnrv4hm_native_supported(
            dict(
                baseline,
                mass1=58.0,
                mass2=1.0,
                spin1z=np.nextafter(0.96, math.inf),
            )
        )

        max_flow = 0.95 * omega_rd55 / (2.0 * math.pi)
        max_delta_t = math.pi / omega_rd55
        assert swt.seobnrv4hm_native_supported(
            dict(baseline, f_lower=max_flow)
        )
        assert not swt.seobnrv4hm_native_supported(
            dict(baseline, f_lower=np.nextafter(max_flow, math.inf))
        )
        assert swt.seobnrv4hm_native_supported(
            dict(baseline, delta_t=max_delta_t)
        )
        assert not swt.seobnrv4hm_native_supported(
            dict(baseline, delta_t=np.nextafter(max_delta_t, math.inf))
        )
        assert swt.seobnrv4hm_native_supported(
            dict(baseline, f_ref=71.0, f_final=123.0)
        )

        unsupported = (
            {"approximant": "SEOBNRv4"},
            {"mass1": 10.0, "mass2": 20.0},
            {"mode_array": ((2, 2),)},
            {"phase_order": 7},
            {"spin_order": 6},
            {"tidal_order": 0},
            {"amplitude_order": 0},
            {"spin1x": np.nextafter(0.0, 1.0)},
            {"spin2y": -np.nextafter(0.0, 1.0)},
            {"lambda1": 1.0},
            {"dquad_mon2": 1.0},
            {"dchi0": 1.0e-6},
            {"frame_axis": 1},
            {"numrel_data": "file.h5"},
            {"f_ref": float("inf")},
            {"f_final": float("nan")},
            {"distance": 0.0},
            {"delta_t": 0.0},
        )
        for overrides in unsupported:
            assert not swt.seobnrv4hm_native_supported(
                dict(baseline, **overrides)
            )

        for device_type in ("cuda", "mps"):
            non_cpu = object.__new__(_scheme.TorchScheme)
            non_cpu.torch_device = torch.device(device_type)
            non_cpu.device = non_cpu.torch_device
            _scheme.mgr.state = non_cpu
            assert not swt.seobnrv4hm_native_supported(baseline)

        _scheme.mgr.state = _scheme.TorchScheme("cpu")
        monkeypatch.setattr(
            swt._dyn,
            "_ported_gsl_multiroot_available",
            lambda: False,
        )
        assert not swt.seobnrv4hm_native_supported(baseline)
    finally:
        _scheme.mgr.state = old_state


def test_seobnrv4hm_direct_assembler_uses_floor_copy_and_zero_tail():
    """Selector 41 directly decimates HiS and preserves allocation zeros."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    mode = (2, 2)
    adas = torch.arange(6, dtype=torch.float64).to(torch.complex128)
    his = (100.0 + torch.arange(8, dtype=torch.float64)).to(
        torch.complex128
    )
    result = swt._seobnrv4hm_direct_assemble_modes(
        {mode: adas},
        {mode: his},
        adas_length=6,
        his_start=4,
        resample_factor=2,
    )[mode]

    expected = torch.tensor(
        [0.0, 1.0, 2.0, 3.0, 100.0, 102.0, 104.0, 106.0, 0.0, 0.0],
        dtype=torch.complex128,
    )
    assert torch.equal(result, expected)


def test_seobnrv4hm_circular_root_is_pure_and_residual_guarded(monkeypatch):
    """Selector 41 never consults host GSL and validates its root residual."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    if not swt._dyn._ported_gsl_multiroot_available():
        pytest.skip("deterministic selector-41 CBLAS backend is unavailable")
    params = swt._native_params_from_kwargs(ALIGNED_V4HM_PARAMS)
    omega = torch.tensor(
        ((math.pi * params.M) * swt._MTSUN_SI) * params.f_lower,
        dtype=torch.float64,
    )
    captured = {}
    original = swt._dyn._gsl_multiroot_hybrids_ported

    def recording_ported(*args, **kwargs):
        result = original(*args, **kwargs)
        captured["result"] = result
        return result

    def unexpected_host_gsl(*_args, **_kwargs):
        raise AssertionError("selector 41 consulted host GSL")

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_INSTALLED_GSL_DERIV", "1")
    monkeypatch.setattr(
        swt._dyn,
        "_gsl_multiroot_hybrids_ported",
        recording_ported,
    )
    monkeypatch.setattr(
        swt._dyn,
        "_native_gsl_multiroot_hybrids",
        unexpected_host_gsl,
    )
    monkeypatch.setattr(
        swt._dyn,
        "_installed_gsl_deriv_central",
        unexpected_host_gsl,
    )
    root = swt._seobnrv4hm_nonoptimized_circular_root(
        params,
        omega,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert root is not None
    solved, residual = captured["result"]
    assert all(math.isfinite(value) for value in (*solved, *residual))
    assert abs(solved[2]) <= 1.0e-12
    assert math.fsum(abs(value) for value in residual) < 1.0e-10

    radial = swt._seobnrv4hm_nonoptimized_ic_radial_summary(
        params,
        root[0],
        root[1],
        omega,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert radial is not None
    assert torch.isfinite(radial["pr_star"])

    monkeypatch.setattr(
        swt._dyn,
        "_gsl_multiroot_hybrids_ported",
        lambda *_args, **_kwargs: ([10.0, 0.3, 0.0], [1.0e-9, 0.0, 0.0]),
    )
    assert (
        swt._seobnrv4hm_nonoptimized_circular_root(
            params,
            omega,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        is None
    )


def test_seobnrv4hm_exact_adas_failure_does_not_retry_relaxed(monkeypatch):
    """A failed exact selector-41 evolution fails closed without a retry."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    params = swt._native_params_from_kwargs(ALIGNED_V4HM_PARAMS)
    calls = []
    root = (
        torch.tensor(10.0, dtype=torch.float64),
        torch.tensor(3.7, dtype=torch.float64),
    )
    monkeypatch.setattr(
        swt,
        "_seobnrv4hm_nonoptimized_circular_root",
        lambda *_args, **_kwargs: root,
    )
    monkeypatch.setattr(
        swt,
        "_seobnrv4hm_nonoptimized_ic_radial_summary",
        lambda *_args, **_kwargs: {
            "pr_star": torch.tensor(-0.004, dtype=torch.float64)
        },
    )

    def empty_integrator(*_args, **_kwargs):
        calls.append(_kwargs)
        return [], {}

    monkeypatch.setattr(swt, "integrate", empty_integrator)
    with pytest.raises(RuntimeError, match="aligned AdaS integration failed"):
        swt._integrate_traj(
            params,
            device=torch.device("cpu"),
            real_dtype=torch.float64,
            f_final=0.0,
            delta_t=ALIGNED_V4HM_PARAMS["delta_t"],
            selector41=True,
        )
    assert len(calls) == 1
    assert calls[0]["rtol"] == pytest.approx(1.0e-9)
    assert calls[0]["atol"] == pytest.approx(1.0e-10)


def test_seobnrv4_native_support_boundary():
    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform.seobnrv4phm_torch import seobnrv4_native_supported

    baseline = dict(ALIGNED_V4_PARAMS, approximant="SEOBNRv4")
    assert seobnrv4_native_supported(baseline)
    assert seobnrv4_native_supported(
        dict(baseline, spin1x=0.5 * dyn.EPS_ALIGN)
    )

    unsupported = (
        {"approximant": "SEOBNRv4P"},
        {"mass1": 10.0, "mass2": 20.0},
        {"phase_order": 7},
        {"eccentricity": 0.1},
        {"lambda1": 100.0},
        {"dchi0": 0.1},
        {"mode_array": ((2, 2),)},
        {"spin1x": dyn.EPS_ALIGN},
        {"spin2y": float("nan")},
    )
    for overrides in unsupported:
        assert not seobnrv4_native_supported(dict(baseline, **overrides))


def test_seobnrv4p_native_support_boundary():
    from pycbc.waveform.seobnrv4phm_torch import (
        seobnrv4p_native_supported,
        seobnrv4p_sequence_native_supported,
    )

    baseline = dict(PARAMS, approximant="SEOBNRv4P")
    assert seobnrv4p_native_supported(baseline)
    for mode_array in (
        ((2, 2),),
        ((2, 2), (2, 1)),
        ((2, 1), (2, 2)),
    ):
        assert seobnrv4p_native_supported(
            dict(baseline, mode_array=mode_array)
        )

    unsupported = (
        {"approximant": "SEOBNRv4PHM"},
        {"mass1": 10.0, "mass2": 20.0},
        {"phase_order": 7},
        {"eccentricity": 0.1},
        {"lambda1": 100.0},
        {"dchi0": 0.1},
        {"frame_axis": 1},
        {"numrel_data": "file.h5"},
        {"mode_array": ()},
        {"mode_array": ((2, 1),)},
        {"mode_array": ((2, 2), (3, 3))},
        {"mode_array": ((2, 2), (2, -1))},
        {"long_asc_nodes": float("nan")},
    )
    for overrides in unsupported:
        assert not seobnrv4p_native_supported(dict(baseline, **overrides))

    assert seobnrv4p_sequence_native_supported(
        dict(baseline, long_asc_nodes=float("nan"))
    )
    assert not seobnrv4p_sequence_native_supported(
        dict(baseline, mode_array=((2, 1),))
    )


def test_seobnrv4phm_mps_selects_supported_dtypes():
    import torch

    from pycbc.waveform import seobnrv4phm_torch

    old_state = _scheme.mgr.state
    try:
        _scheme.mgr.state = type(
            "MPSState",
            (),
            {"device": torch.device("mps"), "dtype": torch.float64},
        )()
        device, real_dtype, complex_dtype = (
            seobnrv4phm_torch._target_device_dtypes()
        )
    finally:
        _scheme.mgr.state = old_state

    assert device.type == "mps"
    assert real_dtype == torch.float32
    assert complex_dtype == torch.complex64


def test_seobnrv4phm_polarization_rotation_matches_lal_convention():
    import torch

    from pycbc.waveform.seobnrv4phm_torch import _rotate_polarizations

    hp = torch.tensor([1.0, -0.5], dtype=torch.float64)
    hc = torch.tensor([0.25, 2.0], dtype=torch.float64)
    nodes = 0.37
    got_hp, got_hc = _rotate_polarizations(hp, hc, nodes)
    expected = (hp - 1j * hc) * np.exp(2j * nodes)

    torch.testing.assert_close(got_hp, expected.real)
    torch.testing.assert_close(got_hc, -expected.imag)


def test_seobnrv4p_lal_matches_phm_restricted_to_default_modes():
    """The shared native engine follows the exact LAL v4P model boundary."""
    params = dict(
        mass1=50.0,
        mass2=30.0,
        spin1x=0.2,
        spin1y=-0.1,
        spin1z=0.3,
        spin2x=-0.08,
        spin2y=0.04,
        spin2z=-0.2,
        delta_t=1.0 / 2048.0,
        f_lower=18.0,
        f_ref=18.0,
        distance=400.0,
        inclination=0.7,
        coa_phase=0.2,
    )

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        hp_v4p, hc_v4p = get_td_waveform(
            approximant="SEOBNRv4P",
            **params,
        )
        hp_phm, hc_phm = get_td_waveform(
            approximant="SEOBNRv4PHM",
            mode_array=[(2, 2), (2, 1)],
            **params,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert hp_v4p.start_time == hp_phm.start_time
    assert hc_v4p.start_time == hc_phm.start_time
    np.testing.assert_array_equal(hp_v4p.numpy(), hp_phm.numpy())
    np.testing.assert_array_equal(hc_v4p.numpy(), hc_phm.numpy())


def test_seobnrv4phm_mode_array_normalizes_like_lal():
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    assert dyn.normalize_mode_array(None) == tuple(dyn.LM_DEFAULT)
    assert dyn.normalize_mode_array(((5, 5), (2, 1), (2, 2), (2, 1))) == (
        (2, 2),
        (2, 1),
        (5, 5),
    )


def test_seobnrv4phm_mode_array_rejects_lal_invalid_modes():
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    with pytest.raises(ValueError, match="negative m"):
        dyn.normalize_mode_array(((2, -2),))
    with pytest.raises(ValueError, match="not available"):
        dyn.normalize_mode_array(((3, 2),))


def test_seobnrv4phm_time_to_fd_matches_pycbc_timeseries_fft():
    """The native TD->FD helper should mirror PyCBC TimeSeries padding/FFT."""
    import torch

    from pycbc.types import TimeSeries
    from pycbc.waveform.seobnrv4phm_torch import _time_to_fd

    delta_t = 1.0 / 128.0
    delta_f = 0.5
    t = np.arange(127, dtype=np.float64) * delta_t
    hp = np.sin(2.0 * np.pi * 7.0 * t) + 0.25 * np.cos(2.0 * np.pi * 13.0 * t)
    hc = 0.5 * np.sin(2.0 * np.pi * 5.0 * t + 0.3)

    hp_torch, hc_torch, df_out = _time_to_fd(
        torch.as_tensor(hp, dtype=torch.float64),
        torch.as_tensor(hc, dtype=torch.float64),
        delta_t,
        torch.complex128,
        delta_f,
    )

    hp_ref = TimeSeries(hp, delta_t=delta_t).to_frequencyseries(delta_f=delta_f)
    hc_ref = TimeSeries(hc, delta_t=delta_t).to_frequencyseries(delta_f=delta_f)

    assert df_out == pytest.approx(delta_f)
    np.testing.assert_allclose(hp_torch.numpy(), hp_ref.numpy(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(hc_torch.numpy(), hc_ref.numpy(), rtol=1e-12, atol=1e-14)


def test_seobnrv4phm_frequency_sequence_matches_rfft_bins_in_chunks():
    """The nonuniform DFT must reproduce rFFT bins, order, and duplicates."""
    import torch

    from pycbc.waveform.seobnrv4phm_torch import (
        _time_to_frequency_sequence,
    )

    delta_t = 1.0 / 128.0
    times = torch.arange(64, dtype=torch.float64) * delta_t
    hp = torch.sin(2.0 * math.pi * 6.0 * times) + 0.2 * torch.cos(
        2.0 * math.pi * 18.0 * times
    )
    hc = 0.4 * torch.sin(2.0 * math.pi * 10.0 * times + 0.2)
    frequencies = torch.tensor(
        [18.0, 6.0, 18.0, 64.0], dtype=torch.float64
    )

    actual_hp, actual_hc = _time_to_frequency_sequence(
        hp,
        hc,
        delta_t,
        frequencies,
        torch.complex128,
        max_chunk_elements=2 * len(hp),
    )
    bins = (frequencies / 2.0).to(torch.int64)
    expected_hp = torch.fft.rfft(hp) * delta_t
    expected_hc = torch.fft.rfft(hc) * delta_t
    torch.testing.assert_close(actual_hp, expected_hp[bins])
    torch.testing.assert_close(actual_hc, expected_hc[bins])


@pytest.mark.parametrize(
    "sample_points, message",
    (
        ([], "non-empty vector"),
        ([[10.0]], "non-empty vector"),
        ([10.0, float("nan")], "finite"),
        ([0.0, 10.0], "positive"),
        ([10.0, 2.1], "Nyquist"),
    ),
)
def test_seobnrv4phm_sequence_frequency_validation(sample_points, message):
    import torch

    from pycbc.waveform.seobnrv4phm_torch import (
        _seobnrv4phm_sequence_frequencies,
    )

    with pytest.raises(ValueError, match=message):
        _seobnrv4phm_sequence_frequencies(
            sample_points,
            device=torch.device("cpu"),
            real_dtype=torch.float64,
            delta_t=0.25,
        )


def _cpu_waveform():
    env_backup = {
        k: os.environ.get(k)
        for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4PHM_NATIVE")
    }
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_SEOBNRV4PHM_NATIVE"] = "0"
        return get_fd_waveform(approximant="SEOBNRv4PHM", **PARAMS)[0]
    except (RuntimeError, ValueError, OSError) as exc:
        pytest.skip(f"SEOBNRv4PHM CPU reference unavailable: {exc}")
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _phm_fd_parity_metrics(h_cpu, h_torch):
    """Return in-band FD parity diagnostics for the native path."""
    n = min(len(h_cpu), len(h_torch))
    cpu = h_cpu.numpy()[:n]
    torch_native = h_torch.numpy()[:n]
    delta_f = float(h_torch.delta_f)
    freqs = np.arange(n, dtype=float) * delta_f

    f_final = PARAMS.get("f_final", 0.0)
    if f_final is None or f_final <= 0.0:
        f_final = freqs[-1] if len(freqs) else 0.0

    band = (freqs >= PARAMS["f_lower"]) & (freqs <= f_final)
    finite = (
        band
        & np.isfinite(cpu)
        & np.isfinite(torch_native)
        & (np.abs(cpu) > 0.0)
        & (np.abs(torch_native) > 0.0)
    )
    if not np.any(finite):
        return {
            "bins": 0,
            "f_min": float(PARAMS["f_lower"]),
            "f_max": float(f_final),
        }

    cpu_band = cpu[finite]
    torch_band = torch_native[finite]
    ratio = np.abs(torch_band / cpu_band)
    phase_diff = np.angle(torch_band * np.conj(cpu_band))
    rel = np.linalg.norm(torch_band - cpu_band) / np.linalg.norm(cpu_band)
    freq_band = freqs[finite]
    if len(freq_band) >= 2:
        phase_unwrapped = np.unwrap(phase_diff)
        fit_slope, fit_intercept = np.polyfit(freq_band, phase_unwrapped, 1)
        linear_phase = fit_slope * freq_band + fit_intercept
        aligned = torch_band * np.exp(-1j * linear_phase)
        aligned_phase = np.angle(aligned * np.conj(cpu_band))
        aligned_rel = np.linalg.norm(aligned - cpu_band) / np.linalg.norm(cpu_band)
        linear_dt_s = fit_slope / (2.0 * np.pi)
        aligned_phase_mean = aligned_phase.mean()
        aligned_phase_std = aligned_phase.std()
    else:
        fit_slope = np.nan
        fit_intercept = np.nan
        linear_dt_s = np.nan
        aligned_rel = np.inf
        aligned_phase_mean = np.nan
        aligned_phase_std = np.inf

    return {
        "bins": int(finite.sum()),
        "f_min": float(freq_band[0]),
        "f_max": float(freq_band[-1]),
        "rel": float(rel),
        "amp_min": float(ratio.min()),
        "amp_median": float(np.median(ratio)),
        "amp_max": float(ratio.max()),
        "phase_mean": float(phase_diff.mean()),
        "phase_std": float(phase_diff.std()),
        "linear_phase_slope": float(fit_slope),
        "linear_phase_intercept": float(fit_intercept),
        "linear_dt_s": float(linear_dt_s),
        "aligned_rel": float(aligned_rel),
        "aligned_phase_mean": float(aligned_phase_mean),
        "aligned_phase_std": float(aligned_phase_std),
    }


def _format_phm_fd_parity_metrics(metrics):
    if metrics["bins"] == 0:
        return (
            "SEOBNRv4PHM torch-native parity has no comparable in-band bins "
            f"for f=[{metrics['f_min']:.2f}, {metrics['f_max']:.2f}] Hz"
        )
    return (
        "SEOBNRv4PHM torch-native parity not within tolerance: "
        f"bins={metrics['bins']} "
        f"f=[{metrics['f_min']:.2f}, {metrics['f_max']:.2f}] Hz "
        f"rel={metrics['rel']:.3e} "
        f"amp_ratio=[{metrics['amp_min']:.3e}, "
        f"{metrics['amp_median']:.3e}, {metrics['amp_max']:.3e}] "
        f"phase_mean={metrics['phase_mean']:.3e} "
        f"phase_std={metrics['phase_std']:.3e} "
        f"linear_dt={metrics['linear_dt_s']:.3e}s "
        f"linear_intercept={metrics['linear_phase_intercept']:.3e} "
        f"aligned_rel={metrics['aligned_rel']:.3e} "
        f"aligned_phase_std={metrics['aligned_phase_std']:.3e}"
    )


def _isolated_lal_phm_dynamics(params, n_rows=0, vector="joined", n_cols=15):
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    payload = dict(params)
    payload["mode_array"] = list(dyn.normalize_mode_array(params.get("mode_array", None)))
    payload["n_rows"] = int(n_rows)
    payload["n_cols"] = int(n_cols)
    payload["vector"] = str(vector)
    script = r"""
import json
import sys

import lal
import lalsimulation as lalsim
import numpy as np

p = json.loads(sys.argv[1])
mode_array = lalsim.SimInspiralCreateModeArray()
for ell, emm in p["mode_array"]:
    lalsim.SimInspiralModeArrayActivateMode(mode_array, int(ell), int(emm))

outs = lalsim.SimIMRSpinPrecEOBWaveformAll(
    0.0,
    p.get("delta_t", 1.0 / 4096.0),
    p["mass1"] * lal.MSUN_SI,
    p["mass2"] * lal.MSUN_SI,
    p["f_lower"],
    p["distance"] * lal.PC_SI,
    p.get("inclination", 0.0),
    p.get("spin1x", 0.0),
    p.get("spin1y", 0.0),
    p.get("spin1z", 0.0),
    p.get("spin2x", 0.0),
    p.get("spin2y", 0.0),
    p.get("spin2z", 0.0),
    mode_array,
    lal.CreateDict(),
)
dyn_vars = 26
vector_indices = {"adas": 4, "his": 5, "joined": 6}
selected = p.get("vector", "joined")
if selected not in vector_indices:
    raise ValueError(f"unknown dynamics vector {selected!r}")

def vector_data(name):
    vec = outs[vector_indices[name]]
    data = np.array(vec.data, copy=False)
    ret_len = vec.length // dyn_vars
    return data, ret_len

def vector_summary(name):
    data, ret_len = vector_data(name)

    def col(k):
        return data[k * ret_len : (k + 1) * ret_len]

    t_m = col(0)
    r = col(18)
    omega = col(22)
    hamiltonian = col(25)
    idx_peak = int(np.nanargmax(omega))

    def sample_rows(n_rows=5):
        n = min(int(n_rows), ret_len)
        rows = []
        for i in range(n):
            rows.append({
                "t_M": float(t_m[i]),
                "r": float(r[i]),
                "omega": float(omega[i]),
                "H": float(hamiltonian[i]),
                "state": [float(data[k * ret_len + i]) for k in range(1, 15)],
            })
        return rows

    out = {
        "n_dyn": int(ret_len),
        "t_start_M": float(t_m[0]),
        "t_end_M": float(t_m[-1]),
        "r_start": float(r[0]),
        "r_end": float(r[-1]),
        "t_peak_omega_M": float(t_m[idx_peak]),
        "omega_peak_dimless": float(omega[idx_peak]),
        "H_start": float(hamiltonian[0]),
        "H_end": float(hamiltonian[-1]),
        "samples": sample_rows(),
    }
    if ret_len > 1:
        out["dt_start_M"] = float(t_m[1] - t_m[0])
        out["dt_end_M"] = float(t_m[-1] - t_m[-2])
    return out

vectors = {name: vector_summary(name) for name in vector_indices}
merger_params = np.array(outs[21].data, copy=False)
p_mode_t = np.array(outs[7].data, copy=False)
his_data, his_len = vector_data("his")
joined_data, joined_len = vector_data("joined")
his_t = his_data[:his_len]
joined_t = joined_data[:joined_len]
index_join_his = int(np.searchsorted(joined_t, his_t[0], side="left"))
first_his_excluded = joined_len - index_join_his
join = {
    "index_join_his": index_join_his,
    "index_join_attach": int(joined_len),
    "n_his_in_join": int(first_his_excluded),
    "t_join_his_M": float(joined_t[index_join_his]),
    "t_his_start_M": float(his_t[0]),
    "t_dynamics_end_M": float(joined_t[-1]),
    "ret_len_pmodes": int(p_mode_t.size),
}
if index_join_his > 0:
    join["t_adas_before_his_M"] = float(joined_t[index_join_his - 1])
if joined_len < p_mode_t.size:
    t_join_attach = float(p_mode_t[joined_len])
    join["t_join_attach_M"] = t_join_attach
    join["t_first_his_excluded_M"] = t_join_attach

seob_vec = outs[vector_indices[selected]]
data = np.array(seob_vec.data, copy=False)
ret_len = seob_vec.length // dyn_vars

def col(k):
    return data[k * ret_len : (k + 1) * ret_len]

t_m = col(0)
r = col(18)
omega = col(22)
hamiltonian = col(25)
idx_peak = int(np.nanargmax(omega))
n_rows = min(int(p["n_rows"]), ret_len)
n_cols = min(max(int(p.get("n_cols", 15)), 0), dyn_vars)
rows = [[float(data[k * ret_len + i]) for k in range(n_cols)] for i in range(n_rows)]
print(json.dumps({
    "rows": rows,
    "vectors": vectors,
    "join": join,
    "merger": {
        "t_peak_omega_M": float(merger_params[0]),
        "t_attach_M": float(merger_params[1]),
        "t_peak_amp_M": float(merger_params[2]),
        "final_mass_frac": float(merger_params[6]),
        "final_spin": float(merger_params[7]),
        "termination_reason": int(round(merger_params[8])),
    },
    "summary": {
    "n_dyn": int(ret_len),
    "t_start_M": float(t_m[0]),
    "t_end_M": float(t_m[-1]),
    "r_start": float(r[0]),
    "r_end": float(r[-1]),
    "t_peak_omega_M": float(t_m[idx_peak]),
    "omega_peak_dimless": float(omega[idx_peak]),
    "H_start": float(hamiltonian[0]),
    "H_end": float(hamiltonian[-1]),
    },
}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, json.dumps(payload)],
            check=True,
            text=True,
            capture_output=True,
            timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        stderr = getattr(exc, "stderr", "") or str(exc)
        pytest.skip(f"isolated lalsimulation reference unavailable: {stderr}")
    return json.loads(proc.stdout)


def _lal_phm_checkpoint_summary(params):
    ref = _isolated_lal_phm_dynamics(params)
    summary = dict(ref["summary"])
    summary["vectors"] = ref["vectors"]
    summary["join"] = ref["join"]
    summary["merger"] = ref["merger"]
    return summary


def _lal_phm_dynamics_rows(params, n_rows, vector="joined", n_cols=15):
    return _isolated_lal_phm_dynamics(params, n_rows=n_rows, vector=vector, n_cols=n_cols)["rows"]


def _torch_replay_lal_rk_interval(params_dict, rows, idx):
    """Replay one LAL dynamics interval with the torch RK/RHS path."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform.seobnrv4phm_ode import RKState, _rk45_step_impl

    y0 = torch.tensor(rows[idx][1:15], dtype=torch.float64)
    y1_lal = torch.tensor(rows[idx + 1][1:15], dtype=torch.float64)
    h_lal = torch.tensor(rows[idx + 1][0] - rows[idx][0], dtype=torch.float64)
    t0 = torch.tensor(rows[idx][0], dtype=torch.float64)

    eob_params = {k: v for k, v in params_dict.items() if k != "delta_t"}
    params = dyn.EOBParams(**eob_params)
    if idx == 0:
        # LAL's first RHS starts from the coefficient cache left by the IC root.
        dyn.initial_conditions(params, device=torch.device("cpu"), dtype=torch.float64)
    else:
        s1_scale = dyn._lal_spin_scale(params.mass1, params.M)
        s2_scale = dyn._lal_spin_scale(params.mass2, params.M)
        L_vec = torch.linalg.cross(y0[0:3], y0[3:6])
        dyn._refresh_hcoeffs(params, L_vec, y0[6:9] / s1_scale, y0[9:12] / s2_scale)

    dydt0 = dyn.rhs_cartesian_full(t0, y0, params)
    state, _, _ = _rk45_step_impl(
        lambda t, y: dyn.rhs_cartesian_full(t, y, params),
        RKState(t0, y0, h_lal),
        dydt_in=dydt0,
        compute_dydt_out=True,
    )
    return state.y, y1_lal


def _phm_boundary_comparison(lal_ref, torch_summary):
    vectors = lal_ref["vectors"]
    join = lal_ref["join"]
    merger = lal_ref["merger"]
    adas_pairs = list(zip(vectors["adas"]["samples"], torch_summary["adas_samples"], strict=False))
    his_pairs = list(zip(vectors["his"]["samples"], torch_summary["his_uniform_samples"], strict=False))
    return {
        "dn_adas": torch_summary["n_adas"] - vectors["adas"]["n_dyn"],
        "dn_his": torch_summary["n_his"] - vectors["his"]["n_dyn"],
        "dn_joined": torch_summary["n_joined"] - vectors["joined"]["n_dyn"],
        "d_index_join_his": torch_summary["index_join_his"] - join["index_join_his"],
        "d_index_join_attach": torch_summary["index_join_attach"] - join["index_join_attach"],
        "dt_start_his_M": torch_summary["t_start_his_M"] - join["t_his_start_M"],
        "dt_join_his_M": torch_summary["t_join_his_M"] - join["t_join_his_M"],
        "dt_attach_M": torch_summary["t_attach_M"] - merger["t_attach_M"],
        "dt_attach_floor_M": torch_summary["t_join_attach_M"] - join["t_join_attach_M"],
        "dt_joined_end_M": torch_summary["joined_end_M"] - vectors["joined"]["t_end_M"],
        "dr_start": torch_summary["r_start"] - vectors["joined"]["r_start"],
        "dr_his_start": torch_summary["r_his_start"] - vectors["his"]["r_start"],
        "dr_joined_end": torch_summary["r_end"] - vectors["joined"]["r_end"],
        "adas_sample_dt_M": [t["t_M"] - l["t_M"] for l, t in adas_pairs],
        "adas_sample_dr": [t["r"] - l["r"] for l, t in adas_pairs],
        "his_sample_dt_M": [t["t_M"] - l["t_M"] for l, t in his_pairs],
        "his_sample_dr": [t["r"] - l["r"] for l, t in his_pairs],
    }


def _phm_ic_step4_comparison(params_dict):
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    def _native_params():
        return swt._native_params_from_kwargs(params_dict)

    def _row_fields(row):
        return {
            "r": float(row[1]),
            "px": float(row[4]),
            "py": float(row[5]),
            "pz": float(row[6]),
        }

    def _row_dynamics(row):
        keys = (
            "vx",
            "vy",
            "vz",
            "polar_r",
            "polar_phi",
            "polar_pr",
            "polar_pphi",
            "omega",
            "s1dotZ",
            "s2dotZ",
            "H",
        )
        return {key: float(row[15 + idx]) for idx, key in enumerate(keys)}

    def _rhs_from_lal_row(row):
        params = _native_params()
        device = torch.device("cpu")
        dtype = torch.float64
        # LAL's first RHS starts with the coefficient cache left by the IC root.
        dyn.initial_conditions(params, device=device, dtype=dtype)
        y_cart = torch.tensor(row[1:15], device=device, dtype=dtype)
        rhs = dyn.rhs_cartesian_full(
            torch.tensor(row[0], device=device, dtype=dtype),
            y_cart,
            params,
        )
        return {"vx": float(rhs[0]), "vy": float(rhs[1]), "vz": float(rhs[2])}

    def _clean_step4_summary(summary):
        keys = (
            "pr_star",
            "pr_non_tortoise",
            "r_dot",
            "d2Hdr2",
            "d2Hdrdpphi",
            "dHdpphi",
            "dEdr",
            "H",
            "flux",
            "probe_dH_dpvec0",
            "dxdt_probe0",
            "csi",
            "dHdpr",
            "p_norm",
            "pphi",
        )
        return {key: float(summary[key]) for key in keys if key in summary}

    def _step4_from_fields(fields):
        params = _native_params()
        device = torch.device("cpu")
        dtype = torch.float64
        omega = torch.tensor(math.pi * params.f_lower * params.M_sec, device=device, dtype=dtype)
        S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], device=device, dtype=dtype)
        S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)
        r = fields["r"]
        py = fields["py"]
        pz = fields["pz"]
        L_vec = torch.tensor([0.0, -r * pz, r * py], device=device, dtype=dtype)
        summary = dyn._precessing_ic_radial_momentum_summary(
            params,
            r,
            py,
            pz,
            L_vec,
            S1,
            S2,
            omega,
            final_tortoise=1,
            include_flux_modes=True,
            device=device,
            dtype=dtype,
        )
        assert summary is not None
        out = _clean_step4_summary(summary)
        out["pr_star_minus_px"] = out["pr_star"] - fields["px"]
        if "flux_modes" in summary:
            out["flux_22_frac"] = float(summary["flux_modes"]["2,2"] / summary["flux"])
        return out

    def _step4_required_scalars(summary, target_pr_star):
        pr_probe = 1.0e-3
        pr_star = float(summary["pr_star"])
        csi = float(summary["csi"])
        flux = float(summary["flux"])
        dEdr = float(summary["dEdr"])
        dHdpr = float(summary["dHdpr"])
        dHdpphi = float(summary["dHdpphi"])
        d2Hdr2 = float(summary["d2Hdr2"])
        d2Hdrdpphi = float(summary["d2Hdrdpphi"])

        required_flux = -target_pr_star * dEdr * dHdpr / (csi * pr_probe)
        required_dEdr = -csi * flux * pr_probe / (target_pr_star * dHdpr)
        required_dHdpr = -csi * flux * pr_probe / (target_pr_star * dEdr)
        required_csi = -target_pr_star * dEdr * dHdpr / (flux * pr_probe)
        required_d2Hdr2 = -required_dEdr * d2Hdrdpphi / dHdpphi
        required_d2Hdrdpphi = -dHdpphi * d2Hdr2 / required_dEdr

        return {
            "pr_star_rel_delta": (pr_star - target_pr_star) / target_pr_star,
            "required_flux_ratio": required_flux / flux,
            "required_dEdr_ratio": required_dEdr / dEdr,
            "required_dHdpr_ratio": required_dHdpr / dHdpr,
            "required_csi_ratio": required_csi / csi,
            "required_d2Hdr2_ratio": required_d2Hdr2 / d2Hdr2,
            "required_d2Hdrdpphi_ratio": required_d2Hdrdpphi / d2Hdrdpphi,
        }

    lal_row = _lal_phm_dynamics_rows(params_dict, 1, vector="adas", n_cols=26)[0]
    lal_fields = _row_fields(lal_row)
    lal_dynamics = _row_dynamics(lal_row)
    torch_rhs_at_lal = _rhs_from_lal_row(lal_row)

    params = _native_params()
    y_reduced = dyn.initial_conditions(params, device=torch.device("cpu"), dtype=torch.float64)
    y_cart = dyn.reduced_state_to_cartesian_state(y_reduced, params)
    torch_fields = {
        "r": float(y_cart[0]),
        "px": float(y_cart[3]),
        "py": float(y_cart[4]),
        "pz": float(y_cart[5]),
    }

    step4_at_lal = _step4_from_fields(lal_fields)
    step4_at_torch = _step4_from_fields(torch_fields)

    return {
        "lal_row0": lal_fields,
        "lal_row0_dynamics": lal_dynamics,
        "torch_rhs_at_lal_row0": torch_rhs_at_lal,
        "rhs_velocity_delta": {
            key: torch_rhs_at_lal[key] - lal_dynamics[key]
            for key in torch_rhs_at_lal
        },
        "torch_row0": torch_fields,
        "row0_delta": {key: torch_fields[key] - lal_fields[key] for key in lal_fields},
        "step4_at_lal_row0": step4_at_lal,
        "step4_lal_row0_required_scalar_ratios": _step4_required_scalars(
            step4_at_lal,
            lal_fields["px"],
        ),
        "step4_at_torch_row0": step4_at_torch,
    }


def _format_boundary_diagnostic(lal_ref, torch_summary, comparison):
    vectors = lal_ref["vectors"]
    join = lal_ref["join"]
    merger = lal_ref["merger"]
    parts = [
        "SEOBNRv4PHM AdaS/HiS/join boundary diagnostic",
        (
            f"counts lal=(adas={vectors['adas']['n_dyn']}, his={vectors['his']['n_dyn']}, "
            f"joined={vectors['joined']['n_dyn']}) "
            f"torch=(adas={torch_summary['n_adas']}, his={torch_summary['n_his']}, "
            f"joined={torch_summary['n_joined']})"
        ),
        (
            f"indices lal=(his={join['index_join_his']}, attach={join['index_join_attach']}) "
            f"torch=(his={torch_summary['index_join_his']}, "
            f"attach={torch_summary['index_join_attach']})"
        ),
        (
            f"times lal=(tstartHiS={join['t_his_start_M']:.12g}, "
            f"tAttach={merger['t_attach_M']:.12g}, "
            f"floor={join['t_join_attach_M']:.12g}, end={vectors['joined']['t_end_M']:.12g}) "
            f"torch=(tstartHiS={torch_summary['t_start_his_M']:.12g}, "
            f"tAttach={torch_summary['t_attach_M']:.12g}, "
            f"floor={torch_summary['t_join_attach_M']:.12g}, "
            f"end={torch_summary['joined_end_M']:.12g})"
        ),
        f"deltas={comparison}",
        (
            f"samples lal_adas={vectors['adas']['samples'][:3]} "
            f"torch_adas={torch_summary['adas_samples'][:3]}"
        ),
        f"torch_stops=(adas={torch_summary['adas_stop_state']}, his={torch_summary['his_stop_state']})",
        f"torch_rhs={torch_summary['rhs_derivative_options']}",
    ]
    return "; ".join(parts)


def _format_ic_step4_diagnostic(comparison):
    return "; ".join(
        [
            "SEOBNRv4PHM IC Step 4 diagnostic outside strict parity tolerance",
            f"lal_row0={comparison['lal_row0']}",
            f"lal_row0_dynamics={comparison['lal_row0_dynamics']}",
            f"torch_rhs_at_lal_row0={comparison['torch_rhs_at_lal_row0']}",
            f"rhs_velocity_delta={comparison['rhs_velocity_delta']}",
            f"torch_row0={comparison['torch_row0']}",
            f"row0_delta={comparison['row0_delta']}",
            f"step4_at_lal_row0={comparison['step4_at_lal_row0']}",
            (
                "step4_lal_row0_required_scalar_ratios="
                f"{comparison['step4_lal_row0_required_scalar_ratios']}"
            ),
            f"step4_at_torch_row0={comparison['step4_at_torch_row0']}",
        ]
    )


def _format_checkpoint_diagnostic(lal_summary, torch_summary, comparison):
    lal_join = lal_summary.get("join", {})
    parts = [
        "SEOBNRv4PHM checkpoint diagnostic outside strict parity tolerance",
        f"lal_n={lal_summary['n_dyn']} torch_n={torch_summary['n_traj']}",
        f"dt_end={comparison['dt_end_M']:.3e}M",
        f"dt_peak={comparison['dt_peak_omega_M']:.3e}M",
        f"domega_peak_rel={comparison['domega_peak_rel']:.3e}",
        f"dr_start={comparison['dr_start']:.3e}",
        f"dr_end={comparison['dr_end']:.3e}",
        f"torch_segments=(adas={torch_summary['n_adas']}, his_src={torch_summary['n_his_src']}, "
        f"his_uniform={torch_summary['n_his_uniform']}, start_his={torch_summary['index_start_his']}, "
        f"join_attach={torch_summary['index_join_attach']})",
        (
            f"lal_join=(his={lal_join.get('index_join_his')}, "
            f"attach={lal_join.get('index_join_attach')}, "
            f"t_his={lal_join.get('t_join_his_M')}, "
            f"t_attach_floor={lal_join.get('t_join_attach_M')})"
        ),
        (
            f"torch_join=(his={torch_summary.get('index_join_his')}, "
            f"attach={torch_summary.get('index_join_attach')}, "
            f"t_his={torch_summary.get('t_join_his_M')}, "
            f"t_attach_floor={torch_summary.get('t_join_attach_M')})"
        ),
        f"torch_stops=(adas={torch_summary['adas_stop_state']}, his={torch_summary['his_stop_state']})",
        f"torch_t_attach={torch_summary['t_attach_M']:.3f}M",
        f"torch_final=(Mf={torch_summary['final_mass_frac']:.6f}, af={torch_summary['final_spin']:.6f})",
        f"torch_cal=(cal21={torch_summary['cal21']:.3e}, cal55={torch_summary['cal55']:.3e})",
        f"torch_rhs={torch_summary['rhs_derivative_options']}",
    ]
    return "; ".join(parts)


def test_seobnrv4phm_torch_flag_off_fallback():
    """With native flags off, torch scheme must match CPU fallback."""
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    env_backup = {k: os.environ.get(k) for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4PHM_NATIVE")}
    try:
        h_cpu = _cpu_waveform()
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ["PYCBC_SEOBNRV4PHM_NATIVE"] = "0"
        h_torch, _ = get_fd_waveform(approximant="SEOBNRv4PHM", **PARAMS)
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single
    n = min(len(h_cpu), len(h_torch))
    np.testing.assert_allclose(h_torch.numpy()[:n], h_cpu.numpy()[:n], rtol=1e-10, atol=1e-14)


@pytest.mark.skipif(not RUN_SLOW_NATIVE_V4, reason=SLOW_NATIVE_V4_REASON)
def test_seobnrv4_torch_optional_td_parity(monkeypatch):
    """Validate the public aligned TD path against its LAL reference."""
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
        monkeypatch.setenv("PYCBC_SEOBNRV4_NATIVE", "0")
        hp_ref, hc_ref = get_td_waveform(
            approximant="SEOBNRv4", **ALIGNED_V4_PARAMS
        )

        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme("cpu")
        monkeypatch.setenv("PYCBC_SEOBNRV4_NATIVE", "1")
        hp_got, hc_got = get_td_waveform(
            approximant="SEOBNRv4", **ALIGNED_V4_PARAMS
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    reference = hp_ref.numpy() - 1j * hc_ref.numpy()
    actual = hp_got.numpy() - 1j * hc_got.numpy()
    assert np.isfinite(actual).all()

    n = min(len(reference), len(actual))
    reference = reference[:n]
    actual = actual[:n]
    phase = np.vdot(actual, reference)
    phase /= abs(phase)
    scale = np.vdot(actual, reference) / np.vdot(actual, actual)
    relative_error = np.linalg.norm(reference - phase * actual) / np.linalg.norm(
        reference
    )

    assert n > 100
    assert relative_error < 4.0e-3
    assert abs(abs(scale) - 1.0) < 2.0e-3
    assert abs(float(hp_got.start_time - hp_ref.start_time)) <= ALIGNED_V4_PARAMS[
        "delta_t"
    ]
    assert abs(np.argmax(abs(actual)) - np.argmax(abs(reference))) <= 1


@pytest.mark.skipif(not RUN_SLOW_NATIVE_V4HM, reason=SLOW_NATIVE_V4HM_REASON)
def test_seobnrv4hm_torch_optional_td_parity(monkeypatch):
    """Validate selector 41 on its exact public physical-time grid."""
    import torch
    from scipy.interpolate import CubicSpline

    from pycbc.waveform import seobnrv4phm_torch, waveform

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    native_calls = 0
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
        monkeypatch.setenv("PYCBC_SEOBNRV4HM_NATIVE", "0")
        hp_ref, hc_ref = get_td_waveform(
            approximant="SEOBNRv4HM",
            **ALIGNED_V4HM_PARAMS,
        )
        reference_hp = hp_ref.numpy().copy()
        reference_hc = hc_ref.numpy().copy()
        reference_epoch = float(hp_ref.start_time)

        assert len(hp_ref) == len(hc_ref) == 719
        assert reference_epoch == pytest.approx(-0.053710930, abs=2.0e-9)
        reference = reference_hp + 1j * reference_hc
        support = np.flatnonzero(reference != 0.0)
        assert support[0] == 0
        assert support[-1] == 292
        pinned = {
            0: (
                1.8479892334058593e-21,
                -8.993540910187601e-23,
            ),
            109: (
                -2.8660564774122255e-21,
                1.2633866226052272e-21,
            ),
            292: (
                -3.44098578687967e-25,
                7.929598164049504e-26,
            ),
        }
        for index, (expected_hp, expected_hc) in pinned.items():
            np.testing.assert_allclose(
                (reference_hp[index], reference_hc[index]),
                (expected_hp, expected_hc),
                # LAL/GSL build and compiler contraction differences reach
                # about 8e-5 relative; these rows guard sign and convention.
                rtol=2.0e-4,
                atol=1.0e-33,
            )

        native_generator = seobnrv4phm_torch.seobnrv4hm_td_torch

        def recording_native(**parameters):
            nonlocal native_calls
            native_calls += 1
            return native_generator(**parameters)

        def unexpected_lalsimulation(*_args, **_kwargs):
            raise AssertionError("native SEOBNRv4HM called LALSimulation")

        monkeypatch.setattr(
            seobnrv4phm_torch,
            "seobnrv4hm_td_torch",
            recording_native,
        )
        monkeypatch.setattr(
            waveform.lalsimulation,
            "SimInspiralChooseTDWaveform",
            unexpected_lalsimulation,
        )
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
        monkeypatch.setenv("PYCBC_SEOBNRV4HM_NATIVE", "1")
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme("cpu")
        hp_got, hc_got = get_td_waveform(
            approximant="SEOBNRv4HM",
            **ALIGNED_V4HM_PARAMS,
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert native_calls == 1
    assert all(
        isinstance(series._data.tensor, torch.Tensor)
        and series._data.tensor.device.type == "cpu"
        for series in (hp_got, hc_got)
    )
    assert hp_got.delta_t == hc_got.delta_t == ALIGNED_V4HM_PARAMS["delta_t"]

    actual_hp = hp_got.numpy()
    actual_hc = hc_got.numpy()
    actual = actual_hp + 1j * actual_hc
    actual_epoch = float(hp_got.start_time)
    delta_t = ALIGNED_V4HM_PARAMS["delta_t"]
    actual_support = np.flatnonzero(actual != 0.0)
    assert np.isfinite(actual).all()
    assert abs(len(actual) - len(reference)) <= 1
    assert abs(int(np.argmax(abs(actual))) - int(np.argmax(abs(reference)))) <= 1
    assert abs(int(actual_support[0]) - int(support[0])) <= 1
    assert abs(int(actual_support[-1]) - int(support[-1])) <= 1

    epoch_sample_shift = round((reference_epoch - actual_epoch) / delta_t)
    assert abs(epoch_sample_shift) <= 1
    residual_epoch = reference_epoch - (
        actual_epoch + epoch_sample_shift * delta_t
    )
    assert abs(residual_epoch) <= 0.1 * delta_t

    actual_start = max(epoch_sample_shift, 0)
    reference_start = max(-epoch_sample_shift, 0)
    overlap = min(
        len(actual) - actual_start,
        len(reference) - reference_start,
    )
    shifted_grid_l2 = np.linalg.norm(
        actual[actual_start : actual_start + overlap]
        - reference[reference_start : reference_start + overlap]
    ) / np.linalg.norm(
        reference[reference_start : reference_start + overlap]
    )
    assert shifted_grid_l2 < 1.0e-2

    actual_times = actual_epoch + np.arange(len(actual)) * delta_t
    reference_times = reference_epoch + np.arange(len(reference)) * delta_t
    common = (reference_times >= actual_times[0]) & (
        reference_times <= actual_times[-1]
    )
    assert np.count_nonzero(common) >= min(len(actual), len(reference)) - 2
    comparison_times = reference_times[common]
    comparison_reference = reference[common]
    interpolated = CubicSpline(
        actual_times,
        actual_hp,
        bc_type="natural",
        extrapolate=False,
    )(comparison_times) + 1j * CubicSpline(
        actual_times,
        actual_hc,
        bc_type="natural",
        extrapolate=False,
    )(comparison_times)
    assert np.isfinite(interpolated).all()
    physical_time_l2 = np.linalg.norm(
        interpolated - comparison_reference
    ) / np.linalg.norm(comparison_reference)
    phase_offset = float(np.angle(np.vdot(interpolated, comparison_reference)))
    norm_ratio = float(
        np.linalg.norm(interpolated) / np.linalg.norm(comparison_reference)
    )
    peak_ratio = float(
        np.max(abs(interpolated)) / np.max(abs(comparison_reference))
    )

    assert physical_time_l2 < 2.0e-3
    assert abs(phase_offset) < 1.0e-3
    assert abs(norm_ratio - 1.0) < 2.0e-3
    assert abs(peak_ratio - 1.0) < 2.0e-3


@pytest.mark.skipif(not RUN_SLOW_NATIVE_V4, reason=SLOW_NATIVE_V4_REASON)
def test_seobnrv4_torch_optional_fd_parity(monkeypatch):
    """Validate the native public TD-to-FD path against its LAL reference."""
    fd_params = dict(ALIGNED_V4_PARAMS)
    fd_params.pop("delta_t")
    fd_params["delta_f"] = 0.25

    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    torch_storage = False
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
        monkeypatch.setenv("PYCBC_SEOBNRV4_NATIVE", "0")
        hp_ref, hc_ref = get_fd_waveform(
            approximant="SEOBNRv4", **fd_params
        )

        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme("cpu")
        monkeypatch.setenv("PYCBC_SEOBNRV4_NATIVE", "1")
        hp_got, hc_got = get_fd_waveform(
            approximant="SEOBNRv4", **fd_params
        )
        torch_storage = all(
            hasattr(series._data, "tensor") for series in (hp_got, hc_got)
        )
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single

    assert torch_storage
    assert len(hp_got) == len(hp_ref)
    assert len(hc_got) == len(hc_ref)
    assert hp_got.delta_f == pytest.approx(hp_ref.delta_f)
    assert hc_got.delta_f == pytest.approx(hc_ref.delta_f)
    assert float(hp_got.start_time) == pytest.approx(
        float(hp_ref.start_time)
    )

    reference = hp_ref.numpy() - 1j * hc_ref.numpy()
    actual = hp_got.numpy() - 1j * hc_got.numpy()
    frequencies = np.arange(len(reference)) * hp_ref.delta_f
    band = (
        (frequencies >= fd_params["f_lower"])
        & (frequencies <= 512.0)
        & np.isfinite(reference)
        & np.isfinite(actual)
        & (np.abs(reference) > 0.0)
    )
    assert band.sum() > 100
    reference = reference[band]
    actual = actual[band]
    phase = np.vdot(actual, reference)
    phase /= abs(phase)
    scale = np.vdot(actual, reference) / np.vdot(actual, actual)
    relative_error = np.linalg.norm(reference - phase * actual) / np.linalg.norm(
        reference
    )

    assert relative_error < 8.0e-3
    assert abs(abs(scale) - 1.0) < 2.0e-3


@pytest.fixture(scope="module")
def native_phm_waveform():
    """Generate the expensive native waveform once for smoke and parity."""
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    env_backup = {
        k: os.environ.get(k)
        for k in ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_SEOBNRV4PHM_NATIVE")
    }
    try:
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "1"
        os.environ["PYCBC_SEOBNRV4PHM_NATIVE"] = "1"
        return get_fd_waveform(approximant="SEOBNRv4PHM", **PARAMS)
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


@pytest.mark.skipif(not RUN_SLOW_NATIVE_PHM, reason=SLOW_NATIVE_PHM_REASON)
def test_seobnrv4phm_torch_native_smoke(native_phm_waveform):
    """Native torch path should run and produce finite spectra."""
    h_torch, _ = native_phm_waveform
    arr = h_torch.numpy()
    assert np.isfinite(arr).all()
    assert arr.ndim == 1 and len(arr) > 8
    assert len(arr) == int(PARAMS["f_final"] / PARAMS["delta_f"]) + 1


@pytest.mark.skipif(not RUN_SLOW_NATIVE_PHM, reason=SLOW_NATIVE_PHM_REASON)
def test_seobnrv4phm_torch_optional_parity(native_phm_waveform):
    """Check in-band numerical parity against the CPU/LAL reference."""
    h_cpu = _cpu_waveform()
    h_torch, _ = native_phm_waveform
    metrics = _phm_fd_parity_metrics(h_cpu, h_torch)
    tol = _PHM_FD_PARITY_TOLERANCES
    within_tolerance = (
        metrics["bins"] > 0
        and metrics["rel"] < tol["raw_rel"]
        and metrics["amp_min"] > tol["amp_min"]
        and metrics["amp_max"] < tol["amp_max"]
        and abs(metrics["phase_mean"]) < tol["raw_phase_mean"]
        and metrics["phase_std"] < tol["raw_phase_std"]
        and abs(metrics["linear_dt_s"]) < tol["linear_dt_s"]
        and metrics["aligned_rel"] < tol["aligned_rel"]
        and metrics["aligned_phase_std"] < tol["aligned_phase_std"]
    )
    assert within_tolerance, _format_phm_fd_parity_metrics(metrics)


@pytest.mark.skipif(not RUN_BOUNDARY_DIAGNOSTIC, reason=BOUNDARY_DIAGNOSTIC_REASON)
def test_seobnrv4phm_torch_optional_boundary_diagnostic(monkeypatch):
    """Fast trajectory-boundary diagnostic before merger/NQC/RD work."""
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = _phm_boundary_params()
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_AUTO_TMAX", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    monkeypatch.setenv(
        "PYCBC_SEOBNRV4PHM_ADAS_MAX_STEPS",
        os.environ.get("PYCBC_SEOBNRV4PHM_ADAS_MAX_STEPS", "4"),
    )
    monkeypatch.setenv(
        "PYCBC_SEOBNRV4PHM_HIS_MAX_STEPS",
        os.environ.get("PYCBC_SEOBNRV4PHM_HIS_MAX_STEPS", "2"),
    )
    lal_ref = _isolated_lal_phm_dynamics(params)
    torch_summary = swt._seobnrv4phm_native_boundary_summary(
        **params,
        t_attach_M=lal_ref["merger"]["t_attach_M"],
    )
    comparison = _phm_boundary_comparison(lal_ref, torch_summary)

    within_tolerance = (
        comparison["dn_adas"] == 0
        and comparison["dn_his"] == 0
        and comparison["dn_joined"] == 0
        and comparison["d_index_join_his"] == 0
        and comparison["d_index_join_attach"] == 0
        and abs(comparison["dt_start_his_M"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["dt_M"]
        and abs(comparison["dt_join_his_M"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["dt_M"]
        and abs(comparison["dt_attach_floor_M"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["dt_M"]
        and abs(comparison["dt_joined_end_M"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["dt_M"]
        and abs(comparison["dr_start"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["r"]
        and abs(comparison["dr_his_start"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["r"]
        and abs(comparison["dr_joined_end"]) < _PHM_CHECKPOINT_PARITY_TOLERANCES["r"]
    )
    assert within_tolerance, _format_boundary_diagnostic(lal_ref, torch_summary, comparison)


@pytest.mark.skipif(not RUN_IC_STEP4_DIAGNOSTIC, reason=IC_STEP4_DIAGNOSTIC_REASON)
def test_seobnrv4phm_torch_optional_ic_step4_diagnostic(monkeypatch):
    """Compare row-0 LAL IC inputs against torch Step 4 ingredients."""
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")

    comparison = _phm_ic_step4_comparison(_phm_boundary_params())
    row0_delta = comparison["row0_delta"]
    step4_at_lal = comparison["step4_at_lal_row0"]

    within_tolerance = (
        abs(row0_delta["r"]) < 1.0e-10
        and abs(row0_delta["px"]) < 1.0e-10
        and abs(row0_delta["py"]) < 1.0e-10
        and abs(row0_delta["pz"]) < 1.0e-10
        and abs(step4_at_lal["pr_star_minus_px"]) < 1.0e-10
    )
    assert within_tolerance, _format_ic_step4_diagnostic(comparison)


@pytest.mark.skipif(not RUN_CHECKPOINT_DIAGNOSTIC, reason=CHECKPOINT_DIAGNOSTIC_REASON)
def test_seobnrv4phm_torch_optional_checkpoint_diagnostic(monkeypatch):
    """Strict checkpoint diagnostic before the full FD parity path."""
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = _phm_boundary_params()
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_AUTO_TMAX", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    lal_summary = _lal_phm_checkpoint_summary(params)
    try:
        torch_summary = swt._seobnrv4phm_native_checkpoint_summary(**params)
    except RuntimeError as exc:
        pytest.fail(f"SEOBNRv4PHM checkpoint path failed: {exc}")

    omega_ref = max(abs(lal_summary["omega_peak_dimless"]), 1.0e-12)
    comparison = {
        "dt_end_M": torch_summary["t_end_M"] - lal_summary["t_end_M"],
        "dt_peak_omega_M": torch_summary["t_peak_omega_argmax_M"] - lal_summary["t_peak_omega_M"],
        "domega_peak_rel": (torch_summary["omega_peak_dimless"] - lal_summary["omega_peak_dimless"]) / omega_ref,
        "dr_start": torch_summary["r_start"] - lal_summary["r_start"],
        "dr_end": torch_summary["r_end"] - lal_summary["r_end"],
    }
    tol = _PHM_CHECKPOINT_PARITY_TOLERANCES
    within_tolerance = (
        abs(comparison["dt_end_M"]) < tol["dt_M"]
        and abs(comparison["dt_peak_omega_M"]) < tol["dt_M"]
        and abs(comparison["domega_peak_rel"]) < tol["omega_peak_rel"]
        and abs(comparison["dr_start"]) < tol["r"]
        and abs(comparison["dr_end"]) < tol["r"]
    )
    assert within_tolerance, _format_checkpoint_diagnostic(lal_summary, torch_summary, comparison)


def test_adas_initial_step_matches_lal_delta_t_scaling():
    """AdaS should use LAL's INdeltaT/mTScaled initial RK step."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=55.0,
        f_ref=55.0,
        mode_array=((2, 2), (2, 1)),
    )
    delta_t = 1.0 / 4096.0
    assert math.isclose(
        swt._lal_adas_initial_step_M(params, delta_t),
        delta_t / params.M_sec,
        rel_tol=1e-15,
        abs_tol=0.0,
    )


def test_lal_phm_dynamics_vector_layout_exposes_adas_his_join():
    """The isolated LAL helper should expose raw AdaS, HiS, and joined vectors."""
    params_dict = _phm_boundary_params()

    ref = _isolated_lal_phm_dynamics(params_dict)
    vectors = ref["vectors"]
    join = ref["join"]
    merger = ref["merger"]

    assert vectors["adas"]["n_dyn"] > join["index_join_his"]
    assert vectors["his"]["dt_start_M"] == pytest.approx(1.0 / 50.0)
    assert join["t_join_his_M"] == pytest.approx(join["t_his_start_M"])
    assert vectors["joined"]["n_dyn"] == join["index_join_his"] + join["n_his_in_join"]
    assert vectors["joined"]["t_end_M"] < vectors["his"]["t_end_M"]
    assert join["t_dynamics_end_M"] < merger["t_attach_M"]
    assert join["t_join_attach_M"] >= merger["t_attach_M"] - 1.0 / 50.0
    assert vectors["adas"]["samples"][0]["t_M"] == pytest.approx(0.0)
    assert vectors["adas"]["samples"][0]["r"] == pytest.approx(vectors["adas"]["r_start"])
    assert len(vectors["adas"]["samples"][0]["state"]) == 14


def test_lal_phm_joined_dynamics_excludes_attachment_floor_sample():
    """LAL's joined dynamics length is indexJoinAttach, excluding that sample."""
    params_dict = _phm_boundary_params()

    ref = _isolated_lal_phm_dynamics(params_dict)
    vectors = ref["vectors"]
    join = ref["join"]

    assert join["index_join_attach"] == vectors["joined"]["n_dyn"]
    assert join["t_dynamics_end_M"] == pytest.approx(vectors["joined"]["t_end_M"])
    assert join["t_join_attach_M"] == pytest.approx(join["t_first_his_excluded_M"])
    assert join["t_dynamics_end_M"] < join["t_join_attach_M"]
    assert join["t_join_attach_M"] - join["t_dynamics_end_M"] == pytest.approx(
        1.0 / 50.0
    )
    assert vectors["joined"]["t_peak_omega_M"] == pytest.approx(vectors["joined"]["t_end_M"])
    assert vectors["his"]["t_peak_omega_M"] > join["t_join_attach_M"]


def test_phm_integration_max_step_env(monkeypatch):
    """Native trajectory diagnostics should support global and stage step caps."""
    from pycbc.waveform import seobnrv4phm_torch as swt

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_MAX_STEPS", "11")
    assert swt._integration_max_steps("adas", 200000) == 11

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_ADAS_MAX_STEPS", "7")
    assert swt._integration_max_steps("adas", 200000) == 7
    assert swt._integration_max_steps("his", 200000) == 11


def test_precessing_initial_conditions_match_lal_spherical_orbit(monkeypatch):
    """Projected ICs should solve the LAL-style precessing spherical orbit."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROJECTED_IC", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PRECESSING_IC_ROOT", "1")

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=PARAMS["spin1x"],
        spin1y=PARAMS["spin1y"],
        spin1z=PARAMS["spin1z"],
        spin2x=PARAMS["spin2x"],
        spin2y=PARAMS["spin2y"],
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )

    device = torch.device("cpu")
    dtype = torch.float64
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    r_vec, p_vec, _, _, _ = dyn._reduced_to_cartesian(y0[0], y0[1], y0[2], y0[3:6], y0[6:9], y0[9:12], params)

    # LAL SimIMRSpinPrecEOBWaveformAll first dynamics row for this fixture.
    torch.testing.assert_close(r_vec, torch.tensor([13.61567447, 0.0, 0.0], device=device, dtype=dtype), rtol=0.0, atol=2e-5)
    torch.testing.assert_close(p_vec[1:], torch.tensor([0.30381503, -0.000622258], device=device, dtype=dtype), rtol=0.0, atol=2e-6)
    torch.testing.assert_close(torch.linalg.cross(r_vec, p_vec), y0[3:6], rtol=1e-11, atol=1e-11)


def test_aligned_initial_conditions_use_v4_energy_balance(monkeypatch):
    """Aligned IC radial momentum should follow LAL's optimized-v4 step 4."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROJECTED_IC", "1")
    params = dyn.EOBParams(
        mass1=50.0,
        mass2=40.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.3,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.2,
        distance=100.0,
        inclination=0.0,
        f_lower=15.0,
        f_ref=15.0,
    )
    y0 = dyn.initial_conditions(
        params,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    # First SEOBNRv4 dynamics row from LALSuite 7.26.1.
    expected = torch.tensor(
        [13.213042632672478, -0.0016211742277193744, 4.096256759858304],
        dtype=torch.float64,
    )
    torch.testing.assert_close(y0[[0, 5]], expected[[0, 2]], rtol=0.0, atol=5.0e-8)
    assert float(y0[1]) == pytest.approx(float(expected[1]), rel=1.0e-4, abs=2.0e-7)


def test_precessing_ic_radial_momentum_boundary_tracks_lal(monkeypatch):
    """STEP 4 and the returned radial momentum should track LAL."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    params_dict = _phm_boundary_params()
    lal_row = _lal_phm_dynamics_rows(params_dict, 1, vector="adas")[0]
    params = swt._native_params_from_kwargs(params_dict)
    device = torch.device("cpu")
    dtype = torch.float64
    px_lal = lal_row[4]
    captured = {}
    radial_momentum_summary = dyn._precessing_ic_radial_momentum_summary

    def capture_radial_momentum_summary(*args, **kwargs):
        # Step 4 precedes LAL's spherical/Cartesian round trip.  Reapplying it
        # to the returned row is not equivalent: py and pz each round by one
        # ulp, enough to steer the adaptive finite-difference retry.  Capture
        # the actual pre-round-trip call instead.
        kwargs["include_flux_modes"] = True
        summary = radial_momentum_summary(*args, **kwargs)
        captured["summary"] = summary
        return summary

    monkeypatch.setattr(
        dyn,
        "_precessing_ic_radial_momentum_summary",
        capture_radial_momentum_summary,
    )
    initial_state = dyn.initial_cartesian_conditions(
        params,
        device=device,
        dtype=dtype,
    )
    summary = captured.get("summary")

    assert summary is not None
    for key in ("flux", "dEdr", "dxdt_probe0", "pr_star"):
        assert torch.isfinite(summary[key])
    assert "flux_modes" in summary
    flux_sum = sum(float(v) for v in summary["flux_modes"].values())
    assert flux_sum == pytest.approx(float(summary["flux"]), rel=0.0, abs=1.0e-18)
    assert float(summary["flux_modes"]["2,2"] / summary["flux"]) > 0.99
    assert 0.0 < float(summary["flux_modes"]["5,5"] / summary["flux"]) < 5.0e-5
    pr_star = float(summary["pr_star"])
    px_torch = float(initial_state[3])
    assert pr_star == pytest.approx(px_torch, rel=0.0, abs=5.0e-15)
    assert px_torch == pytest.approx(px_lal, rel=0.0, abs=5.0e-11)
    assert abs((px_torch - px_lal) / px_lal) < 1.0e-8


def test_precessing_ic_root_radius_uses_lal_extension_order():
    """The IC root callback mirrors the loaded LAL extension's split multiply/add."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    x_scaled = float.fromhex("0x1.0be0ded288ce8p-3")
    assert dyn._precessing_ic_root_radius(x_scaled).hex() == "0x1.80175b315b098p+2"


def test_precessing_spin_weight_uses_lal_v4p_order():
    """v4P stores spin states as chi*m_i*m_i/M/M, not chi*(m_i/M)^2."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    assert dyn._lal_spin_weight(0.03, 12.0, 20.0).hex() == "0x1.61e4f765fd8aep-7"
    assert (0.03 * (12.0 / 20.0) * (12.0 / 20.0)).hex() == "0x1.61e4f765fd8adp-7"


def test_lal_initial_row_rhs_velocity_matches_torch(monkeypatch):
    """At the LAL IC row, remaining drift is upstream of RHS/AdaS replay."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    params_dict = _phm_boundary_params()
    lal_row = _lal_phm_dynamics_rows(params_dict, 1, vector="adas", n_cols=26)[0]
    params = swt._native_params_from_kwargs(params_dict)
    device = torch.device("cpu")
    dtype = torch.float64

    dyn.initial_conditions(params, device=device, dtype=dtype)
    y_cart = torch.tensor(lal_row[1:15], device=device, dtype=dtype)
    rhs = dyn.rhs_cartesian_full(
        torch.tensor(lal_row[0], device=device, dtype=dtype),
        y_cart,
        params,
    )

    torch.testing.assert_close(
        rhs[:3],
        torch.tensor(lal_row[15:18], device=device, dtype=dtype),
        rtol=0.0,
        atol=8.0e-10,
    )
    assert lal_row[18] == pytest.approx(lal_row[1], rel=0.0, abs=1.0e-15)
    assert lal_row[20] == pytest.approx(lal_row[4], rel=0.0, abs=1.0e-15)


def test_precessing_initial_conditions_balance_projected_pdot(monkeypatch):
    """Default projected RHS should not immediately kick the rooted IC outward."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROJECTED_IC", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PRECESSING_IC_ROOT", "1")
    monkeypatch.delenv("PYCBC_SEOBNRV4PHM_TORTOISE_PDOT", raising=False)

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=PARAMS["spin1x"],
        spin1y=PARAMS["spin1y"],
        spin1z=PARAMS["spin1z"],
        spin2x=PARAMS["spin2x"],
        spin2y=PARAMS["spin2y"],
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )

    device = torch.device("cpu")
    dtype = torch.float64
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    dy = dyn.rhs_cartesian_projected(torch.tensor(0.0, device=device, dtype=dtype), y0, params)

    assert abs(float(dy[1])) < 1.0e-5
    assert float(dy[0]) < 0.0


def test_spins_almost_aligned_zeroes_inplane_and_skips_precessing_root(monkeypatch):
    """LAL's SpinsAlmostAligned branch zeroes xy spins and bypasses precessing ICs."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    def fail_precessing_root(*args, **kwargs):
        raise AssertionError("aligned-spin ICs should not call the precessing root")

    monkeypatch.setattr(dyn, "_precessing_spherical_initial_conditions", fail_precessing_root)
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROJECTED_IC", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PRECESSING_IC_ROOT", "1")

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=0.5 * dyn.EPS_ALIGN,
        spin1y=-0.25 * dyn.EPS_ALIGN,
        spin1z=PARAMS["spin1z"],
        spin2x=0.0,
        spin2y=0.5 * dyn.EPS_ALIGN,
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )

    assert params.aligned_spins is True
    assert params.spin1x == params.spin1y == params.spin2x == params.spin2y == 0.0

    device = torch.device("cpu")
    dtype = torch.float64
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)

    torch.testing.assert_close(y0[3:5], torch.zeros(2, device=device, dtype=dtype), rtol=0.0, atol=0.0)
    torch.testing.assert_close(y0[6:8], torch.zeros(2, device=device, dtype=dtype), rtol=0.0, atol=0.0)
    torch.testing.assert_close(y0[9:11], torch.zeros(2, device=device, dtype=dtype), rtol=0.0, atol=0.0)


def test_spins_almost_aligned_threshold_is_strict():
    """LAL uses a strict in-plane spin norm < EPS_ALIGN test."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=dyn.EPS_ALIGN,
        spin1y=0.0,
        spin1z=PARAMS["spin1z"],
        spin2x=0.0,
        spin2y=0.0,
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )

    assert params.aligned_spins is False
    assert params.spin1x == dyn.EPS_ALIGN


def test_cartesian_x_gradient_matches_finite_difference(monkeypatch):
    """Autograd fixed-P Cartesian force should mirror LAL's exact derivative block."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_CARTESIAN_RHS", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROJECTED_IC", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PRECESSING_IC_ROOT", "1")

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=PARAMS["spin1x"],
        spin1y=PARAMS["spin1y"],
        spin1z=PARAMS["spin1z"],
        spin2x=PARAMS["spin2x"],
        spin2y=PARAMS["spin2y"],
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )

    device = torch.device("cpu")
    dtype = torch.float64
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    r_vec, p_vec, _, _, _ = dyn._reduced_to_cartesian(y0[0], y0[1], y0[2], y0[3:6], y0[6:9], y0[9:12], params)

    grad_auto = dyn._dH_dx_cartesian_autograd(r_vec, p_vec, y0[6:9], y0[9:12], params)
    grad_fd = dyn._dH_dx_cartesian_fd(r_vec, p_vec, y0[6:9], y0[9:12], params)
    torch.testing.assert_close(grad_auto, grad_fd, rtol=1e-6, atol=1e-8)


def test_euler_j2p_initial_gamma_alignment():
    """Initial gamma should reproduce (n, lambda, L) alignment w.r.t. J-basis."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    device = torch.device("cpu")
    dtype = torch.float64
    J_hat = torch.tensor([0.3, 0.4, 0.86602540378], device=device, dtype=dtype)
    e1J, e2J, e3J = swt._build_J_frame(J_hat)
    Lhat = torch.tensor([0.2, 0.6, 0.75], device=device, dtype=dtype)
    Lhat = Lhat / torch.linalg.norm(Lhat)
    Lvec = Lhat.unsqueeze(0).repeat(5, 1)
    phi0 = torch.tensor(0.7, device=device, dtype=dtype)
    dt = 0.1

    alpha, beta, gamma = swt._euler_j2p(Lvec, e1J, e2J, e3J, dt, phi0=phi0)
    gamma_expected = swt._initial_gamma_from_sep(Lhat, e1J, e2J, e3J, phi0)

    torch.testing.assert_close(gamma[0], gamma_expected, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(gamma, torch.full_like(gamma, gamma_expected), atol=1e-12, rtol=0.0)


def test_euler_j2p_initial_gamma_uses_supplied_separation():
    """Initial gamma should use the dynamics separation vector, not projected phi."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    device = torch.device("cpu")
    dtype = torch.float64
    J_hat = torch.tensor([0.25, -0.35, 0.90277350426], device=device, dtype=dtype)
    e1J, e2J, e3J = swt._build_J_frame(J_hat)
    Lhat = torch.tensor([0.45, -0.25, 0.85], device=device, dtype=dtype)
    Lhat = Lhat / torch.linalg.norm(Lhat)
    n_raw = torch.tensor([0.1, 0.9, 0.35], device=device, dtype=dtype)
    n_hat = n_raw - torch.dot(n_raw, Lhat) * Lhat
    n_hat = n_hat / torch.linalg.norm(n_hat)
    Lvec = Lhat.unsqueeze(0).repeat(5, 1)
    t_vec = torch.linspace(0.0, 0.4, 5, device=device, dtype=dtype)

    _, _, gamma = swt._euler_j2p(
        Lvec,
        e1J,
        e2J,
        e3J,
        phi0=torch.tensor(1.9, device=device, dtype=dtype),
        n_hat0=n_hat,
        t_vec=t_vec,
    )
    gamma_expected = swt._initial_gamma_from_sep_vector(
        Lhat,
        n_hat,
        e1J,
        e2J,
        e3J,
    )

    torch.testing.assert_close(gamma[0], gamma_expected, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(gamma, torch.full_like(gamma, gamma_expected), atol=1e-12, rtol=0.0)


def test_spins_almost_aligned_euler_angles_are_zero():
    """LAL leaves I/J/P Euler rotations at zero in the aligned-spin branch."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    t = torch.tensor([0.0, 0.3, 0.9, 1.7], dtype=torch.float64)
    alpha, beta, gamma = swt._zero_euler_angles_like(t)

    torch.testing.assert_close(alpha, torch.zeros_like(t), rtol=0.0, atol=0.0)
    torch.testing.assert_close(beta, torch.zeros_like(t), rtol=0.0, atol=0.0)
    torch.testing.assert_close(gamma, torch.zeros_like(t), rtol=0.0, atol=0.0)


def test_lal_output_time_grid_uses_floor_retlen():
    """Output grid length should mirror LAL's retLenTS floor calculation."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    t_modes_m = torch.tensor([0.0, 0.7, 3.74], dtype=torch.float64)
    delta_t = 0.2
    m_sec = 2.0
    out_m = swt._lal_output_time_grid(t_modes_m, delta_t, m_sec)

    expected_len = math.floor((float(t_modes_m[-1]) - float(t_modes_m[0])) / (delta_t / m_sec))
    assert len(out_m) == expected_len
    torch.testing.assert_close(out_m, torch.arange(expected_len, dtype=torch.float64) * (delta_t / m_sec))


def test_amplitude_peak_from_22_21_matches_lal_discrete_rules():
    """The PHM epoch peak is the first discrete max of |h22|^2 plus requested |h21|^2."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    t = torch.tensor([10.0, 11.0, 12.0, 13.0], dtype=torch.float64)
    h22 = torch.tensor([1.0, 3.0, 3.0, 0.5], dtype=torch.complex128)
    h21 = torch.tensor([0.0, 0.0, 5.0, 0.0], dtype=torch.complex128)
    modes = {(2, 2): h22, (2, 1): h21}

    t_peak_22, idx_22 = swt._amplitude_peak_from_22_21(modes, ((2, 2),), t)
    assert idx_22 == 1
    torch.testing.assert_close(t_peak_22, t[1])

    t_peak_both, idx_both = swt._amplitude_peak_from_22_21(modes, ((2, 2), (2, 1)), t)
    assert idx_both == 2
    torch.testing.assert_close(t_peak_both, t[2])

    with pytest.raises(ValueError, match=r"\(2, 2\)"):
        swt._amplitude_peak_from_22_21(modes, ((2, 1),), t)


def test_nyquist_check_rejects_ringdown_above_sample_rate():
    """LAL rejects delta_t values that put the checked ringdown mode above Nyquist."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=25.0,
        mass2=18.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.05,
        spin2x=-0.1,
        spin2y=0.07,
        spin2z=0.2,
        distance=300.0,
        inclination=0.6,
        f_lower=30.0,
        f_ref=50.0,
        mode_array=((2, 2), (5, 5)),
    )
    omega_rd = swt._ringdown_omega_for_nyquist(params, 5)
    dt_limit = math.pi / omega_rd

    swt._check_nyquist_frequency(params, 0.5 * dt_limit, 5, waveform_ell_max=5)
    with pytest.raises(ValueError, match="Nyquist"):
        swt._check_nyquist_frequency(params, 1.1 * dt_limit, 5, waveform_ell_max=5)
    with pytest.raises(ValueError, match=">= 2"):
        swt._check_nyquist_frequency(params, 0.5 * dt_limit, 1, waveform_ell_max=5)


def test_reduced_stop_quantities_match_cartesian_orbital_scalars():
    """Reduced-state stop checks should use the same Cartesian scalars as LAL."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=25.0,
        mass2=18.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.0,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=0.0,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    y = torch.tensor(
        [6.0, -0.02, 0.4, 0.0, 0.0, 3.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2],
        dtype=torch.float64,
    )
    dy = torch.tensor(
        [0.15, -0.04, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    )

    q = swt._lal_stop_quantities_from_reduced(y, dy, params)

    torch.testing.assert_close(q["drdt"], dy[0], rtol=0.0, atol=1e-7)
    torch.testing.assert_close(q["omega"], dy[2], rtol=0.0, atol=1e-7)
    torch.testing.assert_close(q["p_dot_r"], y[1], rtol=0.0, atol=1e-12)
    torch.testing.assert_close(q["pr_dot"], dy[1], rtol=0.0, atol=2e-7)
    assert bool(q["pphi_large"]) is False


def test_minimal_rotation_gamma_integrates_cubic_alpha_spline():
    """Gamma integration should use spline alpha_dot rather than sample differencing."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    dt = 0.25
    t = torch.arange(9, dtype=dtype) * dt
    rate = torch.tensor(0.37, dtype=dtype)
    beta0 = torch.tensor(0.63, dtype=dtype)
    gamma0 = torch.tensor(-0.2, dtype=dtype)
    alpha = 0.4 + rate * t
    beta = torch.full_like(alpha, beta0)

    gamma = swt._integrate_minimal_rotation_gamma(alpha, beta, dt, gamma0)
    expected = gamma0 - rate * torch.cos(beta0) * t

    torch.testing.assert_close(gamma, expected, rtol=1e-12, atol=1e-12)


def test_minimal_rotation_gamma_integrates_nonuniform_times():
    """Joined AdaS/HiS Euler gamma integration should use actual mode times."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t = torch.tensor([0.0, 0.17, 0.41, 0.9, 1.4, 2.1], dtype=dtype)
    rate = torch.tensor(0.22, dtype=dtype)
    beta0 = torch.tensor(0.48, dtype=dtype)
    gamma0 = torch.tensor(0.13, dtype=dtype)
    alpha = -0.2 + rate * t
    beta = torch.full_like(alpha, beta0)

    gamma = swt._integrate_minimal_rotation_gamma_times(alpha, beta, t, gamma0)
    expected = gamma0 - rate * torch.cos(beta0) * t

    torch.testing.assert_close(gamma, expected, rtol=1e-12, atol=1e-12)


def test_euler_post_merger_extension_starts_at_attachment():
    """Post-attach dynamics angles are discarded before QNM precession extension."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    alpha = torch.tensor([0.0, 0.2, 0.5, 9.0, 9.5], dtype=dtype)
    beta = torch.tensor([0.3, 0.4, 0.6, 2.0, 2.5], dtype=dtype)
    gamma = torch.tensor([0.1, 0.05, -0.2, 8.0, 8.5], dtype=dtype)
    dt = 0.5
    prec_rate = torch.tensor(0.2, dtype=dtype)

    alpha_out, beta_out, gamma_out = swt._extend_euler_from_attach(
        alpha,
        beta,
        gamma,
        index_start=3,
        target_len=6,
        dt=dt,
        prec_rate=prec_rate,
    )

    torch.testing.assert_close(alpha_out[:3], alpha[:3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(beta_out[:3], beta[:3], rtol=0.0, atol=0.0)
    torch.testing.assert_close(gamma_out[:3], gamma[:3], rtol=0.0, atol=0.0)

    t = torch.arange(1, 4, dtype=dtype) * dt
    torch.testing.assert_close(alpha_out[3:], alpha[2] + prec_rate * t, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(beta_out[3:], torch.full((3,), beta[2], dtype=dtype), rtol=0.0, atol=0.0)
    torch.testing.assert_close(gamma_out[3:], gamma[2] - torch.cos(beta[2]) * prec_rate * t, rtol=1e-12, atol=1e-12)


def test_euler_post_merger_extension_uses_joined_mode_times():
    """Post-attach extension should follow the non-uniform joined P-mode times."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t_modes = torch.tensor([0.0, 0.4, 1.1, 1.3, 1.32, 1.34], dtype=dtype)
    alpha = torch.tensor([0.0, 0.2, 0.5, 9.0], dtype=dtype)
    beta = torch.tensor([0.3, 0.4, 0.6, 2.0], dtype=dtype)
    gamma = torch.tensor([0.1, 0.05, -0.2, 8.0], dtype=dtype)
    prec_rate = torch.tensor(0.7, dtype=dtype)

    alpha_out, beta_out, gamma_out = swt._extend_euler_from_attach_times(
        alpha,
        beta,
        gamma,
        t_modes,
        index_start=3,
        prec_rate=prec_rate,
    )

    torch.testing.assert_close(alpha_out[:3], alpha[:3], rtol=0.0, atol=0.0)
    dt = t_modes[3:] - t_modes[2]
    torch.testing.assert_close(alpha_out[3:], alpha[2] + prec_rate * dt, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(beta_out[3:], torch.full((3,), beta[2], dtype=dtype), rtol=0.0, atol=0.0)
    torch.testing.assert_close(gamma_out[3:], gamma[2] - torch.cos(beta[2]) * prec_rate * dt, rtol=1e-12, atol=1e-12)


def test_euler_qnm_precession_rate_matches_lal_checkpoint():
    """Use the Cardoso QNM table that drives LAL's Euler extension."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    rate = swt._euler_qnm_precession_rate(
        0.7041349531072489,
        0.9512185112429709,
        1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert float(rate) == pytest.approx(0.08245331084837076, abs=1.0e-14)


def test_euler_attach_index_uses_lal_floor_sample():
    """LAL's joined Euler extension index is the last sample <= tAttach."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    t = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)

    assert swt._last_index_leq_increasing(t, 1.9) == 1
    assert swt._nearest_index_increasing(t, 1.9) == 2


def test_constant_inertial_rotation_matches_vectorized():
    """Constant-angle J→I rotation should match per-sample Wigner application."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    device = torch.device("cpu")
    dtype = torch.float64
    t = torch.linspace(0.0, 0.35, steps=9, device=device, dtype=dtype)
    h22 = torch.exp(1j * t)
    h21 = 0.3 * (torch.sin(t) + 1j * torch.cos(t))
    modes = {
        (2, 2): h22,
        (2, 1): h21,
        (2, -1): ((-1.0) ** 2) * torch.conj(h21),
        (2, -2): ((-1.0) ** 2) * torch.conj(h22),
    }
    alpha = torch.tensor(0.31, device=device, dtype=dtype)
    beta = torch.tensor(0.23, device=device, dtype=dtype)
    gamma = torch.tensor(-0.17, device=device, dtype=dtype)

    rot_vec = swt._rotate_modes(modes, alpha.repeat(len(t)), beta.repeat(len(t)), gamma.repeat(len(t)))
    rot_const = swt._rotate_modes_constant(modes, alpha, beta, gamma)

    for key in rot_vec.keys():
        torch.testing.assert_close(rot_const[key], rot_vec[key], rtol=1e-10, atol=1e-12)


def test_rotate_interpolate_matches_rotation_on_input_grid():
    """LAL-style rotate/interpolate reduces to direct rotation on source samples."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t = torch.tensor([0.0, 0.3, 0.8, 1.2, 1.9], dtype=dtype)
    alpha = 0.2 + 0.13 * t
    beta = 0.4 + 0.03 * torch.sin(t)
    gamma = -0.1 + 0.07 * t
    h22 = (1.0 + 0.2 * t) * torch.exp(0.3j * t)
    h21 = (0.3 + 0.1 * t) * torch.exp(-0.4j * t)
    modes = {
        (2, 2): h22,
        (2, 1): h21,
        (2, -1): ((-1.0) ** 2) * torch.conj(h21),
        (2, -2): ((-1.0) ** 2) * torch.conj(h22),
    }

    got = swt._rotate_interpolate_modes_jframe(modes, alpha, beta, gamma, t, t, ((2, 2), (2, 1)))
    expected = swt._rotate_modes(modes, alpha, beta, gamma)

    for key in expected:
        torch.testing.assert_close(got[key], expected[key], rtol=1e-10, atol=1e-12)


def test_rotate_interpolate_splines_pmode_phase():
    """P-frame modes should be interpolated as amp/phase, not complex samples."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype)
    t_out = torch.tensor([1.5], dtype=dtype)
    amp = torch.tensor([1.0, 1.2, 1.4, 1.6], dtype=dtype)
    phase = torch.tensor([2.8, 3.1, 3.3, 3.5], dtype=dtype)
    h22 = amp * torch.exp(1j * phase)
    modes = {(2, 2): h22, (2, -2): torch.conj(h22)}
    zero = torch.zeros_like(t)

    got = swt._rotate_interpolate_modes_jframe(modes, zero, zero, zero, t, t_out, ((2, 2),))[(2, 2)]
    expected_amp = swt._interp_series_cubic(t_out, t, amp)
    expected_phase = swt._interp_series_cubic(t_out, t, phase)
    expected = expected_amp * torch.exp(1j * expected_phase)

    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_rotate_interpolate_wigner_tensor_contraction_multimode():
    """Verify output-grid Wigner tensor contraction across multiple higher modes."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t = torch.linspace(0.0, 2.0, 50, dtype=dtype)
    t_out = torch.linspace(0.1, 1.9, 100, dtype=dtype)

    alpha = 0.3 * torch.sin(t) + 0.1 * t
    beta = 0.5 + 0.1 * torch.cos(t)
    gamma = -0.2 + 0.05 * t

    h22 = (1.0 + 0.1 * t) * torch.exp(0.5j * t)
    h21 = (0.2 + 0.05 * t) * torch.exp(-0.3j * t)
    h33 = (0.15 + 0.02 * t) * torch.exp(0.8j * t)

    modes = {
        (2, 2): h22,
        (2, 1): h21,
        (3, 3): h33,
    }

    mode_array = ((2, 2), (2, 1), (3, 3))
    got = swt._rotate_interpolate_modes_jframe(modes, alpha, beta, gamma, t, t_out, mode_array)

    assert len(got) == 5 + 7  # 2*2+1 for l=2 and 2*3+1 for l=3
    for (l, m), mode_series in got.items():
        assert mode_series.shape == (100,)
        assert torch.isfinite(mode_series).all()


def test_wigner_d_element_matches_lal():
    """Rotation helper should use LAL's XLALWignerdMatrix sign convention."""
    try:
        import lal
    except Exception as exc:
        pytest.skip(f"lal unavailable: {exc}")

    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    beta = torch.tensor(0.7, dtype=torch.float64)
    for l, m, mp in [(2, 2, 1), (2, 2, 0), (3, -1, 2), (4, 3, -2)]:
        got = swt._wigner_d_element(l, mp, m, beta)
        expected = torch.tensor(lal.WignerdMatrix(l, m, mp, float(beta)), dtype=torch.float64)
        torch.testing.assert_close(got, expected, rtol=1e-13, atol=1e-13)


def test_build_j_frame_matches_lal_projection():
    """The final-J frame e1 axis is projected x, not x cross J."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64

    def projected_basis(J, ref):
        e3 = J / torch.linalg.norm(J)
        e1 = ref - torch.dot(ref, e3) * e3
        e1 = e1 / torch.linalg.norm(e1)
        e2 = torch.cross(e3, e1, dim=0)
        e2 = e2 / torch.linalg.norm(e2)
        return e1, e2, e3

    J = torch.tensor([0.24, -0.31, 0.92], dtype=dtype)
    ex = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    got = swt._build_J_frame(J)
    expected = projected_basis(J, ex)
    for g, e in zip(got, expected, strict=False):
        torch.testing.assert_close(g, e, rtol=1e-13, atol=1e-13)

    J_near_x = torch.tensor([1.0, 1.0e-8, 2.0e-8], dtype=dtype)
    ey = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    got = swt._build_J_frame(J_near_x)
    expected = projected_basis(J_near_x, ey)
    for g, e in zip(got, expected, strict=False):
        torch.testing.assert_close(g, e, rtol=1e-13, atol=1e-13)


def test_euler_from_basis_matches_lal_formula():
    """I->J Euler angles should use LAL's active-rotation matrix entries."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    J = torch.tensor([0.24, -0.31, 0.92], dtype=torch.float64)
    e1, e2, e3 = swt._build_J_frame(J)
    alpha, beta, gamma = swt._euler_from_basis(e1, e2, e3)

    expected_alpha = torch.atan2(e3[1], e3[0])
    expected_beta = torch.acos(e3[2])
    expected_gamma = torch.atan2(e2[2], -e1[2])

    torch.testing.assert_close(alpha, expected_alpha, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(beta, expected_beta, rtol=1e-13, atol=1e-13)
    torch.testing.assert_close(gamma, expected_gamma, rtol=1e-13, atol=1e-13)


def test_final_mass_spin_prec_matches_lal():
    """Torch remnant fit should reproduce LALSimIMREOBFinalMassSpinPrec."""
    try:
        import lalsimulation as lalsim
    except Exception as exc:
        pytest.skip(f"lalsimulation unavailable: {exc}")

    from pycbc.waveform import seobnrv4phm_torch as swt

    m1 = 25.0
    m2 = 18.0
    spin1 = (0.2, -0.15, 0.05)
    spin2 = (-0.1, 0.07, 0.2)

    status, mf_lal, af_lal = lalsim.SimIMREOBFinalMassSpinPrec(m1, m2, spin1, spin2, lalsim.SEOBNRv4PHM)
    if status != 0:
        pytest.skip("LAL remnant fit returned failure")

    mf, af = swt._final_mass_spin_prec(m1, m2, spin1, spin2)
    assert np.isclose(mf, mf_lal, rtol=0, atol=1e-13)
    assert np.isclose(af, af_lal, rtol=0, atol=1e-13)


def test_nqc_peak_delta_t_v4_matches_lal_fit():
    """Peak timing offset should mirror XLALSimIMREOBGetNRSpinPeakDeltaTv4."""
    from pycbc.waveform.seobnrv4phm_nqc import peak_delta_t_v4

    dt22 = peak_delta_t_v4(2, 2, 25.0, 18.0, 0.05, 0.2)
    dt55 = peak_delta_t_v4(5, 5, 25.0, 18.0, 0.05, 0.2)

    assert np.isclose(dt22, 4.561818695562277, rtol=0.0, atol=1e-13)
    assert np.isclose(dt55 - dt22, 10.0, rtol=0.0, atol=1e-15)


def test_nqc_22_peak_amplitude_fits_match_lal():
    """The 22 amplitude targets must include LAL's eta normalization."""
    from pycbc.waveform.seobnrv4phm_nqc import peak_addot_v4, peak_adot_v4, peak_amp_v4

    eta = 0.24
    chiS = 0.1
    chiA = -0.05

    assert np.isclose(peak_amp_v4(2, 2, eta, chiS, chiA), 0.37608626846721882, rtol=0.0, atol=1e-15)
    assert peak_adot_v4(2, 2, eta, chiS, chiA) == 0.0
    assert np.isclose(peak_addot_v4(2, 2, eta, chiS, chiA), -0.00092616014379767926, rtol=0.0, atol=1e-17)


def test_coprecessing_modes_can_skip_distance_scaling_for_nqc():
    """The NQC first pass must use LAL's dimensionless P-frame modes."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=PARAMS["spin1x"],
        spin1y=PARAMS["spin1y"],
        spin1z=PARAMS["spin1z"],
        spin2x=PARAMS["spin2x"],
        spin2y=PARAMS["spin2y"],
        spin2z=PARAMS["spin2z"],
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
        mode_array=((2, 2),),
    )
    dtype = torch.float64
    n = 8
    r = torch.linspace(12.0, 10.5, n, dtype=dtype)
    pr = torch.linspace(-2.0e-3, -4.0e-3, n, dtype=dtype)
    phi = torch.linspace(0.0, 0.4, n, dtype=dtype)
    omega = torch.full((n,), 0.03 / params.M_sec, dtype=dtype)
    Lvec = torch.zeros((n, 3), dtype=dtype)
    Lvec[:, 2] = 3.8
    S1 = torch.tensor([[params.spin1x, params.spin1y, params.spin1z]], dtype=dtype).repeat(n, 1)
    S2 = torch.tensor([[params.spin2x, params.spin2y, params.spin2z]], dtype=dtype).repeat(n, 1)

    h_unscaled = swt._build_coprecessing_modes(
        phi, omega, r, pr, Lvec, S1, S2, params, params.mode_array, distance_scale=False
    )[(2, 2)]
    h_scaled = swt._build_coprecessing_modes(
        phi, omega, r, pr, Lvec, S1, S2, params, params.mode_array, distance_scale=True
    )[(2, 2)]
    scale = (params.M * swt._MRSUN_SI) / (params.distance * 1.0e6 * swt._PC_SI)

    torch.testing.assert_close(h_scaled, h_unscaled * scale, rtol=1e-12, atol=1e-30)

    params.nqc_b_map = {(2, 2): {"b1": 0.7, "b2": 0.0, "b3": 0.0, "b4": 0.0}}
    h_nqc = swt._build_coprecessing_modes(
        phi, omega, r, pr, Lvec, S1, S2, params, params.mode_array, distance_scale=False
    )[(2, 2)]
    rOmega = r * torch.abs(omega * params.M_sec)
    expected_phase = 0.7 * pr / rOmega
    torch.testing.assert_close(h_nqc, h_unscaled * torch.exp(1j * expected_phase), rtol=1e-12, atol=1e-12)


def test_coprecessing_mode_uses_lal_newtonian_phase():
    """With fixed dynamics, the P-frame mode phase follows Y_{l-eps,-m}."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    dtype = torch.float64
    phi = torch.tensor([0.0, 0.37], dtype=dtype)
    r = torch.full((2,), 12.0, dtype=dtype)
    pr = torch.full((2,), -2.0e-3, dtype=dtype)
    omega = torch.full((2,), 0.03 / params.M_sec, dtype=dtype)
    Lvec = torch.zeros((2, 3), dtype=dtype)
    Lvec[:, 2] = 3.8
    S1 = torch.tensor([[0.0, 0.0, params.spin1z]], dtype=dtype).repeat(2, 1)
    S2 = torch.tensor([[0.0, 0.0, params.spin2z]], dtype=dtype).repeat(2, 1)

    h22 = swt._build_coprecessing_modes(
        phi, omega, r, pr, Lvec, S1, S2, params, params.mode_array, distance_scale=False
    )[(2, 2)]
    torch.testing.assert_close(h22[1] / h22[0], torch.exp(-2j * phi[1]), rtol=1e-11, atol=1e-12)


def test_high_mode_delta_vh9_terms_match_lal_nesting():
    """The 33/44/55 d9 terms contain three powers of H*Omega."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_waveform_coeffs as coeffs

    params = dyn.EOBParams(
        mass1=25.0,
        mass2=18.0,
        spin1x=0.2,
        spin1y=-0.15,
        spin1z=0.05,
        spin2x=-0.1,
        spin2y=0.07,
        spin2z=0.2,
        distance=300.0,
        inclination=0.6,
        f_lower=30.0,
        f_ref=50.0,
    )
    v = torch.tensor([0.22, 0.31, 0.39], dtype=torch.float64)
    H = torch.tensor([0.991, 0.976, 0.951], dtype=torch.float64)
    eta = params.eta
    dM = (params.mass1 - params.mass2) / params.M
    chiS = 0.5 * (params.spin1z + params.spin2z)
    chiA = 0.5 * (params.spin1z - params.spin2z)
    a_delta = chiA * dM + chiS * (1.0 - 2.0 * eta)
    omega = v**3
    vh3 = H * omega

    expected = {
        (3, 3): vh3
        * (
            13.0 / 10.0
            + vh3
            * (
                (-81.0 * a_delta) / 20.0 + (39.0 * np.pi) / 7.0
                + vh3
                * (-227827.0 / 3000.0 + (78.0 * np.pi * np.pi) / 7.0)
            )
        )
        + omega * v * v * (-80897.0 * eta / 2430.0),
        (4, 4): vh3
        * (
            (112.0 + 219.0 * eta) / (-120.0 * (-1.0 + 3.0 * eta))
            + vh3
            * (
                (-464.0 * a_delta) / 75.0 + (25136.0 * np.pi) / 3465.0
                + vh3
                * (-55144.0 / 375.0 + 201088.0 * np.pi * np.pi / 10395.0)
            )
        ),
        (5, 5): vh3
        * (
            (96875.0 + 857528.0 * eta) / (131250.0 * (1.0 - 2.0 * eta))
            + vh3
            * (
                3865.0 * np.pi / 429.0
                + vh3
                * ((-7686949127.0 + 954500400.0 * np.pi * np.pi) / 31783752.0)
            )
        ),
    }

    for (ell, emm), expected_delta in expected.items():
        _, _, delta = dyn._factorized_rho_aux_delta(
            ell,
            emm,
            v,
            params,
            waveform=True,
            H=H,
        )
        duplicate_delta = coeffs.delta_lm_full(
            ell,
            emm,
            eta,
            params.spin1z,
            params.spin2z,
            params.mass1,
            params.mass2,
            v,
            H,
            waveform=True,
        )
        torch.testing.assert_close(delta, expected_delta, rtol=2e-15, atol=2e-15)
        torch.testing.assert_close(duplicate_delta, expected_delta, rtol=2e-15, atol=2e-15)


def test_dynamic_spin_projection_tplspin_matches_lal_branches():
    """Main mode generation and 21/55 calibration use different LAL tplspin inputs."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 1),),
    )
    dtype = torch.float64
    Lvec = torch.tensor([[0.0, 0.0, 2.0], [0.0, 3.0, 4.0]], dtype=dtype)
    S1 = torch.tensor([[0.1, 0.2, 0.3], [0.2, -0.1, 0.4]], dtype=dtype)
    S2 = torch.tensor([[-0.2, 0.1, -0.1], [0.1, 0.3, -0.2]], dtype=dtype)

    chi1dotZ, chi2dotZ, chiS, chiA, weighted_tplspin = swt._dynamic_spin_projection_combos(
        Lvec,
        S1,
        S2,
        params,
        weighted_tplspin=True,
    )
    _, _, _, _, unweighted_tplspin = swt._dynamic_spin_projection_combos(
        Lvec,
        S1,
        S2,
        params,
        weighted_tplspin=False,
    )

    Lhat = Lvec / torch.linalg.norm(Lvec, dim=1).unsqueeze(-1)
    torch.testing.assert_close(chi1dotZ, torch.sum(S1 * Lhat, dim=1), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(chi2dotZ, torch.sum(S2 * Lhat, dim=1), rtol=1e-12, atol=1e-12)

    total_mass = params.mass1 + params.mass2
    dM = (params.mass1 - params.mass2) / total_mass
    s1dotZ = (params.mass1 / total_mass) ** 2 * chi1dotZ
    s2dotZ = (params.mass2 / total_mass) ** 2 * chi2dotZ
    expected_weighted = (1.0 - 2.0 * params.eta) * 0.5 * (s1dotZ + s2dotZ) + dM * 0.5 * (s1dotZ - s2dotZ)
    expected_unweighted = (1.0 - 2.0 * params.eta) * chiS + dM * chiA

    torch.testing.assert_close(weighted_tplspin, expected_weighted, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(unweighted_tplspin, expected_unweighted, rtol=1e-12, atol=1e-12)


def test_higher_mode_residual_calibration_matches_lal_formula(monkeypatch):
    """21/55 residual calibration should use the uncalibrated residual at attachment."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=35.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 1), (5, 5)),
    )
    params.cal21 = 12.0
    params.cal55 = -3.0
    params.nqc_a_map = {(2, 1): {"a1": 1.0}}
    params.nqc_b_map = {(2, 1): {"b1": 1.0}}

    dtype = torch.float64
    t_vec = torch.tensor([0.0, 1.0, 2.0], dtype=dtype)
    phi = torch.zeros(3, dtype=dtype)
    omega = torch.full((3,), 8.0 / params.M_sec, dtype=dtype)
    r = torch.full((3,), 10.0, dtype=dtype)
    pr = torch.zeros(3, dtype=dtype)
    Lvec = torch.tensor([[0.0, 0.0, 4.0]], dtype=dtype).repeat(3, 1)
    S1 = torch.zeros((3, 3), dtype=dtype)
    S2 = torch.zeros((3, 3), dtype=dtype)

    def fake_modes(*args, **kwargs):
        assert kwargs["weighted_tplspin"] is False
        assert args[7].cal21 == 0.0
        assert args[7].cal55 == 0.0
        return {
            (2, 1): torch.full((3,), 10.0 + 0.0j, dtype=torch.complex128),
            (5, 5): torch.full((3,), 10.0 + 0.0j, dtype=torch.complex128),
        }

    def fake_residual(mode_l, mode_m, *args, **kwargs):
        assert kwargs["cal21"] == 0.0
        assert kwargs["cal55"] == 0.0
        return torch.full((3,), 2.0 + 0.0j, dtype=torch.complex128), torch.zeros(3, dtype=dtype)

    monkeypatch.setattr(swt, "_build_coprecessing_modes", fake_modes)
    monkeypatch.setattr(swt, "_factorized_residual_power", fake_residual)
    monkeypatch.setattr(swt._dyn, "_eob_potentials", lambda *args, **kwargs: {"H": torch.ones(3, dtype=dtype)})
    monkeypatch.setattr(swt, "peak_delta_t_v4", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(swt, "peak_amp_v4", lambda l, m, eta, chiS, chiA: 20.0 if (l, m) == (2, 1) else 30.0)

    cal = swt._higher_mode_residual_calibration(
        t_vec,
        phi,
        omega,
        r,
        pr,
        Lvec,
        S1,
        S2,
        params,
        t_peak_omega_M=1.0,
        chi1L_peak=0.2,
        chi2L_peak=-0.1,
    )

    assert cal["cal21"] == pytest.approx((20.0 / 5.0 - 2.0) / (8.0 ** (7.0 / 3.0)))
    assert cal["cal55"] == pytest.approx((30.0 / 5.0 - 2.0) / (8.0 ** (5.0 / 3.0)))
    assert params.cal21 == 12.0
    assert params.cal55 == -3.0
    assert params.nqc_a_map == {(2, 1): {"a1": 1.0}}
    assert params.nqc_b_map == {(2, 1): {"b1": 1.0}}


def test_populate_nqc_coeffs_loops_requested_positive_modes(monkeypatch):
    """LAL computes NQC coefficients for requested positive modes, not derived -m modes."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2), (3, 3)),
    )
    calls = []

    def fake_solve(mode_l, mode_m, *args, **kwargs):
        calls.append((mode_l, mode_m))
        return {
            "a1": float(mode_l),
            "a2": float(mode_m),
            "a3": 0.0,
            "a4": 0.0,
            "a5": 0.0,
            "b1": 1.0,
            "b2": 0.0,
            "b3": 0.0,
            "b4": 0.0,
        }

    monkeypatch.setattr(swt, "_solve_nqc_coeffs_lal_series", fake_solve)
    t = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    modes = {
        (2, 2): torch.ones(4, dtype=torch.complex128),
        (2, -2): torch.ones(4, dtype=torch.complex128),
        (3, 3): torch.ones(4, dtype=torch.complex128),
        (3, -3): torch.ones(4, dtype=torch.complex128),
    }

    swt._populate_nqc_coeffs_lal_v4(
        params,
        params.mode_array,
        modes,
        t,
        torch.ones_like(t),
        torch.zeros_like(t),
        torch.ones_like(t),
        t_peak_omega_M=0.5,
        chi1L_peak=0.2,
        chi2L_peak=-0.1,
    )

    assert calls == [(2, 2), (3, 3)]
    assert set(params.nqc_a_map) == {(2, 2), (3, 3)}
    assert set(params.nqc_b_map) == {(2, 2), (3, 3)}
    assert params.nqc_a == params.nqc_a_map[(2, 2)]
    assert params.nqc_b == params.nqc_b_map[(2, 2)]


def test_polarization_projection_uses_lal_observer_azimuth():
    """SEOBNRv4PHM uses phi_sYlm = pi/2 - phiRef and LAL API polarity."""
    import math

    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    device = torch.device("cpu")
    h22 = torch.tensor([1.0 + 0.0j, 0.5 - 0.25j], dtype=torch.complex128)
    inc = 0.7
    phi_ref = 0.3
    hp, hc = swt._polarizations_from_modes(
        {(2, 2): h22},
        inc,
        phi_ref,
        device=device,
        complex_dtype=torch.complex128,
    )
    y_lal = swt._sYlm_torch(
        -2,
        2,
        2,
        torch.tensor(inc, dtype=torch.float64),
        torch.tensor(math.pi / 2.0 - phi_ref, dtype=torch.float64),
    ).to(torch.complex128)
    hpc = h22 * y_lal
    torch.testing.assert_close(hp, -hpc.real, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(hc, hpc.imag, rtol=1e-12, atol=1e-12)


def test_ringdown_attachment_snaps_to_lal_sample():
    """RD attachment should use the nearest HiS sample, not exact interpolation."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    dtype = torch.float64
    dt = 0.5 * params.M_sec
    t_m = torch.arange(80, dtype=dtype) * 0.5
    h22 = (1.0 + 1.0e-3 * t_m) * torch.exp(0.2j * t_m)
    t_attach_m = 10.26
    idx_attach = swt._nearest_index_increasing(t_m, t_attach_m)

    out = swt._attach_ringdown_modes(
        {(2, 2): h22},
        params,
        dt,
        device=torch.device("cpu"),
        dtype=dtype,
        finspin=0.5,
        finmass_frac=0.95,
        t_attach_M=t_attach_m,
        chi1L_attach=0.2,
        chi2L_attach=-0.1,
    )

    torch.testing.assert_close(out[(2, 2)][idx_attach], h22[idx_attach], rtol=1e-10, atol=1e-12)
    assert len(out[(2, 2)]) > idx_attach


def test_ringdown_attachment_respects_his_start_time():
    """HiS ringdown attachment times are absolute PHM times, not zero-based."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    dtype = torch.float64
    dt_m = 0.5
    dt = dt_m * params.M_sec
    t_start_m = 100.0
    t_m = t_start_m + torch.arange(80, dtype=dtype) * dt_m
    h22 = (1.0 + 1.0e-3 * t_m) * torch.exp(0.2j * t_m)
    t_attach_m = 110.26
    idx_attach = swt._nearest_index_increasing(t_m, t_attach_m)

    out = swt._attach_ringdown_modes(
        {(2, 2): h22},
        params,
        dt,
        device=torch.device("cpu"),
        dtype=dtype,
        finspin=0.5,
        finmass_frac=0.95,
        t_attach_M=t_attach_m,
        chi1L_attach=0.2,
        chi2L_attach=-0.1,
        t_start_M=t_start_m,
    )

    torch.testing.assert_close(out[(2, 2)][idx_attach], h22[idx_attach], rtol=1e-10, atol=1e-12)


def test_ringdown_attachment_keeps_lal_patch_length(monkeypatch):
    """LAL keeps the full HiS buffer and appends a longer RD patch."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.2,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.1,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    monkeypatch.setattr(swt, "_qnm_re_im", lambda *args: (1.0, 10.0))

    dtype = torch.float64
    dt_m = 1.0
    dt = dt_m * params.M_sec
    h22 = (1.0 + 1.0e-3 * torch.arange(20, dtype=dtype)) * torch.exp(
        0.2j * torch.arange(20, dtype=dtype)
    )
    idx_attach = 5
    expected_rd_len = max(1, int(swt._EOB_RD_EFOLDS / (10.0 * dt_m)))
    expected_patch_len = max(
        expected_rd_len,
        math.ceil(swt._EOB_RD_PATCH_EFOLDS / (10.0 * dt_m)),
    )

    out = swt._attach_ringdown_modes(
        {(2, 2): h22},
        params,
        dt,
        device=torch.device("cpu"),
        dtype=dtype,
        finspin=0.5,
        finmass_frac=1.0,
        t_attach_M=float(idx_attach),
        chi1L_attach=0.2,
        chi2L_attach=-0.1,
    )[(2, 2)]

    assert len(out) == len(h22) + expected_patch_len
    torch.testing.assert_close(out[:idx_attach], h22[:idx_attach], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        out[idx_attach + expected_rd_len : len(h22)],
        h22[idx_attach + expected_rd_len :],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(out[len(h22) :], torch.zeros_like(out[len(h22) :]), rtol=0.0, atol=0.0)


def test_ringdown_uses_peak_l_frame_spin_projections(monkeypatch):
    """RD coefficient fits use L-frame spins at peak Omega, matching LAL."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.9,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=0.8,
        distance=300.0,
        inclination=0.0,
        f_lower=30.0,
        f_ref=30.0,
        mode_array=((2, 2),),
    )
    seen_chi = []

    def capture(value):
        def _fn(l, m, eta, chi, *, device, dtype):
            seen_chi.append(float(chi))
            return torch.tensor(value, device=device, dtype=dtype)

        return _fn

    monkeypatch.setattr(swt, "_rd_amp_coeff1", capture(0.2))
    monkeypatch.setattr(swt, "_rd_amp_coeff2", capture(0.5))
    monkeypatch.setattr(swt, "_rd_phase_coeff1", capture(0.1))
    monkeypatch.setattr(swt, "_rd_phase_coeff2", capture(1.0))

    dtype = torch.float64
    dt = 0.5 * params.M_sec
    t_m = torch.arange(80, dtype=dtype) * 0.5
    h22 = torch.exp(0.2j * t_m)
    chi1_l = 0.25
    chi2_l = -0.35

    swt._attach_ringdown_modes(
        {(2, 2): h22},
        params,
        dt,
        device=torch.device("cpu"),
        dtype=dtype,
        finspin=0.5,
        finmass_frac=0.95,
        t_attach_M=10.0,
        chi1L_attach=chi1_l,
        chi2L_attach=chi2_l,
    )

    d_m = (params.mass1 - params.mass2) / (params.mass1 + params.mass2)
    expected_chi = 0.5 * (chi1_l + chi2_l) + 0.5 * (chi1_l - chi2_l) * d_m / (1.0 - 2.0 * params.eta)
    assert seen_chi
    assert all(abs(chi - expected_chi) < 1e-14 for chi in seen_chi)


def test_non_keplerian_vphi_matches_fd():
    """non_keplerian_vphi should agree with finite-diff dH/dpphi (pr=0)."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.1,
        spin1y=-0.05,
        spin1z=0.2,
        spin2x=-0.02,
        spin2y=0.04,
        spin2z=0.15,
        distance=400.0,
        inclination=0.4,
        f_lower=20.0,
        f_ref=30.0,
    )
    device, dtype, _ = seobnrv4phm_torch._target_device_dtypes()
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)

    r = y0[0]
    Lvec = y0[3:6]
    zero_pr = y0[1] * 0.0

    r_vec, p_vec, _, _, _ = dyn._reduced_to_cartesian(
        r, zero_pr, y0[2], Lvec, y0[6:9], y0[9:12], params
    )
    omega_fd = torch.abs(
        dyn.dphi_dt_fd(
            r,
            zero_pr,
            y0[2],
            Lvec,
            y0[6:9],
            y0[9:12],
            params,
            p_vec=p_vec,
            r_vec=r_vec,
            step=1e-4,
        )
    )
    coeff = 1.0 / (omega_fd * omega_fd * (r ** 3))
    vphi_fd = torch.abs(r * omega_fd * coeff.pow(1.0 / 3.0))

    rel_err = torch.abs(vphi_fd - dyn.non_keplerian_vphi(r, omega_fd, y0[2], Lvec, y0[6:9], y0[9:12], params)) / torch.clamp(vphi_fd, min=1e-12)
    assert rel_err < 1e-4


def test_non_keplerian_vphi_reuses_lal_rdot():
    """Supplying LAL's rdot derivative should not change vPhi."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    device, dtype, _ = seobnrv4phm_torch._target_device_dtypes()
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    dyn._refresh_hcoeffs(params, y0[3:6], y0[6:9], y0[9:12])
    y_cart = dyn.reduced_state_to_cartesian_state(y0, params)
    r_vec = y_cart[0:3]
    p_vec = y_cart[3:6]
    S1 = y_cart[6:9] / dyn._lal_spin_scale(params.mass1, params.M)
    S2 = y_cart[9:12] / dyn._lal_spin_scale(params.mass2, params.M)
    r = torch.linalg.norm(r_vec)
    L_vec = torch.linalg.cross(r_vec, p_vec)
    omega = torch.tensor(0.027852467644034994, device=device, dtype=dtype)
    rdot_vec = dyn._calcomega_rdot_lal_fd(r_vec, p_vec, S1, S2, params)

    vphi_direct = dyn.non_keplerian_vphi(
        r, omega, y0[2], L_vec, S1, S2, params, r_vec=r_vec, p_vec=p_vec
    )
    vphi_reused = dyn.non_keplerian_vphi(
        r,
        omega,
        y0[2],
        L_vec,
        S1,
        S2,
        params,
        r_vec=r_vec,
        p_vec=p_vec,
        rdot_vec=rdot_vec,
    )
    torch.testing.assert_close(vphi_reused, vphi_direct, rtol=0.0, atol=0.0)


def _aligned_v4_lal_checkpoint():
    """Return the first low-dynamics sample from an aligned LAL v4 run."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    dtype = torch.float64
    pphi = torch.tensor(4.096256759858304, dtype=dtype)
    return dict(
        params=dyn.EOBParams(
            mass1=50.0,
            mass2=40.0,
            spin1x=0.0,
            spin1y=0.0,
            spin1z=0.3,
            spin2x=0.0,
            spin2y=0.0,
            spin2z=-0.2,
            distance=400.0,
            inclination=0.0,
            f_lower=15.0,
            f_ref=15.0,
            mode_array=((2, 2),),
        ),
        phi=torch.tensor(0.0, dtype=dtype),
        r=torch.tensor(13.213042632672478, dtype=dtype),
        pr=torch.tensor(-0.0016211742277193744, dtype=dtype),
        L_vec=torch.stack((pphi * 0.0, pphi * 0.0, pphi)),
        S1=torch.tensor([0.0, 0.0, 0.3], dtype=dtype),
        S2=torch.tensor([0.0, 0.0, -0.2], dtype=dtype),
    )


def test_reduced_aligned_hamiltonian_retains_angular_momentum():
    """An exactly aligned reduced state must build a non-zero orbital basis."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    checkpoint = _aligned_v4_lal_checkpoint()
    params = checkpoint.pop("params")
    potentials = dyn._eob_potentials(
        **checkpoint,
        params=params,
        compute_grad_p=False,
        compute_base_grad=False,
        fd_pphi=False,
    )

    one = torch.ones_like(checkpoint["r"])
    torch.testing.assert_close(torch.linalg.norm(potentials["n_hat"]), one)
    torch.testing.assert_close(torch.linalg.norm(potentials["xi_vec"]), one)
    torch.testing.assert_close(potentials["pf"], checkpoint["L_vec"][2])
    torch.testing.assert_close(
        potentials["H"],
        checkpoint["r"].new_tensor(0.9912994720680695),
        rtol=2.0e-12,
        atol=0.0,
    )


def test_aligned_non_keplerian_omega_one_element_batch():
    """The aligned GSL-derivative mirror must preserve a size-one batch."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    checkpoint = _aligned_v4_lal_checkpoint()
    omega = dyn._aligned_non_keplerian_omega(
        checkpoint["r"].unsqueeze(0),
        checkpoint["L_vec"].unsqueeze(0),
        checkpoint["S1"].unsqueeze(0),
        checkpoint["S2"].unsqueeze(0),
        checkpoint["params"],
    )

    assert omega.shape == (1,)
    torch.testing.assert_close(
        omega[0],
        checkpoint["r"].new_tensor(0.020889746348388673),
        rtol=5.0e-10,
        atol=0.0,
    )


def test_aligned_coprecessing_mode_matches_lal_checkpoint():
    """The native pre-NQC h22 mode must match an aligned LAL checkpoint."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    checkpoint = _aligned_v4_lal_checkpoint()
    params = checkpoint["params"]
    batch = {
        name: checkpoint[name].unsqueeze(0)
        for name in ("phi", "r", "pr", "L_vec", "S1", "S2")
    }
    omega = checkpoint["r"].new_tensor(
        [0.02088974633818432 / params.M_sec]
    )

    h22 = swt._build_coprecessing_modes(
        batch["phi"],
        omega,
        batch["r"],
        batch["pr"],
        batch["L_vec"],
        batch["S1"],
        batch["S2"],
        params,
        ((2, 2),),
        distance_scale=False,
        weighted_tplspin=False,
    )[(2, 2)]

    # LALSuite v7.26.1 SimIMRSpinAlignedEOBModes, with its NQC correction
    # analytically removed at the first low-dynamics sample.
    torch.testing.assert_close(
        h22[0],
        torch.tensor(
            -0.10864441661041561 + 0.024458470045193442j,
            dtype=torch.complex128,
        ),
        rtol=5.0e-6,
        atol=0.0,
    )


def test_coprecessing_mode_uses_lal_calcomega_checkpoint():
    """The first-pass mode must use CalcOmega's non-Keplerian coefficient."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    # LALSuite v7.26.1 HiS checkpoint at 144.02 M.  The final two entries
    # are phiDMod and phiMod in LAL's 14-component Cartesian state.
    cart_state = torch.tensor(
        [
            2.4199303397609335,
            1.6711614236958754,
            -0.065210825480322684,
            -0.69446926858996627,
            0.68481375117182297,
            0.012550200016420781,
            -0.008578474902383915,
            -0.009200114156516219,
            0.043318444957915668,
            -0.0012957142107288893,
            0.0037612009862908126,
            -0.012681265409222112,
            33.367542049771906,
            4.935504129291122,
        ],
        dtype=torch.float64,
    )
    state = dyn.cartesian_state_to_reduced_state(cart_state, params).unsqueeze(0)
    omega_dimless = torch.tensor([0.1490381468581127], dtype=torch.float64)
    h22 = swt._build_coprecessing_modes(
        state[:, 2],
        omega_dimless / params.M_sec,
        state[:, 0],
        state[:, 1],
        state[:, 3:6],
        state[:, 6:9],
        state[:, 9:12],
        params,
        ((2, 2),),
        distance_scale=False,
        H=torch.tensor([0.97278008603797828], dtype=torch.float64),
    )[(2, 2)]

    torch.testing.assert_close(
        torch.abs(h22[0]),
        torch.tensor(0.42015328916193062, dtype=torch.float64),
        rtol=5.0e-8,
        atol=0.0,
    )


def test_calcomega_exact_path_batches_like_scalar():
    """CalcOmega's finite-difference port must preserve each batch member."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    state = dyn.initial_conditions(params, device=torch.device("cpu"), dtype=torch.float64)
    cart = dyn.reduced_state_to_cartesian_state(state, params)
    r_vec = torch.stack((cart[0:3], cart[0:3] * torch.tensor([1.00001, 0.99999, 1.0])))
    p_vec = torch.stack((cart[3:6], cart[3:6] * torch.tensor([0.99999, 1.00001, 1.0])))
    S1 = (cart[6:9] / dyn._lal_spin_scale(params.mass1, params.M)).repeat(2, 1)
    S2 = (cart[9:12] / dyn._lal_spin_scale(params.mass2, params.M)).repeat(2, 1)

    batched = dyn._calcomega_lal_polar_derivative(r_vec, p_vec, S1, S2, params)
    scalar = torch.stack(
        [
            dyn._calcomega_lal_polar_derivative(
                r_vec[i], p_vec[i], S1[i], S2[i], params
            )
            for i in range(2)
        ]
    )
    torch.testing.assert_close(batched, scalar, rtol=1e-14, atol=1e-16)


def test_calcomega_polar_derivative_analytical_parity_and_zero_potentials_calls(monkeypatch):
    """Analytical calcomega polar derivative matches FD and eliminates _eob_potentials calls."""
    import torch
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    state = dyn.initial_conditions(params, device=torch.device("cpu"), dtype=torch.float64)
    cart = dyn.reduced_state_to_cartesian_state(state, params)
    r_vec = cart[0:3]
    p_vec = cart[3:6]
    S1 = cart[6:9] / dyn._lal_spin_scale(params.mass1, params.M)
    S2 = cart[9:12] / dyn._lal_spin_scale(params.mass2, params.M)
    rdot_vec = dyn._calcomega_rdot_lal_fd(r_vec, p_vec, S1, S2, params)

    # 1. Verify 0 calls to _eob_potentials in default analytical mode when rdot_vec is provided
    call_count = [0]
    real_eob_potentials = dyn._eob_potentials

    def counted_eob_potentials(*args, **kwargs):
        call_count[0] += 1
        return real_eob_potentials(*args, **kwargs)

    monkeypatch.setattr(dyn, "_eob_potentials", counted_eob_potentials)
    val_analytic = dyn._calcomega_lal_polar_derivative(
        r_vec, p_vec, S1, S2, params, rdot_vec=rdot_vec
    )
    assert call_count[0] == 0, f"Expected 0 _eob_potentials calls, got {call_count[0]}"

    # 2. Verify FD mode calls _eob_potentials and matches analytical to high precision
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_CALCOMEGA", "1")
    val_fd = dyn._calcomega_lal_polar_derivative(
        r_vec, p_vec, S1, S2, params, rdot_vec=rdot_vec
    )
    assert call_count[0] > 0, "Expected _eob_potentials calls in FD mode"
    torch.testing.assert_close(val_analytic, val_fd, rtol=1e-6, atol=1e-8)


def test_calcomega_rdot_refreshes_hcoeffs_per_momentum_perturbation(monkeypatch):
    """LAL's CalcOmega RvecDerivative recomputes hcoeffs inside dH/dP."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    device, dtype, _ = seobnrv4phm_torch._target_device_dtypes()
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    dyn._refresh_hcoeffs(params, y0[3:6], y0[6:9], y0[9:12])
    y_cart = dyn.reduced_state_to_cartesian_state(y0, params)
    r_vec = y_cart[0:3]
    p_vec = y_cart[3:6]
    S1 = y_cart[6:9] / dyn._lal_spin_scale(params.mass1, params.M)
    S2 = y_cart[9:12] / dyn._lal_spin_scale(params.mass2, params.M)
    seen = []
    real_eob_potentials = dyn._eob_potentials

    def wrapped_eob_potentials(*args, **kwargs):
        seen.append((args[3].detach().clone(), kwargs.get("hcoeffs_override")))
        return real_eob_potentials(*args, **kwargs)

    monkeypatch.setattr(dyn, "_eob_potentials", wrapped_eob_potentials)
    dyn._calcomega_rdot_lal_fd(r_vec, p_vec, S1, S2, params)

    assert len(seen) >= 13
    assert all(hcoeffs_override is None for _, hcoeffs_override in seen)
    assert any(not torch.equal(L_vec, seen[0][0]) for L_vec, _ in seen[1:])


def test_rhs_flux_passes_rdot_vec(monkeypatch):
    """The RHS passes dxdt into _factorized_flux as rdot_vec to optimize flux computation."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    device, dtype, _ = seobnrv4phm_torch._target_device_dtypes()
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    y_cart = dyn.reduced_state_to_cartesian_state(y0, params)
    calls = []

    def fake_flux(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros_like(args[7])

    monkeypatch.setattr(dyn, "_factorized_flux", fake_flux)
    dyn.rhs_cartesian_full(torch.tensor(0.0, device=device, dtype=dtype), y_cart, params)

    assert calls
    assert calls[0].get("rdot_vec") is not None
    assert calls[0].get("velocity_vec") is not None


def test_dynamics_flux_spin_projection_uses_lal_lhat(monkeypatch):
    """v4P flux coefficients use Lhat, even when an LN velocity is available."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.0,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=0.0,
        distance=300.0,
        inclination=0.0,
        f_lower=90.0,
        f_ref=90.0,
    )
    dtype = torch.float64
    r_vec = torch.tensor([1.0, 0.0, 0.0], dtype=dtype)
    p_vec = torch.tensor([0.0, 1.0, 0.0], dtype=dtype)
    velocity_vec = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    L_vec = torch.linalg.cross(r_vec, p_vec)
    S1 = torch.tensor([0.0, 0.7, 0.2], dtype=dtype)
    S2 = torch.tensor([0.0, -0.3, -0.4], dtype=dtype)
    captured = []

    def fake_rho_aux(_l, _m, v, _params, chi1z=None, chi2z=None, **_kwargs):
        captured.append((float(chi1z), float(chi2z)))
        return torch.ones_like(v), torch.zeros_like(v)

    monkeypatch.setattr(dyn, "_rho_aux_flux", fake_rho_aux)
    dyn._factorized_flux(
        torch.tensor(10.0, dtype=dtype),
        torch.tensor(-1.0e-3, dtype=dtype),
        torch.tensor(0.0, dtype=dtype),
        L_vec,
        S1,
        S2,
        params,
        torch.tensor(0.03, dtype=dtype),
        torch.tensor(0.99, dtype=dtype),
        r_vec=r_vec,
        p_vec=p_vec,
        velocity_vec=velocity_vec,
    )

    assert captured
    assert captured[0][0] == pytest.approx(0.2, rel=0.0, abs=1.0e-15)
    assert captured[0][1] == pytest.approx(-0.4, rel=0.0, abs=1.0e-15)


def test_factorized_flux_skips_calcomega_rdot_when_rdot_vec_provided(monkeypatch):
    """When rdot_vec is provided, _factorized_flux avoids calling _calcomega_rdot_lal_fd."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
    )
    dtype = torch.float64
    r_vec = torch.tensor([10.0, 0.0, 0.0], dtype=dtype)
    p_vec = torch.tensor([0.0, 0.35, 0.0], dtype=dtype)
    rdot_vec = torch.tensor([-0.003, 0.30, 0.0], dtype=dtype)
    L_vec = torch.linalg.cross(r_vec, p_vec)
    S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], dtype=dtype)
    S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], dtype=dtype)

    def unexpected_fd(*_args, **_kwargs):
        raise AssertionError("_calcomega_rdot_lal_fd should not be called when rdot_vec is provided")

    monkeypatch.setattr(dyn, "_calcomega_rdot_lal_fd", unexpected_fd)

    flux = dyn._factorized_flux(
        torch.tensor(10.0, dtype=dtype),
        torch.tensor(-1.0e-3, dtype=dtype),
        torch.tensor(0.0, dtype=dtype),
        L_vec,
        S1,
        S2,
        params,
        torch.tensor(0.03, dtype=dtype),
        torch.tensor(0.99, dtype=dtype),
        r_vec=r_vec,
        p_vec=p_vec,
        rdot_vec=rdot_vec,
    )
    assert torch.isfinite(flux)
    assert flux > 0


def test_rhs_cartesian_projected_passes_rdot_vec(monkeypatch):
    """rhs_cartesian_projected passes dxdt into _factorized_flux as rdot_vec."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    device, dtype, _ = seobnrv4phm_torch._target_device_dtypes()
    y0 = dyn.initial_conditions(params, device=device, dtype=dtype)
    calls = []

    def fake_flux(*args, **kwargs):
        calls.append(kwargs)
        return torch.zeros_like(args[7])

    monkeypatch.setattr(dyn, "_factorized_flux", fake_flux)
    dyn.rhs_cartesian_projected(torch.tensor(0.0, device=device, dtype=dtype), y0, params)

    assert calls
    assert calls[0].get("rdot_vec") is not None
    assert calls[0].get("velocity_vec") is not None


def test_ic_radial_momentum_uses_lal_cartesian_flux_probe(monkeypatch):
    """LAL's STEP 4 IC flux probe uses rotated Cartesian dynamics and FD dH/dP."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    device = torch.device("cpu")
    dtype = torch.float64
    r = 10.9
    py = 0.35
    pz = -7.0e-5
    omega = torch.tensor(0.0278, device=device, dtype=dtype)
    S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], device=device, dtype=dtype)
    S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)
    L_vec = torch.tensor([0.0, -r * pz, r * py], device=device, dtype=dtype)
    potential_calls = []
    flux_calls = []

    def fake_second_derivative(_params, idx1, idx2, *_args, **_kwargs):
        assert idx1 == 0
        return torch.tensor(2.0 if idx2 == 0 else 3.0, device=device, dtype=dtype)

    def fake_spherical_derivatives(*_args, **_kwargs):
        raise AssertionError("STEP 4 dH/dpphi should come from Cartesian dH/dp_y / r")

    def fake_cartesian_potential(*args, **kwargs):
        potential_calls.append(kwargs)
        dH_dpvec = torch.tensor([0.2, 0.4 * r, 0.0], device=device, dtype=dtype)
        return {"H": torch.tensor(1.0, device=device, dtype=dtype), "deltaT": omega, "D": omega, "dH_dpvec": dH_dpvec}, L_vec

    def fake_flux(*args, **kwargs):
        flux_calls.append(kwargs)
        return torch.tensor(0.01, device=device, dtype=dtype)

    monkeypatch.setattr(dyn, "_ic_spherical_second_derivative", fake_second_derivative)
    monkeypatch.setattr(dyn, "_ic_spherical_derivatives", fake_spherical_derivatives)
    monkeypatch.setattr(dyn, "_ic_cartesian_potential", fake_cartesian_potential)
    monkeypatch.setattr(dyn, "_factorized_flux", fake_flux)

    pr_star = dyn._precessing_ic_radial_momentum(
        params,
        r,
        py,
        pz,
        L_vec,
        S1,
        S2,
        omega,
        final_tortoise=1,
        device=device,
        dtype=dtype,
    )

    assert torch.isfinite(pr_star)
    assert potential_calls[-1]["compute_grad_p"] is False
    assert potential_calls[-1]["fd_dpvec"] is True
    assert flux_calls
    torch.testing.assert_close(flux_calls[0]["r_vec"], torch.tensor([r, 0.0, 0.0], device=device, dtype=dtype))
    torch.testing.assert_close(flux_calls[0]["p_vec"], torch.tensor([0.0, (py * py + pz * pz) ** 0.5, 0.0], device=device, dtype=dtype))


@pytest.mark.parametrize("lm", [(2, 2), (3, 3), (4, 4), (5, 5)])
def test_tail_factor_complex_matches_scipy(lm):
    """Tail factor T_lm should track the LAL GSL branch (log-gamma)."""
    import torch
    from scipy import special

    from pycbc.waveform import seobnrv4phm_torch as swt

    l, m = lm
    device = torch.device("cpu")
    dtype = torch.float64
    omega = torch.tensor([0.0125, 0.045, 0.08], device=device, dtype=dtype)
    H = torch.tensor([1.05, 0.92, 0.88], device=device, dtype=dtype)

    tlm = swt._tail_factor_complex(l, m, omega, H).to(torch.complex128)

    k = m * omega.detach().cpu().double().numpy()
    hathatk = H.detach().cpu().double().numpy() * k
    ln_gamma = special.loggamma(l + 1 - 2j * hathatk)
    ln_gamma_l = special.gammaln(l + 1)
    ref = np.exp(
        ln_gamma
        - ln_gamma_l
        + np.pi * hathatk
        + 2j * hathatk * np.log(4.0 * np.maximum(np.abs(k), 1e-30) / np.sqrt(np.e))
    )
    ref_t = torch.as_tensor(ref, device=device, dtype=torch.complex128)
    torch.testing.assert_close(tlm, ref_t, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("device_name", ("cpu", "mps", "cuda"))
def test_lanczos_coefficients_are_cached_by_device_and_dtype(device_name):
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is not available")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    dtype = torch.float32 if device_name == "mps" else torch.float64
    swt._clear_device_constant_cache()
    try:
        coefficients = swt._lanczos_coefficients(
            torch.device(device_name),
            dtype,
        )
        assert coefficients is swt._lanczos_coefficients(
            torch.device(device_name),
            dtype,
        )
        assert coefficients.device.type == device_name
        assert coefficients.dtype == dtype

        if device_name == "cpu":
            float32_coefficients = swt._lanczos_coefficients(
                torch.device(device_name),
                torch.float32,
            )
            assert float32_coefficients is not coefficients
            assert float32_coefficients.dtype == torch.float32
    finally:
        swt._clear_device_constant_cache()


def test_calibrated_hcoeffs_include_d1v2_and_dheffSSv2():
    """d1v2/dheffSSv2 should respond to augmented spin chi (v4 calibration)."""
    eta = 0.23
    a = 0.6
    h_chi0 = compute_spin_aligned_hcoeffs(eta, a, chi=0.0)
    h_chi1 = compute_spin_aligned_hcoeffs(eta, a, chi=0.5)

    assert h_chi0["d1"] == 0.0
    assert h_chi0["dheffSS"] == 0.0
    assert h_chi0["KK"] != pytest.approx(h_chi1["KK"])
    assert h_chi1["d1v2"] != 0.0
    assert h_chi1["dheffSSv2"] != 0.0
    assert h_chi0["d1v2"] != pytest.approx(h_chi1["d1v2"])
    assert h_chi0["dheffSSv2"] != pytest.approx(h_chi1["dheffSSv2"])


def test_eobparams_initial_hcoeffs_use_lal_full_sigma():
    """The IC-entry hcoeff cache should match LAL's pre-root v4P setup."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )

    total = params.mass1 + params.mass2
    s1 = np.array([params.spin1x, params.spin1y, params.spin1z]) * (params.mass1 / total) ** 2
    s2 = np.array([params.spin2x, params.spin2y, params.spin2z]) * (params.mass2 / total) ** 2
    sigma = s1 + s2
    sigma_norm = float(np.linalg.norm(sigma))
    denom = 1.0 - 2.0 * params.eta
    chi = sigma[2] / denom
    chi += (np.dot(s1[:2], sigma[:2]) + np.dot(s2[:2], sigma[:2])) / sigma_norm / denom / 2.0
    expected = compute_spin_aligned_hcoeffs(params.eta, sigma_norm, chi=chi)

    assert params.a_sigma == pytest.approx(sigma_norm, rel=0, abs=1.0e-15)
    for key, value in expected.items():
        assert params.hcoeffs[key] == pytest.approx(value, rel=0, abs=1.0e-13)


def test_initial_conditions_leave_lal_ic_probe_hcoeffs():
    """The precessing IC path keeps LAL's final derivative hcoeff state."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2), (2, 1)),
    )
    hcoeffs_before = dict(params.hcoeffs)

    y0 = dyn.initial_conditions(params, device=torch.device("cpu"), dtype=torch.float64)

    assert y0.numel() == 12
    assert params.hcoeffs.keys() == hcoeffs_before.keys()
    expected_hcoeffs, _, _ = dyn._instantaneous_hcoeffs(params, y0[3:6], y0[6:9], y0[9:12])
    assert any(abs(params.hcoeffs[key] - hcoeffs_before[key]) > 1.0e-14 for key in hcoeffs_before)
    for key, expected in expected_hcoeffs.items():
        expected = float(torch.as_tensor(expected).detach())
        assert params.hcoeffs[key] == pytest.approx(expected, rel=0, abs=1.0e-12)


def test_calibrated_hcoeffs_match_polynomial():
    """Check d1v2/dheffSSv2 against the v4 calibration polynomials (LAL)."""
    eta = 0.1875
    a = 0.55
    chi = 0.37

    coeff_K = (
        1.7336, -1.62045, -1.38086, 1.43659,
        10.2573, 2.26831, 0.0, -0.426958,
        -126.687, 17.3736, 6.16466, 0.0,
        267.788, -27.5201, 31.1746, -59.1658,
    )
    coeff_dSO = (
        -44.5324, 0.0, 0.0, 66.1987,
        0.0, 0.0, -343.313, -568.651,
        0.0, 2495.29, 0.0, 147.481,
        0.0, 0.0, 0.0, 0.0,
    )
    coeff_dSS = (
        6.06807, 0.0, 0.0, 0.0,
        -36.0272, 37.1964, 0.0, -41.0003,
        0.0, 0.0, -326.325, 528.511,
        706.958, 0.0, 1161.78, 0.0,
    )

    chi2 = chi * chi
    chi3 = chi2 * chi
    eta2 = eta * eta
    eta3 = eta2 * eta

    def _poly(c):
        (
            c00, c01, c02, c03,
            c10, c11, c12, c13,
            c20, c21, c22, c23,
            c30, c31, c32, c33,
        ) = c
        return (
            c00 + c01 * chi + c02 * chi2 + c03 * chi3
            + c10 * eta + c11 * eta * chi + c12 * eta * chi2 + c13 * eta * chi3
            + c20 * eta2 + c21 * eta2 * chi + c22 * eta2 * chi2 + c23 * eta2 * chi3
            + c30 * eta3 + c31 * eta3 * chi + c32 * eta3 * chi2 + c33 * eta3 * chi3
        )

    expected_KK = _poly(coeff_K)
    m1_plus_eta_KK = -1.0 + eta * expected_KK
    inv_m1_plus_eta_KK = 1.0 / m1_plus_eta_KK
    k0 = expected_KK * (m1_plus_eta_KK - 1.0)
    k1 = -2.0 * (k0 + expected_KK) * m1_plus_eta_KK
    k1p2 = k1 * k1
    k1p3 = k1 * k1p2
    k2 = 0.5 * k1 * (k1 - 4.0 * m1_plus_eta_KK) - a * a * k0 * m1_plus_eta_KK * m1_plus_eta_KK
    k3 = (
        -(k1 * k1) * k1 / 3.0
        + k1 * k2
        + (k1 * k1) * m1_plus_eta_KK
        - 2.0 * (k2 - m1_plus_eta_KK) * m1_plus_eta_KK
        - a * a * k1 * m1_plus_eta_KK * m1_plus_eta_KK
    )
    k4 = (
        (24.0 / 96.0) * (k1 * k1) * (k1 * k1)
        - (96.0 / 96.0) * (k1 * k1) * k2
        + (48.0 / 96.0) * k2 * k2
        - (64.0 / 96.0) * (k1 * k1) * k1 * m1_plus_eta_KK
        + (48.0 / 96.0) * a * a * (k1 * k1 - 2.0 * k2) * m1_plus_eta_KK * m1_plus_eta_KK
        + k1 * (k3 + 2.0 * k2 * m1_plus_eta_KK)
        - m1_plus_eta_KK * (2.0 * k3 + m1_plus_eta_KK * (-(3008.0 / 96.0) + (123.0 / 96.0) * math.pi * math.pi))
    )
    expected_k5 = m1_plus_eta_KK * m1_plus_eta_KK * (
        -4237.0 / 60.0
        + 128.0 / 5.0 * 0.577215664901532860606512090082402431
        + 2275.0 * math.pi * math.pi / 512.0
        - (a * a) * (k1p3 - 3.0 * k1 * k2 + 3.0 * k3) / 3.0
        - (
            k1p3 * k1p2
            - 5.0 * k1p3 * k2
            + 5.0 * k1 * k2 * k2
            + 5.0 * k1p2 * k3
            - 5.0 * k2 * k3
            - 5.0 * k1 * k4
        ) * 0.2 * inv_m1_plus_eta_KK * inv_m1_plus_eta_KK
        + (
            k1p2 * k1p2
            - 4.0 * k1p2 * k2
            + 2.0 * k2 * k2
            + 4.0 * k1 * k3
            - 4.0 * k4
        ) * 0.5 * inv_m1_plus_eta_KK
        + (256.0 / 5.0) * math.log(2.0)
        + (41.0 * math.pi * math.pi / 32.0 - 221.0 / 6.0) * eta
    )
    expected_d1v2 = _poly(coeff_dSO)
    expected_dheffSSv2 = _poly(coeff_dSS)

    h = compute_spin_aligned_hcoeffs(eta, a, chi=chi)
    assert h["KK"] == pytest.approx(expected_KK, rel=0, abs=1e-12)
    assert h["k5"] == pytest.approx(expected_k5, rel=0, abs=1e-12)
    assert h["d1v2"] == pytest.approx(expected_d1v2, rel=0, abs=1e-12)
    assert h["dheffSSv2"] == pytest.approx(expected_dheffSSv2, rel=0, abs=1e-12)


def test_augmented_spin_mapping_matches_lal_formula():
    """Augmented spin used in RHS should follow the LAL v4P mapping."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=24.0,
        spin1x=0.18,
        spin1y=-0.12,
        spin1z=0.27,
        spin2x=-0.08,
        spin2y=0.11,
        spin2z=0.19,
        distance=500.0,
        inclination=0.5,
        f_lower=20.0,
        f_ref=30.0,
    )

    device = torch.device("cpu")
    dtype = torch.float64
    r = torch.tensor(11.0, device=device, dtype=dtype)
    pr = torch.tensor(0.015, device=device, dtype=dtype)
    phi = torch.tensor(0.42, device=device, dtype=dtype)
    L_vec = torch.tensor([2.1, -1.4, 4.6], device=device, dtype=dtype)
    S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], device=device, dtype=dtype)
    S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)
    y = torch.cat([r.view(1), pr.view(1), phi.view(1), L_vec, S1, S2])

    # Trigger RHS to refresh hcoeffs (uses augmented chi internally)
    dyn.rhs(torch.tensor(0.0, device=device, dtype=dtype), y, params)

    M = params.mass1 + params.mass2
    eta = params.mass1 * params.mass2 / (M * M)
    s1_m2 = dyn._lal_spin_weight(S1, params.mass1, M)
    s2_m2 = dyn._lal_spin_weight(S2, params.mass2, M)
    sigma = s1_m2 + s2_m2
    sigma_norm = torch.linalg.norm(sigma)
    Lhat = L_vec / torch.linalg.norm(L_vec)

    S1_perp = s1_m2 - torch.dot(s1_m2, Lhat) * Lhat
    S2_perp = s2_m2 - torch.dot(s2_m2, Lhat) * Lhat
    denom = 1.0 - 2.0 * eta
    chi_aug = torch.dot(sigma, Lhat) / denom
    chi_aug = chi_aug + (torch.dot(S1_perp, sigma) + torch.dot(S2_perp, sigma)) / (sigma_norm * denom * 2.0)

    a_sigma = float(sigma_norm)
    h_expected = compute_spin_aligned_hcoeffs(eta, a_sigma, chi=float(chi_aug))

    assert params.hcoeffs["d1v2"] == pytest.approx(h_expected["d1v2"], rel=0, abs=1e-12)
    assert params.hcoeffs["dheffSSv2"] == pytest.approx(h_expected["dheffSSv2"], rel=0, abs=1e-12)


def test_vectorized_hamiltonian_refreshes_hcoeffs_per_sample():
    """Vector mode-building H should use LAL's per-sample hcoeff refresh."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=24.0,
        spin1x=0.18,
        spin1y=-0.12,
        spin1z=0.27,
        spin2x=-0.08,
        spin2y=0.11,
        spin2z=0.19,
        distance=500.0,
        inclination=0.5,
        f_lower=20.0,
        f_ref=30.0,
    )
    dtype = torch.float64
    r = torch.tensor([11.0, 8.7], dtype=dtype)
    pr = torch.tensor([0.015, -0.002], dtype=dtype)
    phi = torch.tensor([0.42, 1.1], dtype=dtype)
    L_vec = torch.tensor([[2.1, -1.4, 4.6], [-0.6, 0.8, 3.1]], dtype=dtype)
    S1 = torch.tensor([[0.18, -0.12, 0.27], [0.23, 0.04, -0.18]], dtype=dtype)
    S2 = torch.tensor([[-0.08, 0.11, 0.19], [-0.16, 0.07, 0.12]], dtype=dtype)

    coeffs, _, _ = dyn._instantaneous_hcoeffs(params, L_vec, S1, S2)
    assert abs(float(coeffs["KK"][0] - coeffs["KK"][1])) > 1.0e-8

    pot_vec = dyn._eob_potentials(
        r,
        pr,
        phi,
        L_vec,
        S1,
        S2,
        params,
        compute_grad_p=False,
        compute_base_grad=False,
        fd_pphi=False,
    )
    h_loop = []
    for i in range(2):
        pot_i = dyn._eob_potentials(
            r[i],
            pr[i],
            phi[i],
            L_vec[i],
            S1[i],
            S2[i],
            params,
            compute_grad_p=False,
            compute_base_grad=False,
            fd_pphi=False,
        )
        h_loop.append(pot_i["H"])
    torch.testing.assert_close(pot_vec["H"], torch.stack(h_loop), rtol=1e-12, atol=1e-12)


def test_vectorized_fd_dpvec_matches_lal_axis_layout():
    """Vectorized numerical dH/dP should return per-sample Cartesian rows."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=24.0,
        spin1x=0.18,
        spin1y=-0.12,
        spin1z=0.27,
        spin2x=-0.08,
        spin2y=0.11,
        spin2z=0.19,
        distance=500.0,
        inclination=0.5,
        f_lower=20.0,
        f_ref=30.0,
    )
    dtype = torch.float64
    r = torch.tensor([11.0, 8.7], dtype=dtype)
    pr = torch.tensor([0.015, -0.002], dtype=dtype)
    phi = torch.tensor([0.42, 1.1], dtype=dtype)
    L_vec = torch.tensor([[2.1, -1.4, 4.6], [-0.6, 0.8, 3.1]], dtype=dtype)
    S1 = torch.tensor([[0.18, -0.12, 0.27], [0.23, 0.04, -0.18]], dtype=dtype)
    S2 = torch.tensor([[-0.08, 0.11, 0.19], [-0.16, 0.07, 0.12]], dtype=dtype)
    r_vec, p_vec, _, _, _ = dyn._reduced_to_cartesian(r, pr, phi, L_vec, S1, S2, params)

    pot_vec = dyn._eob_potentials(
        r,
        pr,
        phi,
        L_vec,
        S1,
        S2,
        params,
        p_vec=p_vec,
        r_vec=r_vec,
        compute_grad_p=False,
        compute_base_grad=False,
        fd_dpvec=True,
        fd_pphi=False,
    )
    assert pot_vec["dH_dpvec"].shape == p_vec.shape

    dH_loop = []
    for i in range(2):
        pot_i = dyn._eob_potentials(
            r[i],
            pr[i],
            phi[i],
            L_vec[i],
            S1[i],
            S2[i],
            params,
            p_vec=p_vec[i],
            r_vec=r_vec[i],
            compute_grad_p=False,
            compute_base_grad=False,
            fd_dpvec=True,
            fd_pphi=False,
        )
        dH_loop.append(pot_i["dH_dpvec"])
    torch.testing.assert_close(pot_vec["dH_dpvec"], torch.stack(dH_loop), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("tortoise_pdot", ["0", "1"])
def test_cartesian_projected_rhs_reconstructs_reduced_state(monkeypatch, tortoise_pdot):
    """Projected Cartesian RHS should preserve the reduced state's x cross P map."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_TORTOISE_PDOT", tortoise_pdot)

    params = dyn.EOBParams(
        mass1=28.0,
        mass2=17.0,
        spin1x=0.16,
        spin1y=-0.08,
        spin1z=0.22,
        spin2x=-0.12,
        spin2y=0.05,
        spin2z=0.14,
        distance=500.0,
        inclination=0.5,
        f_lower=25.0,
        f_ref=35.0,
    )

    device = torch.device("cpu")
    dtype = torch.float64
    r = torch.tensor(9.5, device=device, dtype=dtype)
    pr = torch.tensor(-3.0e-4, device=device, dtype=dtype)
    phi = torch.tensor(0.73, device=device, dtype=dtype)
    L_vec = torch.tensor([0.36, -0.18, 3.2], device=device, dtype=dtype)
    S1 = torch.tensor([params.spin1x, params.spin1y, params.spin1z], device=device, dtype=dtype)
    S2 = torch.tensor([params.spin2x, params.spin2y, params.spin2z], device=device, dtype=dtype)
    y = torch.cat([r.view(1), pr.view(1), phi.view(1), L_vec, S1, S2])

    r_vec, p_vec, n_hat, _, _ = dyn._reduced_to_cartesian(r, pr, phi, L_vec, S1, S2, params)
    torch.testing.assert_close(torch.linalg.cross(r_vec, p_vec), L_vec, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(torch.dot(p_vec, n_hat), pr, rtol=1e-12, atol=1e-12)

    pot = dyn._eob_potentials(
        r,
        pr,
        phi,
        L_vec,
        S1,
        S2,
        params,
        p_vec=p_vec,
        r_vec=r_vec,
        compute_grad_p=True,
        compute_grad_spin=True,
        fd_dpvec=False,
        fd_pphi=False,
    )
    assert torch.isfinite(pot["dH_dS1"]).all()
    assert torch.isfinite(pot["dH_dS2"]).all()

    dy = dyn.rhs_cartesian_projected(torch.tensor(0.0, device=device, dtype=dtype), y, params)
    assert torch.isfinite(dy).all()
    assert abs(float(dy[2])) > 0.0
    torch.testing.assert_close(torch.dot(dy[6:9], S1), torch.zeros((), device=device, dtype=dtype), rtol=0.0, atol=1e-12)
    torch.testing.assert_close(torch.dot(dy[9:12], S2), torch.zeros((), device=device, dtype=dtype), rtol=0.0, atol=1e-12)


def test_full_cartesian_rhs_roundtrips_reduced_state(monkeypatch):
    """Internal Cartesian trajectory state should project back to reduced data."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_TORTOISE_PDOT", "1")
    params = dyn.EOBParams(
        mass1=26.0,
        mass2=19.0,
        spin1x=0.11,
        spin1y=-0.09,
        spin1z=0.18,
        spin2x=-0.07,
        spin2y=0.03,
        spin2z=0.16,
        distance=450.0,
        inclination=0.4,
        f_lower=25.0,
        f_ref=35.0,
    )
    dtype = torch.float64
    reduced = torch.tensor(
        [9.8, -2.0e-4, 0.4, 0.22, -0.31, 3.35, params.spin1x, params.spin1y, params.spin1z, params.spin2x, params.spin2y, params.spin2z],
        dtype=dtype,
    )
    cart = dyn.reduced_state_to_cartesian_state(reduced, params)
    roundtrip = dyn.cartesian_state_to_reduced_state(cart, params)
    torch.testing.assert_close(roundtrip, reduced, rtol=1e-11, atol=1e-11)

    dy = dyn.rhs_cartesian_full(torch.tensor(0.0, dtype=dtype), cart, params)
    assert torch.isfinite(dy).all()
    assert abs(float(dy[6])) > 0.0
    torch.testing.assert_close(torch.dot(dy[6:9], cart[6:9]), torch.zeros((), dtype=dtype), rtol=0.0, atol=1e-12)
    torch.testing.assert_close(torch.dot(dy[9:12], cart[9:12]), torch.zeros((), dtype=dtype), rtol=0.0, atol=1e-12)


def test_waveform_state_retains_integrated_cartesian_geometry(monkeypatch):
    """Mode and omega builders must receive integrated x/P without reconstruction."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=26.0,
        mass2=19.0,
        spin1x=0.11,
        spin1y=-0.09,
        spin1z=0.18,
        spin2x=-0.07,
        spin2y=0.03,
        spin2z=0.16,
        distance=450.0,
        inclination=0.4,
        f_lower=25.0,
        f_ref=35.0,
    )
    dtype = torch.float64
    reduced = torch.tensor(
        [
            9.8,
            -2.0e-4,
            0.4,
            0.22,
            -0.31,
            3.35,
            params.spin1x,
            params.spin1y,
            params.spin1z,
            params.spin2x,
            params.spin2y,
            params.spin2z,
        ],
        dtype=dtype,
    )
    cart = dyn.reduced_state_to_cartesian_state(reduced, params)
    # The evolved phase variables are not the inertial-plane azimuth of x.
    cart[12] = 1.1
    cart[13] = -0.2
    state = swt._waveform_state_from_cartesian(cart, params).unsqueeze(0)
    assert state.shape == (1, 18)
    torch.testing.assert_close(state[0, 12:18], cart[0:6], rtol=0.0, atol=0.0)

    captured = {}

    def fake_omega(r, pr, phi, Lvec, S1vec, S2vec, params_in, **kwargs):
        assert params_in is params
        captured["omega_r_vec"] = kwargs["r_vec"]
        captured["omega_p_vec"] = kwargs["p_vec"]
        return torch.full_like(r, 0.03)

    monkeypatch.setattr(swt, "_omega_from_hamiltonian_velocity", fake_omega)
    phi, r, pr, Lvec, S1, S2, omega = swt._series_from_states(
        torch.zeros(1, dtype=dtype), state, params
    )
    r_vec, p_vec = cart[0:3].unsqueeze(0), cart[3:6].unsqueeze(0)
    expected_r = torch.linalg.norm(r_vec, dim=-1)
    expected_pr = torch.sum(p_vec * r_vec, dim=-1) / expected_r
    expected_L = torch.cross(r_vec, p_vec, dim=-1)
    torch.testing.assert_close(r, expected_r, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(pr, expected_pr, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(Lvec, expected_L, rtol=1e-15, atol=1e-15)
    torch.testing.assert_close(captured["omega_r_vec"], r_vec, rtol=0.0, atol=0.0)
    torch.testing.assert_close(captured["omega_p_vec"], p_vec, rtol=0.0, atol=0.0)
    torch.testing.assert_close(omega, torch.full_like(r, 0.03 / params.M_sec))

    def fake_potentials(*args, **kwargs):
        captured["mode_r_vec"] = kwargs["r_vec"]
        captured["mode_p_vec"] = kwargs["p_vec"]
        return {"H": torch.ones_like(r)}

    def fake_vphi(*args, **kwargs):
        captured["vphi_r_vec"] = kwargs["r_vec"]
        captured["vphi_p_vec"] = kwargs["p_vec"]
        return torch.ones_like(r)

    monkeypatch.setattr(dyn, "_eob_potentials", fake_potentials)
    monkeypatch.setattr(dyn, "non_keplerian_vphi", fake_vphi)
    modes = swt._build_coprecessing_modes(
        phi,
        omega,
        r,
        pr,
        Lvec,
        S1,
        S2,
        params,
        (),
        r_vec=r_vec,
        p_vec=p_vec,
    )
    assert modes == {}
    for name in ("mode_r_vec", "vphi_r_vec"):
        torch.testing.assert_close(captured[name], r_vec, rtol=0.0, atol=0.0)
    for name in ("mode_p_vec", "vphi_p_vec"):
        torch.testing.assert_close(captured[name], p_vec, rtol=0.0, atol=0.0)


def test_full_cartesian_analytic_spin_torque_matches_weighted_fd(monkeypatch):
    """Analytic chi gradients must produce rates for LAL's weighted spins."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    for name in (
        "PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE",
        "PYCBC_SEOBNRV4PHM_FD_SPIN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_HAMILTONIAN_SPIN", "1")

    params = dyn.EOBParams(
        mass1=25.0,
        mass2=18.0,
        spin1x=0.21,
        spin1y=-0.13,
        spin1z=0.31,
        spin2x=-0.17,
        spin2y=0.09,
        spin2z=0.12,
        distance=450.0,
        inclination=0.7,
        f_lower=25.0,
        f_ref=25.0,
    )
    dtype = torch.float64
    reduced = torch.tensor(
        [
            8.7,
            -3.0e-4,
            0.63,
            0.24,
            -0.19,
            3.15,
            params.spin1x,
            params.spin1y,
            params.spin1z,
            params.spin2x,
            params.spin2y,
            params.spin2z,
        ],
        dtype=dtype,
    )
    cart = dyn.reduced_state_to_cartesian_state(reduced, params)

    def spin_rates():
        L_vec = torch.linalg.cross(cart[0:3], cart[3:6])
        dyn._refresh_hcoeffs(params, L_vec, reduced[6:9], reduced[9:12])
        return dyn.rhs_cartesian_full(torch.tensor(0.0, dtype=dtype), cart, params)[6:12]

    analytic = spin_rates()
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_SPIN", "1")
    finite_difference = spin_rates()

    torch.testing.assert_close(analytic, finite_difference, rtol=2.0e-6, atol=2.0e-10)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(
            mass1=30.0,
            mass2=20.0,
            spin1=(0.1, -0.05, 0.2),
            spin2=(-0.02, 0.04, 0.15),
            f_min=20.0,
            inc=0.4,
        ),
        dict(
            mass1=22.0,
            mass2=16.0,
            spin1=(0.25, 0.12, -0.18),
            spin2=(-0.21, 0.06, 0.14),
            f_min=25.0,
            inc=0.7,
        ),
    ],
)
def test_hamiltonian_rhs_matches_lal_dynamics(cfg, monkeypatch):
    """SEOBNRv4PHM torch Hamiltonian/RHS should track LAL dynamics snapshots."""
    try:
        import lal
        import lalsimulation as lalsim
    except Exception as exc:
        pytest.skip(f"lalsimulation unavailable: {exc}")

    import numpy as np
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_DP", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_SPIN", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_X", "1")

    # Minimal mode set to keep the LAL call light
    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 1)
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 3, 3)
    seobflags = lal.CreateDict()

    phi_c = 0.0
    delta_t = 1.0 / 4096.0
    distance_si = 500.0 * lal.PC_SI
    outs = lalsim.SimIMRSpinPrecEOBWaveformAll(
        phi_c,
        delta_t,
        cfg["mass1"] * lal.MSUN_SI,
        cfg["mass2"] * lal.MSUN_SI,
        cfg["f_min"],
        distance_si,
        cfg["inc"],
        *cfg["spin1"],
        *cfg["spin2"],
        mode_array,
        seobflags,
    )

    # seobdynamicsAdaSHiSVector is index 6 (flattened shape [26 * N])
    seob_vec = outs[6]
    data = np.array(seob_vec.data, copy=False)
    dyn_vars = 26
    ret_len = seob_vec.length // dyn_vars
    snap_indices = sorted(
        set(
            [
                min(40, ret_len - 1),
                min(80, ret_len - 1),
                min(140, ret_len - 1),
            ]
        )
    )

    def col(k, idx):
        return data[k * ret_len + idx]

    for idx in snap_indices:
        r = torch.tensor(col(18, idx), dtype=torch.float64)
        pr_val = torch.tensor(col(20, idx), dtype=torch.float64)

        pos = torch.tensor([col(1, idx), col(2, idx), col(3, idx)], dtype=torch.float64)
        mom = torch.tensor([col(4, idx), col(5, idx), col(6, idx)], dtype=torch.float64)
        L_vec = torch.linalg.cross(pos, mom)

        S1_lal = torch.tensor([col(7, idx), col(8, idx), col(9, idx)], dtype=torch.float64)
        S2_lal = torch.tensor([col(10, idx), col(11, idx), col(12, idx)], dtype=torch.float64)
        S1, S2 = _lal_weighted_spins_to_chi(S1_lal, S2_lal, cfg["mass1"], cfg["mass2"])
        phi = torch.tensor(col(19, idx), dtype=torch.float64)

        params = dyn.EOBParams(
            mass1=cfg["mass1"],
            mass2=cfg["mass2"],
            spin1x=float(S1[0]),
            spin1y=float(S1[1]),
            spin1z=float(S1[2]),
            spin2x=float(S2[0]),
            spin2y=float(S2[1]),
            spin2z=float(S2[2]),
            distance=distance_si / lal.PC_SI,
            inclination=cfg["inc"],
            f_lower=cfg["f_min"],
            f_ref=cfg["f_min"],
        )
        params.tortoise = 1  # match LAL precessing dynamical evolution

        dyn._refresh_hcoeffs(params, L_vec, S1, S2)
        pr = pr_val
        pot = dyn._eob_potentials(r, pr, phi, L_vec, S1, S2, params, p_vec=mom, r_vec=pos)

        H_lal = torch.tensor(col(25, idx), dtype=torch.float64)
        torch.testing.assert_close(pot["H"], H_lal, rtol=1e-12, atol=1e-12)

        # Derivative parity: omega = dphi/dt, rdot from velocity projection
        y_state = torch.cat([r.view(1), pr.view(1), phi.view(1), L_vec, S1, S2])
        rhs_out = dyn.rhs_cartesian_projected(torch.tensor(0.0, dtype=torch.float64), y_state, params)

        omega_lal = torch.tensor(col(22, idx), dtype=torch.float64)
        torch.testing.assert_close(rhs_out[2], omega_lal, rtol=2e-5, atol=1e-8)

        v_vec = torch.tensor([col(15, idx), col(16, idx), col(17, idx)], dtype=torch.float64)
        rhat = pos / torch.linalg.norm(pos)
        rdot_lal = torch.dot(rhat, v_vec)
        torch.testing.assert_close(rhs_out[0], rdot_lal, rtol=1e-4, atol=5e-8)

        nhat_dot_dH = torch.dot(pot["dH_dpvec"], pot["n_hat"])
        v_torch = pot["dH_dpvec"] + (pot["csi"] - 1.0) * nhat_dot_dH * pot["n_hat"]
        torch.testing.assert_close(v_torch, v_vec, rtol=1e-4, atol=2e-6)

        if 0 < idx < ret_len - 1:
            dt_lal = torch.tensor(col(0, idx + 1) - col(0, idx - 1), dtype=torch.float64)
            S1_prev = torch.tensor([col(7, idx - 1), col(8, idx - 1), col(9, idx - 1)], dtype=torch.float64)
            S2_prev = torch.tensor([col(10, idx - 1), col(11, idx - 1), col(12, idx - 1)], dtype=torch.float64)
            S1_next = torch.tensor([col(7, idx + 1), col(8, idx + 1), col(9, idx + 1)], dtype=torch.float64)
            S2_next = torch.tensor([col(10, idx + 1), col(11, idx + 1), col(12, idx + 1)], dtype=torch.float64)
            S1_prev, S2_prev = _lal_weighted_spins_to_chi(S1_prev, S2_prev, cfg["mass1"], cfg["mass2"])
            S1_next, S2_next = _lal_weighted_spins_to_chi(S1_next, S2_next, cfg["mass1"], cfg["mass2"])
            torch.testing.assert_close(rhs_out[6:9], (S1_next - S1_prev) / dt_lal, rtol=2e-1, atol=5e-6)
            torch.testing.assert_close(rhs_out[9:12], (S2_next - S2_prev) / dt_lal, rtol=3e-1, atol=1e-5)

        dphi_fd = dyn.dphi_dt_fd(
            r, pr, phi, L_vec, S1, S2, params, p_vec=mom, r_vec=pos, step=1e-4
        )
        torch.testing.assert_close(pot["dH_dpf"], dphi_fd, rtol=1e-6, atol=1e-8)

        y_full = torch.cat(
            [
                pos,
                mom,
                S1_lal,
                S2_lal,
                torch.tensor([col(13, idx), col(14, idx)], dtype=torch.float64),
            ]
        )
        rhs_full = dyn.rhs_cartesian_full(torch.tensor(0.0, dtype=torch.float64), y_full, params)
        torch.testing.assert_close(rhs_full[:3], v_vec, rtol=1e-4, atol=2e-6)
        torch.testing.assert_close(rhs_full[12] + rhs_full[13], omega_lal, rtol=2e-5, atol=1e-8)


def test_first_clean_lal_rk_step_matches_lal_dynamics(monkeypatch):
    """With LAL's first row and step, torch RK/RHS should reproduce row two."""
    import torch

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    params_dict = dict(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        delta_t=1.0 / 4096.0,
        f_lower=90.0,
        f_ref=90.0,
        distance=300.0,
        inclination=0.4,
        mode_array=((2, 2), (2, 1)),
    )
    rows = _lal_phm_dynamics_rows(params_dict, 2, vector="adas")
    y_torch, y1_lal = _torch_replay_lal_rk_interval(params_dict, rows, 0)

    torch.testing.assert_close(y_torch[:12], y1_lal[:12], rtol=0.0, atol=5.0e-11)
    torch.testing.assert_close(y_torch[12:], y1_lal[12:], rtol=0.0, atol=1.2e-9)


def test_clean_lal_adas_rk_intervals_match_lal_dynamics(monkeypatch):
    """Replay representative raw AdaS LAL intervals with torch RK/RHS."""
    import torch

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    params_dict = dict(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.03,
        spin1y=-0.02,
        spin1z=0.12,
        spin2x=-0.01,
        spin2y=0.02,
        spin2z=-0.08,
        delta_t=1.0 / 4096.0,
        f_lower=90.0,
        f_ref=90.0,
        distance=300.0,
        inclination=0.4,
        mode_array=((2, 2), (2, 1)),
    )
    rows = _lal_phm_dynamics_rows(params_dict, 102, vector="adas")

    for idx in (1, 2, 10, 50, 100):
        y_torch, y_lal = _torch_replay_lal_rk_interval(params_dict, rows, idx)
        torch.testing.assert_close(y_torch[:12], y_lal[:12], rtol=0.0, atol=2.0e-10)
        torch.testing.assert_close(y_torch[12:], y_lal[12:], rtol=0.0, atol=6.0e-9)


def test_dphi_dt_fd_matches_gsl_snapshot():
    """Finite-diff dphi/dt (torch) should match LAL GSL pphi derivative on a snapshot."""
    try:
        import lal
        import lalsimulation as lalsim
    except Exception as exc:
        pytest.skip(f"lalsimulation unavailable: {exc}")

    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    cfg = dict(
        mass1=30.0,
        mass2=20.0,
        spin1=(0.1, -0.05, 0.2),
        spin2=(-0.02, 0.04, 0.15),
        f_min=20.0,
        inc=0.4,
    )

    mode_array = lalsim.SimInspiralCreateModeArray()
    lalsim.SimInspiralModeArrayActivateMode(mode_array, 2, 2)
    seobflags = lal.CreateDict()

    phi_c = 0.0
    delta_t = 1.0 / 4096.0
    distance_si = 500.0 * lal.PC_SI
    outs = lalsim.SimIMRSpinPrecEOBWaveformAll(
        phi_c,
        delta_t,
        cfg["mass1"] * lal.MSUN_SI,
        cfg["mass2"] * lal.MSUN_SI,
        cfg["f_min"],
        distance_si,
        cfg["inc"],
        *cfg["spin1"],
        *cfg["spin2"],
        mode_array,
        seobflags,
    )

    seob_vec = outs[6]
    data = np.array(seob_vec.data, copy=False)
    dyn_vars = 26
    ret_len = seob_vec.length // dyn_vars
    idx = min(80, ret_len - 1)
    def col(k):
        return data[k * ret_len + idx]

    r = torch.tensor(col(18), dtype=torch.float64)
    pr_val = torch.tensor(col(20), dtype=torch.float64)
    pos = torch.tensor([col(1), col(2), col(3)], dtype=torch.float64)
    mom = torch.tensor([col(4), col(5), col(6)], dtype=torch.float64)
    L_vec = torch.linalg.cross(pos, mom)

    S1_lal = torch.tensor([col(7), col(8), col(9)], dtype=torch.float64)
    S2_lal = torch.tensor([col(10), col(11), col(12)], dtype=torch.float64)
    S1, S2 = _lal_weighted_spins_to_chi(S1_lal, S2_lal, cfg["mass1"], cfg["mass2"])
    phi = torch.tensor(col(19), dtype=torch.float64)

    params = dyn.EOBParams(
        mass1=cfg["mass1"],
        mass2=cfg["mass2"],
        spin1x=float(S1[0]),
        spin1y=float(S1[1]),
        spin1z=float(S1[2]),
        spin2x=float(S2[0]),
        spin2y=float(S2[1]),
        spin2z=float(S2[2]),
        distance=distance_si / lal.PC_SI,
        inclination=cfg["inc"],
        f_lower=cfg["f_min"],
        f_ref=cfg["f_min"],
    )
    params.tortoise = 1

    pot = dyn._eob_potentials(r, pr_val, phi, L_vec, S1, S2, params, p_vec=mom, r_vec=pos)
    dphi_fd = dyn.dphi_dt_fd(
        r, pr_val, phi, L_vec, S1, S2, params, p_vec=mom, r_vec=pos, step=1e-4
    )
    omega_lal = torch.tensor(col(22), dtype=torch.float64)

    torch.testing.assert_close(dphi_fd, omega_lal, rtol=1e-3, atol=1e-5)
    torch.testing.assert_close(pot["dH_dpf"], dphi_fd, rtol=1e-6, atol=1e-8)


def test_rhs_derivative_options_enable_lal_numerical_bundle(monkeypatch):
    """One parity switch should enable LAL's numerical derivative choices."""
    from pycbc.waveform import seobnrv4phm_dynamics as dyn

    for name in (
        "PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE",
        "PYCBC_SEOBNRV4PHM_FD_DP",
        "PYCBC_SEOBNRV4PHM_FD_SPIN",
        "PYCBC_SEOBNRV4PHM_FD_X",
        "PYCBC_SEOBNRV4PHM_HAMILTONIAN_SPIN",
        "PYCBC_SEOBNRV4PHM_TORTOISE_PDOT",
        "PYCBC_SEOBNRV4PHM_X_GRAD",
    ):
        monkeypatch.delenv(name, raising=False)

    opts = dyn._rhs_derivative_options()
    assert opts["fd_dpvec"] is False
    assert opts["use_fd_spin"] is False
    assert opts["use_fd_x"] is False

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_LAL_NUMERICAL_DERIVATIVE", "1")
    opts = dyn._rhs_derivative_options()
    assert opts["fd_dpvec"] is True
    assert opts["use_fd_spin"] is True
    assert opts["use_fd_x"] is True
    assert opts["use_hamiltonian_spin"] is True
    assert opts["use_tortoise_pdot"] is True

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_FD_X", "0")
    assert dyn._rhs_derivative_options()["use_fd_x"] is False


def test_rkf45_step_nonzero_error_and_accuracy():
    """rk45_step should produce non-zero error and accurate growth for y'=y."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    def f(t, y):
        return y

    state = ode.RKState(torch.tensor(0.0), torch.tensor([1.0], dtype=torch.float64), torch.tensor(0.5))
    new_state, err = ode.rk45_step(f, state)

    assert torch.all(err > 0)
    expected = torch.exp(torch.tensor(0.5, dtype=new_state.y.dtype, device=new_state.y.device))
    torch.testing.assert_close(new_state.y[0], expected, rtol=1e-3, atol=1e-3)


def test_integrate_simple_exp():
    """integrate should track y'=y over [0,1] within tolerance."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    def f(t, y):
        return y

    y0 = torch.tensor([1.0], dtype=torch.float64)
    traj = ode.integrate(f, y0, t0=0.0, t1=1.0, h0=0.25, rtol=1e-7, atol=1e-10)

    assert len(traj) > 2  # multiple steps taken
    t_end, y_end = traj[-1]
    torch.testing.assert_close(t_end, torch.tensor(1.0, dtype=t_end.dtype, device=t_end.device), atol=1e-6, rtol=0.0)
    expected = torch.exp(torch.tensor(1.0, dtype=y_end.dtype, device=y_end.device))
    torch.testing.assert_close(y_end[0], expected, rtol=2e-3, atol=2e-3)


def test_integrate_reuses_cached_k1_after_rejection():
    """LAL's RK driver does not recompute dydt_in when retrying a rejected step."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    calls = []

    def f(t, y):
        calls.append(float(t))
        return y

    y0 = torch.tensor([1.0], dtype=torch.float64)
    traj = ode.integrate(
        f,
        y0,
        t0=0.0,
        t1=1.0,
        h0=1.0,
        rtol=1e-14,
        atol=1e-14,
        stop_fn=lambda t, y: True,
    )

    assert len(traj) == 1
    assert sum(abs(t) < 1.0e-15 for t in calls) == 1


def test_integrate_accepts_lal_min_step_clamp():
    """NoInterpolate accepts when LAL's min-step clamp prevents a decrease."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    y0 = torch.tensor([1.0], dtype=torch.float64)
    traj = ode.integrate(
        lambda t, y: y,
        y0,
        t0=0.0,
        t1=10.0,
        h0=1.0,
        rtol=1e-14,
        atol=1e-14,
        max_steps=3,
        stop_fn=lambda t, y: True,
        h_min=1.0,
    )

    assert len(traj) == 1
    assert float(traj[0][0]) == 1.0


def test_integrate_debug_progress_label_and_max_steps(monkeypatch, capsys):
    """Debug output should identify bounded native integration stages."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_DEBUG", "1")
    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_PROGRESS_INTERVAL", "0")

    with pytest.warns(UserWarning, match="max_steps"):
        ode.integrate(
            lambda t, y: y,
            torch.tensor([1.0], dtype=torch.float64),
            t0=0.0,
            t1=10.0,
            h0=0.1,
            rtol=1e-7,
            atol=1e-10,
            max_steps=1,
            h_min=0.0,
            progress_label="unit",
        )

    out = capsys.readouterr().out
    assert "ODE unit accept step=0" in out


def test_integrate_can_return_step_diagnostics():
    """Diagnostics expose accepted/rejected attempts for capped runs."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    with pytest.warns(UserWarning, match="max_steps"):
        traj, info = ode.integrate(
            lambda t, y: y,
            torch.tensor([1.0], dtype=torch.float64),
            t0=0.0,
            t1=10.0,
            h0=0.1,
            rtol=1e-7,
            atol=1e-10,
            max_steps=1,
            h_min=0.0,
            return_diagnostics=True,
        )

    assert len(traj) == 1
    assert info["accepted_steps"] == 1
    assert info["rejected_steps"] == 0
    assert info["attempted_steps"] == 1
    assert info["max_steps"] == 1
    assert info["hit_max_steps"] is True
    assert info["t_end"] == pytest.approx(0.1)


def test_integrate_skips_error_argmax_without_step_trace(monkeypatch):
    """Unreturned diagnostics must not synchronize per-component errors."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_TRACE_STEPS", "5")

    def unexpected_argmax(*args, **kwargs):
        raise AssertionError("error argmax should be trace-only")

    monkeypatch.setattr(torch, "argmax", unexpected_argmax)
    traj = ode.integrate(
        lambda t, y: y,
        torch.tensor([1.0], dtype=torch.float64),
        t0=0.0,
        t1=1.0,
        h0=0.25,
        rtol=1e-7,
        atol=1e-10,
    )

    assert traj


def test_integrate_can_trace_lal_hadjust_attempts(monkeypatch):
    """Optional diagnostics should expose accepted/rejected h-adjust decisions."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_TRACE_STEPS", "2")
    y0 = torch.tensor([1.0], dtype=torch.float64)
    calls = []

    def fake_step(f, state, dydt_in=None, compute_dydt_out=False):
        calls.append(float(state.h))
        err = torch.tensor([2.0 if len(calls) == 1 else 0.25], dtype=state.y.dtype)
        return ode.RKState(state.t + state.h, state.y.clone(), state.h), err, torch.zeros_like(state.y)

    monkeypatch.setattr(ode, "_rk45_step_impl", fake_step)
    traj, info = ode.integrate(
        lambda t, y: y,
        y0,
        t0=0.0,
        t1=10.0,
        h0=1.0,
        rtol=0.0,
        atol=1.0,
        stop_fn=lambda t, y: True,
        h_min=0.0,
        return_diagnostics=True,
    )

    expected_shrink = 0.9 * 2.0 ** (-1.0 / 5.0)
    expected_grow = expected_shrink * 0.9 * 0.25 ** (-1.0 / 6.0)
    assert float(traj[0][0]) == pytest.approx(expected_shrink)
    assert info["step_trace"] == [
        {
            "step": 0,
            "action": "reject_error",
            "t": 0.0,
            "h": 1.0,
            "err_norm": 2.0,
            "err_argmax": 0,
            "err_component": 2.0,
            "scale_component": 1.0,
            "h_next": pytest.approx(expected_shrink),
        },
        {
            "step": 1,
            "action": "accept",
            "t": 0.0,
            "h": pytest.approx(expected_shrink),
            "err_norm": 0.25,
            "err_argmax": 0,
            "err_component": 0.25,
            "scale_component": 1.0,
            "h_next": pytest.approx(expected_grow),
        },
    ]


def test_integrate_uses_gsl_rkf45_control_order(monkeypatch):
    """Old gsl_odeiv reports RKF45 step order 5 to the controller."""
    import torch

    from pycbc.waveform import seobnrv4phm_ode as ode

    y0 = torch.tensor([1.0], dtype=torch.float64)

    calls = []

    def fake_step_shrink(f, state, dydt_in=None, compute_dydt_out=False):
        calls.append(float(state.h))
        err = torch.tensor([2.0 if len(calls) == 1 else 0.4], dtype=state.y.dtype)
        return ode.RKState(state.t + state.h, state.y.clone(), state.h), err, torch.zeros_like(state.y)

    monkeypatch.setattr(ode, "_rk45_step_impl", fake_step_shrink)
    traj = ode.integrate(
        lambda t, y: y,
        y0,
        t0=0.0,
        t1=10.0,
        h0=1.0,
        rtol=0.0,
        atol=1.0,
        stop_fn=lambda t, y: True,
        h_min=0.0,
    )

    expected_shrink = 0.9 * 2.0 ** (-1.0 / 5.0)
    assert calls[1] == pytest.approx(expected_shrink)
    assert float(traj[0][0]) == pytest.approx(expected_shrink)

    calls = []

    def fake_step_grow(f, state, dydt_in=None, compute_dydt_out=False):
        calls.append(float(state.h))
        err = torch.tensor([0.25 if len(calls) == 1 else 1.0], dtype=state.y.dtype)
        return ode.RKState(state.t + state.h, state.y.clone(), state.h), err, torch.zeros_like(state.y)

    monkeypatch.setattr(ode, "_rk45_step_impl", fake_step_grow)
    ode.integrate(
        lambda t, y: y,
        y0,
        t0=0.0,
        t1=10.0,
        h0=1.0,
        rtol=0.0,
        atol=1.0,
        stop_fn=lambda t, y: len(calls) >= 2,
        h_min=0.0,
    )

    expected_grow = 0.9 * 0.25 ** (-1.0 / 6.0)
    assert calls[1] == pytest.approx(expected_grow)


def test_final_mass_spin_uses_adas_r10m_lframe_state():
    """Final mass/spin fit should use LAL's AdaS sample nearest r=10M."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    params = dyn.EOBParams(
        mass1=PARAMS["mass1"],
        mass2=PARAMS["mass2"],
        spin1x=0.2,
        spin1y=0.1,
        spin1z=0.3,
        spin2x=-0.1,
        spin2y=0.05,
        spin2z=0.2,
        distance=PARAMS["distance"],
        inclination=PARAMS["inclination"],
        f_lower=PARAMS["f_lower"],
        f_ref=PARAMS["f_ref"],
    )
    t_adas = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype)
    y_adas = torch.zeros((4, 12), dtype=dtype)
    y_adas[:, 0] = torch.tensor([13.0, 11.0, 9.8, 8.0], dtype=dtype)
    y_adas[:, 1] = -1.0e-3
    y_adas[:, 2] = torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=dtype)
    y_adas[:, 3:6] = torch.tensor([0.0, 0.0, 4.0], dtype=dtype)
    y_adas[:, 6:9] = torch.tensor([0.2, 0.1, 0.3], dtype=dtype)
    y_adas[:, 9:12] = torch.tensor([-0.1, 0.05, 0.2], dtype=dtype)

    (mf, af), time_10M, s1_l, s2_l = swt._final_mass_spin_from_adas_10M(t_adas, y_adas, params)

    assert time_10M == pytest.approx(2.0)
    expected_s1, expected_s2 = swt._l_frame_spin_vectors_from_state(y_adas[2], params)
    expected_mf, expected_af = swt._final_mass_spin_prec(
        params.mass1,
        params.mass2,
        tuple(float(x) for x in expected_s1),
        tuple(float(x) for x in expected_s2),
    )
    torch.testing.assert_close(s1_l, expected_s1, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(s2_l, expected_s2, rtol=1e-12, atol=1e-12)
    assert mf == pytest.approx(expected_mf)
    assert af == pytest.approx(expected_af)


def test_final_spin_sign_and_clamp_match_lal():
    """LAL signs finalSpin with Lhat.J and clamps QNM interpolation input."""
    from pycbc.waveform import seobnrv4phm_torch as swt

    assert swt._signed_clamped_final_spin(0.4, 0.1) == pytest.approx(0.4)
    assert swt._signed_clamped_final_spin(0.4, -0.1) == pytest.approx(-0.4)
    assert swt._signed_clamped_final_spin(1.2, 0.1) == pytest.approx(0.9996)
    assert swt._signed_clamped_final_spin(1.2, -0.1) == pytest.approx(-0.9996)


def test_adaptive_his_join_excludes_adas_join_sample():
    """AdaS/HiS join should keep AdaS before tstartHiS and HiS from tstartHiS."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    adas = [(torch.tensor(float(i), dtype=dtype), torch.tensor([float(i)], dtype=dtype)) for i in range(5)]
    his = [
        (torch.tensor(2.0, dtype=dtype), torch.tensor([20.0], dtype=dtype)),
        (torch.tensor(2.5, dtype=dtype), torch.tensor([25.0], dtype=dtype)),
        (torch.tensor(3.0, dtype=dtype), torch.tensor([30.0], dtype=dtype)),
    ]

    assert swt._last_traj_index_leq(adas, 2.6) == 2
    joined = swt._join_adaptive_his_trajectories(adas, his, 2)

    times = [float(t) for t, _ in joined]
    values = [float(y[0]) for _, y in joined]
    assert times == [0.0, 1.0, 2.0, 2.5, 3.0]
    assert values == [0.0, 1.0, 20.0, 25.0, 30.0]


def test_uniform_joined_dynamics_excludes_attachment_index():
    """SEOBJoinDynamics excludes the last sample <= tAttach from dynamics."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t_adas = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype)
    y_adas = torch.arange(4, dtype=dtype).reshape(4, 1)
    t_his = torch.tensor([2.0, 2.5, 3.0, 3.5, 4.0], dtype=dtype)
    y_his = (10.0 + torch.arange(5, dtype=dtype)).reshape(5, 1)

    t_join, y_join, index_attach = swt._join_uniform_dynamics_until_attach(
        t_adas,
        y_adas,
        t_his,
        y_his,
        index_start_his=2,
        t_attach_M=3.5,
    )

    assert index_attach == 5
    assert t_join.tolist() == [0.0, 1.0, 2.0, 2.5, 3.0]
    assert y_join[:, 0].tolist() == [0.0, 1.0, 10.0, 11.0, 12.0]


def test_phm_boundary_summary_from_segments_uses_supplied_attachment():
    """Boundary summaries should avoid merger timing when tAttach is supplied."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t_adas = torch.tensor([0.0, 0.01, 0.02, 0.03], dtype=dtype)
    y_adas = torch.tensor([[10.0], [9.0], [8.0], [7.0]], dtype=dtype)
    t_his = torch.tensor([0.02, 0.04, 0.06, 0.08], dtype=dtype)
    y_his = torch.tensor([[8.0], [6.0], [4.0], [2.0]], dtype=dtype)
    segments = swt._TrajectorySegments(
        traj=[(t_his[i], y_his[i]) for i in range(len(t_his))],
        traj_adas=[(t_adas[i], y_adas[i]) for i in range(len(t_adas))],
        traj_his=[(t_his[i], y_his[i]) for i in range(len(t_his))],
        index_start_his=2,
        tstart_his=0.02,
        adas_stop_state={"stop_reason": "adas"},
        his_stop_state={"stop_reason": "his"},
    )
    params = type("Params", (), {"mode_array": ((2, 2),)})()

    summary = swt._phm_boundary_summary_from_segments(params, segments, 0.06)

    assert summary["n_adas"] == 4
    assert summary["n_his"] == 4
    assert summary["n_joined"] == 4
    assert summary["index_join_his"] == 2
    assert summary["index_join_attach"] == 4
    assert summary["t_join_attach_M"] == pytest.approx(0.06)
    assert summary["joined_end_M"] == pytest.approx(0.04)
    assert summary["r_his_start"] == pytest.approx(8.0)
    assert summary["r_end"] == pytest.approx(6.0)
    assert summary["adas_samples"][1]["t_M"] == pytest.approx(0.01)
    assert summary["adas_samples"][1]["r"] == pytest.approx(9.0)
    assert summary["joined_samples"][2]["state_prefix"] == [8.0]


def test_phm_adas_only_diagnostic_skips_his(monkeypatch):
    """The AdaS-only diagnostic path should not launch the HiS rerun."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=12.0,
        mass2=8.0,
        spin1x=0.0,
        spin1y=0.0,
        spin1z=0.12,
        spin2x=0.0,
        spin2y=0.0,
        spin2z=-0.08,
        distance=300.0,
        inclination=0.4,
        f_lower=90.0,
        f_ref=90.0,
        mode_array=((2, 2),),
    )
    y0 = torch.arange(12, dtype=torch.float64)
    calls = []

    def fake_integrate(_rhs, y_start, *_args, progress_label=None, return_diagnostics=False, **_kwargs):
        calls.append(progress_label)
        traj = [
            (torch.tensor(1.0, dtype=torch.float64), y_start + 1.0),
            (torch.tensor(2.0, dtype=torch.float64), y_start + 2.0),
        ]
        diag = {"accepted_steps": 2, "rejected_steps": 0, "attempted_steps": 2}
        return (traj, diag) if return_diagnostics else traj

    monkeypatch.setenv("PYCBC_SEOBNRV4PHM_ADAS_ONLY", "1")
    monkeypatch.setattr(dyn, "initial_conditions", lambda *_args, **_kwargs: y0)
    monkeypatch.setattr(swt, "integrate", fake_integrate)

    segments = swt._integrate_traj(
        params,
        device=torch.device("cpu"),
        real_dtype=torch.float64,
        f_final=0.0,
        delta_t=1.0 / 4096.0,
        return_segments=True,
    )

    assert calls == ["adas"]
    assert segments.traj_his == []
    assert segments.his_stop_state["stop_reason"] == "adas_only"
    assert len(segments.traj_adas) == 2


def test_lal_local_interpolation_window_indices():
    """SEOBInterpolateDynamicsAtTime uses the last <= sample and +/-20 points."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    t = torch.arange(50, dtype=torch.float64)

    assert swt._interpolation_window_indices(t, 22.4) == (2, 42)
    assert swt._interpolation_window_indices(t, 1.5) == (0, 21)
    assert swt._interpolation_window_indices(t, 48.5) == (28, 49)


def test_his_state_transfer_uses_lal_local_cubic_window():
    """HiS IC transfer should mirror SEOBInterpolateDynamicsAtTime."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t = torch.arange(50, dtype=dtype)
    y = torch.stack(
        [
            torch.sin(0.13 * t),
            torch.cos(0.07 * t),
            0.01 * t * t,
        ],
        dim=1,
    )
    traj = [(t[i], y[i]) for i in range(len(t))]
    q = 22.4

    got = swt._interpolate_traj_at_time(traj, q)
    expected = swt._interp_series_cubic(torch.tensor([q], dtype=dtype), t[2:43], y[2:43])[0]

    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_sample_trajectory_states_batches_columnwise_cubic():
    """Trajectory resampling should match the previous column-wise cubic calls."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t_src = torch.linspace(0.0, 3.0, 16, dtype=dtype)
    t_query = torch.linspace(0.15, 2.85, 23, dtype=dtype)
    y_src = torch.stack(
        [
            torch.sin(t_src),
            torch.cos(0.7 * t_src),
            t_src * t_src,
            torch.exp(-0.2 * t_src),
        ],
        dim=1,
    )

    got = swt._sample_trajectory_states(t_query, t_src, y_src)
    expected = torch.stack(
        [
            swt._interp_series_cubic(t_query, t_src, y_src[:, i])
            for i in range(y_src.shape[1])
        ],
        dim=1,
    )

    torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)


def test_sample_trajectory_states_handles_two_point_vector_fallback():
    """Capped HiS diagnostics may resample vector states from two support rows."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    dtype = torch.float64
    t_src = torch.tensor([0.0, 2.0], dtype=dtype)
    y_src = torch.tensor(
        [
            [1.0, 10.0, -3.0],
            [5.0, 14.0, 9.0],
        ],
        dtype=dtype,
    )
    t_query = torch.tensor([0.5, 1.0, 1.5], dtype=dtype)

    got = swt._sample_trajectory_states(t_query, t_src, y_src)
    expected = torch.tensor(
        [
            [2.0, 11.0, 0.0],
            [3.0, 12.0, 3.0],
            [4.0, 13.0, 6.0],
        ],
        dtype=dtype,
    )

    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


def test_series_from_states_uses_lal_velocity_omega(monkeypatch):
    """Mode series should use LAL's r x rdot omega, not sampled phase slope."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.1,
        spin1y=-0.05,
        spin1z=0.2,
        spin2x=-0.02,
        spin2y=0.04,
        spin2z=0.15,
        distance=400.0,
        inclination=0.4,
        f_lower=20.0,
        f_ref=30.0,
    )
    t = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch.zeros((3, 12), dtype=torch.float64)
    y[:, 0] = torch.tensor([11.0, 10.9, 10.7], dtype=torch.float64)
    y[:, 1] = -1.0e-3
    y[:, 2] = torch.tensor([0.0, 100.0, 200.0], dtype=torch.float64)
    y[:, 3:6] = torch.tensor([0.2, -0.1, 3.4], dtype=torch.float64)
    y[:, 6:9] = torch.tensor([0.1, -0.05, 0.2], dtype=torch.float64)
    y[:, 9:12] = torch.tensor([-0.02, 0.04, 0.15], dtype=torch.float64)
    omega_lal = torch.tensor([0.03, 0.031, 0.032], dtype=torch.float64)

    def fake_omega(r, pr, phi, Lvec, S1vec, S2vec, params_in):
        assert params_in is params
        return omega_lal

    monkeypatch.setattr(swt, "_omega_from_hamiltonian_velocity", fake_omega)
    *_, omega_orb = swt._series_from_states(t, y, params)
    torch.testing.assert_close(omega_orb, omega_lal / params.M_sec, rtol=0.0, atol=0.0)


def test_hamiltonian_velocity_omega_uses_lal_numerical_derivative(monkeypatch):
    """LAL's derived omegaVec path uses numerical Hamiltonian dH/dP by default."""
    import torch

    from pycbc.waveform import seobnrv4phm_dynamics as dyn
    from pycbc.waveform import seobnrv4phm_torch as swt

    params = dyn.EOBParams(
        mass1=30.0,
        mass2=20.0,
        spin1x=0.1,
        spin1y=-0.05,
        spin1z=0.2,
        spin2x=-0.02,
        spin2y=0.04,
        spin2z=0.15,
        distance=400.0,
        inclination=0.4,
        f_lower=20.0,
        f_ref=30.0,
    )
    r = torch.tensor([10.0, 8.0], dtype=torch.float64)
    pr = torch.zeros_like(r)
    phi = torch.zeros_like(r)
    L = torch.zeros((2, 3), dtype=torch.float64)
    S1 = torch.zeros((2, 3), dtype=torch.float64)
    S2 = torch.zeros((2, 3), dtype=torch.float64)
    expected = torch.tensor([0.03, 0.04], dtype=torch.float64)

    def fake_reduced_to_cartesian(r_c, pr_c, phi_c, L_c, S1_c, S2_c, params_in):
        r_vec = torch.stack([r_c, torch.zeros_like(r_c), torch.zeros_like(r_c)], dim=-1)
        p_vec = torch.zeros_like(r_vec)
        n_hat = r_vec / r_c.unsqueeze(-1)
        return r_vec, p_vec, n_hat, torch.zeros_like(r_vec), torch.zeros_like(r_vec)

    def fake_potentials(*args, **kwargs):
        assert kwargs["compute_grad_p"] is False
        assert kwargs["fd_dpvec"] is True
        assert kwargs["fd_pphi"] is False
        r_vec = kwargs["r_vec"]
        omega = expected[: r_vec.shape[0]]
        dH_dpvec = torch.stack(
            [torch.zeros_like(omega), omega * r_vec[:, 0], torch.zeros_like(omega)],
            dim=-1,
        )
        return {
            "dH_dpvec": dH_dpvec,
            "n_hat": r_vec / r_vec[:, 0].unsqueeze(-1),
            "csi": torch.ones_like(omega),
        }

    monkeypatch.setattr(swt._dyn, "_reduced_to_cartesian", fake_reduced_to_cartesian)
    monkeypatch.setattr(swt._dyn, "_eob_potentials", fake_potentials)

    got = swt._omega_from_hamiltonian_velocity(r, pr, phi, L, S1, S2, params)
    torch.testing.assert_close(got, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("device_name", ("cpu", "mps"))
def test_nqc_spline_values_derivatives_stay_on_device(device_name):
    """NQC spline values and derivatives should remain on their device."""
    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt
    from pycbc.waveform.seobnrv4phm_peak import local_derivatives

    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device(device_name)
    dtype = torch.float64 if device_name == "cpu" else torch.float32
    t = torch.linspace(-2.0, 2.0, 31, device=device, dtype=dtype)
    series = (
        torch.sin(0.4 * t),
        torch.cos(0.3 * t) + 0.1 * t,
        0.2 * t * t - 0.05 * t,
    )
    t0 = 0.37
    got = swt._spline_values_derivatives(series, t, t0)
    q = torch.tensor([t0], device=device, dtype=dtype)

    for item, single in zip(got, series, strict=False):
        assert all(value.device == t.device for value in item)
        expected = torch.tensor(
            (
                float(swt._interp_series_cubic(q, t, single)[0]),
                local_derivatives(single, t, t0, order=1),
                local_derivatives(single, t, t0, order=2),
            ),
            device=device,
            dtype=dtype,
        )
        torch.testing.assert_close(
            torch.stack(item),
            expected,
            rtol=0.0,
            atol=1.0e-12 if dtype == torch.float64 else 2.0e-6,
        )


@pytest.mark.parametrize("device_name", ("cpu", "mps"))
def test_nqc_series_linear_solves_stay_on_device(monkeypatch, device_name):
    """Both mode-series NQC systems should be solved on the source device."""
    from types import SimpleNamespace

    import torch

    from pycbc.waveform import seobnrv4phm_torch as swt

    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device(device_name)
    dtype = torch.float64 if device_name == "cpu" else torch.float32
    t = torch.linspace(-20.0, 10.0, 121, device=device, dtype=dtype)
    r = 8.0 - 0.05 * t + 0.002 * t * t
    pr = -0.12 - 0.003 * t + 0.0002 * t * t
    omega = 0.04 + 0.001 * t - 0.00003 * t * t
    amplitude = 0.2 + 0.003 * t - 0.0001 * t * t + 0.000002 * t**3
    phase = 0.3 * t + 0.003 * t * t + 0.00002 * t**3
    hmode = amplitude * torch.exp(1j * phase)
    params = SimpleNamespace(
        mass1=30.0,
        mass2=20.0,
        eta=30.0 * 20.0 / 50.0**2,
    )

    solve_devices = []
    original_solve = swt._solve_nqc_system

    def record_solve(matrix, rhs):
        solve_devices.append((matrix.device, rhs.device))
        return original_solve(matrix, rhs)

    monkeypatch.setattr(swt, "_solve_nqc_system", record_solve)
    result = swt._solve_nqc_coeffs_lal_series(
        2,
        2,
        hmode,
        t,
        r,
        pr,
        omega,
        t_peak_omega_M=4.0,
        chi1L_peak=0.2,
        chi2L_peak=-0.1,
        params=params,
    )

    assert solve_devices == [(t.device, t.device), (t.device, t.device)]
    for key in ("a1", "a2", "a3", "b1", "b2"):
        assert result[key].device == t.device
        assert torch.isfinite(result[key])
    if device_name == "cpu":
        expected = {
            "a1": 190.2372024955108,
            "a2": -3637.470702926369,
            "a3": 6125.146190134807,
            "b1": -21.30900882981344,
            "b2": 399.1299031643519,
        }
        assert {key: float(result[key]) for key in expected} == pytest.approx(
            expected,
            rel=1.0e-12,
            abs=1.0e-12,
        )
