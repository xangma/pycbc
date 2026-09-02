import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import scheme as _scheme  # noqa: E402
from pycbc.waveform import (  # noqa: E402
    get_fd_waveform,
    get_fd_waveform_sequence,
)
from pycbc.waveform.imrphenomxhm_torch import (  # noqa: E402
    _SequenceCore,
    _active_mode_samples,
    imrphenomxhm_fd_native_supported,
    imrphenomxhm_modes_native_supported,
    imrphenomxhm_sequence_native_supported,
)
from pycbc.waveform.imrphenomxhm_mode21_torch import (  # noqa: E402
    _amplitude_21,
    _mode21_state,
)
from pycbc.waveform.imrphenomxhm_mode21_2019_torch import (  # noqa: E402
    _crosses_zero,
)
from pycbc.waveform.imrphenomxhm_mode32_torch import (  # noqa: E402
    _mode32_state,
    imrphenomxhm_h3m2_samples,
)
from pycbc.waveform.imrphenomxhm_mode33_torch import (  # noqa: E402
    _amplitude_33,
    _mode33_state,
)
from pycbc.waveform.imrphenomxhm_mode44_torch import (  # noqa: E402
    _amplitude_44,
    _mode44_state,
)
from pycbc.waveform.waveform_modes import get_fd_waveform_modes  # noqa: E402


_NATIVE_FLAG_ENVS = (
    "PYCBC_TORCH_NATIVE_PORTS",
    "PYCBC_TORCH_NATIVE",
    "PYCBC_IMRPHENOMXHM_NATIVE",
)


def _clear_native_flags(monkeypatch):
    """Remove every native flag so the registry default applies."""
    for name in _NATIVE_FLAG_ENVS:
        monkeypatch.delenv(name, raising=False)


CASES = [
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=300.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=[(2, 2), (2, -2)],
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
        coa_phase=0.6,
        mode_array=[(2, 2)],
    ),
    dict(
        mass1=35.0,
        mass2=28.0,
        spin1z=0.2,
        spin2z=-0.1,
        delta_f=0.5,
        f_lower=20.0,
        f_ref=30.0,
        distance=500.0,
        coa_phase=1.1,
        mode_array=[(2, -2)],
    ),
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=300.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=[(2, -2), (2, -1), (2, 1)],
    ),
    dict(
        mass1=17.0,
        mass2=43.0,
        spin1z=-0.45,
        spin2z=0.65,
        delta_f=0.5,
        f_lower=18.0,
        f_final=220.0,
        f_ref=0.0,
        distance=800.0,
        coa_phase=0.2,
        mode_array=[(2, -1)],
    ),
    dict(
        mass1=54.0,
        mass2=9.0,
        spin1z=0.92,
        spin2z=0.7,
        delta_f=0.5,
        f_lower=15.0,
        f_final=250.0,
        f_ref=30.0,
        distance=600.0,
        coa_phase=0.8,
        mode_array=[(2, -1)],
    ),
    dict(
        mass1=80.0,
        mass2=3.0,
        spin1z=0.6,
        spin2z=-0.4,
        delta_f=0.25,
        f_lower=10.0,
        f_final=150.0,
        f_ref=20.0,
        distance=900.0,
        coa_phase=0.3,
        mode_array=[(2, 1)],
    ),
    dict(
        mass1=30.0,
        mass2=30.0,
        spin1z=0.3,
        spin2z=0.3,
        delta_f=1.0,
        f_lower=20.0,
        f_final=250.0,
        f_ref=20.0,
        distance=500.0,
        coa_phase=0.0,
        mode_array=[(2, -1), (2, 1)],
    ),
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=500.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=[(2, -2), (2, -1), (3, -3), (3, 3)],
    ),
    dict(
        mass1=17.0,
        mass2=43.0,
        spin1z=-0.45,
        spin2z=0.65,
        delta_f=0.5,
        f_lower=18.0,
        f_final=620.0,
        f_ref=0.0,
        distance=800.0,
        coa_phase=0.2,
        mode_array=[(3, -3)],
    ),
    dict(
        mass1=600.0 / 11.0,
        mass2=60.0 / 11.0,
        spin1z=0.98,
        spin2z=0.8,
        delta_f=1.0,
        f_lower=15.0,
        f_final=800.0,
        f_ref=25.0,
        distance=500.0,
        coa_phase=0.37,
        mode_array=[(3, 3)],
    ),
    dict(
        mass1=30.0,
        mass2=30.0,
        spin1z=0.3,
        spin2z=0.3,
        delta_f=1.0,
        f_lower=20.0,
        f_final=600.0,
        f_ref=20.0,
        distance=500.0,
        coa_phase=0.0,
        mode_array=[(3, -3), (3, 3)],
    ),
    dict(
        mass1=30.0,
        mass2=30.0,
        spin1z=0.6,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=600.0,
        f_ref=20.0,
        distance=500.0,
        coa_phase=0.0,
        mode_array=[(3, -3)],
    ),
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=700.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=[(2, -2), (3, -3), (4, -4), (4, 4)],
    ),
    dict(
        mass1=17.0,
        mass2=43.0,
        spin1z=-0.45,
        spin2z=0.65,
        delta_f=0.5,
        f_lower=18.0,
        f_ref=0.0,
        distance=800.0,
        coa_phase=0.2,
        mode_array=[(4, -4)],
    ),
    dict(
        mass1=600.0 / 11.0,
        mass2=60.0 / 11.0,
        spin1z=0.98,
        spin2z=0.8,
        delta_f=1.0,
        f_lower=15.0,
        f_final=900.0,
        f_ref=25.0,
        distance=500.0,
        coa_phase=0.37,
        mode_array=[(4, 4)],
    ),
    dict(
        mass1=30.0,
        mass2=30.0,
        spin1z=0.3,
        spin2z=0.3,
        delta_f=1.0,
        f_lower=20.0,
        f_final=800.0,
        f_ref=20.0,
        distance=500.0,
        coa_phase=0.0,
        mode_array=[(4, -4), (4, 4)],
    ),
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=700.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=[(3, -2), (3, 2)],
    ),
    dict(
        mass1=17.0,
        mass2=43.0,
        spin1z=-0.45,
        spin2z=0.65,
        delta_f=0.5,
        f_lower=18.0,
        f_final=620.0,
        f_ref=0.0,
        distance=800.0,
        coa_phase=0.2,
        mode_array=[(3, -2)],
    ),
    dict(
        mass1=600.0 / 11.0,
        mass2=60.0 / 11.0,
        spin1z=0.98,
        spin2z=0.8,
        delta_f=1.0,
        f_lower=15.0,
        f_final=900.0,
        f_ref=25.0,
        distance=500.0,
        coa_phase=0.37,
        mode_array=[(3, 2)],
    ),
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=700.0,
        f_ref=25.0,
        distance=350.0,
        coa_phase=0.4,
        mode_array=None,
    ),
]

POLARIZATION_CASES = [
    dict(
        mass1=46.0,
        mass2=19.0,
        spin1z=0.35,
        spin2z=-0.2,
        delta_f=1.0,
        f_lower=20.0,
        f_final=700.0,
        f_ref=25.0,
        distance=350.0,
        inclination=0.9,
        coa_phase=0.4,
        long_asc_nodes=0.37,
    ),
    dict(
        mass1=17.0,
        mass2=43.0,
        spin1z=-0.45,
        spin2z=0.65,
        delta_f=0.5,
        f_lower=18.0,
        f_final=620.0,
        f_ref=0.0,
        distance=800.0,
        inclination=1.2,
        coa_phase=0.2,
        long_asc_nodes=-0.21,
        mode_array=[(2, -2), (2, 1), (3, -3), (3, 2), (4, 4)],
    ),
    dict(
        mass1=600.0 / 11.0,
        mass2=60.0 / 11.0,
        spin1z=0.98,
        spin2z=0.8,
        delta_f=1.0,
        f_lower=15.0,
        f_final=900.0,
        f_ref=25.0,
        distance=500.0,
        inclination=0.6,
        coa_phase=0.37,
        mode_array=[(3, -2), (3, 2)],
    ),
]


SEQUENCE_CASES = [
    (
        dict(
            mass1=46.0,
            mass2=19.0,
            spin1z=0.35,
            spin2z=-0.2,
            distance=350.0,
            inclination=0.7,
            coa_phase=0.4,
            f_ref=25.0,
        ),
        [20.0, 31.5, 80.0, 180.0, 500.0, 900.0, 1000.0, 2000.0],
        5.0e-4,
    ),
    (
        dict(
            mass1=43.0,
            mass2=17.0,
            spin1z=0.65,
            spin2z=-0.45,
            distance=800.0,
            inclination=1.2,
            coa_phase=0.2,
            f_ref=0.0,
            long_asc_nodes=0.91,
            mode_array=[(2, -2), (2, 1), (3, -3), (3, 2), (4, 4)],
        ),
        [17.3, 500.0, 22.0, 150.0],
        5.0e-3,
    ),
    (
        dict(
            mass1=600.0 / 11.0,
            mass2=60.0 / 11.0,
            spin1z=0.98,
            spin2z=0.8,
            distance=500.0,
            inclination=0.6,
            coa_phase=0.37,
            f_ref=25.0,
            mode_array=[(3, -2), (3, 2)],
        ),
        [15.0, 25.0, 80.0, 250.0, 900.0],
        5.0e-2,
    ),
]


def test_mode21_ringdown_frequency_overrides_are_isolated():
    params = CASES[3]
    baseline = _mode21_state(params)
    overridden = _mode21_state(
        params,
        ringdown_frequency=0.091,
        damping_frequency=torch.tensor(0.014, dtype=torch.float64),
        carrier_ringdown_frequency=torch.tensor(0.073, dtype=torch.float64),
        carrier_damping_frequency=0.012,
    )

    assert overridden.f_ring_21 == pytest.approx(0.091)
    assert overridden.f_damp_21 == pytest.approx(0.014)
    assert overridden.f_ring_22 == pytest.approx(0.073)
    assert overridden.f_damp_22 == pytest.approx(0.012)
    for name in baseline.__dataclass_fields__:
        if name not in {"f_ring_21", "f_damp_21", "f_ring_22", "f_damp_22"}:
            assert getattr(overridden, name) == getattr(baseline, name)


def test_mode33_ringdown_frequency_overrides_are_isolated():
    params = CASES[3]
    baseline = _mode33_state(params)
    overridden = _mode33_state(
        params,
        ringdown_frequency=0.121,
        damping_frequency=torch.tensor(0.017, dtype=torch.float64),
        carrier_ringdown_frequency=torch.tensor(0.073, dtype=torch.float64),
        carrier_damping_frequency=0.012,
    )

    assert overridden.f_ring_33 == pytest.approx(0.121)
    assert overridden.f_damp_33 == pytest.approx(0.017)
    assert overridden.f_ring_22 == pytest.approx(0.073)
    assert overridden.f_damp_22 == pytest.approx(0.012)
    for name in baseline.base.__dataclass_fields__:
        if name not in {"f_ring_22", "f_damp_22"}:
            assert getattr(overridden.base, name) == getattr(baseline.base, name)


def test_mode32_ringdown_frequency_overrides_are_isolated():
    params = CASES[3]
    baseline = _mode32_state(params)
    overridden = _mode32_state(
        params,
        ringdown_frequency=0.101,
        damping_frequency=torch.tensor(0.016, dtype=torch.float64),
        carrier_ringdown_frequency=torch.tensor(0.073, dtype=torch.float64),
        carrier_damping_frequency=0.012,
    )

    assert overridden.f_ring_32 == pytest.approx(0.101)
    assert overridden.f_damp_32 == pytest.approx(0.016)
    assert overridden.f_ring_22 == pytest.approx(0.073)
    assert overridden.f_damp_22 == pytest.approx(0.012)
    for name in baseline.base.__dataclass_fields__:
        if name not in {"f_ring_22", "f_damp_22"}:
            assert getattr(overridden.base, name) == getattr(baseline.base, name)


def test_mode44_ringdown_frequency_overrides_are_isolated():
    params = CASES[3]
    baseline = _mode44_state(params)
    overridden = _mode44_state(
        params,
        ringdown_frequency=0.141,
        damping_frequency=torch.tensor(0.019, dtype=torch.float64),
        carrier_ringdown_frequency=torch.tensor(0.073, dtype=torch.float64),
        carrier_damping_frequency=0.012,
    )

    assert overridden.f_ring_44 == pytest.approx(0.141)
    assert overridden.f_damp_44 == pytest.approx(0.019)
    assert overridden.f_ring_22 == pytest.approx(0.073)
    assert overridden.f_damp_22 == pytest.approx(0.012)
    for name in baseline.base.__dataclass_fields__:
        if name not in {"f_ring_22", "f_damp_22"}:
            assert getattr(overridden.base, name) == getattr(baseline.base, name)


@pytest.mark.parametrize(
    ("state_builder", "keyword", "value", "message"),
    [
        (_mode21_state, "ringdown_frequency", 0.0, "mode ringdown frequency"),
        (
            _mode33_state,
            "ringdown_frequency",
            float("inf"),
            "mode ringdown frequency",
        ),
        (
            _mode33_state,
            "damping_frequency",
            -0.1,
            "mode damping frequency",
        ),
        (
            _mode32_state,
            "ringdown_frequency",
            float("nan"),
            "mode ringdown frequency",
        ),
        (
            _mode32_state,
            "damping_frequency",
            0.0,
            "mode damping frequency",
        ),
        (
            _mode44_state,
            "ringdown_frequency",
            float("inf"),
            "mode ringdown frequency",
        ),
        (
            _mode44_state,
            "damping_frequency",
            -0.1,
            "mode damping frequency",
        ),
        (
            _mode21_state,
            "carrier_ringdown_frequency",
            float("nan"),
            "carrier ringdown frequency",
        ),
        (
            _mode33_state,
            "carrier_damping_frequency",
            float("inf"),
            "carrier damping frequency",
        ),
    ],
)
def test_higher_modes_reject_invalid_ringdown_frequency_overrides(
    state_builder,
    keyword,
    value,
    message,
):
    with pytest.raises(ValueError, match=message):
        state_builder(CASES[3], **{keyword: value})


def test_active_mode_samples_forwards_mode21_ringdown_overrides(monkeypatch):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch

    captured = {}

    def recording_mode(core, _params, **kwargs):
        captured.update(kwargs)
        return torch.ones_like(core.polarization)

    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_h2m1_samples",
        recording_mode,
    )
    core = _SequenceCore(torch.zeros(4, dtype=torch.complex128))
    mode_ringdown = torch.tensor(0.091, dtype=torch.float64)
    mode_damping = torch.tensor(0.014, dtype=torch.float64)
    carrier_ringdown = torch.tensor(0.073, dtype=torch.float64)
    carrier_damping = torch.tensor(0.012, dtype=torch.float64)
    carrier_deviations = object()

    active_modes = _active_mode_samples(
        core,
        {},
        [(2, 1)],
        ringdown_frequencies={(2, 1): mode_ringdown},
        damping_frequencies={(2, 1): mode_damping},
        carrier_ringdown_frequency=carrier_ringdown,
        carrier_damping_frequency=carrier_damping,
        carrier_coprecessing_deviations=carrier_deviations,
        mode21_amplitude_release=122019,
    )

    assert active_modes.keys() == {(2, 1)}
    assert captured["ringdown_frequency"] is mode_ringdown
    assert captured["damping_frequency"] is mode_damping
    assert captured["carrier_ringdown_frequency"] is carrier_ringdown
    assert captured["carrier_damping_frequency"] is carrier_damping
    assert captured["carrier_coprecessing_deviations"] is carrier_deviations
    assert captured["amplitude_release"] == 122019


def test_active_mode_samples_forwards_mode33_overrides(monkeypatch):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch

    captured = {}

    def recording_mode(core, _params, **kwargs):
        captured.update(kwargs)
        return torch.ones_like(core.polarization)

    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_h3m3_samples",
        recording_mode,
    )
    core = _SequenceCore(torch.zeros(4, dtype=torch.complex128))
    mode_ringdown = torch.tensor(0.121, dtype=torch.float64)
    mode_damping = torch.tensor(0.017, dtype=torch.float64)
    carrier_ringdown = torch.tensor(0.073, dtype=torch.float64)
    carrier_damping = torch.tensor(0.012, dtype=torch.float64)
    carrier_deviations = object()

    active_modes = _active_mode_samples(
        core,
        {},
        [(3, 3)],
        ringdown_frequencies={(3, 3): mode_ringdown},
        damping_frequencies={(3, 3): mode_damping},
        carrier_ringdown_frequency=carrier_ringdown,
        carrier_damping_frequency=carrier_damping,
        carrier_coprecessing_deviations=carrier_deviations,
        mode33_amplitude_release=122019,
    )

    assert active_modes.keys() == {(3, 3)}
    assert captured["ringdown_frequency"] is mode_ringdown
    assert captured["damping_frequency"] is mode_damping
    assert captured["carrier_ringdown_frequency"] is carrier_ringdown
    assert captured["carrier_damping_frequency"] is carrier_damping
    assert captured["carrier_coprecessing_deviations"] is carrier_deviations
    assert captured["amplitude_release"] == 122019


def test_active_mode_samples_forwards_mode32_overrides(monkeypatch):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch

    captured = {}

    def recording_mode(core, _params, **kwargs):
        captured.update(kwargs)
        return torch.ones_like(core.polarization)

    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_h3m2_samples",
        recording_mode,
    )
    core = _SequenceCore(torch.zeros(4, dtype=torch.complex128))
    mode_ringdown = torch.tensor(0.101, dtype=torch.float64)
    mode_damping = torch.tensor(0.016, dtype=torch.float64)
    carrier_ringdown = torch.tensor(0.073, dtype=torch.float64)
    carrier_damping = torch.tensor(0.012, dtype=torch.float64)
    carrier_deviations = object()

    active_modes = _active_mode_samples(
        core,
        {},
        [(3, 2)],
        ringdown_frequencies={(3, 2): mode_ringdown},
        damping_frequencies={(3, 2): mode_damping},
        carrier_ringdown_frequency=carrier_ringdown,
        carrier_damping_frequency=carrier_damping,
        carrier_coprecessing_deviations=carrier_deviations,
        mode32_amplitude_release=122019,
    )

    assert active_modes.keys() == {(3, 2)}
    assert captured["ringdown_frequency"] is mode_ringdown
    assert captured["damping_frequency"] is mode_damping
    assert captured["carrier_ringdown_frequency"] is carrier_ringdown
    assert captured["carrier_damping_frequency"] is carrier_damping
    assert captured["carrier_coprecessing_deviations"] is carrier_deviations
    assert captured["amplitude_release"] == 122019


def test_active_mode_samples_forwards_mode44_overrides(monkeypatch):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch

    captured = {}

    def recording_mode(core, _params, **kwargs):
        captured.update(kwargs)
        return torch.ones_like(core.polarization)

    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_h4m4_samples",
        recording_mode,
    )
    core = _SequenceCore(torch.zeros(4, dtype=torch.complex128))
    mode_ringdown = torch.tensor(0.141, dtype=torch.float64)
    mode_damping = torch.tensor(0.019, dtype=torch.float64)
    carrier_ringdown = torch.tensor(0.073, dtype=torch.float64)
    carrier_damping = torch.tensor(0.012, dtype=torch.float64)
    carrier_deviations = object()

    active_modes = _active_mode_samples(
        core,
        {},
        [(4, 4)],
        ringdown_frequencies={(4, 4): mode_ringdown},
        damping_frequencies={(4, 4): mode_damping},
        carrier_ringdown_frequency=carrier_ringdown,
        carrier_damping_frequency=carrier_damping,
        carrier_coprecessing_deviations=carrier_deviations,
        mode44_amplitude_release=122019,
    )

    assert active_modes.keys() == {(4, 4)}
    assert captured["ringdown_frequency"] is mode_ringdown
    assert captured["damping_frequency"] is mode_damping
    assert captured["carrier_ringdown_frequency"] is carrier_ringdown
    assert captured["carrier_damping_frequency"] is carrier_damping
    assert captured["carrier_coprecessing_deviations"] is carrier_deviations
    assert captured["amplitude_release"] == 122019


def test_phase_anchor_cache_switch_is_strict_and_defaults_off(monkeypatch):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch

    monkeypatch.delenv(xhm_torch._PHASE_ANCHOR_CACHE_ENV, raising=False)
    assert not xhm_torch._phase_anchor_cache_enabled()
    monkeypatch.setenv(xhm_torch._PHASE_ANCHOR_CACHE_ENV, "maybe")
    with pytest.raises(ValueError, match=xhm_torch._PHASE_ANCHOR_CACHE_ENV):
        xhm_torch._phase_anchor_cache_enabled()


def test_phase_anchor_cache_is_recommended_without_batched_tiny_solves():
    benchmark = pytest.importorskip("tools.torch_workflows.experiments.benchmark_xphm_exact_tricks")

    phase_cache = "PYCBC_IMRPHENOMXHM_PHASE_ANCHOR_CACHE"
    tiny_solves = "PYCBC_IMRPHENOMXHM_BATCHED_TINY_SOLVES"

    assert phase_cache in benchmark._SWITCHES
    assert tiny_solves in benchmark._SWITCHES
    assert phase_cache in benchmark._PR_STYLE_EXACT_SWITCHES
    assert tiny_solves not in benchmark._PR_STYLE_EXACT_SWITCHES
    assert benchmark._VARIANTS["pr_style_exact"][phase_cache] == "1"
    assert benchmark._VARIANTS["pr_style_exact"][tiny_solves] == "0"
    assert benchmark._VARIANTS["torch213_cpu_candidate"][phase_cache] == "1"
    assert benchmark._VARIANTS["torch213_cpu_candidate"][tiny_solves] == "0"
    assert benchmark._VARIANTS["batched_tiny_solves_candidate"][tiny_solves] == "1"


def test_phase_anchor_cache_is_request_local_and_bypasses_autograd(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    anchors = []

    def recording_mode(core, _params, **kwargs):
        anchors.append(kwargs["carrier_phase_anchors"])
        return torch.ones_like(core.polarization)

    for name in (
        "imrphenomxhm_h2m1_samples",
        "imrphenomxhm_h3m3_samples",
        "imrphenomxhm_h3m2_samples",
        "imrphenomxhm_h4m4_samples",
    ):
        monkeypatch.setattr(xhm_torch, name, recording_mode)
    monkeypatch.setenv(xhm_torch._PHASE_ANCHOR_CACHE_ENV, "1")
    modes = [(2, 1), (3, 3), (3, 2), (4, 4)]

    core = _SequenceCore(torch.zeros(4, dtype=torch.complex128))
    _active_mode_samples(core, {}, modes)
    first_request = tuple(anchors)
    anchors.clear()
    _active_mode_samples(core, {}, modes)
    second_request = tuple(anchors)

    assert first_request[0] is not None
    assert all(anchor is first_request[0] for anchor in first_request)
    assert all(anchor is second_request[0] for anchor in second_request)
    assert first_request[0] is not second_request[0]

    anchors.clear()
    grad_core = _SequenceCore(
        torch.zeros(4, dtype=torch.complex128, requires_grad=True)
    )
    _active_mode_samples(grad_core, {}, modes)
    assert anchors == [None, None, None, None]

    if torch.cuda.is_available():
        anchors.clear()
        _activate_scheme(_scheme.TorchScheme("cuda"))
        cuda_core = _SequenceCore(
            torch.zeros(4, dtype=torch.complex128, device="cuda")
        )
        _active_mode_samples(cuda_core, {}, modes)
        assert anchors[0] is not None
        assert all(anchor is anchors[0] for anchor in anchors)


def test_phase_anchor_runtime_guard_precedes_autograd_scan(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setenv(xhm_torch._PHASE_ANCHOR_CACHE_ENV, "1")
    monkeypatch.setenv(xhm_torch._BATCHED_TINY_SOLVES_ENV, "0")
    # An explicitly requested eager-only executor must still leave an outer
    # torch.compile graph on the established implementation without a break.
    monkeypatch.setenv(
        "PYCBC_IMRPHENOMXHM_SCRIPTED_PHASE_TRIPLET",
        "1",
    )
    monkeypatch.setattr(
        xhm_torch,
        "_plain_request_runtime_supported",
        lambda: False,
    )

    def unexpected_scan(*_args, **_kwargs):
        pytest.fail("unsupported runtime reached the autograd scanner")

    anchors = []

    def recording_mode(core, _params, **kwargs):
        anchors.append(kwargs["carrier_phase_anchors"])
        return torch.ones_like(core.polarization)

    monkeypatch.setattr(
        xhm_torch,
        "_phase_anchor_inputs_have_autograd",
        unexpected_scan,
    )
    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_h2m1_samples",
        recording_mode,
    )

    core = _SequenceCore(torch.zeros(4, dtype=torch.complex128))
    _active_mode_samples(core, {}, [(2, 1)])
    assert anchors == [None]


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile unavailable")
def test_phase_anchor_cache_real_compile_bypasses_cold_and_warm(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch

    _activate_scheme(_scheme.TorchScheme("cpu"))
    monkeypatch.setenv(xhm_torch._PHASE_ANCHOR_CACHE_ENV, "1")
    monkeypatch.setenv(xhm_torch._BATCHED_TINY_SOLVES_ENV, "0")

    # Isolate the real cache activation in the dispatcher from unrelated
    # optional executors.  A (2, 2)-only request has no higher-mode kernel.
    for name in (
        "_carrier_inspiral_lane_supported",
        "_shared_mode_inputs_supported",
        "_shared_carrier_inspiral_phase_supported",
        "_fixed_schema_amplitude_triplet_supported",
        "_batched_tiny_solves_supported",
    ):
        monkeypatch.setattr(xhm_torch, name, lambda *_args, **_kwargs: False)

    def unexpected_cache_use(*_args, **_kwargs):
        pytest.fail("torch.compile reached the eager-only phase-anchor cache")

    monkeypatch.setattr(
        xhm_torch,
        "_phase_anchor_inputs_have_autograd",
        unexpected_cache_use,
    )
    monkeypatch.setattr(
        xhm_torch,
        "_CarrierPhaseAnchors",
        unexpected_cache_use,
    )

    def mode22_only(polarization):
        core = xhm_torch._SequenceCore(polarization)
        return xhm_torch._active_mode_samples_serial_impl(
            core,
            {},
            [(2, 2)],
        )[2, 2]

    polarization = torch.arange(8, dtype=torch.float64).to(torch.complex128)
    reference = polarization / xhm_torch._XAS_MODE_POLARIZATION_FACTOR
    torch._dynamo.reset()
    try:
        compiled = torch.compile(mode22_only, backend="eager", fullgraph=True)
        cold = compiled(polarization)
        warm = compiled(polarization.clone())
    finally:
        torch._dynamo.reset()

    reference_bits = reference.contiguous().view(torch.int64)
    assert torch.equal(cold.contiguous().view(torch.int64), reference_bits)
    assert torch.equal(warm.contiguous().view(torch.int64), reference_bits)


def test_phase_anchor_cache_bypasses_only_cuda_alignment():
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    like = torch.zeros((), dtype=torch.float64, device=device)
    anchors = xhm_torch._CarrierPhaseAnchors()
    calls = {"alignment_phase": 0, "reference_phase": 0}

    def value(name):
        def factory():
            calls[name] += 1
            return like + calls[name]

        return anchors.get_or_compute(name, like, factory)

    alignment_first = value("alignment_phase")
    alignment_second = value("alignment_phase")
    reference_first = value("reference_phase")
    reference_second = value("reference_phase")

    if like.device.type == "cuda":
        assert calls["alignment_phase"] == 2
        assert not torch.equal(alignment_first, alignment_second)
    else:
        assert calls["alignment_phase"] == 1
        assert torch.equal(alignment_first, alignment_second)
    assert calls["reference_phase"] == 1
    assert torch.equal(reference_first, reference_second)


@pytest.mark.parametrize(
    ("mass1", "mass2", "spin1z", "spin2z"),
    [
        (40.0, 20.0, 0.2, -0.05),
        (45.0, 3.0, 0.3, -0.7),
    ],
)
def test_mode21_2019_amplitude_matches_lal(
    mass1,
    mass2,
    spin1z,
    spin2z,
):
    params = {
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "distance": 400.0,
        "f_lower": 15.0,
        "f_final": 600.0,
        "delta_f": 0.5,
        "coa_phase": 0.3,
        "f_ref": 25.0,
    }
    state = _mode21_state(params)
    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(
        lal_params,
        122019,
    )
    reference = lalsimulation.SimIMRPhenomXHMGenerateFDOneMode(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        spin1z,
        spin2z,
        2,
        -1,
        params["distance"] * 1.0e6 * lal.PC_SI,
        params["f_lower"],
        params["f_final"],
        params["delta_f"],
        params["coa_phase"],
        params["f_ref"],
        lal_params,
    )
    expected = np.abs(np.asarray(reference.data.data)) / state.amp0
    frequencies = torch.arange(len(expected), dtype=torch.float64) * params["delta_f"]
    active = (frequencies >= params["f_lower"]) & torch.as_tensor(expected > 0.0)
    actual = _amplitude_21(
        frequencies[active] * state.total_mass_seconds,
        state,
        122019,
    ).numpy()
    expected = expected[active.numpy()]
    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error < 1.0e-8


def test_legacy_quintic_zero_check_matches_lal_packed_root_order():
    coefficients = torch.tensor(
        [
            -0.04377046076326342,
            17.764539275243695,
            -171.07235541921517,
            1733.1713489529357,
            -14098.847020817848,
            47879.633711294,
        ],
        dtype=torch.float64,
    )

    assert _crosses_zero(
        coefficients,
        0.07707914689970682,
        0.16880765424728308,
    )


@pytest.mark.parametrize(
    ("mass1", "mass2", "spin1z", "spin2z"),
    [
        (40.0, 20.0, 0.2, -0.05),
        (45.0, 3.0, 0.3, -0.7),
        (33.0, 30.0, -0.2, 0.3),
        (40.0, 4.0, 0.97, -0.2),
        (50.0, 1.0, 0.93, -0.2),
    ],
)
def test_mode33_2019_amplitude_matches_lal(
    mass1,
    mass2,
    spin1z,
    spin2z,
):
    params = {
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "distance": 400.0,
        "f_lower": 15.0,
        "f_final": 600.0,
        "delta_f": 0.5,
        "coa_phase": 0.3,
        "f_ref": 25.0,
    }
    state = _mode33_state(params)
    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(
        lal_params,
        122019,
    )
    reference = lalsimulation.SimIMRPhenomXHMGenerateFDOneMode(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        spin1z,
        spin2z,
        3,
        -3,
        params["distance"] * 1.0e6 * lal.PC_SI,
        params["f_lower"],
        params["f_final"],
        params["delta_f"],
        params["coa_phase"],
        params["f_ref"],
        lal_params,
    )
    expected = np.abs(np.asarray(reference.data.data)) / state.amp0
    frequencies = torch.arange(len(expected), dtype=torch.float64) * params["delta_f"]
    active = (frequencies >= params["f_lower"]) & torch.as_tensor(expected > 0.0)
    actual = _amplitude_33(
        frequencies[active] * state.total_mass_seconds,
        state,
        122019,
    ).numpy()
    expected = expected[active.numpy()]
    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error < 1.0e-8


@pytest.mark.parametrize(
    ("mass1", "mass2", "spin1z", "spin2z"),
    [
        (40.0, 20.0, 0.2, -0.05),
        (45.0, 3.0, 0.3, -0.7),
        (33.0, 30.0, -0.2, 0.3),
        (40.0, 4.0, 0.97, -0.2),
        (50.0, 1.0, 0.93, -0.2),
    ],
)
def test_mode44_2019_amplitude_matches_lal(
    mass1,
    mass2,
    spin1z,
    spin2z,
):
    params = {
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "distance": 400.0,
        "f_lower": 15.0,
        "f_final": 600.0,
        "delta_f": 0.5,
        "coa_phase": 0.3,
        "f_ref": 25.0,
    }
    state = _mode44_state(params)
    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(
        lal_params,
        122019,
    )
    reference = lalsimulation.SimIMRPhenomXHMGenerateFDOneMode(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        spin1z,
        spin2z,
        4,
        -4,
        params["distance"] * 1.0e6 * lal.PC_SI,
        params["f_lower"],
        params["f_final"],
        params["delta_f"],
        params["coa_phase"],
        params["f_ref"],
        lal_params,
    )
    expected = np.abs(np.asarray(reference.data.data)) / state.amp0
    frequencies = torch.arange(len(expected), dtype=torch.float64) * params["delta_f"]
    active = (frequencies >= params["f_lower"]) & torch.as_tensor(expected > 0.0)
    actual = _amplitude_44(
        frequencies[active] * state.total_mass_seconds,
        state,
        122019,
    ).numpy()
    expected = expected[active.numpy()]
    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error < 2.0e-8


@pytest.mark.parametrize(
    ("mass1", "mass2", "spin1z", "spin2z"),
    [
        (40.0, 20.0, 0.2, -0.05),
        (45.0, 3.0, 0.3, -0.7),
        (33.0, 30.0, -0.2, 0.3),
        (40.0, 4.0, 0.97, -0.2),
        (50.0, 1.0, 0.93, -0.2),
        (40.0, 10.0, -0.7, 0.2),
        (40.0, 10.0, -0.95, -0.95),
    ],
)
def test_mode32_2019_amplitude_matches_lal(
    mass1,
    mass2,
    spin1z,
    spin2z,
):
    params = {
        "mass1": mass1,
        "mass2": mass2,
        "spin1z": spin1z,
        "spin2z": spin2z,
        "distance": 400.0,
        "f_lower": 15.0,
        "f_final": 600.0,
        "delta_f": 0.5,
        "coa_phase": 0.3,
        "f_ref": 25.0,
    }
    state = _mode32_state(params)
    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertPhenomXHMReleaseVersion(
        lal_params,
        122019,
    )
    reference = lalsimulation.SimIMRPhenomXHMGenerateFDOneMode(
        mass1 * lal.MSUN_SI,
        mass2 * lal.MSUN_SI,
        spin1z,
        spin2z,
        3,
        -2,
        params["distance"] * 1.0e6 * lal.PC_SI,
        params["f_lower"],
        params["f_final"],
        params["delta_f"],
        params["coa_phase"],
        params["f_ref"],
        lal_params,
    )
    expected = np.asarray(reference.data.data)
    frequencies = torch.arange(len(expected), dtype=torch.float64) * params["delta_f"]
    active = (frequencies >= params["f_lower"]) & torch.as_tensor(
        np.abs(expected) > 0.0
    )
    core = _SequenceCore(torch.zeros(1, dtype=torch.complex128))
    actual = imrphenomxhm_h3m2_samples(
        core,
        params,
        frequencies=frequencies[active],
        amplitude_release=122019,
    )
    actual = torch.abs(actual).numpy() / state.amp0
    expected = np.abs(expected[active.numpy()]) / state.amp0
    relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert relative_error < 3.0e-7


def test_mode32_2019_amplitude_rejects_unported_two_region_branch():
    params = dict(CASES[3], mass1=72.0, mass2=1.0, spin1z=0.8)
    core = _SequenceCore(torch.zeros(1, dtype=torch.complex128))

    with pytest.raises(ValueError, match="q > 70 two-region branch"):
        imrphenomxhm_h3m2_samples(
            core,
            params,
            frequencies=torch.tensor([20.0], dtype=torch.float64),
            amplitude_release=122019,
        )


def test_mode33_2019_amplitude_rejects_unported_two_region_branch():
    params = dict(CASES[3], mass1=72.0, mass2=1.0, spin1z=0.8)
    state = _mode33_state(params)

    with pytest.raises(ValueError, match="q > 70 two-region branch"):
        _amplitude_33(torch.tensor([0.01], dtype=torch.float64), state, 122019)


def test_mode44_2019_amplitude_rejects_unported_two_region_branch():
    params = dict(CASES[3], mass1=72.0, mass2=1.0, spin1z=0.8)
    state = _mode44_state(params)

    with pytest.raises(ValueError, match="q > 70 two-region branch"):
        _amplitude_44(torch.tensor([0.01], dtype=torch.float64), state, 122019)


def test_mode33_default_amplitude_release_is_2022():
    state = _mode33_state(CASES[3])
    frequencies = torch.tensor([20.0, 50.0, 200.0], dtype=torch.float64)
    mf = frequencies * state.total_mass_seconds

    torch.testing.assert_close(
        _amplitude_33(mf, state),
        _amplitude_33(mf, state, 122022),
        rtol=0.0,
        atol=0.0,
    )


def test_mode44_default_amplitude_release_is_2022():
    state = _mode44_state(CASES[3])
    frequencies = torch.tensor([20.0, 50.0, 200.0], dtype=torch.float64)
    mf = frequencies * state.total_mass_seconds

    torch.testing.assert_close(
        _amplitude_44(mf, state),
        _amplitude_44(mf, state, 122022),
        rtol=0.0,
        atol=0.0,
    )


def test_mode21_default_amplitude_release_is_2022():
    state = _mode21_state(CASES[3])
    frequencies = torch.tensor([20.0, 50.0, 200.0], dtype=torch.float64)
    mf = frequencies * state.total_mass_seconds

    torch.testing.assert_close(
        _amplitude_21(mf, state),
        _amplitude_21(mf, state, 122022),
        rtol=0.0,
        atol=0.0,
    )


def test_mode32_default_amplitude_release_is_2022():
    params = CASES[3]
    core = _SequenceCore(torch.zeros(1, dtype=torch.complex128))
    frequencies = torch.tensor([20.0, 50.0, 200.0], dtype=torch.float64)

    actual = imrphenomxhm_h3m2_samples(core, params, frequencies=frequencies)
    expected = imrphenomxhm_h3m2_samples(
        core,
        params,
        frequencies=frequencies,
        amplitude_release=122022,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("release", [True, 122020, "122019", None, []])
def test_mode21_rejects_invalid_amplitude_release(release):
    state = _mode21_state(CASES[3])
    mf = torch.tensor([0.01], dtype=torch.float64)

    with pytest.raises(ValueError, match="amplitude_release"):
        _amplitude_21(mf, state, release)


@pytest.mark.parametrize("release", [True, 122020, "122019", None, []])
def test_mode32_rejects_invalid_amplitude_release(release):
    core = _SequenceCore(torch.zeros(1, dtype=torch.complex128))

    with pytest.raises(ValueError, match="amplitude_release"):
        imrphenomxhm_h3m2_samples(
            core,
            CASES[3],
            frequencies=torch.tensor([20.0], dtype=torch.float64),
            amplitude_release=release,
        )


@pytest.mark.parametrize("release", [True, 122020, "122019", None, []])
def test_mode44_rejects_invalid_amplitude_release(release):
    state = _mode44_state(CASES[3])
    mf = torch.tensor([0.01], dtype=torch.float64)

    with pytest.raises(ValueError, match="amplitude_release"):
        _amplitude_44(mf, state, release)


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


@pytest.mark.parametrize("params", CASES)
def test_imrphenomxhm_native_modes_match_lal(params, monkeypatch, preserve_scheme):
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)
    reference_arrays = {
        mode: tuple(series.numpy().copy() for series in polarizations)
        for mode, polarizations in reference.items()
    }

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)

    assert actual.keys() == reference.keys()
    for mode, polarizations in actual.items():
        for expected, expected_array, result in zip(
            reference[mode], reference_arrays[mode], polarizations
        ):
            assert len(result) == len(expected)
            assert result.delta_f == expected.delta_f
            assert float(result.epoch) == float(expected.epoch)
            assert result._data.tensor.device.type == "cpu"
            assert result._data.tensor.dtype == torch.complex128
            np.testing.assert_array_equal(
                result.numpy() == 0.0,
                expected_array == 0.0,
            )
            nonzero = np.abs(expected_array) > 0.0
            if not np.any(nonzero):
                continue
            relative_error = np.linalg.norm(
                result.numpy()[nonzero] - expected_array[nonzero]
            ) / np.linalg.norm(expected_array[nonzero])
            if mode[0] == 3 and abs(mode[1]) == 2:
                mass_ratio = max(params["mass1"], params["mass2"]) / min(
                    params["mass1"], params["mass2"]
                )
                # LAL obtains the mixed-mode phase curvature from a 1e-7
                # finite difference. The native path uses its stable analytic
                # derivative, so LAL's amplified roundoff is visible near the
                # edge of the calibrated parameter space.
                tolerance = 5.0e-2 if mass_ratio >= 8.0 else 5.0e-4
            else:
                # The higher-mode eight-condition intermediate-amplitude
                # systems are ill-conditioned; different LU implementations
                # retain slightly different double-precision roundoff.
                tolerance = 1.0e-8 if mode[0] >= 3 else 1.0e-10
            assert relative_error < tolerance


@pytest.mark.parametrize("params", POLARIZATION_CASES)
def test_imrphenomxhm_native_polarizations_match_lal(
    params, monkeypatch, preserve_scheme
):
    params = {
        **params,
        "phase_order": 2.5,
        "amplitude_order": "3",
        "eccentricity_order": 4,
    }
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform(approximant="IMRPhenomXHM", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform(approximant="IMRPhenomXHM", **params)

    mass_ratio = max(params["mass1"], params["mass2"]) / min(
        params["mass1"], params["mass2"]
    )
    # LAL's ordinary polarization path enables higher-mode multibanding,
    # whereas its one-mode interface and the native kernels perform the full
    # evaluation.  Sparse, sign-asymmetric mode selections can expose a few
    # parts in 1e3 of multibanding error through polarization cancellation.
    tolerance = 5.0e-2 if mass_ratio >= 8.0 else 5.0e-3
    for expected, expected_array, result in zip(reference, reference_arrays, actual):
        assert len(result) == len(expected)
        assert result.delta_f == expected.delta_f
        assert float(result.epoch) == float(expected.epoch)
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        nonzero = np.abs(expected_array) > 0.0
        relative_error = np.linalg.norm(
            result.numpy()[nonzero] - expected_array[nonzero]
        ) / np.linalg.norm(expected_array[nonzero])
        assert relative_error < tolerance


def test_imrphenomxhm_native_support_is_deliberately_narrow():
    params = {"approximant": "IMRPhenomXHM", **CASES[0]}
    assert imrphenomxhm_modes_native_supported(params)
    assert imrphenomxhm_modes_native_supported(
        {**params, "mode_array": [(2, -2), (2, -1), (2, 1)]}
    )
    assert imrphenomxhm_modes_native_supported(
        {**params, "mode_array": [(3, -3), (3, 3)]}
    )
    assert imrphenomxhm_modes_native_supported(
        {**params, "mode_array": [(4, -4), (4, 4)]}
    )
    assert imrphenomxhm_modes_native_supported(
        {**params, "mode_array": [(3, -2), (3, 2)]}
    )
    assert imrphenomxhm_modes_native_supported({**params, "mode_array": None})
    assert imrphenomxhm_modes_native_supported(
        {
            **params,
            "phase_order": 2.5,
            "amplitude_order": "3",
            "eccentricity_order": 4,
        }
    )
    assert not imrphenomxhm_modes_native_supported({**params, "spin_order": 4})
    assert not imrphenomxhm_modes_native_supported({**params, "spin1x": 0.1})
    assert not imrphenomxhm_modes_native_supported({**params, "lambda1": 100.0})
    assert imrphenomxhm_fd_native_supported({**params, "mode_array": None})
    assert not imrphenomxhm_fd_native_supported({**params, "mode_array": []})


def test_imrphenomxhm_unsupported_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch
    import pycbc.waveform.waveform_modes as waveform_modes

    params = {**CASES[0], "mode_array": [(3, 2)], "lambda1": 100.0}
    lal_generator = waveform_modes.lalsimulation.SimIMRPhenomXHMGenerateFDOneMode
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported XHM mode reached the Torch generator")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_modes_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_modes.lalsimulation,
        "SimIMRPhenomXHMGenerateFDOneMode",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    result = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)

    assert lal_calls == 1
    assert result.keys() == {(3, 2)}


def test_imrphenomxhm_polarization_unsupported_options_use_lal_fallback(
    monkeypatch, preserve_scheme
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch
    import pycbc.waveform.waveform as waveform

    params = {**POLARIZATION_CASES[0], "dchi0": 0.01}
    lal_generator = waveform.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported XHM parameters reached the Torch generator")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(xhm_torch, "imrphenomxhm_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    hp, hc = get_fd_waveform(approximant="IMRPhenomXHM", **params)

    assert lal_calls == 1
    assert len(hp) == len(hc)


def test_imrphenomxhm_native_avoids_lal_and_host_transfer(monkeypatch, preserve_scheme):
    import pycbc.waveform.waveform as waveform
    import pycbc.waveform.waveform_modes as waveform_modes
    from pycbc.types.array_torch import TorchArrayData

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXHM mode called LAL")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomXHM transferred data to NumPy")

    monkeypatch.setattr(
        waveform_modes.lalsimulation,
        "SimIMRPhenomXHMGenerateFDOneMode",
        reject_lal,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveform",
        reject_lal,
    )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    params = {**CASES[0], "mode_array": None}
    with torch.no_grad():
        modes = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)
        polarizations = get_fd_waveform(
            approximant="IMRPhenomXHM",
            inclination=0.9,
            long_asc_nodes=0.2,
            **params,
        )

    assert list(modes) == [
        (2, 2),
        (2, 1),
        (3, 3),
        (3, 2),
        (4, 4),
        (2, -2),
        (2, -1),
        (3, -3),
        (3, -2),
        (4, -4),
    ]
    for mode_polarizations in modes.values():
        for series in mode_polarizations:
            assert isinstance(series._data.tensor, torch.Tensor)
    for series in polarizations:
        assert isinstance(series._data.tensor, torch.Tensor)


@pytest.mark.parametrize(
    "opt_out_flag",
    ("PYCBC_TORCH_NATIVE_PORTS", "PYCBC_IMRPHENOMXHM_NATIVE"),
)
def test_imrphenomxhm_native_opt_out_uses_lal_modes(
    opt_out_flag,
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch

    params = {**CASES[0], "mode_array": [(2, 2)]}

    def unexpected_native(**_params):
        raise AssertionError("opted-out IMRPhenomXHM reached the Torch generator")

    monkeypatch.setenv(opt_out_flag, "0")
    for name in _NATIVE_FLAG_ENVS:
        if name != opt_out_flag:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_modes_torch",
        unexpected_native,
    )

    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)
    reference_arrays = tuple(
        series.numpy().copy() for series in reference[(2, 2)]
    )

    _activate_scheme(_scheme.TorchScheme("cpu"))
    result = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)

    assert result.keys() == reference.keys()
    for expected, series in zip(reference_arrays, result[(2, 2)]):
        assert isinstance(series._data.tensor, torch.Tensor)
        np.testing.assert_array_equal(series.numpy(), expected)


@pytest.mark.parametrize(
    "mode",
    [
        (2, -2),
        (2, -1),
        (3, -3),
        (3, -2),
        (3, 2),
        (3, 3),
        (4, -4),
        (4, 4),
    ],
)
@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomxhm_modes_stay_on_requested_device(
    device_name, mode, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = {**CASES[0], "mode_array": [mode]}
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)
    reference_array = reference[mode][0].numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual = get_fd_waveform_modes(approximant="IMRPhenomXHM", **params)
    series = actual[mode][0]

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    assert series._data.tensor.device.type == device_name
    assert series._data.tensor.dtype == expected_dtype
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        series.numpy()[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    if device_name == "mps":
        tolerance = 5.0e-3
    elif mode[0] == 3 and abs(mode[1]) == 2:
        tolerance = 5.0e-4
    else:
        tolerance = 1.0e-8 if mode[0] >= 3 else 1.0e-10
    assert relative_error < tolerance


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomxhm_polarizations_stay_on_requested_device(
    device_name, monkeypatch, preserve_scheme
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params = POLARIZATION_CASES[0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform(approximant="IMRPhenomXHM", **params)
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform(approximant="IMRPhenomXHM", **params)

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual.numpy()[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    tolerance = 5.0e-3 if device_name == "mps" else 5.0e-4
    assert relative_error < tolerance


@pytest.mark.parametrize(
    ("params", "sample_points", "tolerance"),
    SEQUENCE_CASES,
)
def test_imrphenomxhm_sequence_matches_lal(
    params,
    sample_points,
    tolerance,
    monkeypatch,
    preserve_scheme,
):
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    _activate_scheme(_scheme.TorchScheme("cpu"))
    actual = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )

    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        relative_error = np.linalg.norm(result.numpy() - expected) / np.linalg.norm(
            expected
        )
        assert relative_error < tolerance


def test_imrphenomxhm_sequence_support_is_deliberately_narrow():
    params = {"approximant": "IMRPhenomXHM"}
    assert imrphenomxhm_sequence_native_supported(params)
    assert imrphenomxhm_sequence_native_supported(
        {**params, "mode_array": [(2, -1), (3, 2), (4, -4)]}
    )
    assert imrphenomxhm_sequence_native_supported({**params, "mode_array": []})
    assert not imrphenomxhm_sequence_native_supported(
        {**params, "mode_array": [(5, -5)]}
    )
    assert not imrphenomxhm_sequence_native_supported({**params, "spin1x": 0.1})
    assert not imrphenomxhm_sequence_native_supported({**params, "lambda1": 100.0})
    assert not imrphenomxhm_sequence_native_supported({**params, "dchi0": 0.01})


def test_imrphenomxhm_sequence_empty_mode_array_is_zero(
    monkeypatch,
    preserve_scheme,
):
    params, sample_points, _ = SEQUENCE_CASES[0]
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    polarizations = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        mode_array=[],
        **params,
    )

    for polarization in polarizations:
        assert torch.count_nonzero(polarization._data.tensor) == 0


def test_imrphenomxhm_sequence_public_dispatch_avoids_lal_and_host_transfer(
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch
    import pycbc.waveform.waveform as waveform

    params, sample_values, _ = SEQUENCE_CASES[0]
    native = xhm_torch.imrphenomxhm_fd_sequence_torch
    native_calls = 0

    def recording_native(**native_params):
        nonlocal native_calls
        native_calls += 1
        return native(**native_params)

    def reject_lal(*_args, **_kwargs):
        raise AssertionError("native IMRPhenomXHM sequence called LAL")

    def reject_host_transfer(_self):
        raise AssertionError("native IMRPhenomXHM sequence transferred to NumPy")

    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.TorchScheme("cpu"))
    sample_points = Array(sample_values)
    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        reject_lal,
    )
    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        polarizations = get_fd_waveform_sequence(
            approximant="IMRPhenomXHM",
            sample_points=sample_points,
            **params,
        )

    assert native_calls == 1
    for polarization in polarizations:
        assert isinstance(polarization._data.tensor, torch.Tensor)


def test_imrphenomxhm_sequence_unsupported_options_use_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.imrphenomxhm_torch as xhm_torch
    import pycbc.waveform.waveform as waveform

    base, sample_points, _ = SEQUENCE_CASES[0]
    params = {**base, "dchi0": 0.01}
    _clear_native_flags(monkeypatch)
    _activate_scheme(_scheme.CPUScheme())
    reference = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    lal_generator = waveform.lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsupported XHM sequence reached Torch")

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        xhm_torch,
        "imrphenomxhm_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    _activate_scheme(_scheme.TorchScheme("cpu"))
    fallback = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert isinstance(actual._data.tensor, torch.Tensor)
        np.testing.assert_allclose(
            actual.numpy(),
            expected,
            rtol=1.0e-14,
            atol=0.0,
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_imrphenomxhm_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params, sample_points, _ = SEQUENCE_CASES[0]
    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme())
    reference, _ = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_IMRPHENOMXHM_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme(device_name))
    actual, _ = get_fd_waveform_sequence(
        approximant="IMRPhenomXHM",
        sample_points=sample_points,
        **params,
    )

    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    assert actual._data.tensor.device.type == device_name
    assert actual._data.tensor.dtype == expected_dtype
    relative_error = np.linalg.norm(actual.numpy() - reference_array) / np.linalg.norm(
        reference_array
    )
    tolerance = 5.0e-3 if device_name == "mps" else 5.0e-4
    assert relative_error < tolerance
