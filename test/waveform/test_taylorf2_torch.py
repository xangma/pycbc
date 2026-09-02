import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc.waveform import get_fd_waveform, get_fd_waveform_sequence
from pycbc.waveform.spa_tmplt import spa_tmplt
from pycbc.waveform.taylorf2_torch import (
    _eos_q_from_lambda,
    taylorf2_aligned_phasing,
    taylorf2_native_supported,
    taylorf2_sequence_native_supported,
)
from pycbc import scheme as _scheme


def _tol(dtype):
    """Return tolerance tuple keyed by output dtype."""
    if dtype == np.complex64:
        return dict(rel=1e-7, mag=1e-6, phase_mean=1e-3, phase_std=5e-2)
    return dict(rel=1e-11, mag=1e-10, phase_mean=1e-6, phase_std=1e-3)


def _run_case(params):
    env_names = (
        "PYCBC_TORCH_NATIVE_PORTS",
        "PYCBC_SPATPLT_NATIVE",
        "PYCBC_TAYLORF2_NATIVE",
    )
    env_backup = {name: os.environ.get(name) for name in env_names}
    old = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        # CPU reference (uses lalsimulation phasing)
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        _scheme.mgr.state.prefix = "cpu"
        os.environ["PYCBC_TAYLORF2_NATIVE"] = "0"
        h_cpu = spa_tmplt(**params)

        # Torch path (native phasing)
        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        _scheme.mgr.state.prefix = "torch"
        os.environ["PYCBC_TAYLORF2_NATIVE"] = "0"
        os.environ["PYCBC_SPATPLT_NATIVE"] = "1"
        h_torch = spa_tmplt(**params)
    finally:
        for name, value in env_backup.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _scheme.mgr.state = old
        _scheme.Scheme._single = old_single

    cpu = h_cpu.numpy()
    tor = h_torch.numpy()
    # Ignore only exact/near-zero bins; keep tiny but non-zero tails.
    mask = np.abs(cpu) > 1e-26
    assert mask.any(), "waveform contains no non-zero bins"
    rel = np.linalg.norm(tor[mask] - cpu[mask]) / np.linalg.norm(cpu[mask])
    mag_ratio = np.mean(np.abs(tor[mask]) / np.abs(cpu[mask]))
    phase_diff = np.angle(tor[mask] * np.conj(cpu[mask]))
    return (
        rel,
        mag_ratio,
        phase_diff.mean(),
        phase_diff.std(),
        np.nonzero(mask)[0][0],
        np.nonzero(mask)[0][-1],
        tor.dtype,
    )


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=30.0,
            mass2=20.0,
            spin1z=0.3,
            spin2z=-0.1,
            delta_f=0.2,
            f_lower=20.0,
            distance=500.0,
            phase_order=-1,
            spin_order=-1,
        ),
        dict(
            mass1=10.0,
            mass2=8.0,
            spin1z=0.0,
            spin2z=0.4,
            delta_f=0.25,
            f_lower=15.0,
            distance=400.0,
            phase_order=7,
            spin_order=5,
        ),
        dict(
            mass1=1.4,
            mass2=1.3,
            spin1z=0.0,
            spin2z=0.0,
            delta_f=0.1,
            f_lower=10.0,
            distance=100.0,
            phase_order=-1,
            spin_order=-1,
            lambda1=800.0,
            lambda2=700.0,
        ),
    ],
)
def test_taylorf2_torch_parity(params):
    rel, mag_ratio, phase_mean, phase_std, kmin, kmax, dtype = _run_case(params)
    tol = _tol(dtype)
    assert rel < tol["rel"]
    assert abs(mag_ratio - 1.0) < tol["mag"]
    assert abs(phase_mean) < tol["phase_mean"]
    assert phase_std < tol["phase_std"]
    # basic sanity on bin coverage
    assert kmax > kmin


def test_taylorf2_torch_global_switch_falls_back():
    """Global switch should force torch scheme to reuse the CPU/LAL path."""
    params = dict(
        mass1=20.0,
        mass2=15.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=0.2,
        f_lower=20.0,
        distance=300.0,
        phase_order=-1,
        spin_order=-1,
    )

    env_backup = {
        k: os.environ.get(k)
        for k in (
            "PYCBC_TORCH_NATIVE_PORTS",
            "PYCBC_SPATPLT_NATIVE",
            "PYCBC_TAYLORF2_NATIVE",
        )
    }
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        os.environ["PYCBC_TORCH_NATIVE_PORTS"] = "0"
        os.environ.pop("PYCBC_TAYLORF2_NATIVE", None)
        os.environ.pop("PYCBC_SPATPLT_NATIVE", None)

        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.CPUScheme()
        h_cpu = spa_tmplt(**params)

        _scheme.Scheme._single = None
        _scheme.mgr.state = _scheme.TorchScheme()
        h_torch = spa_tmplt(**params)

        np.testing.assert_allclose(
            h_torch.numpy(), h_cpu.numpy(), rtol=1e-12, atol=1e-18
        )
    finally:
        for k, v in env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


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


def _activate_scheme(scheme_type):
    _scheme.Scheme._single = None
    _scheme.mgr.state = scheme_type()


SEQUENCE_CASES = [
    (
        dict(
            mass1=30.0,
            mass2=20.0,
            spin1z=0.3,
            spin2z=-0.1,
            distance=500.0,
            inclination=0.4,
            coa_phase=1.1,
            f_ref=30.0,
            # The LAL sequence API ignores ascending-node rotation.
            long_asc_nodes=0.7,
        ),
        [20.0, 23.5, 30.0, 50.0, 100.0, 500.0, 1000.0],
    ),
    (
        dict(
            mass1=1.4,
            mass2=1.3,
            spin1z=0.02,
            spin2z=-0.01,
            distance=100.0,
            inclination=0.8,
            coa_phase=0.2,
            f_ref=0.0,
            lambda1=800.0,
            lambda2=700.0,
        ),
        # Sequence sampling is not truncated at the regular-grid termination.
        [20.0, 5000.0, 100.0, 1000.0],
    ),
    (
        dict(
            mass1=2.0,
            mass2=1.6,
            spin1z=0.1,
            spin2z=-0.04,
            distance=150.0,
            inclination=1.2,
            coa_phase=0.3,
            f_ref=25.0,
            lambda1=300.0,
            lambda2=100.0,
            dquad_mon1=2.2,
            dquad_mon2=1.5,
            tidal_order=15,
            phase_order=4,
            dchi3=0.02,
            dchi6l=-0.01,
        ),
        [19.3, 25.0, 47.0, 300.0, 1200.0],
    ),
]


@pytest.mark.parametrize(("params", "sample_points"), SEQUENCE_CASES)
def test_taylorf2_sequence_public_torch_parity_and_dispatch(
    params,
    sample_points,
    monkeypatch,
    preserve_scheme,
):
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    reference = get_fd_waveform_sequence(
        approximant="TaylorF2",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.taylorf2_torch as taylorf2_mod
    import pycbc.waveform.waveform as waveform_mod

    native = taylorf2_mod.taylorf2_fd_sequence_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    def unexpected_lal(*_args, **_kwargs):
        raise AssertionError("native TaylorF2 sequence called LAL")

    monkeypatch.setattr(
        taylorf2_mod,
        "taylorf2_fd_sequence_torch",
        recording_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        unexpected_lal,
    )
    monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    monkeypatch.delenv("PYCBC_TORCH_NATIVE", raising=False)
    monkeypatch.delenv("PYCBC_TAYLORF2_NATIVE", raising=False)
    _activate_scheme(_scheme.TorchScheme)
    actual = get_fd_waveform_sequence(
        approximant="TaylorF2",
        sample_points=sample_points,
        **params,
    )

    assert calls == 1
    for expected, result in zip(reference_arrays, actual):
        assert result._data.tensor.device.type == "cpu"
        assert result._data.tensor.dtype == torch.complex128
        result_array = result.numpy()
        assert np.all(result_array != 0.0)
        relative_error = np.linalg.norm(result_array - expected) / np.linalg.norm(
            expected
        )
        assert relative_error < 1.0e-10


def test_taylorf2_sequence_unsupported_amplitude_uses_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    params, sample_points = SEQUENCE_CASES[0]
    params = dict(params, amplitude_order=2)
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    reference = get_fd_waveform_sequence(
        approximant="TaylorF2",
        sample_points=sample_points,
        **params,
    )
    reference_arrays = tuple(array.numpy().copy() for array in reference)

    import pycbc.waveform.taylorf2_torch as taylorf2_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported TaylorF2 sequence reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(
        taylorf2_mod,
        "taylorf2_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_lal,
    )
    monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    monkeypatch.delenv("PYCBC_TORCH_NATIVE", raising=False)
    monkeypatch.delenv("PYCBC_TAYLORF2_NATIVE", raising=False)
    _activate_scheme(_scheme.TorchScheme)
    fallback = get_fd_waveform_sequence(
        approximant="TaylorF2",
        sample_points=sample_points,
        **params,
    )

    assert lal_calls == 1
    for expected, result in zip(reference_arrays, fallback):
        np.testing.assert_allclose(
            result.numpy(),
            expected,
            rtol=1.0e-14,
            atol=0.0,
        )


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda"])
def test_taylorf2_sequence_stays_on_requested_device(
    device_name,
    monkeypatch,
    preserve_scheme,
):
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Torch MPS device is unavailable")
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device is unavailable")

    params, sample_points = SEQUENCE_CASES[1]
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    reference, _ = get_fd_waveform_sequence(
        approximant="TaylorF2",
        sample_points=sample_points,
        **params,
    )
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")
    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.TorchScheme(device_name)
    actual, _ = get_fd_waveform_sequence(
        approximant="TaylorF2",
        sample_points=sample_points,
        **params,
    )

    assert actual._data.tensor.device.type == device_name
    expected_dtype = torch.complex64 if device_name == "mps" else torch.complex128
    assert actual._data.tensor.dtype == expected_dtype
    actual_array = actual.numpy()
    relative_error = np.linalg.norm(actual_array - reference_array) / np.linalg.norm(
        reference_array
    )
    tolerance = 5.0e-3 if device_name == "mps" else 1.0e-10
    assert relative_error < tolerance


def test_taylorf2_sequence_native_avoids_host_transfer(
    monkeypatch,
    preserve_scheme,
):
    from pycbc.types import Array
    from pycbc.types.array_torch import TorchArrayData

    params, sample_values = SEQUENCE_CASES[0]
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme)
    sample_points = Array(sample_values)

    def reject_host_transfer(_self):
        raise AssertionError("native TaylorF2 sequence materialized on the host")

    monkeypatch.setattr(TorchArrayData, "numpy", reject_host_transfer)
    with torch.no_grad():
        hp, hc = get_fd_waveform_sequence(
            approximant="TaylorF2",
            sample_points=sample_points,
            **params,
        )

    assert isinstance(hp._data.tensor, torch.Tensor)
    assert isinstance(hc._data.tensor, torch.Tensor)
    assert hp._data.tensor.device.type == "cpu"
    assert hc._data.tensor.device.type == "cpu"


@pytest.mark.parametrize(
    "params",
    [
        dict(
            mass1=30.0,
            mass2=20.0,
            spin1z=0.3,
            spin2z=-0.1,
            delta_f=0.25,
            f_lower=20.0,
            distance=500.0,
            inclination=0.4,
            coa_phase=1.1,
            f_ref=30.0,
        ),
        dict(
            mass1=10.0,
            mass2=8.0,
            spin2z=0.4,
            delta_f=0.25,
            f_lower=15.0,
            f_final=400.0,
            f_ref=25.0,
            distance=400.0,
            phase_order=4,
            spin_order=3,
            inclination=1.2,
            coa_phase=0.3,
            long_asc_nodes=0.37,
        ),
        dict(
            mass1=1.4,
            mass2=1.3,
            spin1z=0.02,
            spin2z=-0.01,
            delta_f=0.5,
            f_lower=20.0,
            distance=100.0,
            lambda1=800.0,
            lambda2=700.0,
            dquad_mon1=0.0,
            dquad_mon2=0.0,
            inclination=0.8,
            coa_phase=0.2,
            f_ref=30.0,
        ),
        dict(
            mass1=2.0,
            mass2=1.6,
            spin1z=0.1,
            spin2z=-0.04,
            delta_f=0.5,
            f_lower=20.0,
            f_final=300.0,
            distance=150.0,
            lambda1=300.0,
            lambda2=100.0,
            dquad_mon1=2.2,
            dquad_mon2=1.5,
            tidal_order=15,
            dchi3=0.02,
            dchi6l=-0.01,
            f_ref=25.0,
        ),
        dict(
            mass1=3.2,
            mass2=1.7,
            spin1z=0.23,
            spin2z=-0.17,
            delta_f=1.0,
            f_lower=20.0,
            f_final=160.0,
            f_ref=31.7,
            distance=230.0,
            lambda1=450.0,
            lambda2=120.0,
            dquad_mon1=0.0,
            dquad_mon2=0.0,
            spin_order=0,
        ),
    ],
)
def test_taylorf2_public_torch_parity_and_dispatch(
    params, monkeypatch, preserve_scheme
):
    """The public API must select native Torch and retain LAL parity."""
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "0")
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    cpu = get_fd_waveform(approximant="TaylorF2", **params)
    cpu_arrays = tuple(series.numpy().copy() for series in cpu)

    import pycbc.waveform.taylorf2_torch as taylorf2_mod

    native = taylorf2_mod.taylorf2_fd_torch
    calls = 0

    def recording_native(**native_params):
        nonlocal calls
        calls += 1
        return native(**native_params)

    monkeypatch.setattr(taylorf2_mod, "taylorf2_fd_torch", recording_native)
    monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    monkeypatch.delenv("PYCBC_TORCH_NATIVE", raising=False)
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")
    _activate_scheme(_scheme.TorchScheme)
    torch_waveform = get_fd_waveform(approximant="TaylorF2", **params)

    assert calls == 1
    for reference, reference_array, actual in zip(cpu, cpu_arrays, torch_waveform):
        assert len(actual) == len(reference)
        assert actual.delta_f == reference.delta_f
        assert float(actual.epoch) == float(reference.epoch)
        assert actual._data.tensor.device.type == "cpu"
        assert actual._data.tensor.dtype.is_complex

        actual_array = actual.numpy()
        np.testing.assert_array_equal(
            actual_array == 0.0,
            reference_array == 0.0,
        )
        nonzero = np.abs(reference_array) > 0.0
        relative_error = np.linalg.norm(
            actual_array[nonzero] - reference_array[nonzero]
        ) / np.linalg.norm(reference_array[nonzero])
        assert relative_error < 1.0e-11


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_taylorf2_public_torch_uses_mps(monkeypatch, preserve_scheme):
    params = dict(
        mass1=30.0,
        mass2=20.0,
        spin1z=0.2,
        spin2z=-0.1,
        delta_f=0.5,
        f_lower=20.0,
        f_ref=30.0,
        distance=400.0,
        inclination=0.7,
    )
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    reference, _ = get_fd_waveform(approximant="TaylorF2", **params)
    reference_array = reference.numpy().copy()

    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "1")
    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.TorchScheme("mps")
    actual, _ = get_fd_waveform(approximant="TaylorF2", **params)

    assert actual._data.tensor.device.type == "mps"
    assert actual._data.tensor.dtype == torch.complex64
    actual_array = actual.numpy()
    nonzero = np.abs(reference_array) > 0.0
    relative_error = np.linalg.norm(
        actual_array[nonzero] - reference_array[nonzero]
    ) / np.linalg.norm(reference_array[nonzero])
    assert relative_error < 5.0e-5


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_taylorf2_mps_phase_accuracy_boundary(preserve_scheme):
    safe = dict(
        mass1=1.4,
        mass2=1.3,
        f_lower=20.0,
        f_ref=30.0,
    )
    unsafe = dict(
        mass1=0.45,
        mass2=0.07,
        f_lower=3.5,
        f_ref=0.0,
    )

    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.TorchScheme("mps")
    assert taylorf2_native_supported(safe)
    assert not taylorf2_native_supported(unsafe)
    assert not taylorf2_native_supported(dict(safe, f_ref=2.0))

    assert taylorf2_sequence_native_supported(
        dict(safe, sample_points=[100.0, 20.0, 50.0])
    )
    assert not taylorf2_sequence_native_supported(
        dict(safe, sample_points=[100.0, 3.0, 50.0])
    )
    assert not taylorf2_sequence_native_supported(
        dict(safe, sample_points=[100.0, 20.0, 50.0], f_ref=2.0)
    )

    _activate_scheme(_scheme.CPUScheme)
    assert taylorf2_native_supported(unsafe)
    assert taylorf2_sequence_native_supported(
        dict(unsafe, sample_points=[100.0, 3.0, 50.0])
    )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="Torch MPS device is unavailable",
)
def test_taylorf2_mps_phase_boundary_uses_lal_fallback(
    monkeypatch,
    preserve_scheme,
):
    import pycbc.waveform.taylorf2_torch as taylorf2_mod
    import pycbc.waveform.waveform as waveform_mod

    params = dict(
        mass1=0.45,
        mass2=0.07,
        delta_f=2.0,
        f_lower=5.0,
        f_final=64.0,
        distance=100.0,
    )
    regular_lal = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    sequence_lal = (
        waveform_mod.lalsimulation.SimInspiralChooseFDWaveformSequence
    )
    regular_calls = 0
    sequence_calls = 0

    def unexpected_native(**_params):
        raise AssertionError("unsafe MPS TaylorF2 request reached Torch")

    def recording_regular(*args, **kwargs):
        nonlocal regular_calls
        regular_calls += 1
        return regular_lal(*args, **kwargs)

    def recording_sequence(*args, **kwargs):
        nonlocal sequence_calls
        sequence_calls += 1
        return sequence_lal(*args, **kwargs)

    monkeypatch.setattr(taylorf2_mod, "taylorf2_fd_torch", unexpected_native)
    monkeypatch.setattr(
        taylorf2_mod,
        "taylorf2_fd_sequence_torch",
        unexpected_native,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_regular,
    )
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveformSequence",
        recording_sequence,
    )
    monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    monkeypatch.delenv("PYCBC_TORCH_NATIVE", raising=False)
    monkeypatch.delenv("PYCBC_TAYLORF2_NATIVE", raising=False)
    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.TorchScheme("mps")

    regular = get_fd_waveform(approximant="TaylorF2", **params)
    sequence = get_fd_waveform_sequence(
        approximant="TaylorF2",
        sample_points=[64.0, 5.0, 20.0],
        **params,
    )

    assert regular_calls == 1
    assert sequence_calls == 1
    for waveform in (*regular, *sequence):
        assert waveform._data.tensor.device.type == "mps"
        assert waveform._data.tensor.dtype == torch.complex64


def test_taylorf2_unsupported_amplitude_uses_lal_fallback(monkeypatch, preserve_scheme):
    params = dict(
        mass1=20.0,
        mass2=15.0,
        spin1z=0.1,
        spin2z=-0.05,
        delta_f=0.5,
        f_lower=20.0,
        f_final=200.0,
        amplitude_order=2,
    )
    monkeypatch.setenv("PYCBC_TAYLORF2_NATIVE", "0")
    _activate_scheme(_scheme.CPUScheme)
    reference = get_fd_waveform(approximant="TaylorF2", **params)
    reference_arrays = tuple(series.numpy().copy() for series in reference)

    import pycbc.waveform.taylorf2_torch as taylorf2_mod
    import pycbc.waveform.waveform as waveform_mod

    def unexpected_native(**_params):
        raise AssertionError("unsupported TaylorF2 parameters reached Torch")

    lal_generator = waveform_mod.lalsimulation.SimInspiralChooseFDWaveform
    lal_calls = 0

    def recording_lal(*args, **kwargs):
        nonlocal lal_calls
        lal_calls += 1
        return lal_generator(*args, **kwargs)

    monkeypatch.setattr(taylorf2_mod, "taylorf2_fd_torch", unexpected_native)
    monkeypatch.setattr(
        waveform_mod.lalsimulation,
        "SimInspiralChooseFDWaveform",
        recording_lal,
    )
    monkeypatch.delenv("PYCBC_TORCH_NATIVE_PORTS", raising=False)
    monkeypatch.delenv("PYCBC_TORCH_NATIVE", raising=False)
    monkeypatch.delenv("PYCBC_TAYLORF2_NATIVE", raising=False)
    _activate_scheme(_scheme.TorchScheme)
    fallback = get_fd_waveform(approximant="TaylorF2", **params)

    assert lal_calls == 1
    for expected, actual in zip(reference_arrays, fallback):
        assert actual._data.tensor.device.type == "cpu"
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1.0e-14, atol=0.0)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({}, True),
        ({"spin_order": 1}, True),
        ({"lambda1": 800.0, "dchi3": 0.1}, True),
        ({"amplitude_order": 2}, False),
        ({"spin1x": 0.1}, False),
        ({"lambda_octu1": 10.0}, False),
        ({"mode_array": [(2, 2)]}, False),
        ({"dalpha1": 0.1}, False),
        ({"lambda1": -1.0}, False),
    ],
)
def test_taylorf2_native_support_boundary(params, expected):
    assert taylorf2_native_supported(params) is expected
    assert taylorf2_sequence_native_supported(params) is expected


@pytest.mark.parametrize("lambda_tidal", [0.25, 0.5, 1.0, 100.0, 800.0])
def test_taylorf2_eos_quadrupole_fit_matches_lal(lambda_tidal):
    lal_params = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(lal_params, lambda_tidal)
    lalsimulation.SimInspiralWaveformParamsInsertdQuadMon1(lal_params, 0.0)
    lalsimulation.SimInspiralSetQuadMonParamsFromLambdas(lal_params)
    expected = lalsimulation.SimInspiralWaveformParamsLookupdQuadMon1(lal_params) + 1.0
    assert _eos_q_from_lambda(lambda_tidal) == pytest.approx(expected, rel=1.0e-14)


def test_taylorf2_default_tides_stop_at_7pn():
    default = taylorf2_aligned_phasing(1.4, 1.3, 0.0, 0.0, lambda1=800.0, lambda2=700.0)
    explicit_75pn = taylorf2_aligned_phasing(
        1.4,
        1.3,
        0.0,
        0.0,
        lambda1=800.0,
        lambda2=700.0,
        tidal_order=15,
    )
    assert default.v[15] == 0.0
    assert explicit_75pn.v[15] != 0.0
