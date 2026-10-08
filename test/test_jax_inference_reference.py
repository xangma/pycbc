"""Original numerical boundaries for device-resident inference."""

import pickle
import types

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc.scheme import CPUScheme, JAXScheme
from pycbc.types import Array, FrequencySeries
from pycbc.inference.models.gaussian_noise import GaussianNoise
from pycbc.inference.models.gaussian_noise_jax import JAXGaussianNoise
from pycbc.inference.models import relbin_jax, tools_jax
from pycbc.inference.models.tools import marginalize_likelihood


@pytest.fixture(params=("cpu", "cuda"))
def inference_device(request):
    if request.param == "cuda":
        try:
            devices = jax.devices("gpu")
        except RuntimeError:
            devices = []
        if not devices:
            pytest.skip("CUDA is unavailable")
    return request.param


def _bytes(value):
    arr = np.asarray(value)
    return arr.dtype, arr.shape, arr.tobytes()


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("selector", ["inner", "inference_inner"])
def test_original_inner_is_independently_selectable(dtype, selector, inference_device):
    values = np.asarray([1.234567 + 0.3j, -2.125 + 1.1j], dtype=dtype)
    other = np.asarray([0.35 - 1.4j, 2.02 + 0.27j], dtype=dtype)
    with CPUScheme():
        expected = Array(values).inner(Array(other))
    with JAXScheme(inference_device, reference_operations=(selector,)):
        actual = tools_jax.inner(jnp.asarray(values), jnp.asarray(other))
    assert _bytes(actual) == _bytes(expected)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_weighting_and_inner_controls_compose_exactly(dtype, inference_device):
    rng = np.random.default_rng(121)
    h = (rng.normal(size=33) + 1j * rng.normal(size=33)).astype(dtype)
    d = (rng.normal(size=33) + 1j * rng.normal(size=33)).astype(dtype)
    # Match the precision required by original Array arithmetic.
    w = rng.uniform(0.25, 2.0, size=33).astype(h.real.dtype)
    with CPUScheme():
        weighted = Array(h)
        weighted *= Array(w)
        expected = weighted.inner(Array(d)), weighted.inner(weighted).real
    with JAXScheme(
        inference_device, reference_operations=("inference_whitening", "inner")
    ):
        actual = tools_jax.fused_inner_hd_hh(
            jnp.asarray(h), jnp.asarray(d), jnp.asarray(w)
        )
    assert tuple(map(_bytes, actual)) == tuple(map(_bytes, expected))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_default_physical_complex_division_is_finite(dtype):
    value = jnp.asarray([1e-23 + 2e-23j], dtype=dtype)
    real, imag = relbin_jax._cdiv(value.real, value.imag, value.real, value.imag)
    np.testing.assert_allclose(np.asarray(real + 1j * imag), [1 + 0j], atol=2e-7)


def test_summary_uses_local_bins_and_original_route_is_exact():
    values = np.asarray([1e16 + 0j, 1 + 0j], dtype=np.complex128)
    h1 = np.ones(2, dtype=np.complex128)
    psd = np.ones(2)
    freqs = np.asarray([0.0, 1.0])
    bins = np.asarray([[1, 2]], dtype=np.int64)
    with JAXScheme():
        current = relbin_jax.summary_product(
            jnp.asarray(h1),
            jnp.asarray(values),
            jnp.asarray(psd),
            jnp.asarray(freqs),
            bins,
            1.0,
        )
    assert complex(current[0][0]) == 4 + 0j
    with JAXScheme(reference_operations=("relbin_summary",)):
        reference = relbin_jax.summary_product(
            jnp.asarray(h1), jnp.asarray(values), psd, freqs, bins, 1.0
        )
    assert tuple(map(_bytes, reference)) == tuple(map(_bytes, current))


@pytest.mark.parametrize("phase", [False, True])
def test_original_marginalization_is_exact(phase):
    sh = np.asarray([1.35 + 0.21j, -2.01 + 0.3j, 0.75 - 0.35j])
    hh = np.asarray([1.05, 3.11, 0.69])
    weights = np.log(np.asarray([0.2, 0.3, 0.5]))
    with CPUScheme():
        expected = marginalize_likelihood(
            sh, hh, phase=phase, logw=weights, return_peak=True
        )
    with JAXScheme(reference_operations=("inference_marginalization",)):
        actual = marginalize_likelihood(
            jnp.asarray(sh),
            jnp.asarray(hh),
            phase=phase,
            logw=weights,
            return_peak=True,
        )
    assert tuple(map(_bytes, actual)) == tuple(map(_bytes, expected))


def test_default_inner_and_marginalization_remain_differentiable():
    with JAXScheme():
        data = jnp.asarray([1.0 + 0.5j, -0.2 + 0.3j])

        def loss(amplitude):
            hd, hh = tools_jax.fused_inner_hd_hh(amplitude * data, data)
            return marginalize_likelihood(hd, hh, phase=True, skip_vector=True)

        actual = jax.grad(loss)(jnp.asarray(0.7))
        step = 1e-5
        finite = (loss(0.7 + step) - loss(0.7 - step)) / (2 * step)
        np.testing.assert_allclose(actual, finite, rtol=1e-8)


def test_cpu_statistics_keep_original_uncached_method_contract():
    with CPUScheme():
        waveform = FrequencySeries(np.ones(2, dtype=np.complex128), delta_f=1.0)
        data = FrequencySeries(np.ones(2, dtype=np.complex128), delta_f=1.0)
        model = object.__new__(GaussianNoise)
        model.get_waveforms = lambda: {"H1": waveform}
        model._whitened_data = {"H1": data}
        model._weight = {"H1": np.full(2, 2.0)}
        model._kmin = {"H1": 0}
        model._kmax = {"H1": 2}
        model._current_stats = types.SimpleNamespace(lognl=0.0)
        model.waveform_transforms = None
        assert model.det_cplx_loglr("H1") == 0j
        # The original accessor calls _loglr without filling cached loglr.
        assert not hasattr(model._current_stats, "loglr")
        assert model.loglr == -8.0


def test_jax_public_statistics_preserve_complex_scalar_kind():
    from pycbc.inference.models.base_jax import JAXModelStats

    stats = JAXModelStats()
    stats.example = jnp.asarray(1.0 + 2.0j)
    assert stats.getstats(("example",)) == (1.0 + 2.0j,)
    assert stats.getstatsdict(("example",)) == {"example": 1.0 + 2.0j}


def test_factory_preserves_explicit_custom_model_and_generator_overrides():
    class Custom(GaussianNoise):
        def _loglr(self):
            return 123.0

    with JAXScheme():
        native = object.__new__(Custom)
        # __new__ must not replace an inherited user model with a registered class.
        selected = Custom.__new__(Custom)
        assert type(selected) is Custom
        assert type(native) is Custom
        assert type(GaussianNoise.__new__(GaussianNoise)) is JAXGaussianNoise
    # Class identity survives standard sampler serialization.
    assert (
        pickle.loads(pickle.dumps(object.__new__(JAXGaussianNoise))).__class__
        is JAXGaussianNoise
    )


def _relative_arguments(name):
    rng = np.random.default_rng(205)
    f = np.asarray([20.0, 80.0, 200.0])
    wave = lambda: rng.normal(size=3) + 1j * rng.normal(size=3)
    result = dict(freqs=f, hp=wave(), h00=wave(), a0=wave()[:2], a1=wave()[:2])
    if "multi" in name:
        result.update(hp2=wave(), h002=wave(), dtc=0.001, dtc2=-0.002)
    else:
        result.update(b0=np.ones(2), b1=np.full(2, 0.1))
    if name.startswith("snr_predictor"):
        result.update(tstart=-0.01, delta_t=0.001, num_samples=4)
        if name == "snr_predictor":
            result["hc"] = wave()
    elif "det" in name:
        result.setdefault("dtc", 0.001)
    else:
        result.update(hc=wave(), fp=0.7, fc=0.2, dtc=0.001)
        if "multi" in name:
            result.update(hc2=wave(), fp2=0.3, fc2=-0.5)
        if "_v" in name:
            result.update(fp=np.full(3, 0.7), fc=np.full(3, 0.2), dtc=np.full(3, 0.001))
            if "multi" in name:
                result.update(
                    fp2=np.full(3, 0.3), fc2=np.full(3, -0.5), dtc2=np.full(3, -0.002)
                )
        if "time" in name:
            result.update(
                times=np.linspace(-0.001, 0.001, 3),
                dtc=np.asarray([0.002, 0.003, 0.004]),
            )
        if "pol" in name:
            result["pol_phase"] = np.exp(-2j * np.asarray([0.1, 0.2, 0.3]))
    return result


@pytest.mark.parametrize(
    "name",
    [
        "likelihood_parts",
        "likelihood_parts_det",
        "likelihood_parts_v",
        "likelihood_parts_v_pol",
        "likelihood_parts_v_time",
        "likelihood_parts_v_pol_time",
        "likelihood_parts_multi",
        "likelihood_parts_multi_v",
        "likelihood_parts_det_multi",
        "snr_predictor",
        "snr_predictor_dom",
    ],
)
def test_original_relative_kernels_are_independently_exact(name):
    from pycbc.inference.models import relbin_cpu

    arguments = _relative_arguments(name)
    expected = getattr(relbin_cpu, name)(**arguments)
    selected = "relbin_snr" if name.startswith("snr") else "relbin_likelihood"
    with JAXScheme(reference_operations=(selected,)):
        adapted = {
            key: jnp.asarray(value) if np.ndim(value) else value
            for key, value in arguments.items()
        }
        if "det" in name:
            adapted["channel"] = adapted.pop("hp")
            if "multi" in name:
                adapted["channel2"] = adapted.pop("hp2")
        actual = getattr(relbin_jax, name)(**adapted)
    if isinstance(expected, tuple):
        assert tuple(map(_bytes, actual)) == tuple(map(_bytes, expected))
    else:
        assert _bytes(actual) == _bytes(expected)


def test_original_relative_scalar_route_preserves_batched_samples():
    from pycbc.inference.models import relbin_cpu

    arguments = _relative_arguments("likelihood_parts")
    arguments.update(
        fp=np.asarray([0.2, 0.4, 0.8]),
        fc=np.asarray([0.1, 0.3, 0.2]),
        dtc=np.asarray([0.0, 0.001, 0.002]),
    )
    expected = relbin_cpu.likelihood_parts_vector(**arguments)
    with JAXScheme(reference_operations=("relbin_likelihood",)):
        actual = relbin_jax.likelihood_parts(
            **{k: jnp.asarray(v) for k, v in arguments.items()}
        )
    assert tuple(map(_bytes, actual)) == tuple(map(_bytes, expected))


def test_original_sampling_preserves_rng_stream_and_indices():
    from pycbc.inference.models.tools import draw_sample

    logs = np.random.default_rng(11).normal(size=4096).astype(np.float32)
    before = np.random.get_state()
    try:
        np.random.seed(419)
        expected = draw_sample(logs, size=100000)
        expected_next = np.random.random()
        np.random.seed(419)
        with JAXScheme(reference_operations=("inference_sampling",)):
            actual = tools_jax.draw_sample(jnp.asarray(logs), size=100000, host=False)
        actual_next = np.random.random()
        assert _bytes(actual) == _bytes(expected)
        assert actual_next == expected_next
    finally:
        np.random.set_state(before)


def test_original_distance_scalar_and_interpolation_marginalization():
    from scipy.interpolate import RectBivariateSpline

    sh = np.complex128(1.35 + 0.21j)
    hh = np.float64(1.05)
    distance = (np.asarray([0.5, 1.0, 2.0]), np.asarray([0.2, 0.3, 0.5]))
    expected = marginalize_likelihood(sh, hh, phase=True, distance=distance)
    with JAXScheme(reference_operations=("inference_marginalization",)):
        actual = marginalize_likelihood(
            jnp.asarray(sh), jnp.asarray(hh), phase=True, distance=distance
        )
    assert _bytes(actual) == _bytes(expected)
    x = np.linspace(0.1, 5.0, 6)
    y = np.linspace(0.2, 4.0, 6)
    spline = RectBivariateSpline(x, y, np.outer(x, y))
    evaluator = tools_jax.rect_bivariate_spline_evaluator(spline)
    sh = np.asarray([0.5, 1.25, 2.5])
    hh = np.asarray([0.8, 1.0, 2.0])
    expected = marginalize_likelihood(
        sh, hh, interpolator=lambda a, b: spline(a, b, grid=False)
    )
    with JAXScheme(
        reference_operations=("inference_marginalization", "inference_interpolant")
    ):
        actual = marginalize_likelihood(
            jnp.asarray(sh), jnp.asarray(hh), interpolator=evaluator
        )
    assert _bytes(actual) == _bytes(expected)


@pytest.mark.parametrize(
    "model_name", ["GaussianNoise", "MarginalizedTime", "MarginalizedPolarization"]
)
def test_actual_physical_models_restore_original_calculation(
    model_name, monkeypatch, inference_device
):
    from pycbc.fft import backend_cpu
    from pycbc.inference.models import gaussian_noise, marginalized_gaussian_noise

    model_type = getattr(
        (
            gaussian_noise
            if model_name == "GaussianNoise"
            else marginalized_gaussian_noise
        ),
        model_name,
    )
    monkeypatch.setattr(backend_cpu, "cpu_backend", "numpy")
    params = dict(
        approximant="TaylorF2",
        mass1=30.0,
        mass2=20.0,
        distance=500.0,
        inclination=0.4,
        f_lower=20.0,
        coa_phase=0.2,
        tc=1126259460.2,
        ra=1.1,
        dec=-0.3,
    )
    controls = (
        "waveform",
        "time_shift",
        "inference_whitening",
        "inference_inner",
        "inference_projection",
        "inference_time_interpolation",
        "inference_marginalization",
        "correlate",
        "divide",
        "ifft",
        "fft",
        "squared_norm",
        "inner",
        "detector",
    )
    outputs = []
    state = np.random.get_state()
    try:
        for context in (
            CPUScheme(),
            JAXScheme(inference_device, reference_operations=controls),
        ):
            with context:
                data = FrequencySeries(
                    np.full(129, 1e-23 + 1e-23j, np.complex128),
                    delta_f=2.0,
                    epoch=1126259460.0,
                )
                psd = FrequencySeries(np.full(129, 1e-46), delta_f=2.0)
                model = model_type(
                    ("polarization",),
                    {"H1": data},
                    {"H1": 20.0},
                    psds={"H1": psd},
                    static_params=params,
                )
                model.update(
                    polarization=(
                        0.2
                        if model_name == "GaussianNoise"
                        else np.linspace(0.0, 2 * np.pi, 8)
                    )
                )
                outputs.append(_bytes(model.loglr))
        assert outputs[0] == outputs[1]
    finally:
        np.random.set_state(state)


def test_original_time_interpolation_and_error_boundary():
    from pycbc.types import TimeSeries
    from pycbc.inference.models.marginalized_gaussian_noise_jax import _time_value

    samples = np.random.default_rng(18).normal(size=16).astype(np.float64)
    with CPUScheme():
        series = TimeSeries(samples, delta_t=0.125, epoch=100.0)
        expected = series.at_time(100.83, interpolate="quadratic")
    with JAXScheme(reference_operations=("inference_time_interpolation",)):
        actual = _time_value(TimeSeries(samples, delta_t=0.125, epoch=100.0), 100.83)
        assert _bytes(actual) == _bytes(expected)
        with pytest.raises(IndexError):
            _time_value(TimeSeries(samples, delta_t=0.125, epoch=100.0), 103.0)


def test_jax_subset_uses_same_model_rng_as_original():
    from pycbc.inference.models.proposals_jax import _random_permutation

    logs = jnp.arange(100, dtype=jnp.float64)
    expected = np.random.default_rng(27).choice(100, size=7, replace=False)
    device, host = _random_permutation(logs, 7, np.random.default_rng(27))
    assert _bytes(host) == _bytes(expected)
    assert _bytes(device) == _bytes(expected)


def test_actual_jax_model_pickles_and_keeps_class_identity():
    with JAXScheme("cpu", reference_operations=("inner",)):
        data = FrequencySeries(np.zeros(17, dtype=np.complex128), delta_f=1.0)
        model = GaussianNoise(
            (),
            {"H1": data},
            {"H1": 2.0},
            static_params=dict(
                approximant="TaylorF2",
                mass1=10.0,
                mass2=10.0,
                f_lower=2.0,
                distance=100.0,
                tc=0.0,
                ra=0.0,
                dec=0.0,
                polarization=0.0,
            ),
        )
        restored = pickle.loads(pickle.dumps(model))
        assert type(restored) is type(model)
        assert restored.lognl == model.lognl
        np.testing.assert_array_equal(
            restored.data["H1"].numpy(), model.data["H1"].numpy()
        )
        assert restored.data["H1"]._scheme.jax_reference_operations == frozenset(
            {"inner"}
        )


def test_relative_reference_preparation_and_full_model_are_exact(inference_device):
    from pycbc.inference.models.relbin import Relative
    from pycbc.inference.models.relbin_jax import prepare_reference_data

    epoch = 1126259460.0
    delay = 0.007123456789
    values = np.random.default_rng(12).normal(size=17) + 1j * np.random.default_rng(
        13
    ).normal(size=17)
    frequencies = np.arange(17) * 2.0
    with CPUScheme():
        original = FrequencySeries(values, delta_f=2.0, epoch=epoch) * np.conjugate(
            np.exp(-2j * np.pi * frequencies * delay)
        )
    with JAXScheme(inference_device, reference_operations=("time_shift",)):
        series = FrequencySeries(values, delta_f=2.0, epoch=epoch)
        _, prepared = prepare_reference_data(series, series, 17, 0, 2.0, delay)
    assert _bytes(prepared.numpy()) == _bytes(original.numpy())
    static = dict(
        approximant="TaylorF2",
        mass1=30.0,
        mass2=20.0,
        distance=500.0,
        inclination=0.4,
        f_lower=20.0,
        coa_phase=0.2,
        tc=epoch + 0.2,
        ra=1.1,
        dec=-0.3,
        polarization=0.2,
    )
    controls = (
        "detector.Detector.antenna_pattern",
        "detector.Detector.time_delay_from_location",
        "waveform",
        "time_shift",
        "inference_whitening",
        "inference_inner",
        "relbin_summary",
        "relbin_likelihood",
        "relbin_snr",
        "inference_marginalization",
    )
    results = []
    for context in (
        CPUScheme(),
        JAXScheme(inference_device, reference_operations=controls),
    ):
        with context:
            data = FrequencySeries(
                np.full(129, 1e-23 + 1e-23j, np.complex128), delta_f=2.0, epoch=epoch
            )
            psd = FrequencySeries(np.full(129, 1e-46), delta_f=2.0)
            model = Relative(
                (),
                {"H1": data},
                {"H1": 20.0},
                psds={"H1": psd},
                static_params=static,
                fiducial_params=static,
            )
            model.update()
            results.append(_bytes(model.loglr))
    assert results[0] == results[1]


def test_summary_sum_control_uses_original_ndarray_reduction(monkeypatch):
    from pycbc.inference.models.relbin import Relative

    rng = np.random.default_rng(205)
    left = rng.normal(size=19) + 1j * rng.normal(size=19)
    right = rng.normal(size=19) + 1j * rng.normal(size=19)
    bins = np.asarray([[1, 8], [8, 17]])
    psd = np.ones(19)
    freq = np.arange(19, dtype=np.float64)
    state = types.SimpleNamespace(psds={"D": psd}, df={"D": 1.0}, f={"D": freq})
    expected = Relative.summary_product(state, left, right, bins, "D")

    def reject(*args, **kwargs):
        raise AssertionError("device reduction bypassed selected sum")

    with JAXScheme(reference_operations=("sum", "divide")):
        monkeypatch.setattr(jnp, "sum", reject)
        actual = relbin_jax.summary_product(
            jnp.asarray(left), jnp.asarray(right), psd, freq, bins, 1.0
        )
    assert tuple(map(_bytes, actual)) == tuple(map(_bytes, expected))


def test_spline_gradient_before_compilation_preserves_evaluator():
    from scipy.interpolate import RectBivariateSpline

    x = np.linspace(0.1, 5.0, 6)
    y = np.linspace(0.2, 4.0, 6)
    evaluator = tools_jax.rect_bivariate_spline_evaluator(
        RectBivariateSpline(x, y, np.outer(x, y))
    )
    with JAXScheme():
        np.testing.assert_allclose(
            jax.grad(lambda z: evaluator(z, jnp.asarray(0.8)))(jnp.asarray(0.7)), 0.8
        )
        np.testing.assert_allclose(
            jax.jit(evaluator)(jnp.asarray(0.7), jnp.asarray(0.8)), 0.56
        )


def test_selected_device_survives_ambient_override_in_numerical_controls():
    import os
    import subprocess
    import sys

    code = """
import numpy as np
import jax
from scipy.interpolate import RectBivariateSpline
from pycbc.scheme import JAXScheme
from pycbc.types.array_jax import to_jax
from pycbc.inference.models.tools_jax import inner, draw_sample, rect_bivariate_spline_evaluator
x=np.linspace(.1,5.,6);y=np.linspace(.2,4.,6)
evaluate=rect_bivariate_spline_evaluator(RectBivariateSpline(x,y,np.outer(x,y)))
for controls in ((),('inner','inference_sampling','inference_interpolant')):
 with JAXScheme('cpu:1',reference_operations=controls):
  values=to_jax(np.ones(6,np.complex128))
  with jax.default_device(jax.devices('cpu')[0]):
   results=(inner(values,values),draw_sample(values.real,size=2,host=False),evaluate(values.real,values.real))
  assert all(result.device==jax.devices('cpu')[1] for result in results)
"""
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=2",
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def _relative_dominant_model(active):
    from pycbc.inference.models.relbin import RelativeTimeDom
    from pycbc.inference.models.relative_jax import JAXRelativeTimeDom

    rng = np.random.default_rng(205)
    wave = lambda: rng.normal(size=3) + 1j * rng.normal(size=3)
    values = dict(
        hp=wave(),
        hc=wave(),
        h00=wave(),
        a0=wave()[:2],
        a1=wave()[:2],
        b0=np.ones(2),
        b1=np.full(2, 0.1),
    )
    model = object.__new__(JAXRelativeTimeDom if active else RelativeTimeDom)
    model.sample_rate = 1000.0
    model.tstart = {"H1": -0.01}
    model.end_time = {"H1": 0.0}
    model.ta = {"H1": 0.0}
    model.num_samples = {"H1": 8}
    model.fedges = {"H1": np.asarray([20.0, 80.0, 200.0])}
    model.h00_sparse = {"H1": values["h00"]}
    model.sdat = {"H1": {key: values[key] for key in ("a0", "a1", "b0", "b1")}}
    if active:
        model._get_jax_likelihood_data = lambda ifo, hp: tuple(
            jnp.asarray(v)
            for v in (
                model.fedges[ifo],
                model.h00_sparse[ifo],
                values["a0"],
                values["a1"],
                values["b0"],
                values["b1"],
            )
        )
    wfs = {
        "H1": tuple(
            jnp.asarray(values[key]) if active else values[key] for key in ("hp", "hc")
        )
    }
    return model, wfs


def test_actual_relative_dominant_snr_controls_compose(inference_device):
    outputs = []
    for context in (
        CPUScheme(),
        JAXScheme(
            inference_device,
            reference_operations=("relbin_snr", "relbin_snr_normalization", "divide"),
        ),
    ):
        with context:
            model, wfs = _relative_dominant_model(isinstance(context, JAXScheme))
            result = model.get_snr(wfs)["H1"]
            outputs.append((_bytes(result.numpy()), result.delta_t, str(result._epoch)))
    assert outputs[0] == outputs[1]


def test_actual_relative_dominant_likelihood_controls_compose(inference_device):
    outputs = []
    controls = (
        "relbin_snr",
        "relbin_snr_normalization",
        "relbin_likelihood",
        "inference_projection",
        "inference_time_interpolation",
        "divide",
    )
    for context in (
        CPUScheme(),
        JAXScheme(inference_device, reference_operations=controls),
    ):
        with context:
            model, wfs = _relative_dominant_model(isinstance(context, JAXScheme))
            model._current_params = dict(
                inclination=0.41, polarization=0.31, tc=-0.007, ra=1.0, dec=0.2
            )
            model.marginalize_vector_params = {}
            model.vsamples = 1
            model.get_waveforms = lambda *args, **kwargs: wfs
            # Hold the proposed parameters fixed while testing the actual likelihood terms.
            model.snr_draw = lambda **kwargs: None
            model.precalc_antenna_factors = True
            model.get_precalc_antenna_factors = lambda ifo: (0.35687123, 0.673249, 0.0)
            model.return_sh_hh = True
            model.marginalize_loglr = lambda sh, hh: np.real(sh) - 0.5 * hh
            model._data = {"H1": None}
            model._static_params = {}
            model.waveform_transforms = None
            outputs.append(tuple(_bytes(value) for value in model._loglr()))
    assert outputs[0] == outputs[1]


def test_original_time_interpolation_keeps_extrapolation_option():
    from pycbc.types import TimeSeries
    from pycbc.inference.models.marginalized_gaussian_noise_jax import _time_value

    with JAXScheme(reference_operations=("inference_time_interpolation",)):
        samples = TimeSeries(
            np.asarray([1.0 + 2j, 3.0 + 4j]), delta_t=0.125, epoch=100.0
        )
        assert complex(_time_value(samples, 103.0, extrapolate=0.0j)) == 0.0j


def test_original_relative_polarization_projection_is_independent(inference_device):
    rng = np.random.default_rng(81)
    fp, fc = rng.normal(size=(2, 100))
    pol = rng.uniform(0.0, 2 * np.pi, 100)
    inc = rng.uniform(0.0, np.pi, 100)
    phase = np.exp(-2j * pol)
    rotated = (fp + 1j * fc) * phase
    cosine = np.cos(inc)
    plus = 0.5 * (1 + cosine * cosine)
    expected = rotated.real * plus + 1j * rotated.imag * cosine
    with JAXScheme(inference_device, reference_operations=("inference_projection",)):
        like = jnp.ones(100, jnp.complex128)
        actual_phase = relbin_jax.polarization_phase(jnp.asarray(pol), like)
        real, imag = relbin_jax.polarized_antenna_response(
            jnp.asarray(fp), jnp.asarray(fc), actual_phase, like
        )
        actual = relbin_jax.dominant_mode_projection(
            jnp.asarray(fp), jnp.asarray(fc), jnp.asarray(pol), jnp.asarray(inc), like
        )
    assert _bytes(actual_phase) == _bytes(phase)
    assert _bytes(real + 1j * imag) == _bytes(rotated)
    assert _bytes(actual) == _bytes(expected)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_phase_reconstruction_sampling_control_restores_original_grid(
    dtype, inference_device
):
    from pycbc.inference.models.proposals_jax import _phase_reconstruction_values

    sh = np.asarray(1.234567 + 0.4321j, dtype=dtype)
    hh = np.asarray(0.72, dtype=sh.real.dtype)
    phase = np.linspace(0, 2 * np.pi, 17)
    expected = (np.exp(-2j * phase) * sh).real + hh
    with JAXScheme(inference_device):
        default_phase, _ = _phase_reconstruction_values(
            jnp.asarray(sh), jnp.asarray(hh), 17
        )
        assert default_phase.dtype == phase.dtype
    with JAXScheme(inference_device, reference_operations=("inference_sampling",)):
        actual_phase, actual = _phase_reconstruction_values(
            jnp.asarray(sh), jnp.asarray(hh), 17
        )
    assert _bytes(actual_phase) == _bytes(phase)
    assert _bytes(actual) == _bytes(expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sampling_control_restores_reconstruction_weights(dtype, inference_device):
    from pycbc.inference.models.proposals_jax import _weighted_loglr

    rng = np.random.default_rng(881)
    values = rng.normal(size=100).astype(dtype)
    weights = rng.uniform(0.1, 1.0, 100).astype(dtype)
    with JAXScheme(inference_device, reference_operations=("inference_sampling",)):
        actual = _weighted_loglr(jnp.asarray(values), weights)
    assert _bytes(actual) == _bytes(values + np.log(weights))
