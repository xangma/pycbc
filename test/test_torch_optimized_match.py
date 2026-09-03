import numpy as np
import pytest

torch = pytest.importorskip("torch")
scipy_optimize = pytest.importorskip("scipy.optimize")

from pycbc import scheme
from pycbc.filter import matchedfilter
from pycbc.types import FrequencySeries
from pycbc.types.array_torch import TorchArrayData


@pytest.fixture(params=("cpu", "cuda", "mps"))
def torch_device_context(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    if device == "mps":
        if not torch.backends.mps.is_available():
            pytest.skip("Torch MPS device unavailable")
        pytest.skip("Torch MPS does not support complex PyCBC arrays")

    context = scheme.TorchScheme(device)
    try:
        yield context, device
    finally:
        del context
        scheme.Scheme._single = None


def _scipy_brent(function, bracket, **_):
    return scipy_optimize.minimize_scalar(
        function, method="brent", bracket=bracket
    ).x


def _waveform_pair(dtype, seed, phase_rotation):
    rng = np.random.default_rng(seed)
    frequency_count = 1025
    delta_f = 0.75
    frequencies = np.arange(frequency_count) * delta_f
    amplitude = np.exp(
        -0.5 * ((frequencies - 180.0) / 70.0) ** 2
    )
    amplitude += 0.12 * np.exp(
        -0.5 * ((frequencies - 430.0) / 35.0) ** 2
    )
    random_phase = np.cumsum(rng.normal(0.0, 0.012, frequency_count))
    phase = 0.009 * frequencies + 2e-5 * frequencies**2 + random_phase
    data = (amplitude * np.exp(1j * phase)).astype(dtype)

    waveform = FrequencySeries(data, delta_f=delta_f)
    sample_shift = rng.integers(1, 20) + rng.uniform(-0.49, 0.49)
    shifted = waveform.cyclic_time_shift(sample_shift * waveform.delta_t)
    shifted *= np.exp(1j * phase_rotation)
    return waveform, shifted, frequencies


@pytest.mark.parametrize(
    "dtype,use_psd,cutoffs,seed,phase_rotation",
    (
        (np.complex64, False, (None, None), 13, np.pi - 2e-6),
        (np.complex64, True, (20.0, 600.0), 27, -np.pi + 2e-6),
        (np.complex128, False, (35.0, 500.0), 41, 0.73),
        (np.complex128, True, (10.0, None), 59, -1.91),
    ),
)
def test_torch_optimized_match_matches_scipy_controller(
    torch_device_context,
    monkeypatch,
    dtype,
    use_psd,
    cutoffs,
    seed,
    phase_rotation,
):
    context, device = torch_device_context
    with context:
        waveform, shifted, frequencies = _waveform_pair(
            dtype, seed, phase_rotation
        )
        if use_psd:
            real_dtype = np.empty((), dtype=dtype).real.dtype
            psd = FrequencySeries(
                (1.0 + (frequencies / 240.0) ** 2).astype(real_dtype),
                delta_f=waveform.delta_f,
            )
        else:
            psd = None

        with monkeypatch.context() as patch:
            patch.setattr(matchedfilter, "_brent_minimum", _scipy_brent)
            expected = matchedfilter.optimized_match(
                waveform,
                shifted,
                psd=psd,
                low_frequency_cutoff=cutoffs[0],
                high_frequency_cutoff=cutoffs[1],
                return_phase=True,
            )

        def reject_scipy(*_args, **_kwargs):
            raise AssertionError("optimized_match called SciPy")

        with monkeypatch.context() as patch:
            patch.setattr(scipy_optimize, "minimize_scalar", reject_scipy)
            actual = matchedfilter.optimized_match(
                waveform,
                shifted,
                psd=psd,
                low_frequency_cutoff=cutoffs[0],
                high_frequency_cutoff=cutoffs[1],
                return_phase=True,
            )

    tolerance = 3e-5 if dtype == np.complex64 else 2e-10
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=tolerance
    )
    assert -np.pi <= actual[2] <= np.pi
    assert waveform._data.tensor.device.type == device
    assert shifted._data.tensor.device.type == device


def test_torch_optimized_match_uses_one_scalar_per_objective(
    torch_device_context, monkeypatch
):
    context, _ = torch_device_context
    objective_calls = 0
    item_calls = 0
    real_brent = matchedfilter._brent_minimum
    real_item = torch.Tensor.item

    def counting_brent(function, *args, **kwargs):
        def counted_function(value):
            nonlocal objective_calls
            objective_calls += 1
            return function(value)

        return real_brent(counted_function, *args, **kwargs)

    def counting_item(tensor, *args, **kwargs):
        nonlocal item_calls
        item_calls += 1
        return real_item(tensor, *args, **kwargs)

    with context:
        waveform, shifted, _ = _waveform_pair(
            np.complex128, 71, 0.31
        )

        def reject_host_array(_self):
            raise AssertionError("optimized_match copied a waveform to host")

        with monkeypatch.context() as patch:
            patch.setattr(matchedfilter, "_brent_minimum", counting_brent)
            patch.setattr(torch.Tensor, "item", counting_item)
            patch.setattr(TorchArrayData, "numpy", reject_host_array)
            matchedfilter.optimized_match(
                waveform, shifted, return_phase=True
            )

    # Six scalar reductions belong to the coarse match and normalization.
    # The refinement needs one magnitude scalar per objective evaluation;
    # phase is transferred only with the final result.
    assert item_calls <= objective_calls + 6


@pytest.mark.parametrize("single_bin", (False, True))
def test_degenerate_objective_matches_scipy_fallback(
    torch_device_context, monkeypatch, single_bin
):
    context, _ = torch_device_context
    data = np.zeros(33, dtype=np.complex128)
    if single_bin:
        data[7] = np.exp(0.83j)

    with context:
        waveform = FrequencySeries(data, delta_f=2.0)
        shifted = waveform * np.exp(1.17j)
        if not single_bin:
            with pytest.raises(ZeroDivisionError):
                matchedfilter.optimized_match(
                    waveform, shifted, return_phase=True
                )
            return

        with monkeypatch.context() as patch:
            patch.setattr(matchedfilter, "_brent_minimum", _scipy_brent)
            expected = matchedfilter.optimized_match(
                waveform, shifted, return_phase=True
            )
        actual = matchedfilter.optimized_match(
            waveform, shifted, return_phase=True
        )

    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_bracket_failure_and_flat_recovery_match_scipy():
    def flat(_value):
        return 2.0

    scipy_flat = scipy_optimize.minimize_scalar(
        flat, method="brent", bracket=(-1.0, 1.0)
    )
    assert not scipy_flat.success
    assert matchedfilter._brent_minimum(flat, (-1.0, 1.0)) == scipy_flat.x

    def not_finite(_value):
        return np.nan

    scipy_nan = scipy_optimize.minimize_scalar(
        not_finite, method="brent", bracket=(-1.0, 1.0)
    )
    assert not scipy_nan.success
    assert np.isnan(
        matchedfilter._brent_minimum(not_finite, (-1.0, 1.0))
    )

    with pytest.raises(RuntimeError, match="No valid bracket") as actual:
        matchedfilter._bracket_minimum(
            lambda value: -value,
            -1.0,
            1.0,
            max_iterations=2,
        )
    with pytest.raises(RuntimeError, match="No valid bracket") as expected:
        scipy_optimize.bracket(
            lambda value: -value,
            xa=-1.0,
            xb=1.0,
            maxiter=2,
        )
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize(
    "objective,bracket",
    (
        (lambda value: (value - 0.125) ** 2, (-1.0, 1.0)),
        (lambda value: (value - 8.3) ** 2, (-0.1, 0.1)),
        (
            lambda value: -np.abs(
                np.exp(0.37j * value)
                + 0.41 * np.exp(2.9j * value + 0.2j)
                + 0.13 * np.exp(7.1j * value - 0.7j)
            ),
            (-0.25, 0.25),
        ),
    ),
)
def test_bracket_and_brent_updates_match_scipy(objective, bracket):
    expected_bracket = scipy_optimize.bracket(
        objective, xa=bracket[0], xb=bracket[1]
    )
    actual_bracket = matchedfilter._bracket_minimum(
        objective, bracket[0], bracket[1]
    )
    np.testing.assert_allclose(
        actual_bracket, expected_bracket[:6], rtol=0.0, atol=0.0
    )

    expected_minimum = scipy_optimize.minimize_scalar(
        objective, method="brent", bracket=bracket
    ).x
    actual_minimum = matchedfilter._brent_minimum(objective, bracket)
    assert actual_minimum == expected_minimum
