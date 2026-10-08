# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Unit tests for batched live veto evaluation on JAX devices."""

from types import SimpleNamespace

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp
enable_x64 = getattr(jax, "enable_x64", None)
if enable_x64 is None:
    from jax.experimental import enable_x64

from pycbc import scheme
from pycbc.filter.matchedfilter_jax import _batched_live_vetoes_gpu
from pycbc.types import FrequencySeries
from pycbc.types.array_jax import _ensure_x64, to_jax


def _devices():
    devices = ["cpu"]
    try:
        if jax.devices("gpu"):
            devices.append("cuda:0")
    except RuntimeError:
        pass
    return devices


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("chisq_mode", ["cpu-compatible", "direct-phase"])
def test_batched_live_veto_gpu_equivalence(dev_name, chisq_mode):
    """Batched vetoes must retain the selected scalar chi-square arithmetic."""
    _ensure_x64()
    with scheme.JAXScheme(dev_name, chisq_mode=chisq_mode):
        flen = 16384
        delta_f = 0.25
        kmin = 120
        bins1 = np.array([120, 2000, 6000, 10000, 15000], dtype=np.int32)
        bins2 = np.array([120, 2500, 6200, 10000, 15500], dtype=np.int32)

        rng = np.random.default_rng(2026)
        data1 = (rng.normal(size=flen) +
                 1j * rng.normal(size=flen)).astype(np.complex64)
        data2 = (rng.normal(size=flen) +
                 1j * rng.normal(size=flen)).astype(np.complex64)
        s_data = (rng.normal(size=flen) +
                  1j * rng.normal(size=flen)).astype(np.complex64)

        psd = FrequencySeries(
            np.ones(flen, dtype=np.float32), delta_f=delta_f
        )
        stilde = FrequencySeries(s_data, delta_f=delta_f)
        stilde.psd = psd

        tmpl1 = FrequencySeries(data1, delta_f=delta_f)
        tmpl1.f_lower = kmin * delta_f
        tmpl1.cout = np.zeros(flen, dtype=np.complex64)

        tmpl2 = FrequencySeries(data2, delta_f=delta_f)
        tmpl2.f_lower = kmin * delta_f
        tmpl2.cout = np.zeros(flen, dtype=np.complex64)

        bin_map = {id(tmpl1): bins1, id(tmpl2): bins2}

        class MockPowerChisq:
            do = True
            snr_threshold = None

            def cached_chisq_bins(self, tmpl, psd):
                return bin_map.get(id(tmpl))

        control = SimpleNamespace(
            power_chisq=MockPowerChisq(),
            sg_chisq=SimpleNamespace(do=False),
            newsnr_threshold=None,
        )

        matrix = to_jax(np.stack([data1, data2]))
        veto_info = [
            (np.array([5.5 + 2.1j]), 0.12, 1023, tmpl1, stilde, matrix, 0),
            (np.array([4.2 - 3.7j]), 0.15, 2047, tmpl2, stilde, matrix, 1),
        ]

        results = {
            "snr": jnp.array([abs(5.5 + 2.1j) * 0.12, abs(4.2 - 3.7j) * 0.15])
        }

        batched_res = _batched_live_vetoes_gpu(
            control, dict(results), veto_info, control.power_chisq
        )
        assert batched_res is not None

        # Scalar reference computation
        from pycbc.vetoes.chisq_jax import (
            _point_chisq_cpu_compatible, shift_sum,
        )
        ref_chisq = []
        for info in veto_info:
            snrv, norm, l, tmpl, _ = info[:5]
            corr = np.conj(tmpl) * s_data
            bins = bin_map[id(tmpl)]
            correlation = SimpleNamespace(
                _batch_tensor=to_jax(corr)[None, kmin:bins[-1]],
                _batch_pos=0, _kmin=kmin, _tlen=flen,
            )
            shift = (_point_chisq_cpu_compatible(
                np.asarray(corr), [l], bins, flen)
                if chisq_mode == "cpu-compatible" else
                shift_sum(correlation, np.array([l]), bins))
            nb = len(bins) - 1
            dof = nb * 2 - 2
            raw = (float(shift[0]) * nb - abs(snrv[0]) ** 2) * (norm ** 2)
            ref_chisq.append(raw / dof)

        np.testing.assert_allclose(
            np.asarray(batched_res["chisq"]), ref_chisq, rtol=1e-5, atol=1e-5
        )
        np.testing.assert_array_equal(
            np.asarray(batched_res["chisq_dof"]), [6, 6]
        )
        np.testing.assert_array_equal(
            np.asarray(batched_res["sg_chisq"]), [0.0, 0.0]
        )


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("chisq_mode", ["cpu-compatible", "direct-phase"])
@pytest.mark.parametrize("difference", ["source", "data", "geometry", "bins"])
def test_live_veto_preserves_trigger_sources(dev_name, chisq_mode, difference):
    """Combined live chunks must retain each trigger's template and data."""
    from pycbc.vetoes.chisq_jax import power_chisq_at_points_from_precomputed

    _ensure_x64()
    with scheme.JAXScheme(dev_name, chisq_mode=chisq_mode):
        rng = np.random.default_rng(31)
        infos, bin_map = [], {}
        for index in range(2):
            flen = 1025 if difference != "geometry" or index == 0 else 2049
            delta_f = 0.25 if flen == 1025 else 0.125
            kmin = int(8.0 / delta_f)
            template = FrequencySeries(
                (rng.normal(size=flen) + 1j * rng.normal(size=flen))
                .astype(np.complex64), delta_f=delta_f,
            )
            template.f_lower = 8.0
            template.cout = np.zeros((flen - 1) * 2, np.complex64)
            stilde = FrequencySeries(
                (rng.normal(size=flen) + 1j * rng.normal(size=flen))
                .astype(np.complex64), delta_f=delta_f,
            )
            stilde.psd = FrequencySeries(
                np.full(flen, index + 1, np.float32), delta_f=delta_f)
            bin_map[id(template), id(stilde.psd)] = np.linspace(
                kmin, flen - 1,
                7 if difference == "bins" and index == 1 else 5,
                dtype=np.int32)
            infos.append((np.array([2.0 + 1j], np.complex64), 0.125,
                          511 + index * 512, template, stilde,
                          to_jax(template)[None, :], 0))

        if difference == "source":
            second = infos[1]
            infos[1] = (*second[:4], infos[0][4], *second[5:])
            bin_map[id(second[3]), id(infos[0][4].psd)] = bin_map[
                id(second[3]), id(second[4].psd)]
        elif difference == "data":
            source = jnp.stack([to_jax(info[3]) for info in infos])
            infos = [(*info[:5], source, index)
                     for index, info in enumerate(infos)]

        power = SimpleNamespace(
            snr_threshold=None, cached_chisq_bins=lambda template, psd:
            bin_map[id(template), id(psd)],
        )
        control = SimpleNamespace(sg_chisq=SimpleNamespace(do=False),
                                  newsnr_threshold=None)
        actual = _batched_live_vetoes_gpu(
            control, {"snr": jnp.ones(2)}, infos, power)
        assert actual is not None
        expected = []
        for snrv, norm, point, template, stilde, *_ in infos:
            bins = bin_map[id(template), id(stilde.psd)]
            corr = SimpleNamespace(
                _batch_tensor=(jnp.conj(to_jax(template)) * to_jax(stilde))
                [None, :], _batch_pos=0, _kmin=0,
                _tlen=len(template.cout),
            )
            raw = power_chisq_at_points_from_precomputed(
                corr, snrv, norm, bins, [point])
            expected.append(float(raw[0]) / (2 * (len(bins) - 1) - 2))
        np.testing.assert_allclose(actual["chisq"], expected,
                                   rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("lazy_candidates", [False, True])
@pytest.mark.parametrize("dev_name", _devices())
def test_live_veto_threshold_masks_sine_gaussian_inputs(lazy_candidates,
                                                      dev_name):
    """A gated power veto must supply the scalar zero/sentinel to SG chisq."""
    from pycbc.filter.matchedfilter_jax import _LiveVetoCandidate

    with scheme.JAXScheme(dev_name):
        template = FrequencySeries(np.ones(65, np.complex64), delta_f=1.0)
        template.f_lower = 4.0
        template.cout = np.zeros(128, np.complex64)
        stilde = FrequencySeries(np.ones(65, np.complex64), delta_f=1.0)
        stilde.psd = FrequencySeries(np.ones(65, np.float32), delta_f=1.0)
        inputs = []

        def sg_values(stilde, template, psd, snrv, norm, chisq, dof, points):
            np.testing.assert_array_equal(snrv, [1 + 0j])
            assert float(norm) == 0.1
            inputs.append((chisq, dof))
            return chisq

        power = SimpleNamespace(
            snr_threshold=1.0, cached_chisq_bins=lambda template, psd:
            np.array([4, 24, 44, 64]),
        )
        control = SimpleNamespace(
            sg_chisq=SimpleNamespace(do=True, values=sg_values),
            newsnr_threshold=None,
        )
        info = (np.array([1 + 0j], np.complex64), 0.1, 7, template, stilde)
        if lazy_candidates:
            info = _LiveVetoCandidate(jnp.asarray(info[0]),
                                      jnp.asarray([info[1]]), 0,
                                      *info[2:], None, None)
        result = _batched_live_vetoes_gpu(
            control, {"snr": jnp.array([0.1])},
            [info],
            power,
        )
        assert result is not None
        np.testing.assert_array_equal(inputs[0][0], [0.0])
        np.testing.assert_array_equal(inputs[0][1], [-100])
        np.testing.assert_array_equal(result["chisq"], [0.0])
        np.testing.assert_array_equal(result["sg_chisq"], [0.0])


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("chisq_mode", ["cpu-compatible", "direct-phase"])
@pytest.mark.parametrize("selected", [(1, 2, 0), (2, 0)])
def test_sorted_live_candidates_gather_shared_vectors_without_scalar_reads(
        dev_name, chisq_mode, selected, monkeypatch):
    """Concatenated duration/source groups retain sparse candidate order."""
    from pycbc.filter.matchedfilter_jax import _LiveVetoCandidate

    with scheme.JAXScheme(dev_name, chisq_mode=chisq_mode):
        rng = np.random.default_rng(42)
        infos = []
        for flen, peaks, norms in ((65, [2 + 1j, 3 - 2j], [.125, .25]),
                                   (129, [1 + 4j], [.5])):
            matrix = jnp.asarray((rng.normal(size=(len(peaks), flen))
                                  + 1j * rng.normal(size=(len(peaks), flen)))
                                 .astype(np.complex64))
            stilde = FrequencySeries(np.ones(flen, np.complex64),
                                     delta_f=1.0)
            stilde.psd = FrequencySeries(np.ones(flen, np.float32),
                                         delta_f=1.0)
            snrs = jnp.asarray(peaks, jnp.complex64)
            scales = jnp.asarray(norms)
            for row in range(len(peaks)):
                template = FrequencySeries(np.asarray(matrix[row]),
                                            delta_f=1.0)
                template.f_lower = 4.0
                template.cout = np.zeros((flen - 1) * 2, np.complex64)
                infos.append(_LiveVetoCandidate(
                    snrs, scales, row, 11 + row, template, stilde,
                    matrix, row))
        # process_all concatenates batches, then sorts and truncates by SNR.
        infos = [infos[index] for index in selected]
        legacy = [tuple(info) for info in infos]
        power = SimpleNamespace(
            snr_threshold=None,
            cached_chisq_bins=lambda template, psd: np.linspace(
                4, len(template) - 1, 5, dtype=np.int32))
        control = SimpleNamespace(sg_chisq=SimpleNamespace(do=False),
                                  newsnr_threshold=None)
        results = {"snr": jnp.ones(len(infos))}
        expected = _batched_live_vetoes_gpu(control, dict(results), legacy,
                                          power)
        assert expected is not None
        original = _LiveVetoCandidate.__getitem__

        def reject_scalar_reads(self, index):
            if not isinstance(index, slice) and index in (0, 1):
                raise AssertionError("batched veto eagerly read a scalar row")
            return original(self, index)

        with monkeypatch.context() as patch:
            patch.setattr(_LiveVetoCandidate, "__getitem__",
                          reject_scalar_reads)
            actual = _batched_live_vetoes_gpu(control, dict(results), infos,
                                             power)
        assert actual is not None
        for field in ("chisq", "chisq_dof", "sg_chisq"):
            np.testing.assert_array_equal(actual[field], expected[field])


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("selected", [[], [0, 1, 2], [2, 0]])
def test_live_candidate_columns_preserve_phase_dtype_and_order(
        dev_name, selected):
    from pycbc.filter.matchedfilter_jax import _live_candidate_columns

    with scheme.JAXScheme(dev_name):
        peaks = jnp.asarray([3 + 4j, -2 - 3j, -1 + 0j], jnp.complex64)
        norms = jnp.asarray([.25, .5, 2.0], jnp.float64)
        sigmasq = jnp.asarray([1.1, 2.2, 3.3], jnp.float64)
        indices = jnp.asarray(selected, jnp.int32)
        scaled = peaks * norms
        expected = (jnp.abs(scaled[indices]), jnp.angle(scaled[indices]),
                    sigmasq[indices].astype(jnp.float32))
        actual = _live_candidate_columns(scaled, sigmasq, indices)
        for value, reference in zip(actual, expected):
            assert value.dtype == reference.dtype
            np.testing.assert_array_equal(value, reference)


@pytest.mark.parametrize("dev_name", _devices())
def test_live_candidate_bucket_reuses_selection_and_preserves_exact_columns(
        dev_name, monkeypatch):
    from pycbc.filter import matchedfilter_jax as module

    original = jax.stages.Compiled.__call__
    calls = []

    def execute(compiled, *args, **kwargs):
        calls.append(compiled)
        return original(compiled, *args, **kwargs)

    with scheme.JAXScheme(dev_name):
        peaks = jnp.asarray([3 + 4j, -2 - 3j, -1 + 0j, 2 - 1j, 7j],
                            jnp.complex64)
        norms = jnp.asarray([.25, .5, 2.0, .1, .125], jnp.float64)
        sigmasq = jnp.asarray([1.1, 2.2, 3.3, 4.4, 5.5], jnp.float64)
        scaled = peaks * norms
        power = SimpleNamespace()
        control = SimpleNamespace(power_chisq=power)
        monkeypatch.setattr(jax.stages.Compiled, "__call__", execute)
        loaded = None
        empty = None
        for selected in ([], [], [2], [4, 2, 0], [3, 1, 4, 2],
                         [4, 2, 0, 3, 1], []):
            start = len(calls)
            previous_cache = getattr(control, "_jax_live_veto_executables", None)
            previous_power_cache = getattr(
                power, "_jax_live_veto_executables", None)
            previous_handles = (dict(previous_cache)
                                if previous_cache is not None else None)
            actual = module._live_candidate_columns_bucketed(
                control, scaled, peaks, norms, sigmasq, selected)
            jax.block_until_ready(actual)
            indices = jnp.asarray(selected, jnp.int32)
            expected = (abs(scaled[indices]), jnp.angle(scaled[indices]),
                        sigmasq[indices].astype(jnp.float32), peaks[indices],
                        norms[indices])
            for value, reference in zip(actual, expected):
                assert value.shape == (len(selected),)
                assert value.dtype == reference.dtype
                np.testing.assert_array_equal(value, reference)
            if not selected:
                assert len(calls) == start
                assert getattr(control, "_jax_live_veto_executables", None) \
                    is previous_cache
                assert getattr(power, "_jax_live_veto_executables", None) \
                    is previous_power_cache
                if previous_cache is not None:
                    assert previous_cache.keys() == previous_handles.keys()
                    assert all(previous_cache[key] is value for key, value
                               in previous_handles.items())
                if empty is not None:
                    assert all(value is previous for value, previous
                               in zip(actual, empty))
                empty = actual
                continue
            cache = control._jax_live_veto_executables
            assert cache is power._jax_live_veto_executables
            handles = [value for key, value in cache.items()
                       if key[0] is module._live_candidate_columns]
            assert len(handles) == (1 if len(selected) <= 4 else 2)
            if loaded is None:
                loaded = handles[0]
            assert handles[0] is loaded
            # Selection stays bucketed; only exact-length publication has a
            # count-specific handle, and a full bucket needs no final slice.
            assert len(calls) - start == (1 if len(selected) == 4 else 2)
            assert calls[start] is (loaded if len(selected) <= 4 else handles[1])
            assert all(key[2][-1][0] in ((4,), (8,)) for key in cache
                       if key[0] is module._live_candidate_columns)


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("x64", [False, True])
@pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("norm_dtype", [np.float32, np.float64])
def test_live_empty_candidates_match_retained_legacy_without_loading(
        dev_name, x64, complex_dtype, norm_dtype, monkeypatch):
    from pycbc.filter import matchedfilter_jax as module
    from pycbc.vetoes import chisq_jax

    with scheme.JAXScheme(dev_name), enable_x64(x64):
        complex_dtype = jax.dtypes.canonicalize_dtype(complex_dtype)
        norm_dtype = jax.dtypes.canonicalize_dtype(norm_dtype)
        peaks = jnp.asarray([complex(np.nan, np.inf), complex(-0., 0.)],
                            dtype=complex_dtype)
        norms = jnp.asarray([np.nan, -0.], dtype=norm_dtype)
        sigmasq = jnp.asarray([np.inf, -0.], dtype=norm_dtype)
        scaled = peaks * norms
        args = (scaled, peaks, norms, sigmasq, [])
        with monkeypatch.context() as legacy:
            legacy.setattr(module, "_live_empty_candidate_device",
                           lambda arrays: None)
            expected = module._live_candidate_columns_bucketed(
                SimpleNamespace(), *args)
            jax.block_until_ready(expected)
        module._live_empty_candidate_columns.cache_clear()

        def reject(*args, **kwargs):
            raise AssertionError("empty publication entered executable setup")

        monkeypatch.setattr(chisq_jax, "_live_chisq_executable", reject)
        monkeypatch.setattr(jax, "device_get", reject)
        cache = {"sentinel": object()}
        control = SimpleNamespace(_jax_live_veto_executables=cache)
        actual = module._live_candidate_columns_bucketed(control, *args)
        again = module._live_candidate_columns_bucketed(control, *args)
        assert control._jax_live_veto_executables is cache
        assert len(cache) == 1
        assert module._live_empty_candidate_columns.cache_info().maxsize == 32
        assert module._live_empty_candidate_columns.cache_info().hits == 1
        for value, repeated, reference in zip(actual, again, expected):
            assert value is repeated
            assert value.shape == reference.shape == (0,)
            assert value.dtype == reference.dtype
            assert value.weak_type == reference.weak_type is False
            assert value.committed == reference.committed
            assert value.device == reference.device
            assert np.asarray(value).tobytes() == np.asarray(reference).tobytes()


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("warm", [False, True])
def test_live_empty_candidates_closed_over_trace_keeps_concrete_cache(
        dev_name, warm):
    from pycbc.filter import matchedfilter_jax as module

    with scheme.JAXScheme(dev_name):
        peaks = jnp.asarray([1 + 2j, 3 + 4j], dtype=jnp.complex64)
        norms = jnp.asarray([.5, .25], dtype=jnp.float64)
        sigma = jnp.asarray([2., 3.], dtype=jnp.float64)
        scaled = peaks * norms
        control = SimpleNamespace()
        module._live_empty_candidate_columns.cache_clear()

        def publish():
            return module._live_candidate_columns_bucketed(
                control, scaled, peaks, norms, sigma, [])

        if warm:
            publish()
        traced = jax.jit(publish)()
        direct = publish()
        assert all(isinstance(value, jax.Array)
                   and not isinstance(value, jax.core.Tracer)
                   and value.shape == (0,) for value in direct)
        for value, reference in zip(traced, direct):
            assert value.dtype == reference.dtype
            assert value.shape == reference.shape
        assert not hasattr(control, "_jax_live_veto_executables")


@pytest.mark.parametrize("dev_name", _devices())
def test_live_empty_candidates_integer_scaled_keeps_legacy_angle_dtype(
        dev_name, monkeypatch):
    from pycbc.filter import matchedfilter_jax as module

    with scheme.JAXScheme(dev_name):
        scaled = jnp.asarray([1, -2], dtype=jnp.int32)
        peaks = jnp.asarray([1 + 2j, 3 + 4j], dtype=jnp.complex64)
        norms = jnp.asarray([.5, .25], dtype=jnp.float64)
        sigma = jnp.asarray([2., 3.], dtype=jnp.float64)

        def reject(*args, **kwargs):
            raise AssertionError("unsupported dtype entered empty cache")

        monkeypatch.setattr(module, "_live_empty_candidate_columns", reject)
        control = SimpleNamespace()
        actual = module._live_candidate_columns_bucketed(
            control, scaled, peaks, norms, sigma, [])
        assert actual[0].dtype == jnp.int32
        assert actual[1].dtype == jnp.float64
        assert len(control._jax_live_veto_executables) == 2


def test_live_empty_candidate_eligibility_rejects_tracers_and_bad_geometry():
    from pycbc.filter import matchedfilter_jax as module

    with scheme.JAXScheme("cpu"):
        peaks = jnp.asarray([1 + 2j, 3 + 4j], dtype=jnp.complex64)
        norms = jnp.asarray([.5, .25], dtype=jnp.float64)
        arrays = (peaks, peaks, norms, norms)
        assert module._live_empty_candidate_device(arrays) == peaks.device
        for index in range(4):
            for replacement in (arrays[index][:0], arrays[index][:1],
                                arrays[index].reshape((1, 2))):
                malformed = arrays[:index] + (replacement,) + arrays[index + 1:]
                assert module._live_empty_candidate_device(malformed) is None

        weak = jax.lax.broadcast_in_dim(jnp.asarray(1 + 2j), (2,), ())
        assert weak.weak_type
        assert module._live_empty_candidate_device(
            (weak, peaks, norms, norms)) is None

        def trace(value):
            assert module._live_empty_candidate_device(
                (value, peaks, norms, norms)) is None
            return value

        jax.make_jaxpr(trace)(peaks)


def test_live_norm_metadata_does_not_materialize_lazy_template_rows(monkeypatch):
    from pycbc.filter.matchedfilter_jax import (
        batch_template_power_jax, live_template_norms_jax,
    )
    from pycbc.waveform.bank_jax import LazyFrequencySeries

    with scheme.JAXScheme("cpu"):
        source = jnp.asarray([[0, 1 + 2j, 3 + 4j, 0],
                              [0, 2 + 3j, 4 + 5j, 0]], jnp.complex64)
        templates = [LazyFrequencySeries(source, row, .5) for row in range(2)]
        for template in templates:
            template.f_lower = .5
            template.end_frequency = 1.5
        psd = FrequencySeries(np.ones(4, np.float32), delta_f=.5)
        power = batch_template_power_jax(source, .5)

        def reject_row(*args):
            raise AssertionError("geometry read materialized a lazy row")

        monkeypatch.setattr(LazyFrequencySeries, "_data", property(reject_row))
        actual = live_template_norms_jax(
            templates, psd, template_matrix=source, template_power=power)
        np.testing.assert_array_equal(actual, [60., 108.])
        assert all(template._data_inst is None for template in templates)


def _live_bucket_candidates(count):
    """Original-grid rows with sparse, reordered candidate metadata."""
    from pycbc.filter.matchedfilter_jax import _LiveVetoCandidate

    rng = np.random.default_rng(613)
    n_time, flen, base_k = 4096, 2049, 37
    source = (rng.normal(size=(18, flen))
              + 1j * rng.normal(size=(18, flen))).astype(np.complex64)
    # These frequencies lie beyond every bin edge and must not contribute.
    source[:, 237:] *= 1000
    source = jnp.asarray(source)
    stilde = FrequencySeries(
        (rng.normal(size=flen) + 1j * rng.normal(size=flen))
        .astype(np.complex64), delta_f=.5)
    stilde.psd = FrequencySeries(np.ones(flen, np.float32), delta_f=.5)
    snrs = jnp.asarray([3 + .1 * row + (2 + .1 * row) * 1j
                        for row in range(18)], jnp.complex64)
    norms = jnp.asarray([.1 + .01 * row for row in range(18)], jnp.float64)
    order = [15, 2, 17, 0, 13, 6, 8, 1, 16, 11, 7, 4, 10, 14, 3, 12, 9, 5]
    points = [0, 3, 127, 4095, 2048]
    bins, infos = {}, []
    for index, row in enumerate(order[:count]):
        template = FrequencySeries(np.asarray(source[row]), delta_f=.5)
        template.f_lower = base_k * .5
        template.cout = np.zeros(n_time, np.complex64)
        bins[id(template)] = np.array(
            [37, 37, 69, 153, 211] if row % 2 else
            [37, 91, 91, 173, 237], dtype=np.int32)
        infos.append(_LiveVetoCandidate(
            snrs, norms, row, points[index % len(points)], template,
            stilde, source, row))
    power = SimpleNamespace(
        snr_threshold=None,
        cached_chisq_bins=lambda template, psd: bins[id(template)])
    control = SimpleNamespace(sg_chisq=SimpleNamespace(do=False),
                              newsnr_threshold=None)
    return control, power, infos, bins


def _live_bin_group_candidates(dev_name, monkeypatch):
    """Different original grids and bin counts in interleaved source order."""
    from pycbc.filter.matchedfilter_jax import _LiveVetoCandidate
    from pycbc.vetoes import chisq_jax
    from pycbc.vetoes.chisq import SingleDetPowerChisq

    rng = np.random.default_rng(7561)
    sources = [jnp.asarray((rng.normal(size=(6, nfreq))
                            + 1j * rng.normal(size=(6, nfreq)))
                           .astype(np.complex64))
               for nfreq in (129, 257, 129)]
    psds = [FrequencySeries(np.linspace(1, 3, nfreq, dtype=np.float32),
                            delta_f=.5) for nfreq in (129, 257)]
    if dev_name == "cpu":
        # Exercise CUDA scheduling on a CPU-only runner. Arithmetic still uses
        # real JAX executables; only hardware qualification is simulated.
        class CudaSource:
            device = SimpleNamespace(platform="gpu")

            def __init__(self, array):
                self.array = array

        original = chisq_jax.to_jax
        monkeypatch.setattr(
            chisq_jax, "to_jax",
            lambda value, *args, **kwargs: original(
                value.array if isinstance(value, CudaSource) else value,
                *args, **kwargs))
        wrapped_sources = [CudaSource(source) for source in sources]
    else:
        wrapped_sources = sources
    snrs, norms = jnp.ones(6, jnp.complex64), jnp.ones(6, jnp.float64)
    infos = []
    for source_index, row, num_bins in (
            (0, 3, 4), (1, 2, 4), (0, 0, 6),
            (2, 4, 4), (1, 1, 4), (0, 4, 4)):
        template = FrequencySeries(np.asarray(sources[source_index][row]),
                                   delta_f=.5)
        template.params = SimpleNamespace(num_bins=num_bins)
        template.f_lower = 4. + row * .5
        template.approximant = "IMRPhenomD"
        psd = psds[source_index % 2]
        stilde = SimpleNamespace(psd=psd)
        infos.append(_LiveVetoCandidate(
            snrs, norms, row, 0, template, stilde,
            wrapped_sources[source_index], row))
    return SingleDetPowerChisq("params.num_bins"), infos


def _serial_live_bin_reference(power, infos):
    """Use the immediate-collection API on the same original grids."""
    from pycbc.vetoes.chisq_jax import cache_batch_power_chisq_bins_jax
    from pycbc.waveform.bank_jax import TemplateBatchList

    expected = []
    for info in infos:
        template, stilde, source, position = info[3:7]
        batch = TemplateBatchList([template])
        batch._batch_tensor = source
        batch._batch_positions = [position]
        edges = cache_batch_power_chisq_bins_jax(power, batch, stilde.psd)
        expected.append(edges[0].copy())
    for info in infos:
        info[3]._bin_cache.clear()
        info[4].psd._chisq_cached_key.clear()
    return expected


@pytest.mark.parametrize("dev_name", _devices())
def test_live_bin_groups_launch_before_one_collection_and_reuse_cache(
        dev_name, monkeypatch):
    from pycbc.filter.matchedfilter_jax import _cache_live_veto_bins_jax

    with scheme.JAXScheme(dev_name):
        power, infos = _live_bin_group_candidates(dev_name, monkeypatch)
        expected = _serial_live_bin_reference(power, infos)
        handles = dict(power._jax_live_veto_executables)
        events = []
        original_execute = jax.stages.Compiled.__call__
        original_get = jax.device_get

        def execute(compiled, *args, **kwargs):
            events.append("launch")
            return original_execute(compiled, *args, **kwargs)

        def collect(values):
            assert isinstance(values, tuple)
            assert len(values) == 4
            assert events == ["launch"] * 4
            assert all(not info[3]._bin_cache for info in infos)
            events.append("collect")
            return original_get(values)

        monkeypatch.setattr(jax.stages.Compiled, "__call__", execute)
        monkeypatch.setattr(jax, "device_get", collect)
        _cache_live_veto_bins_jax(power, infos)
        assert events == ["launch"] * 4 + ["collect"]
        for info, reference in zip(infos, expected):
            template, stilde = info[3:5]
            actual = template._bin_cache[id(stilde.psd)]
            assert actual.dtype == reference.dtype
            assert actual.tobytes() == reference.tobytes()
            assert id(template.params) in stilde.psd._chisq_cached_key
        assert power._jax_live_veto_executables.keys() == handles.keys()
        assert all(power._jax_live_veto_executables[key] is value
                   for key, value in handles.items())
        _cache_live_veto_bins_jax(power, infos[::-1])
        assert events == ["launch"] * 4 + ["collect"]


@pytest.mark.parametrize("dev_name", _devices())
def test_live_bin_pending_duplicate_keeps_first_source_cache(dev_name,
                                                           monkeypatch):
    from pycbc.filter.matchedfilter_jax import (
        _LiveVetoCandidate, _cache_live_veto_bins_jax,
    )

    with scheme.JAXScheme(dev_name):
        power, infos = _live_bin_group_candidates(dev_name, monkeypatch)
        expected = _serial_live_bin_reference(power, infos[:1])[0]
        first = infos[0]
        # Sequential publication skipped a later source for this template.
        duplicate = _LiveVetoCandidate(
            first.snrs, first.norms, first.row, first[2], first[3],
            first[4], infos[3][5], 0)
        gets = []
        original_get = jax.device_get

        def collect(values):
            gets.append(len(values))
            return original_get(values)

        monkeypatch.setattr(jax, "device_get", collect)
        _cache_live_veto_bins_jax(power, [first, duplicate])
        assert gets == [1]
        assert first[3]._bin_cache[id(first[4].psd)].tobytes() == (
            expected.tobytes())


@pytest.mark.parametrize("dev_name", _devices())
def test_live_bin_overlapping_groups_preserve_bounded_psd_cache_eviction(
        dev_name, monkeypatch):
    from pycbc.filter.matchedfilter_jax import (
        _LiveVetoCandidate, _cache_live_veto_bins_jax,
    )

    with scheme.JAXScheme(dev_name):
        power, infos = _live_bin_group_candidates(dev_name, monkeypatch)
        first = infos[0]
        psds = [first[4].psd] + [
            FrequencySeries(np.linspace(1, 3, len(first[3]),
                                         dtype=np.float32), delta_f=.5)
            for _ in range(4)]
        repeated = [_LiveVetoCandidate(
            first.snrs, first.norms, first.row, 0, first[3],
            SimpleNamespace(psd=psd), first[5], first[6]) for psd in psds]
        repeated.append(_LiveVetoCandidate(
            first.snrs, first.norms, first.row, 0, first[3],
            SimpleNamespace(psd=psds[0]), infos[3][5], 0))
        expected = _serial_live_bin_reference(power, repeated)
        assert expected[0].tobytes() != expected[-1].tobytes()
        gets = []
        original_get = jax.device_get

        def collect(values):
            gets.append(len(values))
            return original_get(values)

        monkeypatch.setattr(jax, "device_get", collect)
        _cache_live_veto_bins_jax(power, repeated)
        assert gets == [1] * 6
        assert len(first[3]._bin_cache) == 4
        assert id(psds[1]) not in first[3]._bin_cache
        assert all(id(psd) in first[3]._bin_cache for psd in psds[2:])
        assert first[3]._bin_cache[id(psds[0])].tobytes() == (
            expected[-1].tobytes())


@pytest.mark.parametrize("dev_name", _devices())
def test_live_bin_cache_rebuilds_for_params_and_psd_identity(dev_name,
                                                          monkeypatch):
    from pycbc.filter.matchedfilter_jax import _cache_live_veto_bins_jax

    with scheme.JAXScheme(dev_name):
        power, infos = _live_bin_group_candidates(dev_name, monkeypatch)
        expected = _serial_live_bin_reference(power, infos)
        _cache_live_veto_bins_jax(power, infos)
        template, stilde = infos[0][3:5]
        old_params = template.params
        template.params = SimpleNamespace(num_bins=old_params.num_bins)
        old_edges = template._bin_cache[id(stilde.psd)]
        old_handles = dict(power._jax_live_veto_executables)
        gets = []
        original_get = jax.device_get

        def collect(values):
            gets.append(len(values))
            return original_get(values)

        monkeypatch.setattr(jax, "device_get", collect)
        _cache_live_veto_bins_jax(power, infos)
        assert gets == [1]
        assert template._bin_cache[id(stilde.psd)] is not old_edges
        assert template._bin_cache[id(stilde.psd)].tobytes() == (
            expected[0].tobytes())
        assert id(template.params) in stilde.psd._chisq_cached_key
        new_psd = FrequencySeries(np.linspace(3, 1, len(stilde.psd),
                                              dtype=np.float32), delta_f=.5)
        stilde.psd = new_psd
        _cache_live_veto_bins_jax(power, infos)
        assert gets == [1, 1]
        assert id(new_psd) in template._bin_cache
        assert id(template.params) in new_psd._chisq_cached_key
        assert id(new_psd) not in infos[2][3]._bin_cache
        assert all(power._jax_live_veto_executables[key] is value
                   for key, value in old_handles.items())


@pytest.mark.parametrize("callback", ["option", "analytic"])
def test_live_bin_fallback_callbacks_see_earlier_published_cache(callback,
                                                              monkeypatch):
    from pycbc.vetoes.chisq_jax import cache_batch_power_chisq_bins_jax
    from pycbc.waveform.bank_jax import TemplateBatchList

    with scheme.JAXScheme("cpu"):
        power, infos = _live_bin_group_candidates("cpu", monkeypatch)
        expected = _serial_live_bin_reference(power, infos[:2])
        first, second = infos[:2]
        pending = [(jnp.asarray(expected[0][None, :]), [first[3]],
                    first[4].psd)]
        calls = []

        def check_prior():
            calls.append(callback)
            assert first[3]._bin_cache[id(first[4].psd)].tobytes() == (
                expected[0].tobytes())
            assert id(first[3].params) in first[4].psd._chisq_cached_key

        if callback == "option":
            original = power.parse_option

            def option(template, argument):
                check_prior()
                return original(template, argument)

            power.parse_option = option
        else:
            second[4].psd.sigmasq_vec = {second[3].approximant: object()}

            def analytic(template, psd):
                check_prior()
                psd._chisq_cached_key[id(template.params)] = True
                template._bin_cache[id(psd)] = expected[1].copy()
                return template._bin_cache[id(psd)]

            power.cached_chisq_bins = analytic

        batch = TemplateBatchList([second[3]])
        batch._batch_tensor = second[5]
        batch._batch_positions = [second[6]]
        got = cache_batch_power_chisq_bins_jax(
            power, batch, second[4].psd, pending=pending)
        assert pending == []
        assert calls == [callback]
        assert got[0].tobytes() == expected[1].tobytes()


def test_live_bin_preparation_error_publishes_earlier_groups(monkeypatch):
    from pycbc.filter.matchedfilter_jax import _cache_live_veto_bins_jax

    with scheme.JAXScheme("cpu"):
        power, infos = _live_bin_group_candidates("cpu", monkeypatch)
        expected = _serial_live_bin_reference(power, infos[:1])[0]
        first, bad = infos[:2]
        bad[3].params = SimpleNamespace()
        with pytest.raises(AttributeError, match="num_bins"):
            _cache_live_veto_bins_jax(power, [first, bad])
        assert first[3]._bin_cache[id(first[4].psd)].tobytes() == (
            expected.tobytes())
        assert id(first[3].params) in first[4].psd._chisq_cached_key
        assert id(bad[4].psd) not in bad[3]._bin_cache


def test_live_bin_collection_error_preserves_ordered_cache_publication(
        monkeypatch):
    from pycbc.vetoes.chisq_jax import _collect_cached_power_chisq_bins_jax

    psd = SimpleNamespace(_chisq_cached_key={})
    templates = [SimpleNamespace(params=object(), _bin_cache={})
                 for _ in range(3)]
    edges = [np.array([[1, 4, 8]], dtype=np.uint32) for _ in templates]
    pending = [(value, [template], psd)
               for value, template in zip(edges, templates)]
    gets = []

    def collect(values):
        if isinstance(values, tuple):
            gets.append("tuple")
            raise RuntimeError("second device group failed")
        if values is edges[1]:
            gets.append("second")
            raise RuntimeError("second device group failed")
        assert values is edges[0], "published after a failed device group"
        gets.append("first")
        return values

    monkeypatch.setattr(jax, "device_get", collect)
    with pytest.raises(RuntimeError, match="second device group"):
        _collect_cached_power_chisq_bins_jax(pending)
    assert gets == ["tuple", "first", "second"]
    assert pending == []
    assert templates[0]._bin_cache[id(psd)].tobytes() == edges[0][0].tobytes()
    assert id(templates[0].params) in psd._chisq_cached_key
    assert templates[1]._bin_cache == templates[2]._bin_cache == {}


@pytest.mark.parametrize("dev_name", _devices())
def test_live_veto_geometry_does_not_resolve_correlation_views(dev_name,
                                                            monkeypatch):
    from pycbc.types import Array
    from pycbc.types.array_jax import JAXArrayData

    with scheme.JAXScheme(dev_name):
        control, power, infos, _ = _live_bucket_candidates(3)
        expected = _batched_live_vetoes_gpu(
            control, {"snr": jnp.ones(3)}, infos, power)
        parent = JAXArrayData(jnp.zeros(3 * 4096, jnp.complex64))
        for row, info in enumerate(infos):
            info[3].cout = Array(parent[row * 4096:(row + 1) * 4096],
                                 copy=False)
        original = JAXArrayData.array.fget

        def reject_view(data):
            if data.parent is not None:
                raise AssertionError("length/dtype read resolved a device view")
            return original(data)

        monkeypatch.setattr(JAXArrayData, "array", property(reject_view))
        actual = _batched_live_vetoes_gpu(
            control, {"snr": jnp.ones(3)}, infos, power)
        assert expected is not None and actual is not None
        for field in ("chisq", "chisq_dof", "sg_chisq"):
            np.testing.assert_array_equal(actual[field], expected[field])


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("chisq_mode", ["cpu-compatible", "direct-phase"])
def test_live_bucket_order_restores_interleaved_truncated_source_groups(
        dev_name, chisq_mode):
    from pycbc.filter.matchedfilter_jax import (
        _bucketed_live_vetoes, _live_restore_veto_order,
    )

    with scheme.JAXScheme(dev_name, chisq_mode=chisq_mode):
        control, power, infos, bins = _live_bucket_candidates(5)
        other_source = infos[0][5] * jnp.complex64(.75 + .125j)
        for info in infos[1::2]:
            point, template, stilde, _, position = info.metadata
            info.metadata = (point, template, stilde, other_source, position)
        loaded = None
        # Simulate the ordering and truncation performed by process_all.
        for indices in ([4, 1, 2], [1, 4]):
            selected = [infos[index] for index in indices]
            expected = _batched_live_vetoes_gpu(
                control, {"snr": jnp.ones(len(selected))},
                [tuple(info) for info in selected], power)
            actual = _bucketed_live_vetoes(
                control, {"snr": jnp.ones(len(selected))}, selected,
                [bins[id(info[3])] for info in selected], power)
            assert expected is not None
            np.testing.assert_allclose(actual["chisq"], expected["chisq"],
                                       rtol=1e-5, atol=1e-5)
            for field in ("chisq_dof", "sg_chisq"):
                np.testing.assert_array_equal(actual[field], expected[field])
            assert actual["chisq"].shape == (len(selected),)
            handles = [value for key, value in
                       power._jax_live_veto_executables.items()
                       if key[0] is _live_restore_veto_order]
            assert len(handles) == 1
            loaded = handles[0] if loaded is None else loaded
            assert handles[0] is loaded


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("chisq_mode", ["cpu-compatible", "direct-phase"])
@pytest.mark.parametrize("count", [1, 3, 4, 5, 14, 16, 17])
def test_bucketed_live_veto_retains_scalar_grid_bins_and_candidate_order(
        dev_name, chisq_mode, count):
    """Masked lanes must preserve the unfused point-veto result and order."""
    from pycbc.vetoes.chisq_jax import power_chisq_at_points_from_precomputed

    with scheme.JAXScheme(dev_name, chisq_mode=chisq_mode):
        control, power, infos, bins = _live_bucket_candidates(count)
        expected = []
        for info in infos:
            snrv, norm, point, template, stilde = info[:5]
            edges = bins[id(template)]
            corr = SimpleNamespace(
                _batch_tensor=(jnp.conj(to_jax(template)) * to_jax(stilde))
                [None, 37:edges[-1]], _batch_pos=0, _kmin=37, _tlen=4096)
            raw = power_chisq_at_points_from_precomputed(
                corr, snrv, norm, edges, [point])
            expected.append(np.asarray(raw)[0] / 6)
        result = _batched_live_vetoes_gpu(
            control, {"snr": jnp.ones(count)}, infos, power)
        assert result is not None
        for field, dtype in (("chisq", jnp.float32),
                             ("chisq_dof", jnp.uint32),
                             ("sg_chisq", jnp.float32)):
            assert result[field].shape == (count,)
            assert result[field].dtype == dtype
        np.testing.assert_allclose(result["chisq"], expected,
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(result["chisq_dof"], 6)
        np.testing.assert_array_equal(result["sg_chisq"], 0)


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("chisq_mode", ["cpu-compatible", "direct-phase"])
@pytest.mark.parametrize("count", [0, 1, 4, 5])
def test_live_veto_group_masks_inactive_bucket_lanes(dev_name, chisq_mode,
                                                  count):
    from pycbc.filter.matchedfilter_jax import _live_veto_group_core
    from pycbc.vetoes.chisq_jax import (
        _use_gpu_ordered_scan, power_chisq_at_points_from_precomputed,
    )

    with scheme.JAXScheme(dev_name, chisq_mode=chisq_mode):
        _, _, infos, bins = _live_bucket_candidates(8)
        first = infos[0]
        # The inactive lanes have genuine nonzero SNR and bin power. Only
        # valid_count may remove them; zero-filled fixtures could hide leaks.
        raw, dof = _live_veto_group_core(
            first[5], to_jax(first[4]), first.snrs, first.norms,
            np.asarray([info[6] for info in infos], np.int32),
            np.asarray([info[2] for info in infos], np.int64),
            np.asarray([info.row for info in infos], np.int32),
            np.asarray([bins[id(info[3])] for info in infos], np.int32),
            np.asarray(count, np.int32), np.asarray(4096, np.int64),
            base_k=37, compatible=chisq_mode == "cpu-compatible",
            use_pallas=_use_gpu_ordered_scan(first[5]), snr_threshold=None)
        assert raw.dtype == jnp.float32
        assert dof.dtype == jnp.int32
        np.testing.assert_array_equal(raw[count:], 0)
        np.testing.assert_array_equal(dof[count:], 0)
        expected = []
        for info in infos[:count]:
            snrv, norm, point, template, stilde = info[:5]
            edges = bins[id(template)]
            corr = SimpleNamespace(
                _batch_tensor=(jnp.conj(to_jax(template)) * to_jax(stilde))
                [None, 37:edges[-1]], _batch_pos=0, _kmin=37, _tlen=4096)
            expected.append(np.asarray(power_chisq_at_points_from_precomputed(
                corr, snrv, norm, edges, [point]))[0])
        np.testing.assert_allclose(raw[:count], expected,
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(dof[:count], 6)


@pytest.mark.parametrize("dev_name", _devices())
def test_bucketed_live_veto_threshold_supplies_scalar_sg_inputs(dev_name):
    """A padded group retains zero/-100 SG inputs for gated real lanes."""
    with scheme.JAXScheme(dev_name):
        control, power, infos, _ = _live_bucket_candidates(3)
        baseline = _batched_live_vetoes_gpu(
            control, {"snr": jnp.ones(3)}, [tuple(info) for info in infos],
            power)
        assert baseline is not None
        expected_raw = np.asarray(baseline["chisq"]) * 6
        power.snr_threshold = .75
        inputs = []

        def sg_values(stilde, template, psd, snrv, norm, raw, dof, points):
            inputs.append((np.asarray(raw).copy(), np.asarray(dof).copy()))
            return raw

        control.sg_chisq = SimpleNamespace(do=True, values=sg_values)
        actual = _batched_live_vetoes_gpu(
            control, {"snr": jnp.ones(3)}, infos, power)
        assert actual is not None
        expected_raw[1] = 0
        expected_dof = np.array([6, -100, 6], np.int32)
        np.testing.assert_allclose(np.concatenate([raw for raw, _ in inputs]),
                                   expected_raw, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(
            np.concatenate([dof for _, dof in inputs]), expected_dof)
        np.testing.assert_allclose(actual["sg_chisq"], expected_raw,
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(actual["chisq_dof"],
                                      expected_dof.astype(np.uint32))
        np.testing.assert_array_equal(actual["chisq"][1:2], 0)


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("chisq_mode", ["cpu-compatible", "direct-phase"])
def test_live_veto_bucket_reuses_numerical_and_order_executables(
        dev_name, chisq_mode, monkeypatch):
    """Warm sparse selections in one bucket reuse the loaded executable."""
    from pycbc.filter.matchedfilter_jax import (
        _bucketed_live_vetoes, _live_veto_group_core, _live_restore_veto_order,
        _live_compact_columns,
    )

    original = jax.stages.Compiled.__call__
    calls = []

    def execute(compiled, *args, **kwargs):
        calls.append(compiled)
        return original(compiled, *args, **kwargs)

    with scheme.JAXScheme(dev_name, chisq_mode=chisq_mode):
        control, power, infos, bins = _live_bucket_candidates(5)
        monkeypatch.setattr(jax.stages.Compiled, "__call__", execute)
        loaded = {}
        for count in (1, 3, 4, 5):
            # Exercise the CUDA dispatch implementation with its portable
            # point kernel on CPU; public CPU execution retains scalar vetoes.
            start = len(calls)
            actual = _bucketed_live_vetoes(
                control, {"snr": jnp.ones(count)}, infos[:count],
                [bins[id(info[3])] for info in infos[:count]], power)
            jax.block_until_ready(actual["chisq"])
            cache = power._jax_live_veto_executables
            assert cache is control._jax_live_veto_executables
            for index, function in enumerate(
                    (_live_veto_group_core, _live_restore_veto_order)):
                handles = [value for key, value in cache.items()
                           if key[0] is function]
                assert len(handles) == (1 if count <= 4 else 2)
                loaded.setdefault(function, handles[0])
                assert handles[0] is loaded[function]
                assert calls[start + index] is handles[0 if count <= 4 else 1]
            assert len(calls) - start == (2 if count == 4 else 3)
            if count != 4:
                compact = [value for key, value in cache.items()
                           if key[0] is _live_compact_columns
                           and key[3] == (("count", count),)]
                assert calls[-1] is compact[0]


@pytest.mark.parametrize("chisq_mode", ["cpu-compatible", "direct-phase"])
def test_cuda_live_veto_demand_loads_and_reuses_bucket_handles(
        chisq_mode, monkeypatch):
    from pycbc.filter import matchedfilter_jax
    from pycbc.vetoes.chisq import SingleDetPowerChisq
    from pycbc.vetoes import chisq_jax
    from pycbc.vetoes.chisq_jax import (
        _power_chisq_bins_from_source, _use_gpu_ordered_scan,
    )

    try:
        jax.devices("gpu")
    except RuntimeError:
        pytest.skip("CUDA JAX device unavailable")
    with scheme.JAXScheme("cuda:0", chisq_mode=chisq_mode):
        control, _, infos, bins = _live_bucket_candidates(18)
        source = infos[0][5]
        if not _use_gpu_ordered_scan(source):
            pytest.skip("CUDA ordered-scan capability unavailable")
        power = control.power_chisq = SingleDetPowerChisq("4")
        templates = [info[3] for info in infos]
        psd = infos[0][4].psd
        psd._chisq_cached_key = {}
        for template in templates:
            template.params = SimpleNamespace()
            template._bin_cache = {id(psd): bins[id(template)]}
            psd._chisq_cached_key[id(template.params)] = True
        calls = []
        original = jax.stages.Compiled.__call__

        def execute(compiled, *args, **kwargs):
            calls.append(compiled)
            return original(compiled, *args, **kwargs)

        def reject_loading(*args, **kwargs):
            raise AssertionError("workspace setup loaded a veto executable")

        def reject_readback(*args, **kwargs):
            raise AssertionError("workspace setup read back scientific data")

        monkeypatch.setattr(jax.stages.Compiled, "__call__", execute)
        with monkeypatch.context() as patch:
            patch.setattr(chisq_jax, "_live_chisq_executable", reject_loading)
            patch.setattr(jax, "device_get", reject_readback)
            matchedfilter_jax.live_batch_matched_filter_init_jax(
                control, templates)
        assert calls == []
        assert not getattr(control, "_jax_live_veto_executables", {})
        assert not getattr(power, "_jax_live_veto_executables", {})
        numerical_functions = {matchedfilter_jax._live_veto_group_core,
                               _power_chisq_bins_from_source}

        def numerical_cache(cache):
            return {key: value for key, value in cache.items()
                    if key[0] in numerical_functions}

        def reduce_count(count):
            # Rebuild bin values so both numerical executables are exercised.
            start = len(calls)
            psd._chisq_cached_key.clear()
            matchedfilter_jax._cache_live_veto_bins_jax(power, infos[:count])
            result = _batched_live_vetoes_gpu(
                control, {"snr": jnp.ones(count)}, infos[:count], power)
            assert result is not None
            jax.block_until_ready(result["chisq"])
            return calls[start:start + 2]

        numeric_calls = reduce_count(1)
        before = numerical_cache(power._jax_live_veto_executables)
        assert len(before) == 2
        veto_keys = [key for key in before
                     if key[0] is matchedfilter_jax._live_veto_group_core]
        bin_keys = [key for key in before
                    if key[0] is _power_chisq_bins_from_source]
        assert len(veto_keys) == len(bin_keys) == 1
        assert veto_keys[0][2][4][0] == (4,)
        assert bin_keys[0][2][2][0] == (4,)
        loaded_veto = before[veto_keys[0]]
        loaded_bins = before[bin_keys[0]]
        assert numeric_calls == [loaded_bins, loaded_veto]

        for count in (3, 4):
            numeric_calls = reduce_count(count)
            after = numerical_cache(power._jax_live_veto_executables)
            assert after.keys() == before.keys()
            assert all(after[key] is value for key, value in before.items())
            assert numeric_calls == [loaded_bins, loaded_veto]

        # Larger sparse selections load on demand. A different count within
        # that same bucket must then reuse the loaded executable.
        for cold_count, warm_count in ((5, 6), (16, 15), (17, 18)):
            previous = numerical_cache(power._jax_live_veto_executables)
            numeric_calls = reduce_count(cold_count)
            cache = power._jax_live_veto_executables
            added = numerical_cache(cache).keys() - previous.keys()
            assert len(added) == 2
            loaded = {key[0]: cache[key] for key in added}
            assert set(loaded) == {matchedfilter_jax._live_veto_group_core,
                                   _power_chisq_bins_from_source}
            assert numeric_calls == [loaded[_power_chisq_bins_from_source],
                                     loaded[matchedfilter_jax._live_veto_group_core]]
            assert all(cache[key] is value for key, value in previous.items())
            after_load = numerical_cache(cache)
            numeric_calls = reduce_count(warm_count)
            assert numerical_cache(cache).keys() == after_load.keys()
            assert all(cache[key] is value for key, value in after_load.items())
            assert numeric_calls == [loaded[_power_chisq_bins_from_source],
                                     loaded[matchedfilter_jax._live_veto_group_core]]
        # Nine calls execute bins, the numerical core and bucketed order;
        # seven non-full buckets additionally publish compact result columns.
        assert len(calls) == 34


@pytest.mark.parametrize("dev_name", _devices())
@pytest.mark.parametrize("precision", ["snrs", "source", "data", "template"])
def test_live_veto_mixed_precision_retains_exact_legacy_promotion(
        dev_name, precision, monkeypatch):
    from pycbc.filter import matchedfilter_jax

    with scheme.JAXScheme(dev_name):
        _, power, original, bins = _live_bucket_candidates(2)
        infos = list(original)
        second = infos[1]
        if precision == "snrs":
            second.snrs = second.snrs.astype(jnp.complex128)
        elif precision == "source":
            point, template, stilde, source, position = second.metadata
            second.metadata = (point, template, stilde,
                               source.astype(jnp.complex128), position)
        elif precision == "data":
            point, template, stilde, source, position = second.metadata
            wide_data = FrequencySeries(np.asarray(stilde).astype(np.complex128),
                                        delta_f=stilde.delta_f)
            wide_data.psd = stilde.psd
            second.metadata = (point, template, wide_data, source, position)
        else:
            point, template, stilde, source, position = second.metadata
            wide_template = FrequencySeries(
                np.asarray(template).astype(np.complex128),
                delta_f=template.delta_f)
            wide_template.f_lower = template.f_lower
            wide_template.cout = template.cout
            bins[id(wide_template)] = bins[id(template)]
            second.metadata = (point, wide_template, stilde, source, position)
        control = SimpleNamespace(sg_chisq=SimpleNamespace(do=False),
                                  newsnr_threshold=None)
        expected = _batched_live_vetoes_gpu(
            control, {"snr": jnp.ones(2)}, [tuple(info) for info in infos],
            power)
        assert expected is not None

        def reject_fusion(*args, **kwargs):
            raise AssertionError("mixed precision changed legacy promotion")

        monkeypatch.setattr(matchedfilter_jax, "_bucketed_live_vetoes",
                            reject_fusion)
        actual = _batched_live_vetoes_gpu(
            control, {"snr": jnp.ones(2)}, infos, power)
        assert actual is not None
        for field in ("chisq", "chisq_dof", "sg_chisq"):
            assert actual[field].dtype == expected[field].dtype
            np.testing.assert_array_equal(actual[field], expected[field])
