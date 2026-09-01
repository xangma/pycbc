import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.types import Array, FrequencySeries
from pycbc.types.array_torch import TorchArrayData
from pycbc.vetoes.chisq import (
    SingleDetPowerChisq,
    SingleDetSkyMaxPowerChisq,
    power_chisq_at_points_from_precomputed,
)


pytest.importorskip("torch")

if not pycbc.HAVE_TORCH:
    pytest.skip("PyCBC built without torch support", allow_module_level=True)


def test_power_chisq_at_points_torch_matches_cpu_and_stays_real():
    rng = np.random.default_rng(1234)
    corr = (rng.normal(size=128) + 1j * rng.normal(size=128)).astype(np.complex64)
    snr = (rng.normal(size=5) + 1j * rng.normal(size=5)).astype(np.complex64)
    indices = np.array([0, 3, 7, 10, 17], dtype=np.int64)
    bins = np.array([4, 12, 24, 40, 80], dtype=np.int64)
    snr_norm = 0.37

    cpu_corr = FrequencySeries(corr.copy(), delta_f=0.1)
    cpu = power_chisq_at_points_from_precomputed(
        cpu_corr, snr.copy(), snr_norm, bins, indices
    )

    with scheme.TorchScheme("cpu"):
        torch_corr = FrequencySeries(corr.copy(), delta_f=0.1)
        chisq = power_chisq_at_points_from_precomputed(
            torch_corr, snr.copy(), snr_norm, bins, indices
        )

    assert chisq.kind == "real"
    assert isinstance(chisq._data, TorchArrayData)
    np.testing.assert_allclose(chisq.numpy(), cpu, rtol=1e-5, atol=1e-5)


def test_power_chisq_at_points_torch_errors_do_not_fall_back(monkeypatch):
    from pycbc.vetoes import chisq_torch

    corr = FrequencySeries(
        np.ones(32, dtype=np.complex64), delta_f=0.1
    )
    snr = np.ones(2, dtype=np.complex64)
    indices = np.array([1, 4], dtype=np.int64)
    bins = np.array([1, 4, 8], dtype=np.int64)

    def fail_on_device(*args, **kwargs):
        raise RuntimeError("torch chisq failure")

    monkeypatch.setattr(
        chisq_torch,
        "power_chisq_at_points_from_precomputed",
        fail_on_device,
    )

    with scheme.TorchScheme("cpu"):
        with pytest.raises(RuntimeError, match="torch chisq failure"):
            power_chisq_at_points_from_precomputed(
                corr, snr, 1.0, bins, indices
            )


@pytest.mark.parametrize("threshold", [3.0, 100.0])
def test_power_chisq_threshold_output_stays_on_torch(
    monkeypatch, threshold
):
    rng = np.random.default_rng(8123)
    corr = (
        rng.normal(size=64) + 1j * rng.normal(size=64)
    ).astype(np.complex64)
    snr = np.array([4.0 + 0.5j, 2.0 - 0.25j], dtype=np.complex64)
    indices = np.array([3, 11], dtype=np.int64)
    bins = np.array([1, 6, 13, 24], dtype=np.int64)

    monkeypatch.setattr(
        SingleDetPowerChisq,
        "cached_chisq_bins",
        lambda self, template, psd: bins,
    )

    def evaluate(corr_values, snr_values):
        veto = SingleDetPowerChisq(
            num_bins="3", snr_threshold=threshold
        )
        return veto.values(
            corr_values,
            snr_values,
            1.0,
            None,
            indices,
            object(),
        )

    expected, expected_dof = evaluate(
        FrequencySeries(corr.copy(), delta_f=0.1), snr.copy()
    )

    with scheme.TorchScheme("cpu"):
        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "numpy",
                lambda self: (_ for _ in ()).throw(
                    AssertionError("power chisq copied Torch data")
                ),
            )
            actual, actual_dof = evaluate(
                FrequencySeries(corr.copy(), delta_f=0.1),
                Array(snr.copy()),
            )

        assert isinstance(actual._data, TorchArrayData)
        actual_values = actual._data.tensor.cpu().numpy()

    np.testing.assert_allclose(
        actual_values, expected, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_array_equal(actual_dof, expected_dof)


@pytest.mark.parametrize("threshold", [None, 4.0])
def test_skymax_power_chisq_keeps_dense_scratch_on_torch(
    monkeypatch, threshold
):
    rng = np.random.default_rng(9182)
    size = 32
    template_plus = (
        rng.normal(size=size) + 1j * rng.normal(size=size)
    ).astype(np.complex64)
    template_cross = (
        rng.normal(size=size) + 1j * rng.normal(size=size)
    ).astype(np.complex64)
    corr_plus = (
        rng.normal(size=size) + 1j * rng.normal(size=size)
    ).astype(np.complex64)
    corr_cross = (
        rng.normal(size=size) + 1j * rng.normal(size=size)
    ).astype(np.complex64)
    snr = np.array([4.0 + 1.5j, 3.5 - 0.75j], dtype=np.complex64)
    indices = np.array([3, 11], dtype=np.int64)
    u_vals = np.array([-0.2, 0.35], dtype=np.float32)
    bins = np.array([1, 6, 13, 24], dtype=np.int64)

    monkeypatch.setattr(
        SingleDetSkyMaxPowerChisq,
        "calculate_chisq_bins",
        lambda self, template, psd: bins,
    )

    def evaluate(array_factory, trigger_factory=lambda values: values):
        hp = array_factory(template_plus.copy())
        hc = array_factory(template_cross.copy())
        cp = array_factory(corr_plus.copy())
        cc = array_factory(corr_cross.copy())
        hp.f_lower = hc.f_lower = 0.1
        hp.params = hc.params = object()
        veto = SingleDetSkyMaxPowerChisq(
            num_bins="3", snr_threshold=threshold
        )
        values, dof = veto.values(
            cp, cc, trigger_factory(snr), None, indices, hp, hc,
            trigger_factory(u_vals),
            hplus_cross_corr=0.15, hpnorm=1.1, hcnorm=0.9,
        )
        return values, dof, veto, hc, cc

    expected, expected_dof, _, _, _ = evaluate(
        lambda values: FrequencySeries(values, delta_f=0.1)
    )

    with scheme.TorchScheme("cpu"):
        with monkeypatch.context() as patch:
            patch.setattr(
                TorchArrayData,
                "numpy",
                lambda self: (_ for _ in ()).throw(
                    AssertionError("sky-max chisq copied dense Torch data")
                ),
            )
            actual, actual_dof, veto, hc, cc = evaluate(
                lambda values: FrequencySeries(values, delta_f=0.1),
                lambda values: Array(values),
            )

        assert isinstance(actual._data, TorchArrayData)
        assert isinstance(veto.template_mem._data, TorchArrayData)
        assert isinstance(veto.corr_mem._data, TorchArrayData)
        assert isinstance(hc._data, TorchArrayData)
        assert isinstance(cc._data, TorchArrayData)
        np.testing.assert_allclose(
            hc._data.tensor.cpu().numpy(), template_cross,
            rtol=0.0, atol=0.0,
        )
        np.testing.assert_allclose(
            cc._data.tensor.cpu().numpy(), corr_cross,
            rtol=0.0, atol=0.0,
        )
        actual_values = actual._data.tensor.cpu().numpy()

    np.testing.assert_allclose(
        actual_values, expected, rtol=2e-5, atol=2e-5
    )
    np.testing.assert_array_equal(actual_dof, expected_dof)
