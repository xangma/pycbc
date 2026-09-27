"""JAX coincidence output contracts."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import pycbc
from pycbc import scheme
from pycbc.events.coinc import LiveCoincTimeslideBackgroundEstimator

if not getattr(pycbc, "HAVE_JAX", False):
    pytest.skip("PyCBC built without JAX support", allow_module_level=True)


def _trigger(time, snr):
    return {
        "snr": np.array([snr], dtype=np.float32),
        "chisq": np.array([2.0], dtype=np.float32),
        "chisq_dof": np.array([2.0], dtype=np.float32),
        "template_id": np.array([0], dtype=np.int32),
        "end_time": np.array([time], dtype=np.float64),
        "mass1": np.array([10.0], dtype=np.float64),
        "mass2": np.array([10.0], dtype=np.float64),
        "approximant": np.array(["TaylorF2"]),
    }


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_jax_single_ranking_only_foreground_stat_dtype_matches_cpu(device):
    """JAX keeps native float64 dtype for the returned coincident statistic."""
    jax.config.update("jax_enable_x64", True)
    if device == "cuda":
        try:
            jax.devices("gpu")
        except RuntimeError:
            pytest.skip("CUDA unavailable")
    kwargs = dict(ifar_limit=1, timeslide_interval=0.1,
                  return_background=True)
    block = {"H1": _trigger(100.0, 5.0),
             "L1": _trigger(100.001, 6.0)}

    cpu = LiveCoincTimeslideBackgroundEstimator(
        2, 10, "single_ranking_only", "snr", [], ["H1", "L1"], **kwargs)
    expected = cpu.add_singles(
        {ifo: dict(values) for ifo, values in block.items()})

    with scheme.JAXScheme(device):
        jax_estimator = LiveCoincTimeslideBackgroundEstimator(
            2, 10, "single_ranking_only", "snr", [], ["H1", "L1"], **kwargs)
        actual = jax_estimator.add_singles(
            {ifo: dict(values) for ifo, values in block.items()})

    assert expected["foreground/stat"].dtype == np.dtype("float64")
    assert np.asarray(actual["foreground/stat"]).dtype == np.dtype("float64")
    np.testing.assert_allclose(
        np.asarray(actual["foreground/stat"]), expected["foreground/stat"])
