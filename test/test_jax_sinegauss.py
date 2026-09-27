import numpy as np
import jax
import pytest

from pycbc.scheme import JAXScheme
from pycbc.waveform.sinegauss import fd_sine_gaussian
from pycbc.waveform.sinegauss_jax import fd_sine_gaussian as fd_sine_gaussian_jax


_DEVICES = ["cpu"]
if any(d.platform in ("cuda", "gpu") for d in jax.devices()):
    _DEVICES.append("cuda")


@pytest.mark.parametrize("device", _DEVICES)
def test_jax_sinegaussian_matches_cpu_cutoffs_and_scaling(device):
    cases = [(1.0, 8.0, 100.0, 20.0, 500.0, .25),
             (2.3, 12.0, 300.0, 250.0, 500.0, .5),
             (.4, 5.0, 80.0, 0.0, 400.0, .125)]
    for args in cases:
        reference = fd_sine_gaussian(*args).numpy()
        with JAXScheme(device):
            result = fd_sine_gaussian_jax(*args).numpy()
        np.testing.assert_allclose(result, reference, rtol=2e-12, atol=2e-14)
