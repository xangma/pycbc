"""Original division controls retain the selected device for host operands."""
import numpy as np
import pytest

jax = pytest.importorskip("jax")

from pycbc.scheme import JAXScheme
from pycbc.types.array_jax import _divide


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("kind", ["python", "numpy_scalar", "numpy_array"])
def test_original_host_division_follows_selected_device(device, kind):
    if device.startswith("cuda") and not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("CUDA device unavailable")
    right = {"python": 7, "numpy_scalar": np.int64(7),
             "numpy_array": np.asarray(7)}[kind]
    expected = np.asarray(4.0 / right)
    with JAXScheme(device, reference_operations=("divide",)) as context:
        result = _divide(4.0, right)
        assert result.device == context.jax_device
        actual = np.asarray(result)
        assert (actual.dtype, actual.shape, actual.tobytes()) == (
            expected.dtype, expected.shape, expected.tobytes())
