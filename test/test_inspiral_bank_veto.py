"""Bank veto accepts sparse host candidates under Torch processing schemes."""
import numpy as np
import pytest

from pycbc.scheme import TorchScheme
from pycbc.types import TimeSeries
from pycbc.vetoes.bank_chisq import bank_chisq_from_filters

torch = pytest.importorskip('torch')
DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('scalar_type', [complex, np.complex64])
def test_sparse_host_snr_bank_veto(device, scalar_type):
    values = np.array([1+2j, 3+1j, 2-1j, 4+3j], dtype=np.complex64)
    points = np.array([0, 3], dtype=np.int64)
    snr = np.array([2+1j, 1+3j], dtype=np.complex64)
    match = scalar_type(0.2+0.1j)
    norm = np.sqrt(1-abs(match)**2)
    expected = np.abs(values[points] * (0.5/norm)
                      - snr * (match.conjugate()*0.8/norm))**2
    with TorchScheme(device=device):
        bank_snr = TimeSeries(values, delta_t=0.1)
        result = bank_chisq_from_filters(
            snr, 0.8, [bank_snr], [0.5], [match], indices=points)
        assert result.data.device.type == device
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6)
