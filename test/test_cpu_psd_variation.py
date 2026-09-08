"""CPU PSD variation must respect the caller's NumPy error policy."""
import numpy as np
import pytest

from pycbc.psd.variation import create_full_filt


def test_cpu_psd_variation_preserves_divide_error():
    with np.errstate(divide="raise"):
        with pytest.raises(FloatingPointError, match="divide by zero"):
            create_full_filt(np.arange(5.), np.ones(5), np.ones(5), 4, 2)


def test_cpu_psd_variation_preserves_divide_warning():
    with np.errstate(divide="warn"):
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            result = create_full_filt(
                np.arange(5.), np.ones(5), np.ones(5), 4, 2
            )
    assert result.shape == (8,)
    assert np.isfinite(result).all()
