"""Preserve the original CPU sky-max edge and exception behavior."""
import numpy as np
import pytest

from pycbc import scheme
from pycbc.types import FrequencySeries
from pycbc.vetoes.chisq import SingleDetSkyMaxPowerChisq


def inputs(indices=(1,)):
    arrays = [FrequencySeries(np.ones(8, dtype=np.complex64), delta_f=.1)
              for _ in range(4)]
    corr_plus, corr_cross, template_plus, template_cross = arrays
    template_plus.f_lower = .1
    template_plus.params = object()
    return dict(corr_plus=corr_plus, corr_cross=corr_cross,
                template_plus=template_plus, template_cross=template_cross,
                snrv=np.ones(len(indices), dtype=np.complex64), psd=None,
                indices=np.array(indices, dtype=np.int64),
                u_vals=np.zeros(len(indices), dtype=np.float32),
                hplus_cross_corr=0., hpnorm=1., hcnorm=1.)


def test_cpu_empty_indices_without_threshold_preserves_error():
    veto = SingleDetSkyMaxPowerChisq(num_bins="3")
    with pytest.raises(UnboundLocalError):
        veto.values(**inputs(()))


def test_cpu_all_below_threshold_preserves_values_and_dof():
    veto = SingleDetSkyMaxPowerChisq(num_bins="3", snr_threshold=2.)
    values, dof = veto.values(**inputs((1, 2)))
    np.testing.assert_array_equal(values, np.zeros(2, dtype=np.float32))
    np.testing.assert_array_equal(dof, np.repeat(-100, 2))
    assert values.dtype == np.float32


@pytest.mark.parametrize("torch_backed", (False, True))
def test_skymax_exception_preserves_backend_buffer_identity(monkeypatch,
                                                          torch_backed):
    def fail(*_args):
        raise RuntimeError("injected bin failure")

    def check():
        veto = SingleDetSkyMaxPowerChisq(num_bins="3")
        args = inputs()
        template_data = args["template_cross"]._data
        corr_data = args["corr_cross"]._data
        monkeypatch.setattr(veto, "calculate_chisq_bins", fail)
        with pytest.raises(RuntimeError, match="injected bin failure"):
            veto.values(**args)
        if torch_backed:
            assert args["template_cross"]._data is template_data
            assert args["corr_cross"]._data is corr_data
        else:
            assert args["template_cross"]._data is veto.template_mem.data
            assert args["corr_cross"]._data is veto.corr_mem.data
            assert args["template_cross"]._data is not template_data
            assert args["corr_cross"]._data is not corr_data

    if torch_backed:
        pytest.importorskip("torch")
        with scheme.TorchScheme("cpu"):
            check()
    else:
        check()
