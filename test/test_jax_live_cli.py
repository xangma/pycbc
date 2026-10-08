"""Live argument, transport and HDF boundaries shared by the JAX search."""

from types import SimpleNamespace

import h5py
import jax
import numpy as np
import pytest

pytest.importorskip('mpi4py.MPI')

from pycbc import scheme  # noqa: E402
from pycbc.live.cli import LiveEventManager  # noqa: E402
from pycbc.live.cli_jax import JAXLiveEventManager  # noqa: E402


def test_followup_options_preserve_scheme_and_validation_controls():
    args = SimpleNamespace(
        processing_scheme='jax:cuda:1', jax_chisq_mode='cpu-compatible',
        jax_highpass_mode='closed-form',
        jax_reference_operations='ifft,newsnr',
    )
    assert JAXLiveEventManager.followup_processing_options(args) == (
        '--processing-scheme jax:cuda:1 --jax-chisq-mode cpu-compatible '
        '--jax-highpass-mode closed-form '
        '--jax-reference-operations ifft,newsnr ')


def test_gather_retains_numerical_device_columns_and_host_labels():
    columns = [
        ({'H1': {'snr': np.array([7., 9.], dtype=np.float32),
                 'label': np.array(['a', 'b'])}, 'L1': False}, 123),
        ({'H1': {'snr': np.array([8.], dtype=np.float32),
                 'label': np.array(['c'])}, 'L1': False}, 123),
    ]
    with scheme.JAXScheme('cpu'):
        manager = object.__new__(JAXLiveEventManager)
        manager.rank = 0
        manager._jax_result_transport = SimpleNamespace(gather=lambda: columns)
        result, end = manager.gather_results()
        assert end == 123
        assert set(result) == {'H1'}
        assert isinstance(result['H1']['snr'], jax.Array)
        np.testing.assert_array_equal(result['H1']['snr'], [7, 9, 8])
        np.testing.assert_array_equal(result['H1']['label'], ['a', 'b', 'c'])
        assert isinstance(result['H1']['label'], np.ndarray)


def test_hdf_output_retains_columns_and_metadata(tmp_path):
    values = np.array([8., 9., 7.], dtype=np.float32)
    chisq = np.array([1., 4., 1.], dtype=np.float32)
    with scheme.JAXScheme('cpu'):
        manager = object.__new__(JAXLiveEventManager)
        manager.live_detectors = {'H1'}
        manager.get_out_dir_path = lambda _: str(tmp_path)
        results = {'H1': {'snr': jax.numpy.asarray(values),
                          'chisq': jax.numpy.asarray(chisq)}}
        manager.dump(results, 'triggers', store_loudest_index=1,
                     raw_results={'label': np.array(['signal', 'noise']),
                                  'index': np.array([2], dtype=np.int32)},
                     gates={'H1': [(123., .1, .2)]})
    with h5py.File(tmp_path / 'triggers.hdf') as output:
        np.testing.assert_array_equal(output['H1/snr'], values)
        np.testing.assert_array_equal(output['H1/chisq'], chisq)
        np.testing.assert_array_equal(output['H1/loudest'], [0, 1])
        np.testing.assert_array_equal(output['label'], [b'signal', b'noise'])
        assert output['index'].dtype == np.dtype('int32')
        assert output.attrs['num_live_detectors'] == 1
        assert output['H1/gates'].dtype.names == (
            'center_time', 'zero_half_width', 'taper_width')


def test_serialization_retains_native_metadata_conversion():
    for value in (np.str_('abc'), np.array(['a', 'b']), np.int64(4)):
        np.testing.assert_array_equal(
            JAXLiveEventManager.serialize_value(value),
            LiveEventManager.serialize_value(value))
