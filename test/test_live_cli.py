"""Native Live defaults, followup options and complete output boundaries."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip('mpi4py.MPI')

from pycbc import scheme  # noqa: E402
from pycbc.live.cli import LiveEventManager, create_parser  # noqa: E402


def test_native_parser_preserves_live_defaults():
    parser = create_parser('Live')
    args = parser.parse_args([
        '--bank-file', 'bank.hdf', '--analysis-chunk', '4',
        '--channel-name', 'H1:STRAIN', '--output-path', 'results',
        '--psd-samples', '3', '--psd-segment-length', '4',
        '--ifar-double-followup-threshold', '1',
        '--pvalue-combination-livetime', '1',
        '--ifar-upload-threshold', '1',
        '--ranking-statistic', 'quadsum', '--sngl-ranking', 'newsnr',
        '--src-class-mchirp-to-delta', '0.01',
        '--src-class-eff-to-lum-distance', '1',
        '--src-class-lum-distance-to-delta', '0.1', '0.1',
    ])
    assert args.processing_scheme == 'cpu'
    assert not hasattr(args, 'replay_clock')
    assert not args.enable_gracedb_upload


def test_native_followup_uses_one_worker():
    args = SimpleNamespace(processing_scheme='cpu:8')
    assert LiveEventManager.followup_processing_options(args) == (
        '--processing-scheme cpu:1 ')


def test_native_hdf_output_retains_columns_and_metadata(tmp_path):
    values = np.array([8., 9., 7.], dtype=np.float32)
    chisq = np.array([1., 4., 1.], dtype=np.float32)
    with scheme.CPUScheme(1):
        manager = object.__new__(LiveEventManager)
        manager.live_detectors = {'H1'}
        manager.get_out_dir_path = lambda _: str(tmp_path)
        manager.dump(
            {'H1': {'snr': values, 'chisq': chisq}}, 'triggers',
            store_loudest_index=1,
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


@pytest.mark.parametrize('value,expected', [
    (np.str_('abc'), [b'a', b'b', b'c']),
    (np.array(['a', 'b']), [b'a', b'b']),
    (np.int64(4), np.int64(4)),
])
def test_native_serialization_retains_original_metadata(value, expected):
    np.testing.assert_array_equal(LiveEventManager.serialize_value(value),
                                  expected)
