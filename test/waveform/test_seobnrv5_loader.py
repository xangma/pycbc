import h5py
import numpy as np
import pytest
import torch

from pycbc.waveform import seobnrv5_torch as rom


@pytest.fixture(autouse=True)
def clear_rom_caches():
    rom._clear_rom_cache()
    yield
    rom._clear_rom_cache()


def _create_empty_dataset(group, name, shape):
    chunks = tuple(min(size, 64) for size in shape)
    group.create_dataset(name, shape=shape, dtype=np.float64, chunks=chunks)


def _write_valid_sparse_rom(path):
    with h5py.File(path, "w") as data:
        data.attrs["CANONICAL_FILE_BASENAME"] = rom._ROM_FILENAME
        data.attrs["version_major"] = 1
        data.attrs["version_minor"] = 0
        data.attrs["version_micro"] = 0
        for patch, (q_size, chi1_size, chi2_size) in rom._PATCH_VECTOR_SIZES.items():
            group = data.create_group(patch)
            qvec = np.linspace(1.0, 100.0, q_size)
            group["qvec"] = qvec
            group["etavec"] = qvec / np.square(1.0 + qvec)
            group["chi1vec"] = np.linspace(-1.0, 1.0, chi1_size)
            group["chi2vec"] = np.linspace(-1.0, 1.0, chi2_size)

            cmode = group.create_group("CF_modes").create_group("22")
            phase = group.create_group("phase_carrier")
            cmode["MF_grid"] = np.linspace(0.001, 1.7, rom._SPARSE_GRID_SIZE)
            phase["MF_grid"] = np.linspace(0.0005, 1.25, rom._SPARSE_GRID_SIZE)
            matrix_shape = (rom._SPARSE_GRID_SIZE, rom._SPARSE_GRID_SIZE)
            _create_empty_dataset(cmode, "basis_re", matrix_shape)
            _create_empty_dataset(cmode, "basis_im", matrix_shape)
            _create_empty_dataset(phase, "basis", matrix_shape)

            parameter_size = (q_size + 2) * (chi1_size + 2) * (chi2_size + 2)
            coefficient_shape = (rom._SPARSE_GRID_SIZE * parameter_size,)
            _create_empty_dataset(cmode, "coeff_re_flattened", coefficient_shape)
            _create_empty_dataset(cmode, "coeff_im_flattened", coefficient_shape)
            _create_empty_dataset(phase, "coeff_flattened", coefficient_shape)


def test_find_rom_file_uses_lal_data_path(tmp_path, monkeypatch):
    expected = tmp_path / rom._ROM_FILENAME
    expected.touch()
    monkeypatch.setenv("LAL_DATA_PATH", str(tmp_path))
    assert rom._find_rom_file() == expected


def test_validate_rom_layout_without_loading_coefficients(tmp_path):
    path = tmp_path / rom._ROM_FILENAME
    _write_valid_sparse_rom(path)

    metadata = rom._validate_rom_file(path)

    assert metadata.path == path
    assert tuple(metadata.patches) == rom._PATCH_NAMES
    assert len(metadata.patches["lowf"].chi1_breaks) == 12
    assert len(metadata.patches["highf"].chi2_breaks) == 21


def test_validate_rom_rejects_wrong_version(tmp_path):
    path = tmp_path / rom._ROM_FILENAME
    with h5py.File(path, "w") as data:
        data.attrs["CANONICAL_FILE_BASENAME"] = rom._ROM_FILENAME
        data.attrs["version_major"] = 2
        data.attrs["version_minor"] = 0
        data.attrs["version_micro"] = 0

    with pytest.raises(ValueError, match="version"):
        rom._validate_rom_file(path)


def test_validate_rom_rejects_truncated_coefficients(tmp_path):
    path = tmp_path / rom._ROM_FILENAME
    _write_valid_sparse_rom(path)
    with h5py.File(path, "r+") as data:
        dataset = data["lowf/CF_modes/22/coeff_re_flattened"]
        wrong_shape = (dataset.shape[0] - 1,)
        del data["lowf/CF_modes/22/coeff_re_flattened"]
        _create_empty_dataset(
            data["lowf/CF_modes/22"], "coeff_re_flattened", wrong_shape
        )

    with pytest.raises(ValueError, match="has shape"):
        rom._validate_rom_file(path)


def test_real_low_frequency_patch_is_cached(monkeypatch):
    try:
        path = rom._find_rom_file()
    except FileNotFoundError:
        pytest.skip("SEOBNRv5 ROM data is not available on LAL_DATA_PATH")
    monkeypatch.setattr(rom, "_find_rom_file", lambda: path)

    device = torch.device("cpu")
    first = rom._load_submodel("lowf", torch.float32, device)
    second = rom._load_submodel("lowf", torch.float32, device)

    assert first is second
    assert first.coeff_real.shape == (300, 59, 14, 14)
    assert first.coeff_real.dtype == torch.float32
    assert first.coeff_real.device == device
    assert first.basis_phase.shape == (300, 300)


def test_loader_rejects_unknown_patch_and_dtype():
    with pytest.raises(ValueError, match="unknown SEOBNRv5 ROM patch"):
        rom._load_host_submodel("middle", torch.float32)
    with pytest.raises(TypeError, match="torch.float32 or torch.float64"):
        rom._numpy_dtype(torch.float16)
