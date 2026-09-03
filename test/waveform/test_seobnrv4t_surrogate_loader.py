import math
import os

import h5py
import numpy as np
import pytest
import torch

from pycbc.waveform import seobnrv4t_surrogate_torch as surrogate


@pytest.fixture(autouse=True)
def clear_surrogate_caches():
    surrogate._clear_surrogate_cache()
    yield
    surrogate._clear_surrogate_cache()


def _write_valid_surrogate(path):
    rng = np.random.default_rng(81208643)
    amplitude_nodes = np.geomspace(2.1e-4, 7.0e-2, 40)
    tf2_nodes = np.geomspace(1.0e-4, 8.0e-2, 1000)
    training_points = np.column_stack(
        (
            rng.uniform(1.0 / 3.0, 1.0, 960),
            rng.uniform(-0.5, 0.5, 960),
            rng.uniform(-0.5, 0.5, 960),
            rng.uniform(0.0, math.log10(51.0), 960),
            rng.uniform(0.0, math.log10(51.0), 960),
        )
    )
    training_points[17] = (0.5, 0.125, -0.25, 0.0, 1.0)

    with h5py.File(path, "w") as data:
        version = surrogate._ROM_VERSIONS[path.name]
        data.attrs["version_major"] = version[0]
        data.attrs["version_minor"] = version[1]
        data.attrs["version_micro"] = version[2]
        if version >= (2, 0, 0):
            data.attrs["CANONICAL_FILE_BASENAME"] = path.name
        data["hyp_amp"] = rng.uniform(0.2, 2.0, (40, 7))
        data["hyp_phi"] = rng.uniform(0.2, 2.0, (39, 7))
        data["kinv_dot_y_amp"] = rng.normal(0.0, 0.01, (40, 960))
        data["kinv_dot_y_phi"] = rng.normal(0.0, 0.01, (39, 960))
        data["x_train"] = training_points
        data["spline_nodes_amp"] = amplitude_nodes
        data["spline_nodes_phase"] = amplitude_nodes[1:]
        data["TF2_Mf_amp_cubic"] = tf2_nodes
        data["TF2_Mf_phi_cubic"] = tf2_nodes
        data["TF2_Mf_amp_linear"] = tf2_nodes
        data["TF2_Mf_phi_linear"] = tf2_nodes
        data["q_bounds"] = (1.0 / 3.0, 1.0)
        data["chi1_bounds"] = (-0.5, 0.5)
        data["chi2_bounds"] = (-0.5, 0.5)
        data["lambda1_bounds"] = (0.0, 5000.0)
        data["lambda2_bounds"] = (0.0, 5000.0)


@pytest.fixture
def surrogate_file(tmp_path):
    path = tmp_path / surrogate._ROM_FILENAME
    _write_valid_surrogate(path)
    return path


def _numpy_gpr(coordinate, hyperparameters, training_points, weights):
    predictions = []
    exact_training_point = np.all(coordinate == training_points, axis=1)
    for hyperparameters_i, weights_i in zip(hyperparameters, weights):
        signal_scale = hyperparameters_i[0]
        length_scales = hyperparameters_i[1:-1]
        noise_scale = hyperparameters_i[-1]
        radius = np.linalg.norm(
            (coordinate - training_points) / length_scales,
            axis=1,
        )
        covariance = (
            signal_scale**2
            * (1.0 + np.sqrt(5.0) * radius + 5.0 * radius**2 / 3.0)
            * np.exp(-np.sqrt(5.0) * radius)
        )
        covariance += noise_scale**2 * exact_training_point
        predictions.append(np.dot(covariance, weights_i))
    return np.asarray(predictions)


def test_find_rom_file_uses_lal_data_path(tmp_path, monkeypatch):
    expected = tmp_path / surrogate._ROM_FILENAME
    expected.touch()
    monkeypatch.setenv("LAL_DATA_PATH", str(tmp_path))
    assert surrogate._find_rom_file() == expected


def test_find_rom_file_accepts_legacy_data(tmp_path, monkeypatch):
    expected = tmp_path / surrogate._ROM_FILENAMES[1]
    expected.touch()
    monkeypatch.setenv("LAL_DATA_PATH", str(tmp_path))
    assert surrogate._find_rom_file() == expected


def test_find_rom_file_prefers_current_data(tmp_path, monkeypatch):
    legacy_dir = tmp_path / "legacy"
    current_dir = tmp_path / "current"
    legacy_dir.mkdir()
    current_dir.mkdir()
    (legacy_dir / surrogate._ROM_FILENAMES[1]).touch()
    expected = current_dir / surrogate._ROM_FILENAMES[0]
    expected.touch()
    monkeypatch.setenv("LAL_DATA_PATH", f"{legacy_dir}{os.pathsep}{current_dir}")
    assert surrogate._find_rom_file() == expected


def test_validate_surrogate_layout(surrogate_file):
    metadata = surrogate._validate_rom_file(surrogate_file)

    assert metadata.path == surrogate_file
    assert metadata.mass_ratio_bounds == pytest.approx((1.0, 3.0))
    assert metadata.chi1_bounds == (-0.5, 0.5)
    assert metadata.lambda2_bounds == (0.0, 5000.0)
    assert metadata.amplitude_frequency_bounds == pytest.approx((2.1e-4, 7.0e-2))


def test_validate_legacy_surrogate_layout(tmp_path):
    path = tmp_path / surrogate._ROM_FILENAMES[1]
    _write_valid_surrogate(path)

    assert surrogate._validate_rom_file(path).path == path


def test_validate_surrogate_rejects_wrong_version(tmp_path):
    path = tmp_path / surrogate._ROM_FILENAME
    with h5py.File(path, "w") as data:
        data.attrs["version_major"] = 3
        data.attrs["version_minor"] = 0
        data.attrs["version_micro"] = 0

    with pytest.raises(ValueError, match="version"):
        surrogate._validate_rom_file(path)


def test_validate_surrogate_rejects_wrong_canonical_basename(surrogate_file):
    with h5py.File(surrogate_file, "r+") as data:
        data.attrs["CANONICAL_FILE_BASENAME"] = "renamed.hdf5"

    with pytest.raises(ValueError, match="canonical basename"):
        surrogate._validate_rom_file(surrogate_file)


def test_validate_surrogate_rejects_bad_shape(surrogate_file):
    with h5py.File(surrogate_file, "r+") as data:
        del data["hyp_phi"]
        data["hyp_phi"] = np.ones((38, 7))

    with pytest.raises(ValueError, match="has shape"):
        surrogate._validate_rom_file(surrogate_file)


def test_validate_surrogate_rejects_inconsistent_phase_nodes(surrogate_file):
    with h5py.File(surrogate_file, "r+") as data:
        data["spline_nodes_phase"][5] *= 1.001

    with pytest.raises(ValueError, match="phase nodes"):
        surrogate._validate_rom_file(surrogate_file)


def test_model_is_cached_in_requested_precision_and_device(surrogate_file, monkeypatch):
    monkeypatch.setattr(surrogate, "_find_rom_file", lambda: surrogate_file)
    device = torch.device("cpu")

    first = surrogate._load_surrogate_data(torch.float32, device)
    second = surrogate._load_surrogate_data(torch.float32, device)

    assert first is second
    assert first.x_train.shape == (960, 5)
    assert first.x_train.dtype == torch.float32
    assert first.x_train.device == device
    assert first.amplitude_nodes.shape == (40,)
    assert first.phase_nodes.shape == (40,)
    assert first.phase_nodes[0] == first.amplitude_nodes[0]


def test_canonical_parameters_move_heavier_body_first():
    parameters = surrogate._canonical_intrinsic_parameters(
        1.2,
        1.8,
        0.1,
        -0.2,
        300.0,
        900.0,
    )

    assert parameters.mass1 == 1.8
    assert parameters.mass2 == 1.2
    assert parameters.chi1 == -0.2
    assert parameters.chi2 == 0.1
    assert parameters.lambda1 == 900.0
    assert parameters.lambda2 == 300.0
    assert parameters.mass_ratio == pytest.approx(1.5)


def test_correction_gpr_matches_independent_numpy_reference(
    surrogate_file, monkeypatch
):
    monkeypatch.setattr(surrogate, "_find_rom_file", lambda: surrogate_file)
    data = surrogate._load_surrogate_data(torch.float64, torch.device("cpu"))
    parameters = surrogate._canonical_intrinsic_parameters(
        2.0,
        1.0,
        0.125,
        -0.25,
        0.0,
        900.0,
    )

    amplitude, phase = surrogate._evaluate_corrections(data, parameters)
    coordinate = surrogate._surrogate_coordinate(
        parameters,
        dtype=torch.float64,
        device=torch.device("cpu"),
    ).numpy()
    with h5py.File(surrogate_file, "r") as reference_data:
        expected_amplitude = _numpy_gpr(
            coordinate,
            reference_data["hyp_amp"][:],
            reference_data["x_train"][:],
            reference_data["kinv_dot_y_amp"][:],
        )
        expected_phase = _numpy_gpr(
            coordinate,
            reference_data["hyp_phi"][:],
            reference_data["x_train"][:],
            reference_data["kinv_dot_y_phi"][:],
        )

    torch.testing.assert_close(
        amplitude,
        torch.from_numpy(expected_amplitude),
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    assert phase[0] == 0.0
    torch.testing.assert_close(
        phase[1:],
        torch.from_numpy(expected_phase),
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_correction_evaluation_checks_model_domain(surrogate_file, monkeypatch):
    monkeypatch.setattr(surrogate, "_find_rom_file", lambda: surrogate_file)
    data = surrogate._load_surrogate_data(torch.float64, torch.device("cpu"))
    parameters = surrogate._canonical_intrinsic_parameters(
        4.0,
        1.0,
        0.0,
        0.0,
        100.0,
        100.0,
    )

    with pytest.raises(ValueError, match="mass ratio"):
        surrogate._evaluate_corrections(data, parameters)


def test_loader_rejects_unsupported_precision():
    with pytest.raises(TypeError, match="torch.float32 or torch.float64"):
        surrogate._numpy_dtype(torch.float16)
