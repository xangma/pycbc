"""Torch parity tests for LALSimulation's P1200087 PSD tables."""

import lal
import numpy as np
import pytest
import torch

from pycbc import scheme
from pycbc.psd import analytical
from pycbc.psd import analytical_torch
from pycbc.types.array_torch import TorchArrayData


P1200087_MODELS = tuple(sorted(
    analytical_torch.P1200087_TORCH_ANALYTICAL_MODELS
))


@pytest.fixture(params=("cpu", "cuda"))
def torch_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    return request.param


def _lal_psd(model_name, **parameters):
    """Generate the public default-scheme LAL oracle."""
    return analytical.from_string(model_name, **parameters).numpy()


def _lal_file_psd(path, *, length=18, delta_f=1.0, cutoff=1.0):
    """Generate an oracle through XLALSimNoisePSDFromFile directly."""
    series = lal.CreateREAL8FrequencySeries(
        "",
        lal.LIGOTimeGPS(0),
        0.0,
        delta_f,
        lal.DimensionlessUnit,
        length,
    )
    analytical.lalsimulation.SimNoisePSDFromFile(
        series,
        cutoff,
        str(path),
    )
    values = series.data.data.copy()
    values[:int(cutoff / delta_f)] = 0.0
    return values


def _install_p1200087_override(monkeypatch, path):
    """Resolve ``path`` natively and make the LAL fallback read it too."""
    model_name = "aLIGODesignSensitivityP1200087"
    function_name = f"SimNoisePSD{model_name}"
    file_reader = analytical.lalsimulation.SimNoisePSDFromFile
    calls = []

    monkeypatch.setattr(
        analytical_torch.lal,
        "FileResolvePath",
        lambda _filename: str(path),
    )

    def read_override(series, cutoff):
        calls.append((series, cutoff))
        return file_reader(series, cutoff, str(path))

    monkeypatch.setattr(
        analytical.lalsimulation,
        function_name,
        read_override,
    )
    return calls


def test_p1200087_models_follow_installed_lalsimulation_discovery():
    assert set(P1200087_MODELS) <= set(analytical.get_lalsim_psd_list())


def test_p1200087_torch_family_matches_lalsimulation(
    torch_device,
    monkeypatch,
):
    parameters = dict(
        length=4101,
        delta_f=2.0,
        low_freq_cutoff=11.7,
    )
    expected = {
        model_name: _lal_psd(model_name, **parameters)
        for model_name in P1200087_MODELS
    }

    def reject_lal_output(*_args, **_kwargs):
        raise AssertionError("Torch P1200087 PSD generated a host LAL series")

    def reject_torch_numpy(*_args, **_kwargs):
        raise AssertionError("Torch P1200087 PSD copied its output to host")

    with scheme.TorchScheme(torch_device):
        with monkeypatch.context() as patch:
            patch.setattr(
                analytical.lal,
                "CreateREAL8FrequencySeries",
                reject_lal_output,
            )
            patch.setattr(TorchArrayData, "numpy", reject_torch_numpy)
            actual = {
                model_name: analytical.from_string(
                    model_name,
                    ignored_lalsimulation_keyword=True,
                    **parameters,
                )
                for model_name in P1200087_MODELS
            }

        for model_name, result in actual.items():
            tensor = result._data.tensor
            assert tensor.device.type == torch_device
            assert tensor.dtype == torch.float64
            assert result.epoch == lal.LIGOTimeGPS(0)
            assert result.delta_f == parameters["delta_f"]
            assert tensor[0] == 0.0
            assert tensor[-1] == 0.0
            torch.testing.assert_close(
                tensor,
                torch.as_tensor(
                    expected[model_name],
                    dtype=torch.float64,
                    device=torch_device,
                ),
                rtol=2e-12,
                atol=0.0,
            )


def test_p1200087_layout_cutoff_sentinel_and_extrapolation():
    parameters = dict(
        length=8202,
        delta_f=1.0,
        low_freq_cutoff=0.0,
    )
    expected = {
        model_name: _lal_psd(model_name, **parameters)
        for model_name in (
            "aLIGODesignSensitivityP1200087",
            "AdVBNSOptimizedSensitivityP1200087",
        )
    }

    with scheme.TorchScheme("cpu"):
        actual = {
            model_name: analytical.from_string(model_name, **parameters)
            for model_name in expected
        }
        cutoff = analytical.aLIGODesignSensitivityP1200087(
            64,
            1.0,
            11.7,
        )
        one_bin = analytical.aLIGODesignSensitivityP1200087(
            1,
            10.0,
            0.0,
        )
        two_bins = analytical.aLIGODesignSensitivityP1200087(
            2,
            10.0,
            0.0,
        )
        negative_cutoff = analytical.aLIGODesignSensitivityP1200087(
            16,
            1.0,
            -1.0,
        )

        for model_name, result in actual.items():
            torch.testing.assert_close(
                result._data.tensor,
                torch.as_tensor(expected[model_name]),
                rtol=5e-15,
                atol=0.0,
            )

        aligo = actual["aLIGODesignSensitivityP1200087"]._data.tensor
        adv = actual["AdVBNSOptimizedSensitivityP1200087"]._data.tensor
        torch.testing.assert_close(
            aligo[[9, 100, 8000, 8200]],
            torch.tensor(
                [
                    2.617705357617846e-42,
                    1.590920124364387e-47,
                    1.1910162992988085e-45,
                    1.2516115123584623e-45,
                ],
                dtype=torch.float64,
            ),
            rtol=5e-15,
            atol=0.0,
        )
        torch.testing.assert_close(
            adv[[9, 10, 100]],
            torch.tensor(
                [0.0, 1.2267912339843334e-42, 2.2831788334654577e-47],
                dtype=torch.float64,
            ),
            rtol=5e-15,
            atol=0.0,
        )
        assert torch.count_nonzero(cutoff._data.tensor[:11]) == 0
        assert cutoff._data.tensor[11] != 0.0
        assert cutoff._data.tensor[-1] == 0.0
        assert torch.count_nonzero(one_bin._data.tensor) == 0
        assert torch.count_nonzero(two_bins._data.tensor) == 0
        assert torch.count_nonzero(negative_cutoff._data.tensor) == 0


def test_p1200087_resolver_is_rechecked_and_failures_propagate(
    tmp_path,
    monkeypatch,
):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    frequencies = np.array([10.0, 20.0, 40.0])
    np.savetxt(first, np.column_stack((frequencies, [1.0, 2.0, 4.0])))
    np.savetxt(second, np.column_stack((frequencies, [2.0, 4.0, 8.0])))
    paths = iter((str(first), str(second), str(second)))
    calls = []

    def changing_resolver(filename):
        calls.append(filename)
        return next(paths)

    with scheme.TorchScheme("cpu"):
        monkeypatch.setattr(
            analytical_torch.lal,
            "FileResolvePath",
            changing_resolver,
        )
        first_result = analytical.from_string(
            "aLIGODesignSensitivityP1200087",
            41,
            1.0,
            10.0,
        )._data.tensor.clone()
        second_result = analytical.from_string(
            "aLIGODesignSensitivityP1200087",
            41,
            1.0,
            10.0,
        )._data.tensor.clone()
        np.savetxt(
            second,
            np.column_stack((frequencies, [3.0, 6.0, 12.0])),
        )
        third_result = analytical.from_string(
            "aLIGODesignSensitivityP1200087",
            41,
            1.0,
            10.0,
        )._data.tensor.clone()

    assert len(calls) == 3
    torch.testing.assert_close(
        second_result[10:-1],
        4.0 * first_result[10:-1],
        rtol=2e-15,
        atol=0.0,
    )
    torch.testing.assert_close(
        third_result[10:-1],
        9.0 * first_result[10:-1],
        rtol=2e-15,
        atol=0.0,
    )

    monkeypatch.setattr(
        analytical_torch.lal,
        "FileResolvePath",
        lambda _filename: None,
    )
    with scheme.TorchScheme("cpu"):
        with pytest.raises(RuntimeError, match="Unable to resolve"):
            analytical.from_string(
                "aLIGODesignSensitivityP1200087",
                41,
                1.0,
                10.0,
            )


@pytest.mark.parametrize(
    "table",
    (
        "1 1 99\n2 2\n4 4\n8 8\n",
        "1 1\n2 2\n4 4\n16 8\n8 16\n",
        "-4 1\n-2 2\n1 4\n4 8\n",
        "1 1\n2 0\n4 4\n8 8\n",
        "1 1\n2 2\nnan 4\n8 8\n",
        "1 1\n2 2\n4 nan\n8 8\n",
    ),
    ids=(
        "extra-column",
        "nonmonotonic-frequency",
        "nonpositive-frequency",
        "nonpositive-asd",
        "nonfinite-frequency",
        "nonfinite-asd",
    ),
)
def test_p1200087_unsafe_override_tables_use_lalsimulation(
    tmp_path,
    monkeypatch,
    table,
):
    path = tmp_path / "override.txt"
    path.write_text(table, encoding="ascii")
    expected = _lal_file_psd(path)
    calls = _install_p1200087_override(monkeypatch, path)

    with scheme.TorchScheme("cpu"):
        actual = analytical.from_string(
            "aLIGODesignSensitivityP1200087",
            18,
            1.0,
            1.0,
        ).numpy()

    assert len(calls) == 1
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "table",
    (
        "1 1\n\n2 2\n4 4\n",
        "1 1\n  # comment\n2 2\n4 4\n",
        "1 1\n2\n4 4\n",
    ),
    ids=("blank-line", "leading-space-comment", "missing-column"),
)
def test_p1200087_malformed_override_errors_come_from_lalsimulation(
    tmp_path,
    monkeypatch,
    table,
):
    path = tmp_path / "override.txt"
    path.write_text(table, encoding="ascii")
    with pytest.raises(RuntimeError):
        _lal_file_psd(path)
    calls = _install_p1200087_override(monkeypatch, path)

    with scheme.TorchScheme("cpu"):
        with pytest.raises(RuntimeError):
            analytical.from_string(
                "aLIGODesignSensitivityP1200087",
                18,
                1.0,
                1.0,
            )

    assert len(calls) == 1


def test_p1200087_default_and_mps_keep_lalsimulation_fallback(monkeypatch):
    model_name = "aLIGODesignSensitivityP1200087"
    function_name = f"SimNoisePSD{model_name}"
    original = getattr(analytical.lalsimulation, function_name)
    calls = []

    def record_lal_call(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(
        analytical.lalsimulation,
        function_name,
        record_lal_call,
    )
    result = analytical.from_string(model_name, 33, 1.0, 10.0)
    assert len(calls) == 1
    assert isinstance(result._data, np.ndarray)

    if not torch.backends.mps.is_available():
        return
    with scheme.TorchScheme("mps"):
        with pytest.raises(TypeError, match="MPS backend only supports"):
            analytical.from_string(model_name, 33, 1.0, 10.0)
    assert len(calls) == 2


def test_p1200087_preserves_real8_and_uses_static_torch_discovery(
        monkeypatch):
    model_name = "aLIGODesignSensitivityP1200087"
    cutoff = np.array(10.0)

    with pytest.raises(TypeError, match="argument 2 of type '(double|REAL8)'"):
        analytical.from_string(model_name, 33, 1.0, cutoff)
    with scheme.TorchScheme("cpu"):
        with pytest.raises(
            TypeError,
            match="argument 2 of type '(double|REAL8)'",
        ):
            analytical.from_string(model_name, 33, 1.0, cutoff)

    monkeypatch.setattr(analytical, "get_lalsim_psd_list", lambda: [])
    with scheme.TorchScheme("cpu"):
        result = analytical.from_string(model_name, 33, 1.0, 10.0)
    assert result._data.tensor.device.type == "cpu"
