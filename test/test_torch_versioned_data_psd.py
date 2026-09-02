"""Torch parity tests for LALSimulation's versioned ASD data files."""

import lal
import numpy as np
import pytest
import torch

from pycbc import scheme
from pycbc.psd import analytical
from pycbc.psd import analytical_torch
from pycbc.types.array_torch import TorchArrayData


VERSIONED_DATA_FILES = {
    "aLIGONoSRMLowPowerGWINC": "LIGO-T0900288-v3-NO_SRM.txt",
    "aLIGOZeroDetLowPowerGWINC": "LIGO-T0900288-v3-ZERO_DET_low_P.txt",
    "aLIGOZeroDetHighPowerGWINC": "LIGO-T0900288-v3-ZERO_DET_high_P.txt",
    "aLIGONSNSOptGWINC": "LIGO-T0900288-v3-NSNS_Opt.txt",
    "aLIGOBHBH20DegGWINC": "LIGO-T0900288-v3-BHBH_20deg.txt",
    "aLIGOHighFrequencyGWINC": "LIGO-T0900288-v3-High_Freq.txt",
    "CosmicExplorerP1600143": "LIGO-P1600143-v18-CE.txt",
    "CosmicExplorerPessimisticP1600143": (
        "LIGO-P1600143-v18-CE_Pessimistic.txt"
    ),
    "CosmicExplorerWidebandP1600143": (
        "LIGO-P1600143-v18-CE_Wideband.txt"
    ),
    "EinsteinTelescopeP1600143": "LIGO-P1600143-v18-ET_D.txt",
    "KAGRAOpeningSensitivityT1600593": (
        "LIGO-T1600593-v1-KAGRA_Opening.txt"
    ),
    "KAGRAEarlySensitivityT1600593": (
        "LIGO-T1600593-v1-KAGRA_Early.txt"
    ),
    "KAGRAMidSensitivityT1600593": "LIGO-T1600593-v1-KAGRA_Mid.txt",
    "KAGRALateSensitivityT1600593": "LIGO-T1600593-v1-KAGRA_Late.txt",
    "KAGRADesignSensitivityT1600593": (
        "LIGO-T1600593-v1-KAGRA_Design.txt"
    ),
    "aLIGOAPlusDesignSensitivityT1800042": (
        "LIGO-T1800042-v5-aLIGO_APLUS.txt"
    ),
    "aLIGODesignSensitivityT1800044": (
        "LIGO-T1800044-v5-aLIGO_DESIGN.txt"
    ),
    "aLIGOaLIGODesignSensitivityT1800044": (
        "LIGO-T1800044-v5-aLIGO_DESIGN.txt"
    ),
    "aLIGOO3LowT1800545": "LIGO-T1800545-v1-aLIGO_O3low.txt",
    "aLIGOaLIGOO3LowT1800545": "LIGO-T1800545-v1-aLIGO_O3low.txt",
    "aLIGO140MpcT1800545": "LIGO-T1800545-v1-aLIGO_140Mpc.txt",
    "aLIGOaLIGO140MpcT1800545": "LIGO-T1800545-v1-aLIGO_140Mpc.txt",
    "aLIGO175MpcT1800545": "LIGO-T1800545-v1-aLIGO_175Mpc.txt",
    "aLIGOaLIGO175MpcT1800545": "LIGO-T1800545-v1-aLIGO_175Mpc.txt",
    "AdVO4IntermediateT1800545": (
        "LIGO-T1800545-v1-AdV_O4intermediate.txt"
    ),
    "aLIGOAdVO4IntermediateT1800545": (
        "LIGO-T1800545-v1-AdV_O4intermediate.txt"
    ),
    "AdVO4T1800545": "LIGO-T1800545-v1-AdV_O4.txt",
    "aLIGOAdVO4T1800545": "LIGO-T1800545-v1-AdV_O4.txt",
    "AdVO3LowT1800545": "LIGO-T1800545-v1-AdV_O3low.txt",
    "aLIGOAdVO3LowT1800545": "LIGO-T1800545-v1-AdV_O3low.txt",
    "KAGRA128MpcT1800545": "LIGO-T1800545-v1-KAGRA_128Mpc.txt",
    "aLIGOKAGRA128MpcT1800545": "LIGO-T1800545-v1-KAGRA_128Mpc.txt",
    "KAGRA25MpcT1800545": "LIGO-T1800545-v1-KAGRA_25Mpc.txt",
    "aLIGOKAGRA25MpcT1800545": "LIGO-T1800545-v1-KAGRA_25Mpc.txt",
    "KAGRA80MpcT1800545": "LIGO-T1800545-v1-KAGRA_80Mpc.txt",
    "aLIGOKAGRA80MpcT1800545": "LIGO-T1800545-v1-KAGRA_80Mpc.txt",
}
VERSIONED_DATA_MODELS = tuple(sorted(VERSIONED_DATA_FILES))
ALIASES = (
    (
        "aLIGODesignSensitivityT1800044",
        "aLIGOaLIGODesignSensitivityT1800044",
    ),
    ("aLIGOO3LowT1800545", "aLIGOaLIGOO3LowT1800545"),
    ("aLIGO140MpcT1800545", "aLIGOaLIGO140MpcT1800545"),
    ("aLIGO175MpcT1800545", "aLIGOaLIGO175MpcT1800545"),
    ("AdVO4IntermediateT1800545", "aLIGOAdVO4IntermediateT1800545"),
    ("AdVO4T1800545", "aLIGOAdVO4T1800545"),
    ("AdVO3LowT1800545", "aLIGOAdVO3LowT1800545"),
    ("KAGRA128MpcT1800545", "aLIGOKAGRA128MpcT1800545"),
    ("KAGRA25MpcT1800545", "aLIGOKAGRA25MpcT1800545"),
    ("KAGRA80MpcT1800545", "aLIGOKAGRA80MpcT1800545"),
)


@pytest.fixture(params=("cpu", "cuda"))
def torch_device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("Torch CUDA device unavailable")
    return request.param


def _lal_psd(model_name, **parameters):
    return analytical.from_string(model_name, **parameters).numpy()


def _lal_file_psd(path, *, length=18, delta_f=1.0, cutoff=1.0):
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


def _install_override(monkeypatch, path):
    model_name = "CosmicExplorerP1600143"
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


def test_versioned_data_file_catalog_and_discovery_match_source():
    remaining = (
        analytical_torch.DATA_FILE_TORCH_ANALYTICAL_MODELS
        - analytical_torch.P1200087_TORCH_ANALYTICAL_MODELS
    )
    assert remaining == set(VERSIONED_DATA_FILES)
    assert {
        name: analytical_torch._VERSIONED_DATA_FILES[name]
        for name in remaining
    } == VERSIONED_DATA_FILES
    assert remaining <= set(analytical.get_lalsim_psd_list())


def test_versioned_data_file_family_matches_lalsimulation(
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
        for model_name in VERSIONED_DATA_MODELS
    }
    original_resolver = analytical_torch.lal.FileResolvePath
    resolved = []

    def record_resolver(filename):
        resolved.append(filename)
        return original_resolver(filename)

    def reject_lal_output(*_args, **_kwargs):
        raise AssertionError("Torch data-file PSD generated a host LAL series")

    def reject_torch_numpy(*_args, **_kwargs):
        raise AssertionError("Torch data-file PSD copied its output to host")

    with scheme.TorchScheme(torch_device):
        with monkeypatch.context() as patch:
            patch.setattr(
                analytical_torch.lal,
                "FileResolvePath",
                record_resolver,
            )
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
                for model_name in VERSIONED_DATA_MODELS
            }

        assert sorted(resolved) == sorted(VERSIONED_DATA_FILES.values())
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


@pytest.mark.parametrize(("canonical", "alias"), ALIASES)
def test_versioned_data_file_deprecated_aliases_are_exact(canonical, alias):
    assert VERSIONED_DATA_FILES[canonical] == VERSIONED_DATA_FILES[alias]
    with scheme.TorchScheme("cpu"):
        canonical_result = analytical.from_string(
            canonical,
            257,
            1.0,
            7.25,
        )
        alias_result = analytical.from_string(alias, 257, 1.0, 7.25)
    assert torch.equal(
        canonical_result._data.tensor,
        alias_result._data.tensor,
    )


def test_versioned_data_file_layout_extrapolation_and_source_pins():
    pins = {
        "aLIGOZeroDetHighPowerGWINC": (
            (9, 100, 8200),
            (
                2.6142986077545313e-42,
                1.5909197486115115e-47,
                1.2507961519099038e-45,
            ),
        ),
        "CosmicExplorerP1600143": (
            (5, 100, 8200),
            (
                1.3637609886916297e-44,
                2.524883371410501e-50,
                3.44710004034311e-47,
            ),
        ),
        "KAGRADesignSensitivityT1600593": (
            (1, 100, 8200),
            (
                6.0733801089655185e-31,
                1.6521533266627296e-47,
                6.739983289779136e-44,
            ),
        ),
        "aLIGOAPlusDesignSensitivityT1800042": (
            (5, 100, 8200),
            (
                3.995844567303201e-40,
                4.6066615364099216e-48,
                4.546582937880486e-46,
            ),
        ),
        "aLIGODesignSensitivityT1800044": (
            (5, 100, 8200),
            (
                3.996643596805695e-40,
                1.6621994643911396e-47,
                2.202956316287601e-45,
            ),
        ),
        "KAGRA80MpcT1800545": (
            (1, 100, 8200),
            (
                6.045180242208668e-31,
                5.462240435999981e-47,
                9.755988970134529e-45,
            ),
        ),
    }
    with scheme.TorchScheme("cpu"):
        for model_name, (indices, values) in pins.items():
            result = analytical.from_string(model_name, 8202, 1.0, 0.0)
            tensor = result._data.tensor
            assert tensor[0] == 0.0
            assert tensor[-1] == 0.0
            torch.testing.assert_close(
                tensor[list(indices)],
                torch.tensor(values, dtype=torch.float64),
                rtol=5e-14,
                atol=0.0,
            )

        cutoff = analytical.from_string(
            "CosmicExplorerP1600143",
            64,
            1.0,
            11.7,
        )
        one_bin = analytical.from_string(
            "CosmicExplorerP1600143",
            1,
            10.0,
            0.0,
        )
        two_bins = analytical.from_string(
            "CosmicExplorerP1600143",
            2,
            10.0,
            0.0,
        )
        negative_cutoff = analytical.from_string(
            "CosmicExplorerP1600143",
            16,
            1.0,
            -1.0,
        )

    assert torch.count_nonzero(cutoff._data.tensor[:11]) == 0
    assert cutoff._data.tensor[11] != 0.0
    assert cutoff._data.tensor[-1] == 0.0
    assert torch.count_nonzero(one_bin._data.tensor) == 0
    assert torch.count_nonzero(two_bins._data.tensor) == 0
    assert torch.count_nonzero(negative_cutoff._data.tensor) == 0


def test_versioned_data_file_resolver_is_rechecked(tmp_path, monkeypatch):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    frequencies = np.array([10.0, 20.0, 40.0])
    np.savetxt(first, np.column_stack((frequencies, [1.0, 2.0, 4.0])))
    np.savetxt(second, np.column_stack((frequencies, [2.0, 4.0, 8.0])))
    paths = iter((str(first), str(second)))
    resolved = []

    def changing_resolver(filename):
        resolved.append(filename)
        return next(paths)

    monkeypatch.setattr(
        analytical_torch.lal,
        "FileResolvePath",
        changing_resolver,
    )
    with scheme.TorchScheme("cpu"):
        first_result = analytical.from_string(
            "CosmicExplorerP1600143",
            41,
            1.0,
            10.0,
        )._data.tensor.clone()
        second_result = analytical.from_string(
            "CosmicExplorerP1600143",
            41,
            1.0,
            10.0,
        )._data.tensor.clone()

    assert resolved == 2 * [VERSIONED_DATA_FILES["CosmicExplorerP1600143"]]
    torch.testing.assert_close(
        second_result[10:-1],
        4.0 * first_result[10:-1],
        rtol=2e-15,
        atol=0.0,
    )


@pytest.mark.parametrize(
    "table",
    (
        "1 1 99\n2 2\n4 4\n8 8\n",
        "1 1\n2 2\n4 4\n16 8\n8 16\n",
        "1 1\n2 2\n4 4\n8 1_0\n",
        "1 1\n2 2\n4 4" + " " * 1050 + "\n",
    ),
    ids=(
        "extra-column",
        "nonmonotonic-frequency",
        "underscore-number",
        "conservative-long-line",
    ),
)
def test_versioned_data_file_unsafe_overrides_use_lalsimulation(
    tmp_path,
    monkeypatch,
    table,
):
    path = tmp_path / "override.txt"
    path.write_text(table, encoding="ascii")
    expected = _lal_file_psd(path)
    calls = _install_override(monkeypatch, path)

    with scheme.TorchScheme("cpu"):
        actual = analytical.from_string(
            "CosmicExplorerP1600143",
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
        "1 1\n2 2\n4 4",
        "1 1\n2 2\n4 4\n# final comment",
        "1 1\r2 2\r4 4\r",
    ),
    ids=(
        "blank-line",
        "unterminated-data",
        "unterminated-comment",
        "carriage-return-only",
    ),
)
def test_versioned_data_file_malformed_override_uses_lal_error(
    tmp_path,
    monkeypatch,
    table,
):
    path = tmp_path / "override.txt"
    path.write_text(table, encoding="ascii")
    with pytest.raises(RuntimeError):
        _lal_file_psd(path)
    calls = _install_override(monkeypatch, path)

    with scheme.TorchScheme("cpu"):
        with pytest.raises(RuntimeError):
            analytical.from_string(
                "CosmicExplorerP1600143",
                18,
                1.0,
                1.0,
            )
    assert len(calls) == 1


def test_versioned_data_file_default_and_mps_use_lalsimulation(monkeypatch):
    model_name = "CosmicExplorerP1600143"
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


def test_versioned_data_file_preserves_real8_boundaries(monkeypatch):
    model_name = "CosmicExplorerP1600143"
    rejected = (np.bool_(True), np.array(1.0), torch.tensor(1.0))

    for delta_f in rejected:
        with pytest.raises(TypeError):
            analytical.from_string(model_name, 8, delta_f, 1.0)
        with scheme.TorchScheme("cpu"):
            with pytest.raises(TypeError):
                analytical.from_string(model_name, 8, delta_f, 1.0)

    with scheme.TorchScheme("cpu"):
        with monkeypatch.context() as patch:
            patch.setattr(
                analytical.lal,
                "CreateREAL8FrequencySeries",
                lambda *_args, **_kwargs: pytest.fail(
                    "accepted scalar unexpectedly used LAL fallback"
                ),
            )
            for delta_f in (1.0, 1, np.float64(1.0), np.int64(1)):
                result = analytical.from_string(
                    model_name,
                    8,
                    delta_f,
                    np.float64(1.0),
                )
                assert result._data.tensor.device.type == "cpu"
