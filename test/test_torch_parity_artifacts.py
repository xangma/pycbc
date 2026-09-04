"""Negative controls for the standalone scientific parity evidence tools."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.torch_parity import compare, generate


SETTINGS = {
    "relative_l2": 1e-9, "allclose": False, "zero_pattern": False,
}


@pytest.mark.parametrize(
    "reference,candidate",
    [([np.inf], [np.nan]), ([np.inf], [-np.inf]), ([np.nan], [np.nan]),
     ([1.0, np.inf], [1.0, np.inf]), ([], []),
     ([1.0 + np.inf * 1j], [1.0 + np.nan * 1j])],
)
def test_invalid_numeric_records_cannot_pass(reference, candidate):
    result = compare._compare_record(
        "invalid", np.asarray(reference), np.asarray(candidate), {}, {},
        SETTINGS,
    )
    assert not result["passed"]


def test_finite_equal_records_pass():
    result = compare._compare_record(
        "valid", np.arange(4.0), np.arange(4.0), {}, {}, SETTINGS,
    )
    assert result["passed"]


@pytest.mark.parametrize("magnitude", (1e-300, 1e308))
def test_extreme_magnitudes_do_not_hide_relative_error(magnitude):
    with np.errstate(over="ignore", invalid="ignore"):
        result = compare._compare_record(
            "extreme", np.array([magnitude]), np.array([-magnitude]), {}, {},
            SETTINGS,
        )
    assert not result["passed"]
    assert result["metrics"]["relative_l2"] == pytest.approx(2.0)


@pytest.mark.parametrize("sign", (1, -1))
def test_finite_complex_components_with_overflowing_magnitude(sign):
    reference = np.array([complex(1.5e308, 1.5e308)])
    with np.errstate(over="ignore", invalid="ignore"):
        result = compare._compare_record(
            "extreme_complex", reference, sign * reference, {}, {}, SETTINGS,
        )
    assert result["passed"] == (sign == 1)
    assert result["metrics"]["relative_l2"] == pytest.approx(1 - sign)
    if sign == -1:
        assert "nonfinite computed error metric" in result["failures"]


def test_empty_corpus_fails_cli(tmp_path, monkeypatch):
    for label in ("reference", "candidate"):
        (tmp_path / f"{label}.json").write_text(json.dumps({
            "records": {}, "label": label,
            "runtime": {"source_revision": "test-revision"},
        }))
        np.savez(tmp_path / f"{label}.npz")
    report = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", [
        "compare.py", str(tmp_path / "reference"),
        str(tmp_path / "candidate"), "--profile", "torch", "--report",
        str(report),
    ])
    assert compare.main() == 1
    assert not json.loads(report.read_text())["passed"]


def test_capture_rejects_host_fallback_for_torch_device():
    with pytest.raises(AssertionError, match="host storage"):
        generate._capture("result", np.arange(4), {}, {}, "cuda:1")


@pytest.mark.parametrize("requested", ("cpu", "cpu:0"))
def test_capture_supports_raw_torch_tensor(requested):
    torch = pytest.importorskip("torch")
    arrays, records = {}, {}
    value = torch.arange(4.0, requires_grad=True)
    generate._capture("result", value, arrays, records, requested)
    np.testing.assert_array_equal(arrays["result"], value.detach().numpy())
    assert records["result"]["storage"] == "torch:cpu"
    assert value.requires_grad
    assert generate._scalar(value[2]) == 2.0


@pytest.mark.parametrize("requested", ("mps", "mps:0"))
def test_capture_accepts_actual_mps_tensor(requested):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    arrays, records = {}, {}
    value = torch.arange(4.0, device="mps")
    generate._capture("result", value, arrays, records, requested)
    np.testing.assert_array_equal(arrays["result"], np.arange(4.0))
    assert records["result"]["storage"] == "torch:mps:0"


@pytest.mark.parametrize("requested", ("cuda:1", "cuda"))
def test_capture_checks_cuda_index_without_cuda_hardware(monkeypatch, requested):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    tensor = SimpleNamespace(device=torch.device("cuda:0"))
    monkeypatch.setattr(generate, "_tensor_from", lambda value: tensor)
    with pytest.raises(AssertionError, match="expected"):
        generate._capture("result", object(), {}, {}, requested)


def test_capture_allows_explicit_host_artifacts():
    records = {}
    generate._capture("host_summary", np.arange(4), {}, records)
    assert records["host_summary"]["storage"] == "numpy"
