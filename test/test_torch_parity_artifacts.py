"""Negative controls for the standalone scientific parity evidence tools."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from tools.torch_parity import compare, generate, manifest


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
def test_capture_checks_cuda_index_without_cuda_hardware(
        monkeypatch, requested):
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


@pytest.fixture
def manifest_campaign(tmp_path, monkeypatch):
    """Real Git/build inventories; substitute only installed-runtime probes."""
    specifications = {}
    for label in ("original", "current"):
        source = tmp_path / label
        source.mkdir()
        subprocess.run(["git", "init", "-q", str(source)], check=True)
        (source / ".gitignore").write_text("native.so\n__pycache__/\n")
        (source / "source.py").write_text("version = 1\n")
        manifest.git(source, "add", ".")
        manifest.git(source, "-c", "user.name=Parity test", "-c",
                     "user.email=parity@example.invalid", "commit", "-qm",
                     "fixture")
        (source / "native.so").write_bytes(b"first build")
        python = tmp_path / f"python-{label}"
        python.symlink_to(sys.executable)
        specifications[label] = (source, python)

    def identity(python):
        source = next(source for source, executable in specifications.values()
                      if executable == python)
        return {"source": str(source),
                "revision": manifest.git(source, "rev-parse", "HEAD")}

    monkeypatch.setattr(manifest, "import_identity", identity)
    monkeypatch.setattr(manifest, "packages", lambda python: {"numpy": "1.0"})
    monkeypatch.setattr(manifest, "current_runtime", lambda python: {
        "python": "test", "environment": {"OMP_NUM_THREADS": "1"},
    })
    dependency = tmp_path / "dependencies.json"
    deployment = tmp_path / "deployment.json"
    return specifications, dependency, deployment


def test_manifest_cli_prepare_then_verify(manifest_campaign, tmp_path):
    specifications, dependency, deployment = manifest_campaign
    arguments = ["--dependencies", str(dependency), "--deployment",
                 str(deployment)]
    for label, (source, python) in specifications.items():
        arguments.extend([f"--{label}-source", str(source),
                          f"--{label}-python", str(python)])
    assert manifest.main(["prepare", *arguments]) == 0
    launch = tmp_path / "launch.json"
    assert manifest.main(["verify", *arguments, "--launch", str(launch)]) == 0
    assert json.loads(launch.read_text())["valid"]
    # Preparation cannot silently reseal changed inputs over existing evidence.
    assert manifest.main(["prepare", *arguments]) == 1


@pytest.mark.parametrize("changed", ("deployment", "dependencies"))
def test_manifest_rejects_tampered_files(manifest_campaign, changed):
    specifications, dependency, deployment = manifest_campaign
    manifest.prepare(*manifest_campaign)
    target = deployment if changed == "deployment" else dependency
    payload = json.loads(target.read_text())
    payload["schema_version"] = 999
    target.chmod(0o644)
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="seal|SHA256"):
        manifest.verify(specifications, dependency, deployment)


@pytest.mark.parametrize("change", ("revision", "dirty", "build"))
def test_manifest_rejects_changed_source_or_build(manifest_campaign, change):
    specifications, dependency, deployment = manifest_campaign
    manifest.prepare(*manifest_campaign)
    source = specifications["current"][0]
    if change == "build":
        (source / "native.so").write_bytes(b"other build")
        message = "build-artifact"
    else:
        (source / "source.py").write_text("version = 2\n")
        message = "not clean"
        if change == "revision":
            manifest.git(source, "-c", "user.name=Parity test", "-c",
                         "user.email=parity@example.invalid", "commit", "-qam",
                         "new revision")
            message = "revision"
    with pytest.raises(ValueError, match=message):
        manifest.verify(specifications, dependency, deployment)


@pytest.mark.parametrize("change", ("packages", "environment"))
def test_manifest_rejects_changed_environment(
        manifest_campaign, monkeypatch, tmp_path, change):
    manifest.prepare(*manifest_campaign)
    if change == "packages":
        monkeypatch.setattr(manifest, "packages",
                            lambda python: {"numpy": "2.0"})
    else:
        monkeypatch.setattr(manifest, "current_runtime", lambda python: {
            "python": "test", "environment": {"OMP_NUM_THREADS": "4"},
        })
    launch = tmp_path / "failed-launch.json"
    with pytest.raises(ValueError, match="changed"):
        manifest.verify(*manifest_campaign, launch_path=launch)
    assert not json.loads(launch.read_text())["valid"]


def test_manifest_rejects_wrong_import_before_sealing(
        manifest_campaign, monkeypatch):
    monkeypatch.setattr(manifest, "import_identity", lambda python: {
        "source": "/some/other/checkout", "revision": "wrong",
    })
    with pytest.raises(ValueError, match="import identity"):
        manifest.prepare(*manifest_campaign)
    assert not manifest_campaign[1].exists()


def test_manifest_rejects_mismatched_installed_packages(
        manifest_campaign, monkeypatch):
    monkeypatch.setattr(manifest, "packages", lambda python: {
        "numpy": str(python),
    })
    with pytest.raises(ValueError, match="package fingerprints differ"):
        manifest.prepare(*manifest_campaign)
    assert not manifest_campaign[1].exists()


def test_manifest_scrubs_scheme_but_retains_material_environment(monkeypatch):
    monkeypatch.setenv("PYCBC_SCHEME", "torch:cuda")
    monkeypatch.setenv("PYCBC_TORCH_NATIVE_PORTS", "1")
    monkeypatch.setenv("PYTHONPATH", "/wrong/checkout")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    environment = manifest.clean_environment()
    assert not any(key.startswith("PYCBC_") for key in environment)
    assert "PYTHONPATH" not in environment
    assert environment["OMP_NUM_THREADS"] == "4"


def test_manifest_ignores_bytecode_but_keeps_native_inventory(
        manifest_campaign):
    specifications, _, _ = manifest_campaign
    manifest.prepare(*manifest_campaign)
    cache = specifications["current"][0] / "__pycache__"
    cache.mkdir()
    (cache / "source.pyc").write_bytes(b"bytecode")
    assert manifest.verify(*manifest_campaign)["valid"]


def test_manifest_package_fingerprint_uses_active_distribution(
        tmp_path, monkeypatch):
    active, shadowed = tmp_path / "venv", tmp_path / "system"
    for root, version in ((active, "2.13.0"), (shadowed, "2.1.1")):
        distribution = root / f"torch-{version}.dist-info"
        distribution.mkdir(parents=True)
        (distribution / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: torch\nVersion: {version}\n"
        )
    monkeypatch.setattr(sys, "path", [str(active), str(shadowed)])
    assert manifest.metadata.version("torch") == "2.13.0"
    # Both copies are discoverable, reproducing --system-site-packages.
    assert {dist.version for dist in manifest.metadata.distributions()} == {
        "2.13.0", "2.1.1",
    }
    assert manifest.installed_packages() == {"torch": "2.13.0"}
    monkeypatch.setattr(sys, "path", [str(shadowed), str(active)])
    assert manifest.installed_packages() == {"torch": "2.1.1"}
