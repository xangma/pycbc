"""CI scheme selection preserves explicit choices and CPU fixtures."""

import pytest

import utils
from pycbc import scheme


@pytest.mark.parametrize("device", ("cpu", "cuda", "cuda:1"))
def test_environment_selects_torch_device(monkeypatch, device):
    monkeypatch.setenv("PYCBC_TEST_SCHEME", "torch:" + device)
    monkeypatch.setattr("sys.argv", ["pytest", "-q"])
    monkeypatch.setattr(scheme, "TorchScheme", lambda device: device)
    selected, context = utils.parse_args_all_schemes("selection")
    assert selected == "torch"
    assert context == device


def test_command_line_overrides_environment_scheme(monkeypatch):
    monkeypatch.setenv("PYCBC_TEST_SCHEME", "torch:cuda")
    monkeypatch.setattr("sys.argv", ["test.py", "--scheme", "cpu"])
    cpu_context = object()
    monkeypatch.setattr(utils, "CPUScheme", lambda: cpu_context)
    assert utils.parse_args_all_schemes("selection") == ["cpu", cpu_context]


def test_command_line_overrides_environment_device(monkeypatch):
    monkeypatch.setenv("PYCBC_TEST_SCHEME", "torch:cuda")
    monkeypatch.setattr("sys.argv", ["test.py", "--scheme", "torch"])
    monkeypatch.setattr(scheme, "TorchScheme", lambda device: device)
    assert utils.parse_args_all_schemes("selection") == ["torch", "cpu"]


def test_cpu_only_tests_still_run_with_torch_environment(monkeypatch, capsys):
    monkeypatch.setenv("PYCBC_TEST_SCHEME", "torch:cuda")
    monkeypatch.setattr("sys.argv", ["pytest", "-q"])
    utils.parse_args_cpu_only("CPU fixture")
    assert "Running CPU unit tests for CPU fixture" in capsys.readouterr().out


@pytest.mark.parametrize("selection", ("torch:", "cpu:cuda", "typo"))
def test_invalid_environment_selector_fails(monkeypatch, selection):
    monkeypatch.setenv("PYCBC_TEST_SCHEME", selection)
    with pytest.raises(ValueError, match="PYCBC_TEST_SCHEME"):
        utils.parse_args_all_schemes("selection")
