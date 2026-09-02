"""Tests for the shared FFTW command-line wisdom helpers."""

import argparse
from types import SimpleNamespace

import pytest

import pycbc.fft
from pycbc.fft import parser_support
from pycbc.fft import fftw


BACKEND_SHAPES = ([], ["fftw"], ["torch"])


class RecordingFFTW:
    """Small FFTW stand-in that records wisdom operations in order."""

    def __init__(self):
        self.calls = []

    def import_sys_wisdom(self):
        self.calls.append(("import_system",))

    def import_single_wisdom_from_filename(self, filename):
        self.calls.append(("import_float", filename))

    def import_double_wisdom_from_filename(self, filename):
        self.calls.append(("import_double", filename))

    def export_single_wisdom_to_filename(self, filename):
        self.calls.append(("export_float", filename))

    def export_double_wisdom_to_filename(self, filename):
        self.calls.append(("export_double", filename))


class RecordingCache:
    """Automatic-cache stand-in used to check parser lifecycle wiring."""

    def __init__(self, pending=False):
        self.pending = pending
        self.calls = []

    def configure_from_cli(self, opt):
        self.calls.append(("configure", opt))

    def has_pending_export(self):
        self.calls.append(("pending",))
        return self.pending

    def export_pending(self, fftw):
        self.calls.append(("export", fftw))


@pytest.mark.parametrize("fft_backends", BACKEND_SHAPES)
@pytest.mark.parametrize("options_present", (False, True))
def test_wisdom_helpers_ignore_backend_shape_and_absent_options(
    monkeypatch, fft_backends, options_present
):
    loads = []
    monkeypatch.setattr(
        parser_support,
        "_load_fftw_for_wisdom",
        lambda: loads.append("fftw"),
    )
    options = {"fft_backends": fft_backends}
    if options_present:
        options.update(
            fftw_import_system_wisdom=False,
            fftw_input_float_wisdom_file=None,
            fftw_input_double_wisdom_file=None,
            fftw_output_float_wisdom_file=None,
            fftw_output_double_wisdom_file=None,
        )
    opt = SimpleNamespace(**options)

    assert parser_support.import_wisdom_from_cli(opt) is None
    assert parser_support.export_wisdom_from_cli(opt) is None
    assert loads == []


@pytest.mark.parametrize("fft_backends", BACKEND_SHAPES)
def test_import_wisdom_from_cli_uses_required_order(
    monkeypatch, fft_backends
):
    fftw = RecordingFFTW()
    monkeypatch.setattr(
        parser_support, "_load_fftw_for_wisdom", lambda: fftw
    )
    opt = SimpleNamespace(
        fft_backends=fft_backends,
        fftw_import_system_wisdom=True,
        fftw_input_float_wisdom_file="float.wisdom",
        fftw_input_double_wisdom_file="double.wisdom",
    )

    parser_support.import_wisdom_from_cli(opt)

    assert fftw.calls == [
        ("import_system",),
        ("import_float", "float.wisdom"),
        ("import_double", "double.wisdom"),
    ]


@pytest.mark.parametrize("fft_backends", BACKEND_SHAPES)
def test_export_wisdom_from_cli_uses_required_order(
    monkeypatch, fft_backends
):
    fftw = RecordingFFTW()
    monkeypatch.setattr(
        parser_support, "_load_fftw_for_wisdom", lambda: fftw
    )
    opt = SimpleNamespace(
        fft_backends=fft_backends,
        fftw_output_float_wisdom_file="float.wisdom",
        fftw_output_double_wisdom_file="double.wisdom",
    )

    parser_support.export_wisdom_from_cli(opt)

    assert fftw.calls == [
        ("export_float", "float.wisdom"),
        ("export_double", "double.wisdom"),
    ]


@pytest.mark.parametrize(
    ("helper", "opt"),
    (
        (
            parser_support.import_wisdom_from_cli,
            SimpleNamespace(fftw_import_system_wisdom=True),
        ),
        (
            parser_support.export_wisdom_from_cli,
            SimpleNamespace(fftw_output_float_wisdom_file="float.wisdom"),
        ),
    ),
)
def test_wisdom_helpers_propagate_unavailable_fftw(
    monkeypatch, helper, opt
):
    failure = ImportError("FFTW is unavailable")

    def fail_load():
        raise failure

    monkeypatch.setattr(parser_support, "_load_fftw_for_wisdom", fail_load)

    with pytest.raises(ImportError) as exc_info:
        helper(opt)
    assert exc_info.value is failure


@pytest.mark.parametrize(
    ("helper", "opt", "method"),
    (
        (
            parser_support.import_wisdom_from_cli,
            SimpleNamespace(fftw_input_float_wisdom_file="float.wisdom"),
            "import_single_wisdom_from_filename",
        ),
        (
            parser_support.export_wisdom_from_cli,
            SimpleNamespace(fftw_output_double_wisdom_file="double.wisdom"),
            "export_double_wisdom_to_filename",
        ),
    ),
)
def test_wisdom_helpers_propagate_native_failures(
    monkeypatch, helper, opt, method
):
    fftw = RecordingFFTW()
    failure = RuntimeError("wisdom operation failed")

    def fail_operation(_filename):
        raise failure

    setattr(fftw, method, fail_operation)
    monkeypatch.setattr(
        parser_support, "_load_fftw_for_wisdom", lambda: fftw
    )

    with pytest.raises(RuntimeError) as exc_info:
        helper(opt)
    assert exc_info.value is failure


def test_wisdom_helpers_are_exported_from_fft_package():
    assert (
        pycbc.fft.import_wisdom_from_cli
        is parser_support.import_wisdom_from_cli
    )
    assert (
        pycbc.fft.export_wisdom_from_cli
        is parser_support.export_wisdom_from_cli
    )


def test_automatic_cache_configures_without_eager_fftw_load(monkeypatch):
    cache = RecordingCache()
    fftw_loads = []
    monkeypatch.setattr(
        parser_support, "_load_wisdom_cache", lambda: cache
    )
    monkeypatch.setattr(
        parser_support,
        "_load_fftw_for_wisdom",
        lambda: fftw_loads.append("fftw"),
    )
    opt = SimpleNamespace(fftw_wisdom_cache=True)

    parser_support.import_wisdom_from_cli(opt)

    assert cache.calls == [("configure", opt)]
    assert fftw_loads == []


def test_automatic_cache_exports_through_shared_lifecycle(monkeypatch):
    cache = RecordingCache(pending=True)
    fftw = RecordingFFTW()
    monkeypatch.setattr(
        parser_support, "_load_wisdom_cache", lambda: cache
    )
    monkeypatch.setattr(
        parser_support, "_load_fftw_for_wisdom", lambda: fftw
    )

    parser_support.export_wisdom_from_cli(SimpleNamespace())

    assert cache.calls == [("pending",), ("export", fftw)]


def test_automatic_cache_cli_defaults_and_overrides(tmp_path):
    parser = argparse.ArgumentParser()
    fftw.insert_fft_options(parser)

    assert parser.parse_args([]).fftw_wisdom_cache is True
    assert parser.parse_args(
        ["--no-fftw-wisdom-cache"]
    ).fftw_wisdom_cache is False
    assert parser.parse_args(
        ["--fftw-wisdom-cache-dir", str(tmp_path)]
    ).fftw_wisdom_cache_dir == str(tmp_path)


def test_automatic_cache_cli_rejects_directory_when_disabled(tmp_path):
    parser = argparse.ArgumentParser()
    fftw.insert_fft_options(parser)
    options = parser.parse_args(
        [
            "--no-fftw-wisdom-cache",
            "--fftw-wisdom-cache-dir",
            str(tmp_path),
        ]
    )

    with pytest.raises(SystemExit):
        fftw.verify_fft_options(options, parser)
