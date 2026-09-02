"""Tests for the bounded automatic FFTW wisdom cache."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pycbc.fft import wisdom_cache


class FakeFFTW:
    """Small FFTW stand-in with file-backed import/export operations."""

    _fftw_threaded_set = True
    _fftw_threaded_lib = "pthreads"
    _float_threaded_lib = None
    float_lib = object()

    def __init__(self, import_failure=None, export_failure=None):
        self.calls = []
        self.import_failure = import_failure
        self.export_failure = export_failure

    def import_single_wisdom_from_filename(self, filename):
        self.calls.append(("import", filename))
        if self.import_failure is not None:
            raise self.import_failure

    def export_single_wisdom_to_filename(self, filename):
        self.calls.append(("export", filename))
        if self.export_failure is not None:
            raise self.export_failure
        Path(filename).write_bytes(b"qualified-wisdom")


def _options(tmp_path, **overrides):
    values = {
        "fftw_wisdom_cache": True,
        "fftw_wisdom_cache_dir": str(tmp_path),
        "fftw_import_system_wisdom": False,
        "fftw_input_float_wisdom_file": None,
        "fftw_input_double_wisdom_file": None,
        "fftw_output_float_wisdom_file": None,
        "fftw_output_double_wisdom_file": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _prepare(fftw, **overrides):
    values = {
        "size": 131072,
        "forward": False,
        "direct": True,
        "aligned": True,
        "nthreads": 1,
        "batch": 1,
        "requested_measure_level": 0,
    }
    values.update(overrides)
    return wisdom_cache.prepare_plan(fftw, **values)


@pytest.fixture(autouse=True)
def _reset_cache():
    wisdom_cache.configure_from_cli(
        SimpleNamespace(fftw_wisdom_cache=False)
    )
    yield
    wisdom_cache.configure_from_cli(
        SimpleNamespace(fftw_wisdom_cache=False)
    )


@pytest.fixture
def stable_fingerprint(monkeypatch):
    monkeypatch.setattr(
        wisdom_cache,
        "_fingerprint_payload",
        lambda *_args, **_kwargs: {"fingerprint": "stable"},
    )


def test_cache_miss_measure_export_and_warm_import(
    monkeypatch, tmp_path, stable_fingerprint
):
    fftw = FakeFFTW()
    wisdom_cache.configure_from_cli(_options(tmp_path))

    level, entry = _prepare(fftw)

    assert level == 1
    assert entry is not None
    assert fftw.calls == []
    wisdom_cache.record_plan(fftw, entry, level)
    assert not wisdom_cache.has_pending_export()
    cache_files = list(tmp_path.glob("*.wisdom"))
    assert cache_files == [entry.path]
    assert entry.path.read_bytes() == b"qualified-wisdom"
    assert entry.path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.glob("*.tmp")) == []

    warm_fftw = FakeFFTW()
    wisdom_cache.configure_from_cli(_options(tmp_path))
    warm_level, warm_entry = _prepare(warm_fftw)

    assert warm_level == 0
    assert warm_entry.path == entry.path
    assert warm_fftw.calls == [("import", str(entry.path))]
    assert not wisdom_cache.has_pending_export()


def test_cancelled_plan_releases_cold_writer_lock(
    tmp_path, stable_fingerprint
):
    fftw = FakeFFTW()
    wisdom_cache.configure_from_cli(_options(tmp_path))

    _, entry = _prepare(fftw)
    wisdom_cache.cancel_plan(entry)

    assert entry.lock_file is None
    assert list(tmp_path.glob("*.wisdom")) == []


def test_failed_immediate_export_is_retried_by_cli_lifecycle(
    tmp_path, stable_fingerprint
):
    failure = OSError("temporary cache failure")
    fftw = FakeFFTW(export_failure=failure)
    wisdom_cache.configure_from_cli(_options(tmp_path))
    level, entry = _prepare(fftw)

    wisdom_cache.record_plan(fftw, entry, level)

    assert entry.lock_file is None
    assert wisdom_cache.has_pending_export()
    assert not entry.path.exists()

    fftw.export_failure = None
    wisdom_cache.export_pending(fftw)

    assert not wisdom_cache.has_pending_export()
    assert entry.path.read_bytes() == b"qualified-wisdom"


def test_corrupt_cache_falls_back_to_bounded_measurement(
    caplog, tmp_path, stable_fingerprint
):
    wisdom_cache.configure_from_cli(_options(tmp_path))
    _, seed_entry = _prepare(FakeFFTW())
    seed_entry.path.write_bytes(b"corrupt")

    failure = RuntimeError("invalid wisdom")
    fftw = FakeFFTW(import_failure=failure)
    wisdom_cache.configure_from_cli(_options(tmp_path))
    level, entry = _prepare(fftw)

    assert level == 1
    assert entry.path == seed_entry.path
    assert fftw.calls == [
        ("import", str(entry.path)),
        ("import", str(entry.path)),
    ]
    assert "Ignoring unusable cached FFTW wisdom" in caplog.text


@pytest.mark.parametrize(
    "override",
    (
        {"size": 65536},
        {"forward": True},
        {"direct": False},
        {"aligned": False},
        {"nthreads": 2},
        {"batch": 2},
        {"requested_measure_level": 2},
        {"requested_measure_level": 3},
    ),
)
def test_cache_never_changes_unqualified_plans(
    tmp_path, stable_fingerprint, override
):
    wisdom_cache.configure_from_cli(_options(tmp_path))
    requested = override.get("requested_measure_level", 0)

    level, entry = _prepare(FakeFFTW(), **override)

    assert level == requested
    assert entry is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "explicit",
    (
        {"fftw_import_system_wisdom": True},
        {"fftw_input_float_wisdom_file": "manual.wisdom"},
        {"fftw_input_double_wisdom_file": "manual.wisdom"},
        {"fftw_output_float_wisdom_file": "manual.wisdom"},
        {"fftw_output_double_wisdom_file": "manual.wisdom"},
    ),
)
def test_manual_wisdom_options_take_precedence(
    tmp_path, stable_fingerprint, explicit
):
    wisdom_cache.configure_from_cli(_options(tmp_path, **explicit))

    level, entry = _prepare(FakeFFTW())

    assert level == 0
    assert entry is None
    assert list(tmp_path.iterdir()) == []


def test_fingerprint_change_selects_a_new_cache_file(
    monkeypatch, tmp_path
):
    identity = {"value": "first"}
    monkeypatch.setattr(
        wisdom_cache,
        "_fingerprint_payload",
        lambda *_args, **_kwargs: identity.copy(),
    )
    wisdom_cache.configure_from_cli(_options(tmp_path))
    _, first = _prepare(FakeFFTW())
    identity["value"] = "second"
    _, second = _prepare(FakeFFTW())

    assert first.path != second.path


def test_default_cache_directory_honors_absolute_xdg_home(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert wisdom_cache._default_cache_dir() == tmp_path / "pycbc" / "fftw"

    monkeypatch.setenv("XDG_CACHE_HOME", "relative-cache")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    assert wisdom_cache._default_cache_dir() == (
        tmp_path / "home" / ".cache" / "pycbc" / "fftw"
    )
