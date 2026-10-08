# Copyright (C) 2026
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Tests for PyCBC JAXScheme and CLI integration."""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import numpy as np
import pytest
import pycbc
from pycbc import scheme

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from pycbc.types.array_jax import JAXArrayData


def test_jax_scheme_availability():
    """Verify HAVE_JAX is True when JAX is installed."""
    assert getattr(pycbc, "HAVE_JAX", False) is True


def test_jax_scheme_init_default():
    """Verify JAXScheme initializes with CPU device by default."""
    ctx = scheme.JAXScheme()
    assert ctx.device_spec == "cpu"
    assert ctx.prefix == "jax"
    assert ctx.jax_device is not None
    assert ctx.jax_device.platform == "cpu"
    assert ctx.jax_chisq_mode == "cpu-compatible"
    assert ctx.jax_highpass_mode == "lal-coefficients"
    assert ctx.jax_reference_operations == frozenset()


@pytest.mark.parametrize("operations", [
    ["sum", "cumsum"], ("cumsum", "sum", "sum"), " sum, cumsum, sum ",
])
def test_jax_scheme_reference_operations_accept_independent_names(operations):
    ctx = scheme.JAXScheme(reference_operations=operations)
    assert ctx.jax_reference_operations == frozenset({"sum", "cumsum"})


def test_jax_scheme_reference_operations_detach_from_mutable_configuration():
    operations = ["sum"]
    ctx = scheme.JAXScheme(reference_operations=iter(operations))
    operations.append("dot")
    assert ctx.jax_reference_operations == frozenset({"sum"})


@pytest.mark.parametrize("operations", [(), [], "", " "])
def test_jax_scheme_empty_reference_operations_keep_jax(operations):
    ctx = scheme.JAXScheme(reference_operations=operations)
    assert not ctx.jax_reference_operations


@pytest.mark.parametrize("operations", [
    ["unknown"], "sum,unknown", "sum,", "sum,,dot", ["sum,dot"], None, [1],
])
def test_jax_scheme_rejects_invalid_reference_operations(operations):
    with pytest.raises(ValueError, match="reference_operations"):
        scheme.JAXScheme(reference_operations=operations)


def test_jax_scheme_chisq_modes():
    assert scheme.JAXScheme(chisq_mode="direct-phase").jax_chisq_mode == "direct-phase"
    with pytest.raises(ValueError, match="chisq_mode"):
        scheme.JAXScheme(chisq_mode="unknown")


def test_jax_scheme_highpass_modes():
    assert scheme.JAXScheme(highpass_mode="closed-form").jax_highpass_mode == "closed-form"
    with pytest.raises(ValueError, match="highpass_mode"):
        scheme.JAXScheme(highpass_mode="unknown")


def test_jax_scheme_context_manager():
    """Verify JAXScheme enters and exits cleanly, restoring default scheme."""
    initial_scheme = scheme.mgr.state
    with scheme.JAXScheme() as ctx:
        assert scheme.mgr.state is ctx
        assert scheme.current_prefix() == "jax"
    assert scheme.mgr.state is initial_scheme
    assert scheme.current_prefix() != "jax"


@pytest.mark.parametrize("failure", ["create", "enter"])
def test_jax_scheme_failed_device_entry_restores_scheme(monkeypatch, failure):
    initial_scheme = scheme.mgr.state
    ctx = scheme.JAXScheme()

    class DeviceContext:
        def __enter__(self):
            raise RuntimeError("device entry failed")

    def default_device(device):
        if failure == "create":
            raise RuntimeError("device entry failed")
        return DeviceContext()

    monkeypatch.setattr(jax, "default_device", default_device)
    with pytest.raises(RuntimeError, match="device entry failed"):
        with ctx:
            pytest.fail("failed device entry reached the context body")
    assert scheme.mgr.state is initial_scheme
    assert not scheme.mgr._lock
    assert ctx._prev_default_device is None
    with scheme.CPUScheme():
        assert scheme.current_prefix() == "cpu"


def test_jax_scheme_failed_device_exit_restores_scheme(monkeypatch):
    initial_scheme = scheme.mgr.state
    ctx = scheme.JAXScheme()

    class DeviceContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise RuntimeError("device exit failed")

    monkeypatch.setattr(jax, "default_device", lambda device: DeviceContext())
    with pytest.raises(RuntimeError, match="device exit failed"):
        with ctx:
            assert scheme.mgr.state is ctx
    assert scheme.mgr.state is initial_scheme
    assert not scheme.mgr._lock
    assert ctx._prev_default_device is None
    with scheme.CPUScheme():
        assert scheme.current_prefix() == "cpu"


def test_jax_scheme_body_exception_restores_default_device():
    initial_scheme = scheme.mgr.state
    initial_device = jax.config.jax_default_device
    with pytest.raises(RuntimeError, match="body failed"):
        with scheme.JAXScheme() as ctx:
            assert jax.config.jax_default_device is ctx.jax_device
            raise RuntimeError("body failed")
    assert scheme.mgr.state is initial_scheme
    assert not scheme.mgr._lock
    assert jax.config.jax_default_device is initial_device


def test_jax_scheme_prefix_mapping():
    """Verify scheme_prefix dictionary has JAXScheme mapped to 'jax'."""
    assert scheme.scheme_prefix[scheme.JAXScheme] == "jax"


def test_jax_scheme_from_cli():
    """Verify from_cli parses --processing-scheme jax properly."""
    parser = argparse.ArgumentParser()
    scheme.insert_processing_option_group(parser)

    # Test default jax
    opts = parser.parse_args(["--processing-scheme", "jax"])
    ctx = scheme.from_cli(opts)
    assert isinstance(ctx, scheme.JAXScheme)
    assert ctx.jax_device.platform == "cpu"
    assert ctx.jax_highpass_mode == "lal-coefficients"

    # Test explicit jax:cpu
    opts = parser.parse_args(["--processing-scheme", "jax:cpu"])
    ctx = scheme.from_cli(opts)
    assert isinstance(ctx, scheme.JAXScheme)
    assert ctx.jax_device.platform == "cpu"

    opts = parser.parse_args([
        "--processing-scheme", "jax:cpu", "--jax-chisq-mode", "direct-phase"
    ])
    assert scheme.from_cli(opts).jax_chisq_mode == "direct-phase"

    opts = parser.parse_args([
        "--processing-scheme", "jax:cpu", "--jax-highpass-mode", "closed-form"
    ])
    assert scheme.from_cli(opts).jax_highpass_mode == "closed-form"


def test_jax_reference_operations_cli():
    parser = argparse.ArgumentParser()
    scheme.insert_processing_option_group(parser)
    opts = parser.parse_args([
        "--processing-scheme", "jax:cpu",
        "--jax-reference-operations", " sum, dot, sum ",
    ])
    scheme.verify_processing_options(opts, parser)
    assert scheme.from_cli(opts).jax_reference_operations == frozenset({"sum", "dot"})
    default = parser.parse_args(["--processing-scheme", "jax:cpu"])
    assert not scheme.from_cli(default).jax_reference_operations


@pytest.mark.parametrize("operations", ["unknown", "sum,"])
def test_jax_reference_operations_cli_rejects_invalid_names(operations):
    parser = argparse.ArgumentParser()
    scheme.insert_processing_option_group(parser)
    opts = parser.parse_args([
        "--processing-scheme", "jax:cpu", "--jax-reference-operations", operations,
    ])
    with pytest.raises(ValueError, match="reference_operations"):
        scheme.from_cli(opts)
    with pytest.raises(SystemExit) as exc:
        scheme.verify_processing_options(opts, parser)
    assert exc.value.code == 2


@pytest.mark.parametrize("operations", ["sum", ""])
def test_jax_reference_operations_cli_rejects_cpu_scheme(operations):
    parser = argparse.ArgumentParser()
    scheme.insert_processing_option_group(parser)
    opts = parser.parse_args([
        "--processing-scheme", "cpu", "--jax-reference-operations", operations,
    ])
    with pytest.raises(ValueError, match="only valid with a JAX"):
        scheme.from_cli(opts)
    with pytest.raises(SystemExit) as exc:
        scheme.verify_processing_options(opts, parser)
    assert exc.value.code == 2


@pytest.mark.parametrize("backend", ["cuda", "gpu"])
@pytest.mark.parametrize("explicit_index,option_index,expected", [
    (None, 0, 0), (None, 1, 1), (1, 0, 1), (0, 1, 0),
])
def test_jax_scheme_cli_gpu_device_selection(
        monkeypatch, backend, explicit_index, option_index, expected):
    devices = [object(), object()]
    monkeypatch.setattr(jax, "devices", lambda platform=None: devices)
    specification = f"jax:{backend}"
    if explicit_index is not None:
        specification += f":{explicit_index}"
    parser = argparse.ArgumentParser()
    scheme.insert_processing_option_group(parser)
    opts = parser.parse_args([
        "--processing-scheme", specification,
        "--processing-device-id", str(option_index),
    ])
    ctx = scheme.from_cli(opts)
    assert ctx.device_spec == f"{backend}:{expected}"
    assert ctx.jax_device is devices[expected]
    assert ctx.jax_highpass_mode == "closed-form"


@pytest.mark.parametrize("index", [-1, 2])
def test_jax_scheme_cli_rejects_invalid_explicit_gpu_index(monkeypatch, index):
    monkeypatch.setattr(jax, "devices", lambda platform=None: [object(), object()])
    parser = argparse.ArgumentParser()
    scheme.insert_processing_option_group(parser)
    opts = parser.parse_args(["--processing-scheme", f"jax:gpu:{index}"])
    with pytest.raises(ValueError, match="out of range"):
        scheme.from_cli(opts)


@pytest.mark.parametrize("specification,threads", [("cpu", 1), ("cpu:2", 2)])
def test_cpu_cli_thread_selection_is_unchanged(specification, threads):
    parser = argparse.ArgumentParser()
    scheme.insert_processing_option_group(parser)
    ctx = scheme.from_cli(parser.parse_args(["--processing-scheme", specification]))
    assert isinstance(ctx, scheme.CPUScheme)
    assert ctx.num_threads == threads


def test_jax_chisq_mode_rejected_for_cpu():
    parser = argparse.ArgumentParser()
    scheme.insert_processing_option_group(parser)
    opts = parser.parse_args([
        "--processing-scheme", "cpu", "--jax-chisq-mode", "direct-phase"
    ])
    with pytest.raises(ValueError, match="only valid with a JAX"):
        scheme.from_cli(opts)

    opts = parser.parse_args([
        "--processing-scheme", "cpu", "--jax-highpass-mode", "lal-coefficients"
    ])
    with pytest.raises(ValueError, match="only valid with a JAX"):
        scheme.from_cli(opts)


def test_jax_scheme_invalid_device():
    """Verify invalid device specifier raises ValueError."""
    with pytest.raises(ValueError):
        scheme.JAXScheme(device="non_existent_accelerator_99")


def test_current_backend_key():
    """Verify current_backend_key returns a consistent hashable key."""
    key1 = scheme.current_backend_key()
    assert isinstance(key1, tuple)
    with scheme.JAXScheme():
        key2 = scheme.current_backend_key()
        assert key2[0] == "jax"
        assert key1 != key2
    with scheme.JAXScheme(highpass_mode="closed-form"):
        assert scheme.current_backend_key() != key2


def test_reference_operations_distinguish_backend_resources():
    with scheme.JAXScheme():
        default = scheme.current_backend_key()
    with scheme.JAXScheme(reference_operations=("sum",)):
        sum_only = scheme.current_backend_key()
    with scheme.JAXScheme(reference_operations=("sum", "dot")):
        both = scheme.current_backend_key()
    with scheme.JAXScheme(reference_operations=("dot", "sum", "sum")):
        reordered = scheme.current_backend_key()
    assert len({default, sum_only, both}) == 3
    assert reordered == both


def test_jax_storage_assignment_preserves_destination_dtype():
    data = JAXArrayData(jnp.zeros(4, dtype=jnp.complex64))
    values = jnp.asarray([1 + 2j, 3 + 4j], dtype=jnp.complex128)

    data.set_slice(slice(1, 3), values)
    assert data.array.dtype == jnp.complex64
    np.testing.assert_array_equal(
        np.asarray(data.array),
        np.asarray([0, 1 + 2j, 3 + 4j, 0], np.complex64),
    )

    data.set_array(jnp.ones(4, dtype=jnp.complex128))
    assert data.array.dtype == jnp.complex64


def _cache_process(tmp_path, source, environment=None):
    """Test import-time cache policy without altering this process's config."""
    env = os.environ.copy()
    for name in (
        "JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_COMPILATION_CACHE",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
        "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
        "JAX_COMPILATION_CACHE_MAX_SIZE", "JAX_DISABLE_JIT",
        "PYCBC_JAX_COMPILATION_AUDIT_DIR",
    ):
        env.pop(name, None)
    env["JAX_COMPILATION_CACHE_DIR"] = str(tmp_path / "cache")
    if environment:
        env.update(environment)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)], env=env,
        check=True, capture_output=True, text=True, timeout=45,
    )
    return json.loads(result.stdout.splitlines()[-1])


def test_jax_cache_default_applies_after_earlier_import(tmp_path):
    result = _cache_process(tmp_path, """
        import json
        import jax
        before = jax.config.jax_persistent_cache_min_compile_time_secs
        from pycbc.scheme import JAXScheme
        JAXScheme('cpu')
        first = jax.config.jax_persistent_cache_min_compile_time_secs
        JAXScheme('cpu')
        print(json.dumps(dict(before=before, first=first,
            repeated=jax.config.jax_persistent_cache_min_compile_time_secs,
            directory=jax.config.jax_compilation_cache_dir,
            enabled=jax.config.jax_enable_compilation_cache)))
    """)
    assert result == dict(before=1., first=0., repeated=0.,
                          directory=str(tmp_path / "cache"), enabled=True)


@pytest.mark.parametrize("source,environment,expected", [
    ("", {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "2.75"}, 2.75),
    ("", {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "1"}, 1.),
    ("os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '2.5'",
     {}, 2.5),
    ("jax.config.update('jax_persistent_cache_min_compile_time_secs', .25)",
     {"JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "2.75"}, .25),
])
def test_jax_cache_preserves_explicit_threshold(
        tmp_path, source, environment, expected):
    result = _cache_process(tmp_path, f"""
        import json
        import os
        import jax
        {source}
        from pycbc.scheme import JAXScheme
        JAXScheme('cpu')
        print(json.dumps(jax.config.jax_persistent_cache_min_compile_time_secs))
    """, environment)
    assert result == expected


def test_jax_cache_preserves_configured_directory_and_disabled_policy(tmp_path):
    result = _cache_process(tmp_path, """
        import json
        import os
        import jax
        directory = os.environ['JAX_COMPILATION_CACHE_DIR'] + '-configured'
        jax.config.update('jax_compilation_cache_dir', directory)
        from pycbc.scheme import JAXScheme
        JAXScheme('cpu')
        first = jax.config.jax_compilation_cache_dir
        jax.config.update('jax_enable_compilation_cache', False)
        jax.config.update('jax_persistent_cache_min_compile_time_secs', .75)
        JAXScheme('cpu')
        print(json.dumps(dict(directory=first,
            enabled=jax.config.jax_enable_compilation_cache,
            threshold=jax.config.jax_persistent_cache_min_compile_time_secs)))
    """)
    assert result == dict(directory=str(tmp_path / "cache-configured"),
                          enabled=False, threshold=.75)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("programmatic", [False, True])
def test_jax_cache_disabled_configuration_does_not_initialize(
        tmp_path, programmatic):
    statement = ("jax.config.update('jax_enable_compilation_cache', False)"
                 if programmatic else "")
    result = _cache_process(tmp_path, f"""
        import json
        import jax
        {statement}
        from pycbc.scheme import JAXScheme
        JAXScheme('cpu')
        print(json.dumps(dict(enabled=jax.config.jax_enable_compilation_cache,
            threshold=jax.config.jax_persistent_cache_min_compile_time_secs)))
    """, {} if programmatic else {"JAX_ENABLE_COMPILATION_CACHE": "false"})
    assert result == dict(enabled=False, threshold=1.)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("directory", ["", "0", "false", "none", "off"])
def test_jax_cache_off_policy_is_effective(tmp_path, directory):
    result = _cache_process(tmp_path, """
        import json
        import jax
        from pycbc.scheme import JAXScheme
        JAXScheme('cpu')
        print(json.dumps(dict(enabled=jax.config.jax_enable_compilation_cache)))
    """, {"JAX_COMPILATION_CACHE_DIR": directory})
    assert result == {"enabled": False}
    assert not (tmp_path / "cache").exists()


def test_jax_cache_initialization_error_is_ignored(tmp_path):
    invalid_dir = tmp_path / "file"
    invalid_dir.write_text("not a directory")
    result = _cache_process(tmp_path, """
        import json
        from pycbc.scheme import JAXScheme
        JAXScheme('cpu')
        print(json.dumps(dict(error=None)))
    """, {"JAX_COMPILATION_CACHE_DIR": str(invalid_dir)})
    assert result == dict(error=None)


def test_jax_short_kernel_persists_and_reuses_without_repeat_requests(tmp_path):
    source = """
        import json
        import numpy as np
        import jax
        from pycbc.scheme import JAXScheme
        context = JAXScheme('cpu')
        events = dict(requests=0, hits=0, writes=0)
        names = {
            '/jax/compilation_cache/compile_requests_use_cache': 'requests',
            '/jax/compilation_cache/cache_hits': 'hits',
            '/jax/compilation_cache/cache_misses': 'writes'}
        def listener(event, **metadata):
            if event in names:
                events[names[event]] += 1
        jax.monitoring.register_event_listener(listener)
        traces = []
        @jax.jit
        def small_kernel(value):
            traces.append(value.shape)
            return value * 3. + 7.
        def run(count, offset):
            data = jax.device_put(np.arange(count, dtype=np.float32) + offset)
            assert data.devices() == {context.jax_device}
            output = small_kernel(data)
            output.block_until_ready()
            assert output.devices() == {context.jax_device}
            np.testing.assert_array_equal(np.asarray(output),
                                          np.asarray(data) * 3. + 7.)
        with context:
            run(17, 0.)
            first = dict(events)
            run(17, 2.)
            repeated = dict(events)
            run(19, 0.)
        print(json.dumps(dict(first=first, repeated=repeated,
                              changed=dict(events), traces=len(traces))))
    """
    first = _cache_process(tmp_path, source)
    assert first["first"] == dict(requests=1, hits=0, writes=1)
    assert first["repeated"] == first["first"]
    assert first["changed"] == dict(requests=2, hits=0, writes=2)
    assert first["traces"] == 2
    assert list((tmp_path / "cache").glob("*-cache"))

    second = _cache_process(tmp_path, source)
    assert second["first"] == dict(requests=1, hits=1, writes=0)
    assert second["repeated"] == second["first"]
    assert second["changed"] == dict(requests=2, hits=2, writes=0)
    assert second["traces"] == 2
