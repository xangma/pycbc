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
import numpy as np
import pytest
import pycbc
from pycbc import scheme

pytest.importorskip("jax")
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
    assert ctx.jax_highpass_mode == "lal-serial"


def test_jax_scheme_chisq_modes():
    assert scheme.JAXScheme(chisq_mode="direct-phase").jax_chisq_mode == "direct-phase"
    with pytest.raises(ValueError, match="chisq_mode"):
        scheme.JAXScheme(chisq_mode="unknown")


def test_jax_scheme_highpass_modes():
    assert scheme.JAXScheme(highpass_mode="parallel").jax_highpass_mode == "parallel"
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
    assert ctx.jax_highpass_mode == "lal-serial"

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
        "--processing-scheme", "jax:cpu", "--jax-highpass-mode", "parallel"
    ])
    assert scheme.from_cli(opts).jax_highpass_mode == "parallel"


def test_jax_chisq_mode_rejected_for_cpu():
    parser = argparse.ArgumentParser()
    scheme.insert_processing_option_group(parser)
    opts = parser.parse_args([
        "--processing-scheme", "cpu", "--jax-chisq-mode", "direct-phase"
    ])
    with pytest.raises(ValueError, match="only valid with a JAX"):
        scheme.from_cli(opts)

    opts = parser.parse_args([
        "--processing-scheme", "cpu", "--jax-highpass-mode", "lal-serial"
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
    with scheme.JAXScheme(highpass_mode="parallel"):
        assert scheme.current_backend_key() != key2


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
