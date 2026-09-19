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
import pytest
import pycbc
from pycbc import scheme

pytest.importorskip("jax")


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

    # Test explicit jax:cpu
    opts = parser.parse_args(["--processing-scheme", "jax:cpu"])
    ctx = scheme.from_cli(opts)
    assert isinstance(ctx, scheme.JAXScheme)
    assert ctx.jax_device.platform == "cpu"


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
