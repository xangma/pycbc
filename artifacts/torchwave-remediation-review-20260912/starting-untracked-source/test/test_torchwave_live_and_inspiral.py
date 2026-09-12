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

"""Tests for torchwave integration in LiveFilterBank and FilterBank."""

import os
import tempfile
import h5py
import numpy as np
import pytest
import torch

try:
    import torchwave  # noqa: F401
    HAS_TORCHWAVE = True
except ImportError:
    HAS_TORCHWAVE = False

from pycbc.waveform.bank import FilterBank, LiveFilterBank
from pycbc.filter.matchedfilter import match


pytestmark = pytest.mark.skipif(
    not HAS_TORCHWAVE, reason="torchwave not installed in environment"
)


@pytest.fixture
def dummy_bank_file():
    """Create a temporary HDF5 template bank file."""
    fd, path = tempfile.mkstemp(suffix=".hdf")
    os.close(fd)
    with h5py.File(path, "w") as f:
        f["mass1"] = np.array([1.4, 1.8, 25.0, 30.0], dtype=np.float32)
        f["mass2"] = np.array([1.2, 1.4, 20.0, 20.0], dtype=np.float32)
        f["spin1z"] = np.array([0.0, 0.0, 0.2, 0.0], dtype=np.float32)
        f["spin2z"] = np.array([0.0, 0.0, -0.1, 0.0], dtype=np.float32)
        f["f_lower"] = np.array([30.0, 30.0, 20.0, 20.0], dtype=np.float32)
        f.attrs["parameters"] = [
            "mass1",
            "mass2",
            "spin1z",
            "spin2z",
            "f_lower",
        ]
    yield path
    if os.path.exists(path):
        os.remove(path)


def test_live_filter_bank_torchwave_parity(dummy_bank_file):
    """Verify LiveFilterBank batched synthesis matches scalar generation."""
    lfb_tw = LiveFilterBank(
        dummy_bank_file,
        sample_rate=2048,
        minimum_buffer=4,
        approximant="TaylorF2",
        enable_torchwave=True,
    )
    lfb_ref = LiveFilterBank(
        dummy_bank_file,
        sample_rate=2048,
        minimum_buffer=4,
        approximant="TaylorF2",
        enable_torchwave=False,
    )

    assert lfb_tw.can_use_torchwave() is True
    assert lfb_ref.can_use_torchwave() is False

    wfs_tw = list(lfb_tw)
    wfs_ref = list(lfb_ref)
    assert len(wfs_tw) == len(wfs_ref) == 4

    for i in range(2):
        overlap, _ = match(wfs_ref[i], wfs_tw[i], low_frequency_cutoff=30.0)
        assert (
            overlap > 0.99999
        ), f"Template {i} match {overlap} below threshold"
        assert hasattr(wfs_tw[i], "params")
        assert hasattr(wfs_tw[i], "sigmasq")
        assert wfs_tw[i].f_lower == wfs_ref[i].f_lower


def test_live_filter_bank_slicing(dummy_bank_file):
    """Verify sliced LiveFilterBank inherits enable_torchwave and iterates."""
    lfb = LiveFilterBank(
        dummy_bank_file,
        sample_rate=2048,
        minimum_buffer=4,
        approximant="TaylorF2",
        enable_torchwave=True,
    )
    # Simulate MPI rank slicing: bank[rank::size]
    slice_bank = lfb[1::2]
    assert slice_bank.can_use_torchwave() is True
    wfs = list(slice_bank)
    assert len(wfs) == 2


def test_filter_bank_get_batch_tensor(dummy_bank_file):
    """Verify FilterBank.get_batch_tensor produces correct 2D tensor."""
    fb = FilterBank(
        dummy_bank_file,
        filter_length=1024,
        delta_f=0.25,
        dtype=np.complex64,
        approximant="TaylorF2",
        enable_torchwave=True,
    )
    assert fb.can_use_torchwave() is True

    tnums = [0, 1]
    tensor, batch_templates = fb.get_batch_tensor(tnums, device="cpu")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (2, 1024)
    assert tensor.dtype == torch.complex64
    assert len(batch_templates) == 2

    for i, t_num in enumerate(tnums):
        scalar_ref = fb[t_num]
        overlap, _ = match(
            scalar_ref, batch_templates[i], low_frequency_cutoff=30.0
        )
        assert overlap > 0.99999, f"Batch template {i} match {overlap} low"


def test_fallback_behavior(dummy_bank_file):
    """Verify clean fallback when disabled or approximant is unsupported."""
    # Explicitly disabled
    fb_dis = FilterBank(
        dummy_bank_file,
        filter_length=1024,
        delta_f=0.25,
        dtype=np.complex64,
        approximant="TaylorF2",
        enable_torchwave=False,
    )
    assert fb_dis.can_use_torchwave() is False

    # Unsupported approximant
    fb_unsup = FilterBank(
        dummy_bank_file,
        filter_length=1024,
        delta_f=0.25,
        dtype=np.complex64,
        approximant="SpinTaylorT4",
        enable_torchwave=None,
    )
    assert fb_unsup.can_use_torchwave() is False


def test_cli_flags():
    """Verify CLI flags in pycbc_live and pycbc_inspiral parse cleanly."""
    import subprocess
    import sys

    # Test pycbc_inspiral --help contains --enable-torchwave
    res = subprocess.run(
        [sys.executable, "bin/pycbc_inspiral", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--enable-torchwave" in res.stdout
    assert "--disable-torchwave" in res.stdout

    # For pycbc_live, verify arguments are registered
    with open("bin/pycbc_live", "r") as f:
        live_src = f.read()
    assert "--enable-torchwave" in live_src
    assert "--disable-torchwave" in live_src
    assert "enable_torchwave=enable_torchwave" in live_src
