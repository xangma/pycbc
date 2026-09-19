# Copyright (C) 2026 PyCBC contributors
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
"""Integration checks for diffgw JAX-native IMR waveform generation (PhenomD/PhenomXAS)."""

import h5py
import numpy as np
import pytest

from pycbc import scheme
from pycbc.filter import match
from pycbc.waveform.bank import FilterBank

jax = pytest.importorskip("jax")
pytest.importorskip("diffgw")


def write_phenom_bank(path, rows):
    """Write waveform parameter rows to an HDF5 bank."""
    with h5py.File(path, "w") as bank:
        for name in rows[0]:
            values = [r[name] for r in rows]
            if isinstance(values[0], str):
                values = np.array(values, dtype=h5py.string_dtype())
            bank[name] = values
        bank.attrs["parameters"] = list(rows[0])
    return str(path)


def phenom_row(
    mass1=30.0,
    mass2=20.0,
    spin1z=0.1,
    spin2z=-0.2,
    f_lower=20.0,
    approximant="IMRPhenomD",
):
    return dict(
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        f_lower=f_lower,
        approximant=approximant,
    )


@pytest.mark.parametrize("approximant", ["IMRPhenomD", "IMRPhenomXAS"])
def test_phenom_batch_parity_and_match(tmp_path, approximant):
    rows = [
        phenom_row(30.0, 25.0, 0.1, -0.2, 20.0, approximant=approximant),
        phenom_row(40.0, 35.0, -0.3, 0.4, 20.0, approximant=approximant),
        phenom_row(15.0, 10.0, 0.0, 0.0, 25.0, approximant=approximant),
    ]
    filename = write_phenom_bank(tmp_path / f"{approximant}_bank.hdf", rows)
    kwargs = dict(
        filter_length=4097,
        delta_f=0.25,
        dtype=np.complex128,
        low_frequency_cutoff=20.0,
    )

    ref_bank = FilterBank(filename, enable_diffgw=False, **kwargs)
    ref_templates = [ref_bank[i] for i in range(len(rows))]

    bank = FilterBank(filename, enable_diffgw=True, **kwargs)
    assert bank.can_use_diffgw()

    with scheme.JAXScheme():
        batch, templates = bank.get_batch_tensor(list(range(len(rows))))
        assert batch.shape == (len(rows), 4097)

        for i, (tmpl, ref) in enumerate(zip(templates, ref_templates)):
            assert tmpl.waveform_provider in ("diffgw", "jaxwave")
            # Verify match against PyCBC LAL reference waveform
            tmpl_fs = ref.copy()
            tmpl_fs.data[:] = np.asarray(batch[i])
            m, _ = match(
                ref,
                tmpl_fs,
                low_frequency_cutoff=ref.f_lower,
                high_frequency_cutoff=1024.0,
            )
            assert m > 0.99999, f"Match {m} below threshold for {approximant} template {i}"


def test_phenom_mixed_fallback(tmp_path):
    rows = [
        phenom_row(30.0, 25.0, 0.1, -0.2, 20.0, approximant="IMRPhenomD"),
        phenom_row(35.0, 25.0, 0.0, 0.0, 20.0, approximant="IMRPhenomPv2"),  # unsupported
        phenom_row(40.0, 30.0, -0.1, 0.2, 20.0, approximant="IMRPhenomXAS"),
    ]
    filename = write_phenom_bank(tmp_path / "mixed_bank.hdf", rows)
    bank = FilterBank(
        filename,
        filter_length=4097,
        delta_f=0.25,
        enable_diffgw=True,
        dtype=np.complex128,
        low_frequency_cutoff=20.0,
    )

    with scheme.JAXScheme():
        batch, templates = bank.get_batch_tensor([0, 1, 2])
        assert templates[0].waveform_provider in ("diffgw", "jaxwave")
        assert templates[1].waveform_provider == "reference"
        assert templates[2].waveform_provider in ("diffgw", "jaxwave")
