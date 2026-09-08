# Copyright (C) 2026 The PyCBC Collaboration
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.

"""Real power-chisq regression coverage for reused live-batch workspaces."""

from types import SimpleNamespace

import numpy as np
import pytest

import pycbc
from pycbc import scheme
from pycbc.filter import matchedfilter
from pycbc.types import FrequencySeries
from pycbc.vetoes.sgchisq import SingleDetSGChisq


SIZE = 64
DELTA_F = 0.25
TEMPLATE_COUNT = 4
BINS = 4
THRESHOLD = 0.1


@pytest.fixture
def inputs():
    rng = np.random.default_rng(7219)
    frequency_size = SIZE // 2 + 1
    templates = (
        rng.normal(size=(TEMPLATE_COUNT, frequency_size))
        + 1j * rng.normal(size=(TEMPLATE_COUNT, frequency_size))
    ).astype(np.complex64)
    templates *= np.array([0.6, 1.1, 1.7, 2.3], np.float32)[:, None]
    # Match the physical filtering band: no DC or Nyquist contribution.
    templates[:, [0, -1]] = 0
    data = (
        rng.normal(size=(2, frequency_size))
        + 1j * rng.normal(size=(2, frequency_size))
    ).astype(np.complex64)
    psds = np.array([
        np.linspace(1.0, 2.5, frequency_size),
        np.linspace(2.0, 0.7, frequency_size),
    ], dtype=np.float32)
    return templates, data, psds


def _build_batch(template_values, batch_size):
    templates = []
    for index, values in enumerate(template_values):
        template = FrequencySeries(values, delta_f=DELTA_F)
        template.id = 10 + index
        template.params = np.array(
            [(20.0 + index,)], dtype=[("mass1", np.float32)]
        )[0]
        template.f_lower = DELTA_F
        template.sigmasq = lambda psd, h=template: matchedfilter.sigmasq(
            h, psd, low_frequency_cutoff=h.f_lower
        )
        templates.append(template)
    batch = matchedfilter.LiveBatchMatchedFilter(
        templates, snr_threshold=THRESHOLD, chisq_bins=str(BINS),
        sg_chisq=SingleDetSGChisq(None),
        maxelements=batch_size * SIZE,
        enable_cuda_graphs=False, enable_async_streams=False,
    )
    assert batch.power_chisq.do
    assert not batch.sg_chisq.do
    assert [len(group) for group in batch.tgroups] == (
        [batch_size] * (TEMPLATE_COUNT // batch_size)
    )
    assert len(set(batch.mids)) == len(batch.cout_mem) == 1
    workspace = batch.cout_mem[batch.mids[0]]
    assert workspace.shape == (batch_size * SIZE,)
    for group in batch.tgroups:
        for row, template in enumerate(group):
            assert template.cout.shape == (SIZE,)
            assert template.cout.ptr == (
                workspace.ptr + row * SIZE * np.dtype(np.complex64).itemsize
            )
    return batch, templates


def _check_direct_cpu(result, templates, stilde, reader, batch):
    """Check the single-group oracle against explicit complex128 DFT sums."""
    valid_end = SIZE - reader.trim_padding
    valid_start = valid_end - reader.blocksize * reader.sample_rate
    for row, template in enumerate(templates):
        corr = np.zeros(SIZE, np.complex128)
        corr[:len(stilde)] = (
            template.numpy().astype(np.complex128).conj() * stilde.numpy()
        )
        snr = np.fft.ifft(corr) * SIZE
        peak = valid_start + np.abs(snr[valid_start:valid_end]).argmax()
        norm = 4 * DELTA_F / np.sqrt(template.sigmasq(stilde.psd))
        bins = batch.power_chisq.cached_chisq_bins(template, stilde.psd)
        shifted = corr * np.exp(2j * np.pi * peak * np.arange(SIZE) / SIZE)
        bin_sums = np.array([
            shifted[start:end].sum()
            for start, end in zip(bins[:-1], bins[1:])
        ])
        reduced_chisq = (
            BINS * np.sum(np.abs(bin_sums) ** 2) - abs(snr[peak]) ** 2
        ) * norm**2 / (2 * BINS - 2)
        np.testing.assert_allclose(
            result["chisq"][row], reduced_chisq, rtol=3e-6, atol=2e-6
        )
        np.testing.assert_allclose(
            result["snr"][row], abs(snr[peak]) * norm, rtol=3e-6
        )
        assert result["end_time"][row] == (
            reader.start_time + (peak - valid_start) / reader.sample_rate
        )


def _run_blocks(inputs, batch_size, monkeypatch, check_direct=False):
    template_values, data_values, psd_values = inputs
    batch, templates = _build_batch(template_values, batch_size)
    arrays = list(batch.cout_mem.values()) + [t.cout for t in templates]
    storage = [(array.shape, array.ptr) for array in arrays]
    calls = []
    real_values = batch.power_chisq.values

    def checked_values(corr, snrv, norm, psd, indices, template):
        # Observe the real veto call: a shortened row changes its phase
        # denominator even if the owning allocation and pointer survive.
        assert corr is template.cout
        assert corr.shape == (SIZE,)
        assert [(array.shape, array.ptr) for array in arrays] == storage
        result = real_values(corr, snrv, norm, psd, indices, template)
        assert [(array.shape, array.ptr) for array in arrays] == storage
        calls.append(template.id)
        return result

    monkeypatch.setattr(batch.power_chisq, "values", checked_values)
    blocks = []
    for data, psd in zip(data_values, psd_values):
        stilde = FrequencySeries(data / psd, delta_f=DELTA_F)
        stilde.psd = FrequencySeries(psd, delta_f=DELTA_F)
        blocks.append(stilde)

    results = []
    # Reuse both the batch and the original data/PSD identities after a
    # different block, exercising correlation and power-bin cache reuse.
    for block_index, stilde in enumerate([blocks[0], blocks[1], blocks[0]]):
        reader = SimpleNamespace(
            overwhitened_data=lambda _df, block=stilde: block,
            trim_padding=4, blocksize=3, sample_rate=16,
            start_time=1000.0 + 3 * block_index,
        )
        calls.clear()
        result = batch.process_data(reader)
        assert [(array.shape, array.ptr) for array in arrays] == storage
        expected_ids = [template.id for template in templates]
        np.testing.assert_array_equal(result["template_id"], expected_ids)
        assert calls == expected_ids
        # Every row survives: B2 must evaluate both siblings in both groups.
        assert np.all(result["snr"] > THRESHOLD)
        assert np.all(np.isfinite(result["chisq"]))
        assert np.all(result["chisq"] >= 0)
        np.testing.assert_array_equal(result["chisq_dof"], 2 * BINS - 2)
        np.testing.assert_array_equal(result["sg_chisq"], 0)
        for template in templates:
            np.testing.assert_array_equal(
                template.cout.numpy()[len(stilde):], 0
            )
        if check_direct:
            _check_direct_cpu(result, templates, stilde, reader, batch)
        results.append(result)
    for key in ("template_id", "snr", "chisq", "chisq_dof"):
        np.testing.assert_array_equal(results[0][key], results[2][key])
    np.testing.assert_array_equal(
        results[2]["end_time"], results[0]["end_time"] + 6
    )
    return results


@pytest.fixture
def cpu_oracle(inputs, monkeypatch):
    monkeypatch.delenv("PYCBC_BATCH_MAXELEMENTS", raising=False)
    # All four rows retain their own correlation until veto calculation.
    with scheme.DefaultScheme():
        return _run_blocks(
            inputs, TEMPLATE_COUNT, monkeypatch, check_direct=True
        )


@pytest.mark.parametrize("backend", ["cpu", "torch-cpu", "torch-mps"])
@pytest.mark.parametrize(
    "batch_size", [1, 2, TEMPLATE_COUNT], ids=["B1", "B2", "single-group"]
)
def test_process_data_power_chisq_workspace_reuse(
        backend, batch_size, inputs, cpu_oracle, monkeypatch):
    if backend == "cpu":
        context = scheme.DefaultScheme()
    else:
        torch = pytest.importorskip("torch")
        if not pycbc.HAVE_TORCH:
            pytest.skip("PyCBC built without Torch support")
        device = backend.removeprefix("torch-")
        if device == "mps" and not torch.backends.mps.is_available():
            pytest.skip("MPS unavailable")
        context = scheme.TorchScheme(device)
    with context:
        results = _run_blocks(inputs, batch_size, monkeypatch)
    for result, expected in zip(results, cpu_oracle):
        assert result.keys() == expected.keys()
        for key in (
                "template_id", "end_time", "chisq_dof", "mass1", "sg_chisq"):
            np.testing.assert_array_equal(result[key], expected[key])
        for key in ("snr", "coa_phase", "sigmasq", "chisq"):
            np.testing.assert_allclose(
                result[key], expected[key], rtol=2e-5, atol=2e-6,
                err_msg=f"{backend} B{batch_size}: {key}",
            )
