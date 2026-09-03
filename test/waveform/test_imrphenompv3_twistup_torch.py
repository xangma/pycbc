import numpy as np
import pytest

torch = pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")

from pycbc import pnutils, scheme as _scheme  # noqa: E402
from pycbc.waveform import get_fd_waveform  # noqa: E402
from pycbc.waveform.imrphenompv3_torch import (  # noqa: E402
    _imrphenompv3hm_model,
    _imrphenompv3hm_polarizations,
)


FREQUENCIES = (19.0, 27.5, 43.0, 71.0, 110.0, 173.0, 281.0)

PRECESSING_HM_CASES = (
    dict(
        mass1=40.0,
        mass2=20.0,
        spin1x=0.3,
        spin1y=-0.2,
        spin1z=0.45,
        spin2x=-0.1,
        spin2y=0.25,
        spin2z=-0.3,
        distance=400.0,
        inclination=1.1,
        coa_phase=0.4,
        f_ref=30.0,
    ),
    dict(
        mass1=18.0,
        mass2=47.0,
        spin1x=-0.12,
        spin1y=0.31,
        spin1z=-0.22,
        spin2x=0.4,
        spin2y=0.05,
        spin2z=0.37,
        distance=720.0,
        inclination=2.0,
        coa_phase=-0.8,
        f_ref=24.0,
        long_asc_nodes=0.63,
        mode_array=[(2, 2), (2, 1), (4, 3)],
    ),
    dict(
        mass1=55.0,
        mass2=8.0,
        spin1x=0.05,
        spin1y=0.45,
        spin1z=0.1,
        spin2x=-0.2,
        spin2y=0.05,
        spin2z=0.6,
        distance=310.0,
        inclination=2.2,
        coa_phase=1.2,
        f_ref=40.0,
        mode_array=[(2, 2), (3, 3), (3, 2), (4, 4)],
    ),
)


@pytest.fixture
def preserve_scheme():
    old_scheme = _scheme.mgr.state
    old_single = _scheme.Scheme._single
    try:
        yield
    finally:
        _scheme.mgr.state = old_scheme
        _scheme.Scheme._single = old_single


def _activate_torch_cpu():
    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.TorchScheme("cpu")


def _lal_frequencies(values):
    frequencies = lal.CreateREAL8Vector(len(values))
    frequencies.data[:] = values
    return frequencies


def _lal_mode_dictionary(modes):
    dictionary = lal.CreateDict()
    if modes is None:
        return dictionary
    mode_array = lalsimulation.SimInspiralCreateModeArray()
    for ell, emm in modes:
        lalsimulation.SimInspiralModeArrayActivateMode(mode_array, ell, emm)
    lalsimulation.SimInspiralWaveformParamsInsertModeArray(dictionary, mode_array)
    return dictionary


def _lal_pv3hm(params):
    return lalsimulation.SimIMRPhenomPv3HMGetHplusHcross(
        _lal_frequencies(FREQUENCIES),
        params["mass1"] * lal.MSUN_SI,
        params["mass2"] * lal.MSUN_SI,
        params.get("spin1x", 0.0),
        params.get("spin1y", 0.0),
        params.get("spin1z", 0.0),
        params.get("spin2x", 0.0),
        params.get("spin2y", 0.0),
        params.get("spin2z", 0.0),
        pnutils.megaparsecs_to_meters(params["distance"]),
        params["inclination"],
        params["coa_phase"],
        0.0,
        params["f_ref"],
        _lal_mode_dictionary(params.get("mode_array")),
    )


def _assert_lal_parity(actual, reference):
    for result, expected in zip(actual, reference):
        result_array = result.detach().cpu().numpy()
        expected_array = np.asarray(expected.data.data)
        scale = np.max(np.abs(expected_array))
        assert result.device.type == "cpu"
        assert result.dtype == torch.complex128
        np.testing.assert_allclose(
            result_array,
            expected_array,
            rtol=8.0e-11,
            atol=scale * 2.0e-12,
        )


@pytest.mark.parametrize("params", PRECESSING_HM_CASES)
def test_private_pv3hm_twistup_matches_lalsimulation(params, preserve_scheme):
    reference = _lal_pv3hm(params)
    _activate_torch_cpu()
    model = _imrphenompv3hm_model(
        {"approximant": "IMRPhenomPv3HM", **params},
        params["f_ref"],
        sequence=True,
    )
    actual = _imrphenompv3hm_polarizations(model, FREQUENCIES)

    assert model.precessing
    _assert_lal_parity(actual, reference)


def test_private_pv3_dominant_mode_matches_lalsimulation(preserve_scheme):
    params = PRECESSING_HM_CASES[0]
    reference = lalsimulation.SimIMRPhenomPv3(
        _lal_frequencies(FREQUENCIES),
        params["mass1"] * lal.MSUN_SI,
        params["mass2"] * lal.MSUN_SI,
        params["spin1x"],
        params["spin1y"],
        params["spin1z"],
        params["spin2x"],
        params["spin2y"],
        params["spin2z"],
        pnutils.megaparsecs_to_meters(params["distance"]),
        params["inclination"],
        params["coa_phase"],
        0.0,
        params["f_ref"],
        lal.CreateDict(),
    )
    _activate_torch_cpu()
    model = _imrphenompv3hm_model(
        {"approximant": "IMRPhenomPv3", **params},
        params["f_ref"],
        sequence=True,
    )
    actual = _imrphenompv3hm_polarizations(model, FREQUENCIES)

    assert model.carrier.inputs.active_modes == ((2, 2),)
    _assert_lal_parity(actual, reference)


def test_private_pv3hm_outer_rotation_matches_regular_lal(preserve_scheme):
    params = {
        "approximant": "IMRPhenomPv3HM",
        **PRECESSING_HM_CASES[0],
        "delta_f": 1.0,
        "f_lower": 19.0,
        "f_final": 300.0,
        "long_asc_nodes": 0.63,
    }
    _scheme.Scheme._single = None
    _scheme.mgr.state = _scheme.CPUScheme()
    reference = get_fd_waveform(**params)

    _activate_torch_cpu()
    model = _imrphenompv3hm_model(
        params,
        params["f_ref"],
        sequence=False,
    )
    frequencies = torch.arange(19.0, 300.0, dtype=torch.float64)
    actual = _imrphenompv3hm_polarizations(model, frequencies)

    for result, expected in zip(actual, reference):
        expected_array = expected.numpy()[19:300]
        scale = np.max(np.abs(expected_array))
        np.testing.assert_allclose(
            result.detach().cpu().numpy(),
            expected_array,
            rtol=8.0e-11,
            atol=scale * 2.0e-12,
        )


def test_private_pv3hm_nonprecessing_frame_flip_matches_lalsimulation(
    preserve_scheme,
):
    params = dict(
        approximant="IMRPhenomPv3HM",
        mass1=80.0,
        mass2=8.0,
        spin1z=-0.99,
        spin2z=0.0,
        distance=540.0,
        inclination=0.7,
        coa_phase=0.4,
        f_ref=20.0,
        mode_array=[(2, 2), (2, 1), (3, 3), (4, 3)],
    )
    reference = _lal_pv3hm(params)
    _activate_torch_cpu()
    model = _imrphenompv3hm_model(params, params["f_ref"], sequence=True)
    actual = _imrphenompv3hm_polarizations(model, FREQUENCIES)

    assert not model.precessing
    assert model.theta_jn == pytest.approx(np.pi - params["inclination"])
    _assert_lal_parity(actual, reference)
