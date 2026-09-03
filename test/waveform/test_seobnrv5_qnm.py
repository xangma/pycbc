import numpy as np
import pytest

pytest.importorskip("torch")
lal = pytest.importorskip("lal")
lalsimulation = pytest.importorskip("lalsimulation")
from pycbc.waveform._seobnrv5_qnm import (  # noqa: E402
    fundamental_qnm_omega,
    seobnrv5_final_mass_spin,
    seobnrv5_qnm_omega,
)


_CASES = (
    (10.0, 10.0, 0.0, 0.0),
    (14.0, 10.0, 0.7, -0.3),
    (6.0, 14.0, -0.4, 0.8),
    (40.0, 10.0, -0.8, 0.6),
    (1.0, 100.0, -0.998, 0.998),
)

_QNM_MODES = (
    (2, 2),
    (2, 1),
    (3, 3),
    (3, 2),
    (4, 4),
    (4, 3),
    (5, 5),
    (2, -1),
    (4, -3),
)


@pytest.mark.parametrize("parameters", _CASES)
def test_seobnrv5_remnant_matches_lal(parameters):
    mass1, mass2, spin1z, spin2z = parameters
    result = lalsimulation.SimIMREOBFinalMassSpin(
        mass1,
        mass2,
        [0.0, 0.0, spin1z],
        [0.0, 0.0, spin2z],
        lalsimulation.SEOBNRv5_ROM,
    )
    assert result[0] == 0
    expected = (result[1] * (mass1 + mass2), result[2])
    np.testing.assert_allclose(
        seobnrv5_final_mass_spin(*parameters),
        expected,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


@pytest.mark.parametrize("parameters", _CASES)
@pytest.mark.parametrize("ell, emm", _QNM_MODES)
def test_seobnrv5_qnm_matches_lal(parameters, ell, emm):
    mass1, mass2, spin1z, spin2z = parameters
    frequencies = lal.CreateCOMPLEX16Vector(1)
    result = lalsimulation.SimIMREOBGenerateQNMFreqV5(
        frequencies,
        mass1,
        mass2,
        [0.0, 0.0, spin1z],
        [0.0, 0.0, spin2z],
        ell,
        emm,
        1,
        lalsimulation.SEOBNRv5_ROM,
    )
    assert result == 0
    expected = frequencies.data[0].real * (mass1 + mass2) * lal.MTSUN_SI
    assert seobnrv5_qnm_omega(*parameters, ell, emm) == pytest.approx(
        expected, rel=2.0e-14, abs=2.0e-14
    )


def test_seobnrv5_remnant_and_qnm_are_mass_order_invariant():
    ordered = (14.0, 6.0, 0.8, -0.4)
    swapped = (6.0, 14.0, -0.4, 0.8)
    np.testing.assert_allclose(
        seobnrv5_final_mass_spin(*ordered),
        seobnrv5_final_mass_spin(*swapped),
        rtol=0.0,
        atol=2.0e-14,
    )
    assert seobnrv5_qnm_omega(*ordered) == pytest.approx(
        seobnrv5_qnm_omega(*swapped), rel=2.0e-14
    )


def test_seobnrv5_qnm_validates_inputs_and_clips_table_boundary():
    with pytest.raises(ValueError, match="unsupported"):
        fundamental_qnm_omega(0.7, 5, 4)
    with pytest.raises(ValueError, match="finite"):
        fundamental_qnm_omega(float("nan"))
    with pytest.raises(ValueError, match="positive"):
        seobnrv5_final_mass_spin(0.0, 10.0, 0.0, 0.0)

    assert fundamental_qnm_omega(1.2) == fundamental_qnm_omega(0.9996)
    assert fundamental_qnm_omega(-1.2) == fundamental_qnm_omega(-0.9996)
