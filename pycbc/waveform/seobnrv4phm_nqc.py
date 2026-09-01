"""NQC calibration fits for SEOBNRv4/PHM (aligned-spin base).

Ported from lalsimulation/lib/LALSimIMREOBNQCCorrection.c:
- The simple calibrated fits near lines ~410 for a1,a2,a3 (spin-aligned).
- v4HM peak amplitude/slope/curvature fits (lines 730-1185).
- v4HM peak orbital frequency / slope fits (lines 1336-1680).
- v4HM peak time-offset fit (lines 1699-1754).
- Linear NQC solve (amplitude/phase) mirroring XLALSimIMREOBCalculateNQCCoefficients
  and XLALSimIMRSpinEOBCalculateNQCCoefficientsV4 (lines ~547-907, 2359-2840).
"""

from __future__ import annotations

import math

import torch


def _solve_nqc_system(
    matrix: torch.Tensor, rhs: torch.Tensor
) -> torch.Tensor:
    """Solve a small NQC system, retaining least-squares fallback behavior."""

    try:
        return torch.linalg.solve(matrix, rhs)
    except RuntimeError:
        try:
            return torch.linalg.lstsq(matrix, rhs).solution
        except (NotImplementedError, RuntimeError):
            if matrix.device.type == "cpu":
                raise
            # MPS does not currently implement linalg.lstsq. This path is
            # reached only for a singular NQC system that will normally be
            # rejected by the coefficient sanity check.
            solution = torch.linalg.lstsq(
                matrix.detach().cpu().double(),
                rhs.detach().cpu().double(),
            ).solution
            return solution.to(device=matrix.device, dtype=matrix.dtype)


def calibrated_a123(eta: float):
    """Return (a1, a2, a3) calibrated fits as functions of eta (SEOBNRv4).
    COMPLETE port of LALSimIMREOBNQCCorrection.c:410-418."""
    a1 = -4.55919 + 18.761 * eta - 24.226 * eta * eta
    a2 = 37.683 - 201.468 * eta + 324.591 * eta * eta
    a3 = -39.6024 + 228.899 * eta - 387.222 * eta * eta
    return a1, a2, a3


def _combine_tpleqm(eta: float, A1: float, f_eq: float, f_tpl: float) -> float:
    """Blend equal-mass and TPL fits (LALSimIMREOBNQCCorrection.c:710-719)."""
    eta2 = eta * eta
    A0 = -0.00099601593625498 * A1 - 0.00001600025600409607 * f_eq + 1.000016000256004 * f_tpl
    A2 = -3.984063745019967 * A1 + 16.00025600409607 * f_eq - 16.0002560041612 * f_tpl
    return A0 + A1 * eta + A2 * eta2


def peak_delta_t_v4(mode_l: int, mode_m: int, m1: float, m2: float, chi1: float, chi2: float) -> float:
    """NR-fit offset between orbital-frequency peak and mode peak.

    COMPLETE port of ``XLALSimIMREOBGetNRSpinPeakDeltaTv4``
    (LALSimIMREOBNQCCorrection.c:1699-1754).  LAL defines the sampled NQC
    time as ``tPeakOmega - DeltaT_lm``; for the 55 mode this fit includes the
    extra 10M shift used by the PHM attachment code.
    """

    eta = m1 * m2 / ((m1 + m2) * (m1 + m2))
    chi = 0.5 * (chi1 + chi2) + 0.5 * (chi1 - chi2) * (m1 - m2) / (m1 + m2) / (1.0 - 2.0 * eta)
    eta2 = eta * eta
    eta3 = eta2 * eta
    chi2p = chi * chi
    chi3p = chi2p * chi

    res = (
        2.50499
        + 13.0064 * chi
        + 11.5435 * chi2p
        + 45.8838 * eta
        - 40.3183 * eta * chi
        - 19.0538 * eta * chi3p
        + 13.0879 * eta2
        + 0.192775 * eta2 * chi3p
        - 716.044 * eta3
    )
    if mode_l == 5 and mode_m == 5:
        res += 10.0
    return res


def peak_amp_v4(mode_l: int, mode_m: int, eta: float, chiS: float, chiA: float):
    """NR-fit peak amplitude (A_peak) for SEOBNRv4 HM modes.
    COMPLETE port of LALSimIMREOBNQCCorrection.c:730-834."""
    dM = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
    eta2 = eta * eta
    chi = chiS + chiA * dM / (1.0 - 2.0 * eta)
    chi21 = chiS * dM / (1.0 - 1.3 * eta) + chiA
    chi33 = chiS * dM + chiA
    chi44 = chiS * (1.0 - 5.0 * eta) + chiA * dM
    if mode_l == 2 and mode_m == 2:
        fTPL = 1.4528573105413543 + 0.16613449160880395 * chi + 0.027355646661735258 * chi * chi - 0.020072844926136438 * chi * chi * chi
        fEQ = 1.577457498227 - 0.0076949474494639085 * chi + 0.02188705616693344 * chi * chi + 0.023268366492696667 * chi * chi * chi
        e0 = -0.03442402416125921
        e1 = -1.218066264419839
        e2 = -0.5683726304811634
        e3 = 0.4011143761465342
        A1 = e0 + e1 * chi + e2 * chi * chi + e3 * chi * chi * chi
        return eta * _combine_tpleqm(eta, A1, fEQ, fTPL)
    if mode_l == 2 and mode_m == 1:
        return -(
            (0.29256703361640224 - 0.19710255145276584 * eta) * eta * chi21
            + dM
            * eta
            * (
                -0.42817941710649793
                + 0.11378918021042442 * eta
                - 0.7736772957051212 * eta2
                + chi21 * chi21 * (0.047004057952214004 - eta * 0.09326128322462478)
            )
            + dM * eta * chi21 * (-0.010195081244587765 + 0.016876911550777807 * chi21 * chi21)
        )
    if mode_l == 3 and mode_m == 3:
        return (
            (0.10109183988848384 * eta - 0.4704095462146807 * eta2 + 1.0735457779890183 * eta2 * eta) * chi33
            + dM
            * (
                0.5636580081367962 * eta
                - 0.054609013952480856 * eta2
                + 2.3093699480319234 * eta2 * eta
                + chi33 * chi33 * (0.029812986680919126 * eta - 0.09688097244145283 * eta2)
            )
        )
    if mode_l == 4 and mode_m == 4:
        return (
            eta
            * (0.2646580063832686 + 0.067584186955327 * chi44 + 0.02925102905737779 * chi44 * chi44)
            + eta2 * (-0.5658246076387973 - 0.8667455348964268 * chi44 + 0.005234192027729502 * chi44 * chi44)
            + eta2 * eta * (-2.5008294352355405 + 6.880772754797872 * chi44 - 1.0234651570264885 * chi44 * chi44)
            + eta2 * eta2 * (7.6974501716202735 - 16.551524307203252 * chi44)
        )
    if mode_l == 5 and mode_m == 5:
        return (
            dM * eta * (0.128621 - 0.474201 * eta + 1.0833 * eta2)
            + eta * (0.0322784 * chi33 - 0.134511 * chi33 * eta + 0.0990202 * chi33 * eta2)
        )
    raise ValueError(f"peak_amp_v4: unsupported mode ({mode_l},{mode_m})")


def peak_adot_v4(mode_l: int, mode_m: int, eta: float, chiS: float, chiA: float):
    """NR-fit peak A_dot (time derivative) for SEOBNRv4 HM modes.
    COMPLETE port of LALSimIMREOBNQCCorrection.c:864-1006."""
    dM = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
    dM2 = dM * dM
    eta2 = eta * eta
    chi21 = chiS * dM / (1.0 - 2.0 * eta) + chiA
    chi33 = chiS * dM + chiA
    chi44 = chiS * (1.0 - 7.0 * eta) + chiA * dM
    if mode_l == 2 and mode_m == 2:
        return 0.0
    if mode_l == 2 and mode_m == 1:
        return (
            dM * eta * (0.007147528020812309 - eta * 0.035644027582499495)
            + dM * eta * chi21 * (-0.0087785131749995 + eta * 0.03054672006241107)
            + eta * 0.00801714459112299 * abs(-dM * (0.7875612917853588 + eta * 1.161274164728927 + eta2 * 11.306060006923605) + chi21)
        )
    if mode_l == 3 and mode_m == 3:
        return (
            dM * eta * (-0.00309943555972098 + eta * 0.010076527264663805) * chi33 * chi33
            + eta * 0.0016309606446766923 * (dM2 * (8.811660714437027 + 104.47752236009688 * eta) + dM * chi33 * (-5.352043503655119 + eta * 49.68621807460999) + chi33 * chi33) ** 0.5
        )
    if mode_l == 4 and mode_m == 4:
        return eta * (
            (0.004347588211099233 - 0.0014612210699052148 * chi44 - 0.002428047910361957 * chi44 * chi44)
            + eta * (0.023320670701084355 - 0.02240684127113227 * chi44 + 0.011427087840231389 * chi44 * chi44)
            + eta2 * (-0.46054477257132803 + 0.433526632115367 * chi44)
            + eta2 * eta * (1.2796262150829425 - 1.2400051122897835 * chi44)
        )
    if mode_l == 5 and mode_m == 5:
        return eta * (
            dM * (-0.008389798844109389 + 0.04678354680410954 * eta)
            + dM * chi33 * (-0.0013605616383929452 + 0.004302712487297126 * eta)
            + dM * chi33 * chi33 * (-0.0011412109287400596 + 0.0018590391891716925 * eta)
            + 0.0002944221308683548 * abs(dM * (37.11125499129578 - 157.79906814398277 * eta) + chi33)
        )
    raise ValueError(f"peak_adot_v4: unsupported mode ({mode_l},{mode_m})")


def peak_addot_v4(mode_l: int, mode_m: int, eta: float, chiS: float, chiA: float):
    """NR-fit peak A_ddot (second derivative) for SEOBNRv4 HM modes.
    COMPLETE port of LALSimIMREOBNQCCorrection.c:1010-1185."""
    dM = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
    eta2 = eta * eta
    chi = chiS + chiA * dM / (1.0 - 2.0 * eta)
    chi21 = chiS * dM / (1.0 - 2.0 * eta) + chiA
    chi33 = chiS * dM + chiA
    chi_minus_1 = chi - 1.0
    if mode_l == 2 and mode_m == 2:
        fTPL = 0.002395610769995033 * chi_minus_1 - 0.00019273850675004356 * chi_minus_1 * chi_minus_1 - 0.00029666193167435337 * chi_minus_1 * chi_minus_1 * chi_minus_1
        fEQ = -0.004126509071377509 + 0.002223999138735809 * chi
        e0 = -0.005776537350356959
        e1 = 0.001030857482885267
        A1 = e0 + e1 * chi
        return eta * _combine_tpleqm(eta, A1, fEQ, fTPL)
    if mode_l == 2 and mode_m == 1:
        return (
            eta * dM * 0.00037132201959950333
            - abs(
                dM * eta * (-0.0003650874948532221 - eta * 0.003054168419880019)
                + dM * eta * chi21 * chi21 * (-0.0006306232037821514 - eta * 0.000868047918883389 + eta2 * 0.022306229435339213)
                + eta * chi21 * chi21 * chi21 * 0.0003402427901204342
                + dM * eta * chi21 * 0.00028398490492743
            )
        )
    if mode_l == 3 and mode_m == 3:
        return (
            dM
            * eta
            * (0.0009605689249339088 - 0.00019080678283595965 * eta)
            * chi33
            - 0.00015623760412359145
            * eta
            * abs(
                dM
                * (
                    4.676662024170895
                    + 79.20189790272218 * eta
                    - 1097.405480250759 * eta2
                    + 6512.959044311574 * eta * eta2
                    - 13263.36920919937 * eta2 * eta2
                )
                + chi33
            )
        )
    if mode_l == 4 and mode_m == 4:
        return eta * (
            (-0.000301722928925693 + 0.0003215952388023551 * chi)
            + eta * (0.006283048344165004 + 0.0011598784110553046 * chi)
            + eta2 * (-0.08143521096050622 - 0.013819464720298994 * chi)
            + eta2 * eta * (0.22684871200570564 + 0.03275749240408555 * chi)
        )
    if mode_l == 5 and mode_m == 5:
        return eta * (
            dM * (0.00012727220842255978 + 0.0003211670856771251 * eta)
            + dM * chi33 * (-0.00006621677859895541 + 0.000328855327605536 * eta)
            + chi33 * chi33 * (-0.00005824622885648688 + 0.00013944293760663706 * eta)
        )
    raise ValueError(f"peak_addot_v4: unsupported mode ({mode_l},{mode_m})")


def peak_omega_v4(mode_l: int, mode_m: int, eta: float, chiS: float, chiA: float):
    """NR-fit peak GW frequency (Omega) for SEOBNRv4 HM modes.
    COMPLETE port of LALSimIMREOBNQCCorrection.c:1336-1504."""
    dM = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
    eta2 = eta * eta
    chi = chiS + chiA * dM / (1.0 - 2.0 * eta)
    if mode_l == 2 and mode_m == 2:
        c0 = 0.5626787200433265
        c1 = -0.08706198756945482
        c2 = 25.81979479453255
        c3 = 25.85037751197443
        d2 = 7.629921628648589
        d3 = 10.26207326082448
        A4 = d2 + 4.0 * (d2 - c2) * (eta - 0.25)
        A3 = d3 + 4.0 * (d3 - c3) * (eta - 0.25)
        c4 = 0.00174345193125868
        arg = max(A3 - A4 * chi, 1e-12)
        return c0 + (c1 + c4 * chi) * math.log(arg)
    if mode_l == 2 and mode_m == 1:
        return (
            0.1743194440996283
            + eta * 0.1938944514123048
            + 0.1670063050527942 * eta2
            + 0.053508705425291826 * chi
            - eta * chi * 0.18460213023455802
            + eta2 * chi * 0.2187305149636044
            + chi * chi * 0.030228846150378793
            - eta * chi * chi * 0.11222178038468673
        )
    if mode_l == 3 and mode_m == 3:
        chi2 = chi * chi
        chi3 = chi2 * chi
        return (
            0.3973947703114506
            + 0.16419332207671075 * chi
            + 0.1635531186118689 * chi2
            + 0.06140164491786984 * chi3
            + eta * (0.6995063984915486 - 0.3626744855912085 * chi - 0.9775469868881651 * chi2)
            + eta2 * (-0.3455328417046369 + 0.31952307610699876 * chi + 1.9334166149686984 * chi2)
        )
    if mode_l == 4 and mode_m == 4:
        chi2 = chi * chi
        chi3 = chi2 * chi
        return (
            0.5389359134370971
            + 0.16635177426821202 * chi
            + 0.2075386047689103 * chi2
            + 0.15268115749910835 * chi3
            + eta * (0.7617423831337586 + 0.009587856087825369 * chi - 1.302303785053009 * chi2 - 0.5562751887042064 * chi3)
            + eta2 * (0.9675153069365782 - 0.22059322127958586 * chi + 2.678097398558074 * chi2)
            - 4.895381222514275 * eta2 * eta
        )
    if mode_l == 5 and mode_m == 5:
        chi2 = chi * chi
        chi3 = chi2 * chi
        return (
            0.6437545281817488
            + 0.22315530037902315 * chi
            + 0.2956893357624277 * chi2
            + 0.17327819169083758 * chi3
            + eta * (-0.47017798518175785 - 0.3929010618358481 * chi - 2.2653368626130654 * chi2 - 0.5512998466154311 * chi3)
            + eta2 * (2.311483807604238 + 0.8829339243493562 * chi + 5.817595866020152 * chi2)
        )
    raise ValueError(f"peak_omega_v4: unsupported mode ({mode_l},{mode_m})")


def peak_omegadot_v4(mode_l: int, mode_m: int, eta: float, chiS: float, chiA: float):
    """NR-fit peak GW frequency slope (Omega_dot) for SEOBNRv4 HM modes.
    COMPLETE port of LALSimIMREOBNQCCorrection.c:1566-1678."""
    dM = math.sqrt(max(1.0 - 4.0 * eta, 0.0))
    eta2 = eta * eta
    chi = chiS + chiA * dM / (1.0 - 2.0 * eta)
    chi2 = chi * chi
    chi3 = chi2 * chi
    if mode_l == 2 and mode_m == 2:
        fTPL = -0.011209791668428353 + (0.0040867958978563915 + 0.0006333925136134493 * chi) * math.log(
            68.47466578100956 - 58.301487557007206 * chi
        )
        fEQ = 0.01128156666995859 + 0.0002869276768158971 * chi
        e0 = 0.01574321112717377
        e1 = 0.02244178140869133
        A1 = e0 + e1 * chi
        return _combine_tpleqm(eta, A1, fEQ, fTPL)
    if mode_l == 2 and mode_m == 1:
        eta3 = eta2 * eta
        return (
            0.0070987396362959514
            + eta * 0.024816844694685373
            - eta2 * 0.050428973182277494
            + eta3 * 0.03442040062259341
            - chi * 0.0017751850002442097
            + eta * chi * 0.004244058872768811
            - eta2 * chi * 0.031996494883796855
            - chi2 * 0.0035627260615894584
            + eta * chi2 * 0.01471807973618255
            - chi3 * 0.0019020967877681962
        )
    if mode_l == 3 and mode_m == 3:
        return (
            0.010337157192240338
            - 0.0053067782526697764 * chi2
            - 0.005087932726777773 * chi3
            + eta * (0.027735564986787684 + 0.018864151181629343 * chi + 0.021754491131531044 * chi2 + 0.01785477515931398 * chi3)
            + eta2 * (0.018084233854540898 - 0.08204268775495138 * chi)
        )
    if mode_l == 4 and mode_m == 4:
        return (
            0.013997911323773867
            - 0.0051178205260273574 * chi
            - 0.0073874256262988 * chi2
            + eta * (0.0528489379269367 + 0.01632304766334543 * chi + 0.02539072293029433 * chi2)
            + eta2 * (-0.06529992724396189 + 0.05782894076431308 * chi)
        )
    if mode_l == 5 and mode_m == 5:
        eta3 = eta2 * eta
        return (
            0.01763430670755021
            - 0.00024925743340389135 * chi
            - 0.009240404217656968 * chi2
            - 0.007907831334704586 * chi3
            + eta * (-0.1366002854361568 + 0.0561378177186783 * chi + 0.16406275673019852 * chi2 + 0.07736232247880881 * chi3)
            + eta2 * (0.9875890632901151 - 0.31392112794887855 * chi - 0.5926145463423832 * chi2)
            - 1.6943356548192614 * eta3
        )
    raise ValueError(f"peak_omegadot_v4: unsupported mode ({mode_l},{mode_m})")


def solve_nqc_coeffs(eta: float, chiS: float, chiA: float, *, mode_l=2, mode_m=2, eob_peak=None):
    """Compute NQC coefficients using a Torch linear solve.

    LALSimIMREOBNQCCorrection.c:547-907, 2359-2840 (XLALSimIMREOBCalculateNQCCoefficients
    and XLALSimIMRSpinEOBCalculateNQCCoefficientsV4). We reuse the calibrated
    a1–a3 fits when no peak data is available; with peak data we solve the same
    3×3/2×2 linear systems for (a1,a2,a3) and (b1,b2) using local derivatives.
    """

    a1, a2, a3 = calibrated_a123(eta)
    peak_amp_nr = peak_amp_v4(mode_l, mode_m, eta, chiS, chiA)
    peak_adot_nr = peak_adot_v4(mode_l, mode_m, eta, chiS, chiA)
    peak_addot_nr = peak_addot_v4(mode_l, mode_m, eta, chiS, chiA)
    peak_omega_nr = peak_omega_v4(mode_l, mode_m, eta, chiS, chiA)
    peak_omegadot_nr = peak_omegadot_v4(mode_l, mode_m, eta, chiS, chiA)

    # Defaults (LAL zeroes a4,a5,b3,b4).
    a4 = a5 = 0.0
    b1 = b2 = b3 = b4 = 0.0

    if eob_peak is not None:
        amp_e = max(eob_peak.get("amp", 0.0), 1e-16)
        adot_e = eob_peak.get("adot", 0.0)
        addot_e = eob_peak.get("addot", 0.0)
        omega_raw = eob_peak.get("omega", 0.0)
        omegadot_raw = eob_peak.get("omegadot", 0.0)
        omega_e = max(abs(omega_raw), 1e-16)
        if omega_raw * omegadot_raw > 0.0:
            omegadot_e = abs(omegadot_raw)
        else:
            omegadot_e = -abs(omegadot_raw)
        omegaddot_e = eob_peak.get("omegaddot", 0.0)
        pr = eob_peak.get("pr", 0.0)
        prdot = eob_peak.get("prdot", 0.0)
        prddot = eob_peak.get("prddot", 0.0)
        pr2 = max(eob_peak.get("pr2", pr * pr), 0.0)
        pr2dot = eob_peak.get("pr2dot", 2.0 * pr * prdot)
        pr2ddot = eob_peak.get("pr2ddot", 2.0 * (prdot * prdot + pr * prddot))
        r_peak = max(eob_peak.get("r", 0.0), 1e-12)
        rdot = eob_peak.get("rdot", 0.0)
        rddot = eob_peak.get("rddot", 0.0)

        rOmega = r_peak * omega_e
        rOmega_dot = rdot * omega_e + r_peak * omegadot_e
        rOmega_ddot = rddot * omega_e + 2.0 * rdot * omegadot_e + r_peak * omegaddot_e

        rOmega_sq = max(rOmega * rOmega, 1e-16)
        rOmega_sq_dot = 2.0 * rOmega * rOmega_dot
        rOmega_sq_ddot = 2.0 * (rOmega_dot * rOmega_dot + rOmega * rOmega_ddot)

        n1 = pr2 / rOmega_sq
        n1dot = (pr2dot * rOmega_sq - pr2 * rOmega_sq_dot) / (rOmega_sq * rOmega_sq)
        n1ddot = (
            pr2ddot / rOmega_sq
            - 2.0 * pr2dot * rOmega_sq_dot / (rOmega_sq * rOmega_sq)
            - pr2 * rOmega_sq_ddot / (rOmega_sq * rOmega_sq)
            + 2.0 * pr2 * (rOmega_sq_dot ** 2) / (rOmega_sq ** 3)
        )

        n2 = n1 / r_peak
        n2dot = n1dot / r_peak - n1 * rdot / (r_peak * r_peak)
        n2ddot = (
            n1ddot / r_peak
            - 2.0 * n1dot * rdot / (r_peak * r_peak)
            - n1 * rddot / (r_peak * r_peak)
            + 2.0 * n1 * (rdot ** 2) / (r_peak ** 3)
        )

        sqrt_r = math.sqrt(r_peak)
        inv_sqrt_r = 1.0 / sqrt_r
        inv_sqrt_r_dot = -(0.5 * rdot) / (r_peak * sqrt_r)
        inv_sqrt_r_ddot = (0.75 * (rdot ** 2)) / (r_peak * r_peak * sqrt_r) - 0.5 * rddot / (r_peak * sqrt_r)

        n3 = n2 * inv_sqrt_r
        n3dot = n2dot * inv_sqrt_r + n2 * inv_sqrt_r_dot
        n3ddot = n2ddot * inv_sqrt_r + 2.0 * n2dot * inv_sqrt_r_dot + n2 * inv_sqrt_r_ddot

        # Amplitude system (3x3) for a1,a2,a3
        q_matrix = torch.tensor(
            [
                [amp_e * n1, amp_e * n2, amp_e * n3],
                [adot_e * n1 + amp_e * n1dot, adot_e * n2 + amp_e * n2dot, adot_e * n3 + amp_e * n3dot],
                [
                    addot_e * n1 + 2.0 * adot_e * n1dot + amp_e * n1ddot,
                    addot_e * n2 + 2.0 * adot_e * n2dot + amp_e * n2ddot,
                    addot_e * n3 + 2.0 * adot_e * n3dot + amp_e * n3ddot,
                ],
            ],
            dtype=torch.float64,
        )
        rhs_a = torch.tensor(
            [peak_amp_nr - amp_e, peak_adot_nr - adot_e, peak_addot_nr - addot_e],
            dtype=torch.float64,
        )
        a_sol = _solve_nqc_system(q_matrix, rhs_a)
        a1, a2, a3 = [float(x) for x in a_sol]

        # Phase system (2x2) for b1,b2 (b3,b4 stay zero as in LAL)
        p1 = pr / max(rOmega, 1e-16)
        p1dot = (prdot * rOmega - pr * rOmega_dot) / max(rOmega_sq, 1e-16)
        p1ddot = (
            prddot / max(rOmega, 1e-16)
            - 2.0 * prdot * rOmega_dot / max(rOmega_sq, 1e-16)
            - pr * rOmega_ddot / max(rOmega_sq, 1e-16)
            + 2.0 * pr * (rOmega_dot ** 2) / max(rOmega * rOmega_sq, 1e-16)
        )
        p2dot = p1dot * pr2 + p1 * pr2dot
        p2ddot = p1ddot * pr2 + 2.0 * p1dot * pr2dot + p1 * pr2ddot

        p_matrix = torch.tensor(
            [[-p1dot, -p2dot], [-p1ddot, -p2ddot]],
            dtype=torch.float64,
        )
        rhs_b = torch.tensor(
            [peak_omega_nr - omega_e, peak_omegadot_nr - omegadot_e],
            dtype=torch.float64,
        )
        b_sol = _solve_nqc_system(p_matrix, rhs_b)
        b1, b2 = [float(x) for x in b_sol]

    coeffs = {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "a5": a5,
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "b4": b4,
        "peak_amp": peak_amp_nr,
        "peak_adot": peak_adot_nr,
        "peak_addot": peak_addot_nr,
    }
    return coeffs


__all__ = [
    "calibrated_a123",
    "peak_delta_t_v4",
    "solve_nqc_coeffs",
    "peak_amp_v4",
    "peak_adot_v4",
    "peak_addot_v4",
    "peak_omega_v4",
    "peak_omegadot_v4",
]
