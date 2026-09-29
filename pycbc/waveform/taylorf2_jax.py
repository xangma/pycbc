import functools
import math
import jax
import jax.numpy as jnp
import numpy as np

from pycbc.constants import MTSUN_SI, MRSUN_SI, PC_SI, PI
from pycbc.types import Array, FrequencySeries
from pycbc.types.array_jax import JAXArrayData, _ensure_x64, to_jax

GAMMA = 0.5772156649015329


def _order(params, name, default=-1):
    try:
        return int(params.get(name, default))
    except (TypeError, ValueError):
        return None


def _supported(params):
    """Whether parameters are covered by the exact aligned-spin port."""
    if params.get("approximant") != "TaylorF2":
        return False
    if _order(params, "amplitude_order", 0) not in (-1, 0):
        return False
    if _order(params, "phase_order") not in (-1, 0, 1, 2, 3, 4, 5, 6, 7):
        return False
    if _order(params, "spin_order") not in (-1, 0, 1, 2, 3, 4, 5, 6, 7):
        return False
    if _order(params, "tidal_order") not in (-1, 0):
        return False
    if any(float(params.get(name, 0.0) or 0.0) != 0.0 for name in
           ("spin1x", "spin1y", "spin2x", "spin2y", "lambda1", "lambda2")):
        return False
    return True


def fd_supported(params):
    return _supported(params) and params.get("sample_points") is None


def sequence_supported(params):
    return _supported(params) and params.get("sample_points") is not None


def _coefficients(params):
    """Construct the LAL PN series with efficient NumPy operations."""
    import math
    import numpy as np
    m1 = float(params["mass1"])
    m2 = float(params["mass2"])
    s1 = float(params.get("spin1z", 0.0))
    s2 = float(params.get("spin2z", 0.0))
    mt = m1 + m2
    eta = m1 * m2 / mt**2
    x1, x2 = m1 / mt, m2 / mt
    pn = 3.0 / (128.0 * eta)
    v = np.zeros(16, dtype=np.float64)
    vl = np.zeros(16, dtype=np.float64)
    v[0] = 1.0
    v[2] = 5.0 * (74.3 / 8.4 + 11.0 * eta) / 9.0
    v[3] = -16.0 * PI
    v[4] = 5.0 * (3058.673 / 7.056 + 5429.0 / 7.0 * eta + 617.0 * eta**2) / 72.0
    v[5] = 5.0 / 9.0 * (772.9 / 8.4 - 13.0 * eta) * PI
    vl[5] = 5.0 / 3.0 * (772.9 / 8.4 - 13.0 * eta) * PI
    vl[6] = -684.8 / 2.1
    v[6] = (
        11583.231236531 / 4.694215680 - 640.0 / 3.0 * PI**2 - 684.8 / 2.1 * GAMMA
        + eta * (-15737.765635 / 3.048192 + 225.5 / 1.2 * PI**2)
        + eta**2 * 76.055 / 1.728 - eta**3 * 127.825 / 1.296 + vl[6] * math.log(4.0)
    )
    v[7] = PI * (770.96675 / 2.54016 + 378.515 / 1.512 * eta - 740.45 / 7.56 * eta**2)
    so3 = lambda x: x * (25.0 + 38.0 / 3.0 * x)
    so5 = lambda x: -x * (1391.5 / 8.4 - x * (1.0 - x) * 10.0 / 3.0 + x * (1276.0 / 8.1 + x * (1.0 - x) * 170.0 / 9.0))
    so6 = lambda x: PI * x * (1490.0 / 3.0 + x * 260.0)
    so7 = lambda x: x * (-17097.8035 / 4.8384 + eta * 28764.25 / 6.72 + eta**2 * 47.35 / 1.44 + x * (-7189.233785 / 1.524096 + eta * 458.555 / 3.024 - eta**2 * 534.5 / 7.2))
    order = _order(params, "spin_order")
    if order in (-1, 7):
        v[7] += so7(x1) * s1 + so7(x2) * s2
    if order in (-1, 6, 7):
        q6 = lambda x: (4703.5 / 8.4 + 2935.0 / 6.0 * x - 120.0 * x**2) * x**2
        self6 = lambda x: (-4108.25 / 6.72 - 108.5 / 1.2 * x + 125.5 / 3.6 * x**2) * x**2
        v[6] += (
            so6(x1) * s1 + so6(x2) * s2
            + (326.75 / 1.12 + 557.5 / 1.8 * eta) * eta * s1 * s2
            + (q6(x1) + self6(x1)) * s1**2 + (q6(x2) + self6(x2)) * s2**2
        )
    if order in (-1, 5, 6, 7):
        spin5 = so5(x1) * s1 + so5(x2) * s2
        v[5] += spin5
        vl[5] += 3.0 * spin5
    if order in (-1, 4, 5, 6, 7):
        q4 = (-720.0 / 9.6 + 1.0 / 9.6 + 240.0 / 9.6 - 7.0 / 9.6)
        v[4] += (
            247.0 / 4.8 * eta * s1 * s2 - 721.0 / 4.8 * eta * s1 * s2
            + q4 * x1**2 * s1**2 + q4 * x2**2 * s2**2
        )
    if order in (-1, 3, 4, 5, 6, 7):
        v[3] += so3(x1) * s1 + so3(x2) * s2
    phase_order = _order(params, "phase_order")
    if phase_order != -1:
        v[phase_order + 1:8] = 0.0
        vl[phase_order + 1:8] = 0.0
    return jnp.asarray(v * pn), jnp.asarray(vl * pn), phase_order


@functools.partial(jax.jit, static_argnames=("dtype", "has_f_ref"))
def _samples_core(frequencies, pi_mass, coeff, coeff_log, f_ref, coa_phase,
                  m1, m2, distance, eta, dtype, has_f_ref=False):
    real_dtype = jnp.float64
    f = jnp.asarray(frequencies, dtype=real_dtype)
    v = jnp.cbrt(pi_mass * f)
    lv = jnp.log(v)
    powers = jnp.arange(16, dtype=real_dtype)
    terms = (jnp.asarray(coeff, dtype=real_dtype)
             + jnp.asarray(coeff_log, dtype=real_dtype) * lv[..., None])
    phase = jnp.sum(terms * v[..., None] ** powers, axis=-1) / v**5
    if has_f_ref:
        vr = jnp.cbrt(pi_mass * f_ref)
        terms_ref = (jnp.asarray(coeff, dtype=real_dtype)
                     + jnp.asarray(coeff_log, dtype=real_dtype) * jnp.log(vr))
        phase = phase - jnp.sum(terms_ref * vr ** powers, axis=-1) / vr**5
    phase = phase - 2.0 * coa_phase
    amp0 = (-4.0 * m1 * m2 / distance * MRSUN_SI * MTSUN_SI * math.sqrt(PI / 12.0))
    amplitude = amp0 * jnp.sqrt(5.0 / (32.0 * eta)) * v ** (-3.5)
    return (amplitude * (jnp.cos(phase - PI / 4.0) -
                        1j * jnp.sin(phase - PI / 4.0))).astype(dtype)


@functools.partial(jax.jit, static_argnames=("kmin", "n", "dtype", "has_f_ref"))
def _generate_fd_core(kmin, n, delta_f, pi_mass, coeff, coeff_log, f_ref,
                      coa_phase, m1, m2, distance, eta, inclination, dtype,
                      has_f_ref=False):
    frequencies = jnp.arange(kmin, n, dtype=jnp.float64 if dtype == jnp.complex128 else jnp.float32) * delta_f
    active = _samples_core(frequencies, pi_mass, coeff, coeff_log, f_ref,
                          coa_phase, m1, m2, distance, eta, dtype, has_f_ref=has_f_ref)
    out = jnp.zeros(n, dtype=dtype).at[kmin:n].set(active)
    c = jnp.cos(inclination)
    hp = out * (1.0 + c * c) / 2.0
    hc = -1j * c * out
    return hp, hc


@functools.partial(jax.jit, static_argnames=("dtype", "has_f_ref"))
def _generate_sequence_core(frequencies, pi_mass, coeff, coeff_log, f_ref,
                            coa_phase, m1, m2, distance, eta, inclination, dtype,
                            has_f_ref=False):
    active = _samples_core(frequencies, pi_mass, coeff, coeff_log, f_ref,
                          coa_phase, m1, m2, distance, eta, dtype, has_f_ref=has_f_ref)
    c = jnp.cos(inclination)
    hp = active * (1.0 + c * c) / 2.0
    hc = -1j * c * active
    return hp, hc


def generate_fd(**params):
    if not fd_supported(params):
        raise ValueError("TaylorF2 parameters are unsupported by the JAX port")
    _ensure_x64()
    coeffs = _coefficients(params)
    delta_f, f_lower = float(params["delta_f"]), float(params["f_lower"])
    if delta_f <= 0 or f_lower <= 0:
        raise ValueError("TaylorF2 delta_f and f_lower must be positive")
    m1, m2 = float(params["mass1"]), float(params["mass2"])
    pi_mass = PI * (m1 + m2) * MTSUN_SI
    f_final = float(params.get("f_final", 0.0) or 0.0)
    if f_final <= 0:
        f_final = (1.0 / math.sqrt(6.0)) ** 3 / pi_mass
    n, kmin = int(f_final / delta_f + 1.0), int(math.ceil(f_lower / delta_f))
    if n <= kmin:
        raise ValueError("TaylorF2 ending frequency must exceed f_lower")
    dtype = jnp.complex64 if params.get("dtype") in (np.complex64, "complex64") else jnp.complex128
    distance = float(params.get("distance", 1.0)) * 1.0e6 * PC_SI
    eta = m1 * m2 / (m1 + m2) ** 2
    f_ref = float(params.get("f_ref", 0.0) or 0.0)
    has_f_ref = f_ref > 0.0
    coa_phase = float(params.get("coa_phase", 0.0) or 0.0)
    inclination = float(params.get("inclination", 0.0) or 0.0)

    hp, hc = _generate_fd_core(
        kmin, n, delta_f, pi_mass, coeffs[0], coeffs[1], f_ref,
        coa_phase, m1, m2, distance, eta, inclination, dtype,
        has_f_ref=has_f_ref
    )
    epoch = -1.0 / delta_f
    return (FrequencySeries(Array(JAXArrayData(hp), copy=False), delta_f=delta_f, epoch=epoch, copy=False),
            FrequencySeries(Array(JAXArrayData(hc), copy=False), delta_f=delta_f, epoch=epoch, copy=False))


def generate_sequence(**params):
    if not sequence_supported(params):
        raise ValueError("TaylorF2 sequence parameters are unsupported by the JAX port")
    _ensure_x64()
    coeffs = _coefficients(params)
    m1, m2 = float(params["mass1"]), float(params["mass2"])
    pi_mass = PI * (m1 + m2) * MTSUN_SI
    distance = float(params.get("distance", 1.0)) * 1.0e6 * PC_SI
    eta = m1 * m2 / (m1 + m2) ** 2
    f_ref = float(params.get("f_ref", 0.0) or 0.0)
    has_f_ref = f_ref > 0.0
    coa_phase = float(params.get("coa_phase", 0.0) or 0.0)
    inclination = float(params.get("inclination", 0.0) or 0.0)
    frequencies = to_jax(params["sample_points"])
    hp, hc = _generate_sequence_core(
        frequencies, pi_mass, coeffs[0], coeffs[1], f_ref,
        coa_phase, m1, m2, distance, eta, inclination, jnp.complex128,
        has_f_ref=has_f_ref
    )
    return Array(JAXArrayData(hp), copy=False), Array(JAXArrayData(hc), copy=False)
