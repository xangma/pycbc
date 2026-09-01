#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <tuple>
#include <vector>

// Exact fixed-schema MSA reference-plus-mode helper and fused CPU mode twisting.
// Eliminates ATen dispatcher overhead and temporary allocations.

namespace {

using at::Tensor;

enum Field : int64_t {
  OMEGAZ0,
  OMEGAZ1,
  OMEGAZ2,
  OMEGAZ3,
  OMEGAZ4,
  OMEGAZ5,
  OMEGAZETA0,
  OMEGAZETA1,
  OMEGAZETA2,
  OMEGAZETA3,
  OMEGAZETA4,
  OMEGAZETA5,
  S1N2,
  S2N2,
  SAV,
  SAV2,
  S0N2,
  SEFF,
  C1,
  C1_OVER_ETA,
  DELTA,
  ETA,
  ETA2,
  ETA4,
  G0,
  INV_SAV,
  INV_SAV2,
  INV_ETA,
  INV_ETA2,
  INV_ETA3,
  INV_ETA4,
  PHIZ0,
  PSI0,
  PSI1,
  PSI2,
  QQ,
  SQRT_INV_ETA,
  ZETA0,
  L0,
  L1,
  L2,
  L3,
  L4,
  FIELD_COUNT,
};

struct Values {
  const double* data;
  double operator[](Field field) const { return data[static_cast<int64_t>(field)]; }
};

Values checked_values(const Tensor& state) {
  TORCH_CHECK(state.device().is_cpu(), "state must be CPU");
  TORCH_CHECK(state.scalar_type() == at::kDouble, "state must be float64");
  TORCH_CHECK(state.is_contiguous(), "state must be contiguous");
  TORCH_CHECK(state.storage_offset() == 0, "state must own its storage");
  TORCH_CHECK(state.dim() == 1 && state.numel() == FIELD_COUNT,
              "state must have 43 fields");
  return Values{state.const_data_ptr<double>()};
}

inline double j_norm_scalar(double lnorm, const Values& v) {
  double val = lnorm * lnorm + 2.0 * lnorm * v[C1_OVER_ETA] + v[SAV2];
  return std::sqrt(std::max(val, 0.0));
}

inline double orbital_angular_momentum_3pn_scalar(double velocity, const Values& v) {
  double v2 = velocity * velocity;
  return (1.0 / velocity) * v[ETA] *
      (1.0 + v2 *
          (v[L0] + velocity * v[L1] +
           v2 * (v[L2] + velocity * v[L3] + v2 * v[L4])));
}

inline double pruned_jacobi_scalar(double argument, double input_parameter, double upper_parameter) {
  double parameter = std::clamp(input_parameter, 0.0, upper_parameter);
  double a = 1.0;
  double b = std::sqrt(1.0 - parameter);
  double ratios[5];
  for (int64_t index = 0; index < 5; ++index) {
    double next_a = 0.5 * (a + b);
    ratios[index] = 0.5 * (a - b) / next_a;
    b = std::sqrt(a * b);
    a = next_a;
  }
  bool stable = (a == b) && (std::sqrt(a * a) == a);
  double final_a = a;
  double tail_ratios[7];
  if (!stable) {
    double active_a = a;
    double active_b = b;
    for (int64_t index = 0; index < 7; ++index) {
      double next_a = 0.5 * (active_a + active_b);
      tail_ratios[index] = 0.5 * (active_a - active_b) / next_a;
      active_b = std::sqrt(active_a * active_b);
      active_a = next_a;
    }
    final_a = active_a;
  }

  double complete_integral = (1.0 / (2.0 * final_a)) * 3.141592653589793;
  double period = 2.0 * complete_integral;
  double shifted = argument + complete_integral;
  double rem = shifted - std::floor(shifted / period) * period;
  double reduced = rem - complete_integral;

  double amplitude = 4096.0 * final_a * reduced;
  if (!stable) {
    double active_amplitude = amplitude;
    for (int64_t index = 6; index >= 0; --index) {
      double ratio = tail_ratios[index];
      active_amplitude = 0.5 *
          (active_amplitude +
           std::asin(std::clamp(ratio * std::sin(active_amplitude), -1.0, 1.0)));
    }
    amplitude = active_amplitude;
  } else {
    amplitude = amplitude * (1.0 / 128.0);
  }

  for (int64_t index = 4; index >= 0; --index) {
    double ratio = ratios[index];
    amplitude = 0.5 *
        (amplitude +
         std::asin(std::clamp(ratio * std::sin(amplitude), -1.0, 1.0)));
  }
  double sn = std::sin(amplitude);
  return sn * sn;
}

inline void eval_msa_scalar_point(
    double velocity,
    const Values& v,
    bool static_fallback,
    double& out_phiz_val,
    double& out_zeta_val,
    double& out_cos_beta,
    double& out_phiz_unshifted,
    double& out_phiz_corr,
    double& out_zeta_unshifted,
    double& out_zeta_corr) {
  const double lnorm = (1.0 / velocity) * v[ETA];
  const double jnorm = j_norm_scalar(lnorm, v);
  const double lnorm3pn = orbital_angular_momentum_3pn_scalar(velocity, v);
  const double jnorm3pn = j_norm_scalar(lnorm3pn, v);

  // Spin evolution coefficients
  const double j2 = jnorm * jnorm;
  const double l2 = lnorm * lnorm;
  const double s1n2 = v[S1N2];
  const double s2n2 = v[S2N2];
  const double q = v[QQ];
  const double eta = v[ETA];
  const double j2ml2 = j2 - l2;
  const double j2ml2sq = j2ml2 * j2ml2;
  const double delta = v[DELTA];
  const double seff = v[SEFF];
  const double b =
      (l2 + s1n2) * q + 2.0 * lnorm * seff - 2.0 * j2 - s1n2 -
      s2n2 + (l2 + s2n2) / q;
  const double c =
      j2ml2sq - 2.0 * lnorm * seff * j2ml2 -
      2.0 * ((1.0 - q) / q) * l2 * (s1n2 - q * s2n2) +
      4.0 * eta * l2 * seff * seff -
      2.0 * delta * (s1n2 - s2n2) * seff * lnorm +
      2.0 * ((1.0 - q) / q) * (q * s1n2 - s2n2) * j2;
  const double d =
      ((1.0 - q) / q) * (s2n2 - q * s1n2) * j2ml2sq +
      delta * delta * (s1n2 - s2n2) * (s1n2 - s2n2) * l2 / eta +
      2.0 * delta * lnorm * seff * (s1n2 - s2n2) * j2ml2;

  const double b2 = b * b;
  const double p = c - b2 / 3.0;
  const double q_coefficient = (2.0 / 27.0) * b2 * b - b * c / 3.0 + d;
  const double sqrt_argument = std::sqrt(std::max(-p / 3.0, 0.0));
  const double denominator = p * sqrt_argument;
  const double acos_argument = (denominator != 0.0)
      ? (1.5 * q_coefficient / denominator)
      : 0.0;
  const double theta = std::acos(std::clamp(acos_argument, -1.0, 1.0)) / 3.0;
  const double root1 =
      2.0 * sqrt_argument * std::cos(theta - 2.0 * 6.283185307179586 / 3.0) -
      b / 3.0;
  const double root2 =
      2.0 * sqrt_argument * std::cos(theta - 6.283185307179586 / 3.0) -
      b / 3.0;
  const double root3 = 2.0 * sqrt_argument * std::cos(theta) - b / 3.0;

  const double maximum = std::max(root1, std::max(root2, root3));
  const double minimum = std::min(root1, std::min(root2, root3));
  const bool root3_is_middle = (maximum > root3) && (minimum < root3);
  const bool root1_is_middle = (maximum > root1) && (minimum < root1);
  const double middle = root3_is_middle ? root3 : (root1_is_middle ? root1 : root2);

  bool fallback = (p >= 0.0) || (sqrt_argument == 0.0) || !std::isfinite(theta) || static_fallback;
  const double fallback_middle = v[S0N2];
  const double s32 = fallback ? 0.0 : minimum;
  const double smi2 = fallback ? fallback_middle : std::abs(middle);
  const double spl2 = fallback ? (fallback_middle + 1.0e-9) : std::abs(maximum);

  const bool separated = std::abs(smi2 - spl2) > 1.0e-5;

  // MSA corrections
  const double v2 = velocity * velocity;
  const double v3 = velocity * v2;
  const double v4 = v2 * v2;
  const double v6 = v4 * v2;
  const double eta2 = v[ETA2];
  const double c0 =
      -0.75 *
      (std::pow(j2 - spl2, 2) * v4 / eta -
       4.0 * eta * seff * (j2 - spl2) * v3 -
       2.0 *
           (j2 - spl2 + 2.0 * (v[S1N2] - v[S2N2]) * delta) *
           eta * v2 +
       (4.0 * seff * velocity + 1.0) * eta * eta2) *
      jnorm * v2 * (seff * velocity - 1.0);
  const double c2 =
      1.5 * (smi2 - spl2) * jnorm *
      ((j2 - spl2) / eta * v2 - 2.0 * eta * seff * velocity - eta) *
      (seff * velocity - 1.0) * v4;
  const double c4 =
      -0.75 * jnorm * (seff * velocity - 1.0) * (spl2 - smi2) * (spl2 - smi2) *
      v6 / eta;

  const double spl = std::sqrt(std::max(spl2, 0.0));
  const double d0 =
      -(j2 - std::pow(lnorm + spl, 2)) * (j2 - std::pow(lnorm - spl, 2));
  const double d2 = -2.0 * (spl2 - smi2) * (j2 + l2 - spl2);
  const double d4 = -(spl2 - smi2) * (spl2 - smi2);

  const double two_d0 = 2.0 * d0;
  const double sd = std::sqrt(std::max(d2 * d2 - 4.0 * d0 * d4, 0.0));
  const double a_theta_l =
      0.5 * (jnorm / lnorm + lnorm / jnorm - spl2 / (jnorm * lnorm));
  const double b_theta_l = 0.5 * (spl2 - smi2) / (jnorm * lnorm);
  const double nc_num = 2.0 * (d0 + d2 + d4);
  const double nc_denom = two_d0 + d2 + sd;
  const double nc = nc_num / nc_denom;
  const double nd = nc_denom / two_d0;
  const double sqrt_nc = std::sqrt(std::abs(nc));
  const double sqrt_nd = std::sqrt(std::abs(nd));
  const double psi_phase =
      (0.0 - 0.75 * v[G0] * v[DELTA] * (1.0 + v[PSI1] * velocity + v[PSI2] * v2) / (v2 * velocity)) + v[PSI0];
  const double tangent = std::tan(psi_phase);
  const double arctangent = std::atan(tangent);
  const double psi_dot =
      -0.75 * (v2 * v2 * v2) * (1.0 - velocity * v[SEFF]) *
      v[SQRT_INV_ETA] * std::sqrt(std::max(spl2 - s32, 0.0));
  const double c2_denominator = 2.0 * d0 * sd * (d0 + d2 + d4);
  double c_prefactor = std::abs(
      (c4 * d0 * (two_d0 + d2 + sd) -
       c2 * d0 * (d2 + 2.0 * d4 - sd) -
       c0 * (two_d0 * d4 - (d2 + d4) * (d2 - sd))) /
      c2_denominator);
  double d_prefactor = std::abs(
      (-c4 * d0 * (two_d0 + d2 - sd) +
       c2 * d0 * (d2 + 2.0 * d4 + sd) -
       c0 * (-two_d0 * d4 + (d2 + d4) * (d2 + sd))) /
      c2_denominator);
  double c_term = ((c_prefactor * sqrt_nc / (nc - 1.0)) *
      (arctangent - std::atan(sqrt_nc * tangent))) / psi_dot;
  double d_term = ((d_prefactor * sqrt_nd / (nd - 1.0)) *
      (arctangent - std::atan(sqrt_nd * tangent))) / psi_dot;
  if ((std::abs(nc - 1.0) < 1.0e-14) || (psi_dot == 0.0)) {
    c_term = 0.0;
  }
  if ((std::abs(nd - 1.0) < 1.0e-14) || (psi_dot == 0.0)) {
    d_term = 0.0;
  }
  double phiz_correction = std::isnan(c_term + d_term) ? 0.0 : (c_term + d_term);
  double zeta_correction = std::isnan(
      a_theta_l * phiz_correction +
      (2.0 * b_theta_l * d0) *
          (c_term / (sd - d2) - d_term / (sd + d2)))
      ? 0.0
      : (a_theta_l * phiz_correction +
         (2.0 * b_theta_l * d0) *
             (c_term / (sd - d2) - d_term / (sd + d2)));

  if (!separated) {
    phiz_correction = 0.0;
    zeta_correction = 0.0;
  }

  // phiz
  const double invv = 1.0 / velocity;
  const double invv2 = invv * invv;
  const double l_newtonian = invv * v[ETA];
  const double c1 = v[C1];
  const double c12 = c1 * c1;
  const double sav2 = v[SAV2];
  const double sav = v[SAV];
  const double invsav = v[INV_SAV];
  const double invsav2 = v[INV_SAV2];
  const double invsav2_squared = invsav2 * invsav2;
  const double log1 = std::log(std::abs(
      c1 + jnorm * v[ETA] + v[ETA] * l_newtonian));
  const double log2 = std::log(std::abs(
      c1 + jnorm * sav * velocity + sav2 * velocity));
  const double phiz0 =
      jnorm * v[INV_ETA4] *
          (0.5 * c12 - c1 * v[ETA2] * invv / 6.0 -
           sav2 * v[ETA2] / 3.0 - v[ETA4] * invv2 / 3.0) -
      0.5 * c1 * v[INV_ETA] *
          (c12 * v[INV_ETA4] - sav2 * v[INV_ETA2]) * log1;
  const double phiz1 =
      -0.5 * jnorm * v[INV_ETA2] *
          (c1 + v[ETA] * l_newtonian) +
      0.5 * v[INV_ETA3] * (c12 - v[ETA2] * sav2) * log1;
  const double phiz2 = -jnorm + sav * log2 - c1 * log1 * v[INV_ETA];
  const double phiz3 =
      jnorm * velocity - v[ETA] * log1 + c1 * log2 * invsav;
  const double phiz4 =
      0.5 * jnorm * invsav2 * velocity * (c1 + velocity * sav2) -
      0.5 * (invsav2 * invsav) * (c12 - v[ETA2] * sav2) * log2;
  const double phiz5 =
      -jnorm * velocity *
          (0.5 * c12 * invsav2_squared -
           c1 * velocity * invsav2 / 6.0 - velocity * velocity / 3.0 -
           v[ETA2] * invsav2 / 3.0) +
      0.5 * c1 * invsav2_squared * invsav *
          (c12 - v[ETA2] * sav2) * log2;
  const double phiz_unshifted =
      phiz0 * v[OMEGAZ0] + phiz1 * v[OMEGAZ1] +
      phiz2 * v[OMEGAZ2] + phiz3 * v[OMEGAZ3] +
      phiz4 * v[OMEGAZ4] + phiz5 * v[OMEGAZ5];
  const double phiz_val = std::isnan(phiz_unshifted + v[PHIZ0]) ? 0.0 : (phiz_unshifted + v[PHIZ0]);

  // zeta
  const double zeta_unshifted = v[ETA] *
      (v[OMEGAZETA0] * invv2 * invv +
       v[OMEGAZETA1] * invv2 + v[OMEGAZETA2] * invv +
       v[OMEGAZETA3] * std::log(velocity) +
       v[OMEGAZETA4] * velocity +
       v[OMEGAZETA5] * velocity * velocity);
  const double zeta_val = std::isnan(zeta_unshifted + v[ZETA0]) ? 0.0 : (zeta_unshifted + v[ZETA0]);

  // jacobi & cos_beta
  const double parameter = (std::abs(smi2 - spl2) >= 1.0e-5)
      ? ((smi2 - spl2) / (s32 - spl2))
      : 0.0;
  const double psi_value = v[PSI0] - 0.75 * v[G0] * v[DELTA] *
      (1.0 + v[PSI1] * velocity + v[PSI2] * v2) / (v2 * velocity);
  const double sn_squared = pruned_jacobi_scalar(
      psi_value,
      parameter,
      1.0 - std::numeric_limits<double>::epsilon());
  double spin_squared = spl2 + (smi2 - spl2) * sn_squared;
  if (std::abs(smi2 - spl2) < 1.0e-5) {
    spin_squared = spl2;
  }
  const double snorm = std::sqrt(std::max(spin_squared, 0.0));
  const double cos_beta =
      0.5 * (jnorm3pn * jnorm3pn + lnorm3pn * lnorm3pn - snorm * snorm) /
      (lnorm3pn * jnorm3pn);
  const double clean_cos_beta = std::clamp(
      std::isnan(cos_beta) ? 1.0 : cos_beta, -1.0, 1.0);

  out_phiz_val = phiz_val + phiz_correction;
  out_zeta_val = zeta_val + zeta_correction;
  out_cos_beta = clean_cos_beta;
  out_phiz_unshifted = phiz_unshifted;
  out_phiz_corr = phiz_correction;
  out_zeta_unshifted = zeta_unshifted;
  out_zeta_corr = zeta_correction;
}

using ReferenceAndModes =
    std::tuple<Tensor, Tensor, Tensor, double, double>;

ReferenceAndModes reference_and_modes(
    const Tensor& velocity_rows,
    const Tensor& state,
    double reference_velocity,
    bool static_fallback) {
  TORCH_CHECK(velocity_rows.device().is_cpu(), "velocity rows must be CPU");
  TORCH_CHECK(
      velocity_rows.scalar_type() == at::kDouble,
      "velocity rows must be float64");
  TORCH_CHECK(
      velocity_rows.dim() == 2 && velocity_rows.size(0) == 4 &&
          velocity_rows.size(1) != 0,
      "velocity rows must have shape (4, N) with N > 0");
  TORCH_CHECK(velocity_rows.is_contiguous(), "velocity rows must be contiguous");
  TORCH_CHECK(
      velocity_rows.storage_offset() == 0,
      "velocity rows must own their storage");

  const auto values = checked_values(state);
  const int64_t N = velocity_rows.size(1);
  const double* v_ptr = velocity_rows.const_data_ptr<double>();

  // Evaluate reference point
  double ref_phiz, ref_zeta, ref_cos_beta;
  double ref_phiz_unshifted, ref_phiz_corr, ref_zeta_unshifted, ref_zeta_corr;
  eval_msa_scalar_point(
      reference_velocity,
      values,
      static_fallback,
      ref_phiz,
      ref_zeta,
      ref_cos_beta,
      ref_phiz_unshifted,
      ref_phiz_corr,
      ref_zeta_unshifted,
      ref_zeta_corr);

  const double phiz_shift = -ref_phiz;
  const double zeta_shift = -ref_zeta;

  double ref_phiz_shifted = ref_phiz_unshifted + phiz_shift;
  double reference_phiz_residual =
      (std::isnan(ref_phiz_shifted) ? 0.0 : ref_phiz_shifted) + ref_phiz_corr;
  double ref_zeta_shifted = ref_zeta_unshifted + zeta_shift;
  double reference_zeta_residual =
      (std::isnan(ref_zeta_shifted) ? 0.0 : ref_zeta_shifted) + ref_zeta_corr;

  auto mode_phiz = torch::empty({4, N}, velocity_rows.options());
  auto mode_zeta = torch::empty({4, N}, velocity_rows.options());
  auto packed_cos_beta = torch::empty({5, N}, velocity_rows.options());

  double* phiz_out = mode_phiz.data_ptr<double>();
  double* zeta_out = mode_zeta.data_ptr<double>();
  double* cos_out = packed_cos_beta.data_ptr<double>();

  // Set reference row of packed_cos_beta
  for (int64_t j = 0; j < N; ++j) {
    cos_out[j] = ref_cos_beta;
  }

  const int64_t total_mode_points = 4 * N;
#pragma omp parallel for schedule(static)
  for (int64_t k = 0; k < total_mode_points; ++k) {
    double v = v_ptr[k];
    double p_val, z_val, c_beta, p_unsh, p_c, z_unsh, z_c;
    eval_msa_scalar_point(
        v,
        values,
        static_fallback,
        p_val,
        z_val,
        c_beta,
        p_unsh,
        p_c,
        z_unsh,
        z_c);
    double p_shifted = p_unsh + phiz_shift;
    phiz_out[k] = (std::isnan(p_shifted) ? 0.0 : p_shifted) + p_c;
    double z_shifted = z_unsh + zeta_shift;
    zeta_out[k] = (std::isnan(z_shifted) ? 0.0 : z_shifted) + z_c;
    cos_out[N + k] = c_beta;
  }

  auto mode_cos_beta = packed_cos_beta.narrow(0, 1, 4);

  return {
      mode_phiz,
      mode_zeta,
      mode_cos_beta,
      reference_phiz_residual,
      reference_zeta_residual,
  };
}

inline void eval_mode_wigner_twist(
    int ell,
    int mprime,
    double alpha,
    double epsilon,
    double cos_half,
    double sin_half,
    std::complex<double> h_mode,
    const std::complex<double>* mode_harmonics,
    std::complex<double>& total_plus,
    std::complex<double>& total_cross) {
  
  double c = cos_half;
  double s = sin_half;
  double c2 = c * c;
  double c3 = c2 * c;
  double c4 = c3 * c;
  double s2 = s * s;
  double s3 = s2 * s;
  double s4 = s3 * s;

  double pos[9] = {0.0};
  double neg[9] = {0.0};

  if (ell == 2) {
    if (mprime == 2) {
      pos[0] = s4;
      pos[1] = 2.0 * c * s3;
      pos[2] = 2.449489742783178 * s2 * c2;
      pos[3] = 2.0 * c3 * s;
      pos[4] = c4;
      neg[0] = pos[4]; neg[1] = -pos[3]; neg[2] = pos[2]; neg[3] = -pos[1]; neg[4] = pos[0];
    } else if (mprime == 1) {
      pos[0] = 2.0 * c * s3;
      pos[1] = 3.0 * c2 * s2 - s4;
      pos[2] = 2.449489742783178 * (c3 * s - c * s3);
      pos[3] = c2 * (c2 - 3.0 * s2);
      pos[4] = -2.0 * c3 * s;
      neg[0] = -pos[4]; neg[1] = pos[3]; neg[2] = -pos[2]; neg[3] = pos[1]; neg[4] = -pos[0];
    }
  } else if (ell == 3) {
    double c5 = c4 * c;
    double c6 = c5 * c;
    double s5 = s4 * s;
    double s6 = s5 * s;
    if (mprime == 3) {
      pos[0] = s6;
      pos[1] = 2.449489742783178 * c * s5;
      pos[2] = 3.872983346207417 * c2 * s4;
      pos[3] = 4.47213595499958 * c3 * s3;
      pos[4] = 3.872983346207417 * c4 * s2;
      pos[5] = 2.449489742783178 * c5 * s;
      pos[6] = c6;
      neg[0] = pos[6]; neg[1] = -pos[5]; neg[2] = pos[4]; neg[3] = -pos[3]; neg[4] = pos[2]; neg[5] = -pos[1]; neg[6] = pos[0];
    } else if (mprime == 2) {
      pos[0] = 2.449489742783178 * c * s5;
      pos[1] = s4 * (5.0 * c2 - s2);
      pos[2] = 3.1622776601683795 * s3 * (2.0 * c3 - c * s2);
      pos[3] = 5.477225575051661 * c2 * (c2 - s2) * s2;
      pos[4] = 3.1622776601683795 * c3 * (c2 * s - 2.0 * s3);
      pos[5] = c4 * (c2 - 5.0 * s2);
      pos[6] = -2.449489742783178 * c5 * s;
      neg[0] = -pos[6]; neg[1] = pos[5]; neg[2] = -pos[4]; neg[3] = pos[3]; neg[4] = -pos[2]; neg[5] = pos[1]; neg[6] = -pos[0];
    }
  } else if (ell == 4) {
    double c5 = c4 * c;
    double c6 = c5 * c;
    double c7 = c6 * c;
    double c8 = c7 * c;
    double s5 = s4 * s;
    double s6 = s5 * s;
    double s7 = s6 * s;
    double s8 = s7 * s;
    if (mprime == 4) {
      pos[0] = s8;
      pos[1] = 2.8284271247461903 * c * s7;
      pos[2] = 5.291502622129181 * c2 * s6;
      pos[3] = 7.483314773547883 * c3 * s5;
      pos[4] = 8.366600265340756 * c4 * s4;
      pos[5] = 7.483314773547883 * c5 * s3;
      pos[6] = 5.291502622129181 * c6 * s2;
      pos[7] = 2.8284271247461903 * c7 * s;
      pos[8] = c8;
      neg[0] = pos[8]; neg[1] = -pos[7]; neg[2] = pos[6]; neg[3] = -pos[5]; neg[4] = pos[4]; neg[5] = -pos[3]; neg[6] = pos[2]; neg[7] = -pos[1]; neg[8] = pos[0];
    }
  }

  std::complex<double> plus_sum(0.0, 0.0);
  std::complex<double> cross_sum(0.0, 0.0);

  for (int emm = -ell; emm <= ell; ++emm) {
    int idx = emm + ell;
    double cos_ma = std::cos(emm * alpha);
    double sin_ma = std::sin(emm * alpha);
    std::complex<double> exp_pos(cos_ma, sin_ma);
    std::complex<double> exp_neg(cos_ma, -sin_ma);

    std::complex<double> Y = mode_harmonics[idx];
    std::complex<double> Y_bar = std::conj(Y);

    std::complex<double> neg_term = exp_neg * neg[idx] * Y;
    std::complex<double> pos_term = exp_pos * pos[idx] * Y_bar;

    if (ell % 2 != 0) {
      plus_sum += (neg_term - pos_term);
      std::complex<double> sum_terms = neg_term + pos_term;
      cross_sum += std::complex<double>(-sum_terms.imag(), sum_terms.real());
    } else {
      plus_sum += (neg_term + pos_term);
      std::complex<double> diff_terms = neg_term - pos_term;
      cross_sum += std::complex<double>(-diff_terms.imag(), diff_terms.real());
    }
  }

  double factor_phase = -mprime * epsilon;
  std::complex<double> exp_factor(std::cos(factor_phase), std::sin(factor_phase));
  std::complex<double> factor = exp_factor * h_mode * 0.5;

  total_plus += factor * plus_sum;
  total_cross += factor * cross_sum;
}

std::tuple<Tensor, Tensor> fused_twist_cpu(
    const std::vector<std::pair<int64_t, int64_t>>& modes,
    const std::vector<Tensor>& mode_samples,
    const std::vector<Tensor>& alpha_by_mprime,
    const std::vector<Tensor>& epsilon_by_mprime,
    const std::vector<Tensor>& cos_half_by_mprime,
    const std::vector<Tensor>& sin_half_by_mprime,
    const Tensor& harmonics_tensor,
    double pol_rotation,
    double long_asc_nodes,
    const c10::optional<Tensor>& out_plus_opt,
    const c10::optional<Tensor>& out_cross_opt
) {
  int64_t num_modes = modes.size();
  int64_t N = mode_samples[0].size(0);

  Tensor out_plus = out_plus_opt.has_value() && out_plus_opt->defined() ? *out_plus_opt : torch::empty({N}, mode_samples[0].options());
  Tensor out_cross = out_cross_opt.has_value() && out_cross_opt->defined() ? *out_cross_opt : torch::empty({N}, mode_samples[0].options());

  std::complex<double>* plus_ptr = reinterpret_cast<std::complex<double>*>(out_plus.data_ptr<c10::complex<double>>());
  std::complex<double>* cross_ptr = reinterpret_cast<std::complex<double>*>(out_cross.data_ptr<c10::complex<double>>());

  const std::complex<double>* harmonics_ptr = reinterpret_cast<const std::complex<double>*>(harmonics_tensor.const_data_ptr<c10::complex<double>>());

  const std::complex<double>* harmonics_ell2 = harmonics_ptr;       // 5 elements (m = -2..2)
  const std::complex<double>* harmonics_ell3 = harmonics_ptr + 5;   // 7 elements (m = -3..3)
  const std::complex<double>* harmonics_ell4 = harmonics_ptr + 12;  // 9 elements (m = -4..4)

  std::vector<const std::complex<double>*> sample_ptrs(num_modes);
  std::vector<const double*> alpha_ptrs(num_modes);
  std::vector<const double*> eps_ptrs(num_modes);
  std::vector<const double*> ch_ptrs(num_modes);
  std::vector<const double*> sh_ptrs(num_modes);
  std::vector<const std::complex<double>*> mode_harmonics_ptrs(num_modes);

  for (int m_idx = 0; m_idx < num_modes; ++m_idx) {
    int ell = modes[m_idx].first;
    int mprime = modes[m_idx].second;
    sample_ptrs[m_idx] = reinterpret_cast<const std::complex<double>*>(mode_samples[m_idx].const_data_ptr<c10::complex<double>>());
    alpha_ptrs[m_idx] = alpha_by_mprime[mprime].const_data_ptr<double>();
    eps_ptrs[m_idx] = epsilon_by_mprime[mprime].const_data_ptr<double>();
    ch_ptrs[m_idx] = cos_half_by_mprime[mprime].const_data_ptr<double>();
    sh_ptrs[m_idx] = sin_half_by_mprime[mprime].const_data_ptr<double>();
    if (ell == 2) mode_harmonics_ptrs[m_idx] = harmonics_ell2;
    else if (ell == 3) mode_harmonics_ptrs[m_idx] = harmonics_ell3;
    else if (ell == 4) mode_harmonics_ptrs[m_idx] = harmonics_ell4;
  }

  const double phi_total = 2.0 * (pol_rotation + long_asc_nodes);
  const double c_rot = std::cos(phi_total);
  const double s_rot = std::sin(phi_total);

#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < N; ++i) {
    std::complex<double> total_plus(0.0, 0.0);
    std::complex<double> total_cross(0.0, 0.0);

    for (int m_idx = 0; m_idx < num_modes; ++m_idx) {
      int ell = modes[m_idx].first;
      int mprime = modes[m_idx].second;
      eval_mode_wigner_twist(
          ell,
          mprime,
          alpha_ptrs[m_idx][i],
          eps_ptrs[m_idx][i],
          ch_ptrs[m_idx][i],
          sh_ptrs[m_idx][i],
          sample_ptrs[m_idx][i],
          mode_harmonics_ptrs[m_idx],
          total_plus,
          total_cross);
    }

    plus_ptr[i] = c_rot * total_plus + s_rot * total_cross;
    cross_ptr[i] = c_rot * total_cross - s_rot * total_plus;
  }

  return std::make_tuple(out_plus, out_cross);
}

std::tuple<Tensor, Tensor> evaluate_xas_native(
    const Tensor& frequencies,
    double total_mass_seconds,
    double eta,
    double overall_amp,
    double f1_Ms,
    double f2_Ms,
    double fMs_AmpMatchIN,
    double fMs_AmpRDMin,
    const Tensor& insp_phase_coeffs,
    const Tensor& int_phase_coeffs,
    const Tensor& mrd_phase_coeffs,
    const Tensor& insp_amp_coeffs,
    const Tensor& int_amp_coeffs,
    const Tensor& mrd_amp_coeffs,
    double lin_phase_coeff,
    double const_phase,
    double cosi,
    double long_asc_nodes
) {
  int64_t N = frequencies.size(0);
  auto options = torch::TensorOptions().dtype(torch::kComplexDouble).device(frequencies.device());
  Tensor out_plus = torch::empty({N}, options);
  Tensor out_cross = torch::empty({N}, options);

  const double* f_ptr = frequencies.const_data_ptr<double>();
  std::complex<double>* plus_ptr = reinterpret_cast<std::complex<double>*>(out_plus.data_ptr<c10::complex<double>>());
  std::complex<double>* cross_ptr = reinterpret_cast<std::complex<double>*>(out_cross.data_ptr<c10::complex<double>>());

  const double* p_ins = insp_phase_coeffs.const_data_ptr<double>();
  double phi0 = p_ins[0], phi1 = p_ins[1], phi2 = p_ins[2], phi3 = p_ins[3];
  double phi4 = p_ins[4], phi5 = p_ins[5], phi5L = p_ins[6], phi6 = p_ins[7];
  double phi6L = p_ins[8], phi7 = p_ins[9], phi8 = p_ins[10], phi8L = p_ins[11];
  double sigma1 = p_ins[12], sigma2 = p_ins[13], sigma3 = p_ins[14], sigma4 = p_ins[15];
  double phiN = -(3.0 * std::pow(M_PI, -5.0 / 3.0)) / 128.0;

  const double* p_int = int_phase_coeffs.const_data_ptr<double>();
  double b0 = p_int[0], b1 = p_int[1], b2 = p_int[2], b3 = p_int[3], b4 = p_int[4];
  double cL_int = p_int[5], fMs_RD_int = p_int[6], fMs_damp_int = p_int[7];
  double alpha0 = p_int[8], alpha1 = p_int[9];

  const double* p_mrd = mrd_phase_coeffs.const_data_ptr<double>();
  double c0 = p_mrd[0], c1 = p_mrd[1], c2 = p_mrd[2], c4ov3 = p_mrd[3], cLovfda = p_mrd[4];
  double fMs_RD_mrd = p_mrd[5], fMs_damp_mrd = p_mrd[6];
  double beta0 = p_mrd[7], beta1 = p_mrd[8];

  const double* a_ins = insp_amp_coeffs.const_data_ptr<double>();
  double A0 = a_ins[0], A2 = a_ins[1], A3 = a_ins[2], A4 = a_ins[3], A5 = a_ins[4], A6 = a_ins[5];
  double rho1 = a_ins[6], rho2 = a_ins[7], rho3 = a_ins[8];

  const double* a_int = int_amp_coeffs.const_data_ptr<double>();
  double delta0 = a_int[0], delta1 = a_int[1], delta2 = a_int[2], delta3 = a_int[3], delta4 = a_int[4];

  const double* a_mrd = mrd_amp_coeffs.const_data_ptr<double>();
  double fMs_RD_amp = a_mrd[0], gammaR = a_mrd[1], gammaD2 = a_mrd[2], gammaD13 = a_mrd[3];

  double cos_nodes = std::cos(2.0 * long_asc_nodes);
  double sin_nodes = std::sin(2.0 * long_asc_nodes);
  double plus_factor = -0.5 * (1.0 + cosi * cosi);

#pragma omp parallel for schedule(static)
  for (int64_t i = 0; i < N; ++i) {
    double f = f_ptr[i];
    double fM_s = f * total_mass_seconds;

    // 1. Phase
    double phase;
    if (fM_s < f1_Ms) {
      double f13 = std::cbrt(fM_s);
      double f23 = f13 * f13;
      double f43 = fM_s * f13;
      double f53 = fM_s * f23;
      double f2 = fM_s * fM_s;
      double f73 = f2 * f13;
      double f83 = f2 * f23;
      double f3 = f2 * fM_s;
      double f103 = f3 * f13;
      double f113 = f3 * f23;
      double log_f = std::log(fM_s);

      double phi_TF2 = phi0 + phi1*f13 + phi2*f23 + phi3*fM_s + phi4*f43
                     + phi5*f53 + phi5L*f53*log_f + phi6*f2 + phi6L*f2*log_f
                     + phi7*f73 + phi8*f83 + phi8L*f83*log_f;
      double phi_Ins = phi_TF2 + (sigma1*f83 + sigma2*f3 + sigma3*f103 + sigma4*f113);
      phase = (phi_Ins * phiN / f53) / eta;
    } else if (fM_s < f2_Ms) {
      double phi_Int = b0*fM_s + b1*std::log(fM_s) - b2/fM_s - 0.5*b3/(fM_s*fM_s) - (b4/3.0)/(fM_s*fM_s*fM_s)
                     + (2.0*cL_int/fMs_damp_int) * std::atan((fM_s - fMs_RD_int) / (2.0 * fMs_damp_int))
                     + alpha1*fM_s + alpha0;
      phase = phi_Int / eta;
    } else {
      double fM_23 = std::cbrt(fM_s * fM_s);
      double phi_MRD = c0*fM_s + 1.5*c1*fM_23 - c2/fM_s - c4ov3/(fM_s*fM_s*fM_s)
                     + cLovfda * std::atan((fM_s - fMs_RD_mrd) / fMs_damp_mrd)
                     + beta0 + beta1*fM_s;
      phase = phi_MRD / eta;
    }
    phase = phase + lin_phase_coeff * f + const_phase;

    // 2. Amplitude
    double amp;
    if (fM_s < fMs_AmpMatchIN) {
      double f13 = std::cbrt(fM_s);
      double f23 = f13 * f13;
      double f33 = fM_s;
      double f43 = fM_s * f13;
      double f53 = fM_s * f23;
      double f63 = fM_s * fM_s;
      double f73 = f63 * f13;
      double f83 = f63 * f23;
      double f93 = f63 * fM_s;

      double amp_TF2 = A0 + A2*f23 + A3*f33 + A4*f43 + A5*f53 + A6*f63;
      amp = amp_TF2 + rho1*f73 + rho2*f83 + rho3*f93;
    } else if (fM_s < fMs_AmpRDMin) {
      amp = std::pow(fM_s, 7.0 / 6.0) / (delta0 + fM_s * (delta1 + fM_s * (delta2 + fM_s * (delta3 + fM_s * delta4))));
    } else {
      double df = fM_s - fMs_RD_amp;
      amp = std::exp(-df * gammaR) * gammaD13 / (df * df + gammaD2);
    }

    double full_amp = overall_amp * amp * std::pow(fM_s, -7.0 / 6.0);
    std::complex<double> h22 = full_amp * std::complex<double>(std::cos(phase), std::sin(phase));

    if (cosi < -900.0) {
      plus_ptr[i] = h22;
      cross_ptr[i] = std::complex<double>(0.0, 0.0);
    } else {
      std::complex<double> plus0 = plus_factor * h22;
      std::complex<double> cross0 = std::complex<double>(0.0, cosi) * h22;

      plus_ptr[i] = cos_nodes * plus0 + sin_nodes * cross0;
      cross_ptr[i] = cos_nodes * cross0 - sin_nodes * plus0;
    }
  }

  return std::make_tuple(out_plus, out_cross);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "reference_and_modes",
      &reference_and_modes,
      py::call_guard<py::gil_scoped_release>());
  module.def(
      "fused_twist_cpu",
      &fused_twist_cpu,
      py::call_guard<py::gil_scoped_release>());
  module.def(
      "evaluate_xas_native",
      &evaluate_xas_native,
      py::call_guard<py::gil_scoped_release>());
}
