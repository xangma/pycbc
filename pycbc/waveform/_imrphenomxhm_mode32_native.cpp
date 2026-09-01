// Copyright (C) 2026 PyCBC contributors
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3 of the License, or (at your
// option) any later version.

#include <torch/extension.h>
#include <pybind11/complex.h>

#include <cmath>
#include <complex>

namespace {

void validate_packed(const torch::Tensor& value, int64_t count) {
    TORCH_CHECK(value.device().is_cpu(), "expected a CPU tensor");
    TORCH_CHECK(value.scalar_type() == torch::kFloat64, "expected float64");
    TORCH_CHECK(value.dim() == 1, "expected a one-dimensional tensor");
    TORCH_CHECK(value.numel() == count, "unexpected packed tensor size");
    TORCH_CHECK(value.is_contiguous(), "expected a contiguous tensor");
    TORCH_CHECK(value.storage_offset() == 0, "expected zero storage offset");
}

double component(
    double frequency,
    const double* tensor_real,
    double total_mass_seconds,
    double f_ring_32,
    double f_damp_32,
    double amp_norm,
    std::complex<double> mixing_322,
    std::complex<double> mixing_323,
    const double* auxiliary
) {
    // Preserve every eager subtraction and left-to-right intermediate.
    double sub = frequency - tensor_real[18];
    double neg = -sub;
    double mul = neg * tensor_real[19];
    double exponential = std::exp(mul);
    double numerator = exponential * tensor_real[21];
    double sub_1 = frequency - tensor_real[18];
    double sub_2 = frequency - tensor_real[18];
    double square = sub_1 * sub_2;
    double denominator = square + tensor_real[20];
    double carrier_amplitude = numerator / denominator;

    double carrier_frequency = frequency / total_mass_seconds;
    double carrier_fMs = carrier_frequency * tensor_real[7];
    double phiRD = tensor_real[9] * carrier_fMs;
    double coefficient = 1.5 * tensor_real[10];
    double power = std::pow(carrier_fMs, 2.0 / 3.0);
    phiRD = phiRD + coefficient * power;
    power = std::pow(carrier_fMs, -1.0);
    phiRD = phiRD - tensor_real[11] * power;
    power = std::pow(carrier_fMs, -3.0);
    phiRD = phiRD - tensor_real[12] * power;
    double scaled = (carrier_fMs - tensor_real[14]) / tensor_real[15];
    phiRD = phiRD + tensor_real[13] * std::atan(scaled);
    double carrier_phase = phiRD + tensor_real[16];
    carrier_phase = carrier_phase + tensor_real[17] * carrier_fMs;
    double reciprocal_eta = 1.0 / tensor_real[8];
    carrier_phase = reciprocal_eta * carrier_phase;
    double phase22 = carrier_phase + tensor_real[0] * frequency;
    phase22 = phase22 + tensor_real[1];
    double carrier_scale = carrier_amplitude * amp_norm;
    carrier_scale = carrier_scale * std::pow(frequency, -7.0 / 6.0);
    std::complex<double> h22(
        carrier_scale * std::cos(phase22),
        carrier_scale * std::sin(phase22)
    );

    double spheroidal_amplitude = 0.0;
    double frequency_power = 1.0;
    for (int index = 0; index != 4; ++index) {
        double term = auxiliary[index] * frequency_power;
        spheroidal_amplitude = spheroidal_amplitude + term;
        frequency_power = frequency_power * frequency;
    }

    double spheroidal_phase = tensor_real[6] + tensor_real[2] * frequency;
    spheroidal_phase = spheroidal_phase - tensor_real[4] / frequency;
    // Torch's scalar integer-power specialization uses this ordered tree.
    double cubic = frequency * frequency;
    cubic = cubic * frequency;
    double cubic_denominator = 3.0 * cubic;
    spheroidal_phase = spheroidal_phase - tensor_real[5] / cubic_denominator;
    double spheroidal_scaled = (frequency - f_ring_32) / f_damp_32;
    spheroidal_phase = spheroidal_phase
        + tensor_real[3] * std::atan(spheroidal_scaled);
    std::complex<double> h32(
        spheroidal_amplitude * std::cos(spheroidal_phase),
        spheroidal_amplitude * std::sin(spheroidal_phase)
    );

    mixing_322 = std::conj(mixing_322);
    mixing_323 = std::conj(mixing_323);
    // The qualified arm64 eager complex kernel contracts precisely these
    // leading multiply-adds. Explicit fma is still checked byte-for-byte on
    // the first request, so a platform with different semantics fails closed.
    double first_real = std::fma(
        mixing_322.real(), h22.real(),
        -(mixing_322.imag() * h22.imag())
    );
    double first_imag = std::fma(
        mixing_322.real(), h22.imag(),
        mixing_322.imag() * h22.real()
    );
    double second_real = std::fma(
        mixing_323.real(), h32.real(),
        -(mixing_323.imag() * h32.imag())
    );
    double second_imag = std::fma(
        mixing_323.real(), h32.imag(),
        mixing_323.imag() * h32.real()
    );
    double mixed_real = first_real + second_real;
    double mixed_imag = first_imag + second_imag;
    return std::hypot(mixed_real, mixed_imag);
}

torch::Tensor evaluate_packed(
    double point,
    const torch::Tensor& tensor_real,
    double total_mass_seconds,
    double f_ring_32,
    double f_damp_32,
    double amp_norm,
    std::complex<double> mixing_322,
    std::complex<double> mixing_323,
    const torch::Tensor& auxiliary
) {
    TORCH_CHECK(std::isfinite(point), "non-finite point");
    TORCH_CHECK(std::isfinite(total_mass_seconds), "non-finite total mass");
    TORCH_CHECK(std::isfinite(f_ring_32), "non-finite ringdown frequency");
    TORCH_CHECK(std::isfinite(f_damp_32), "non-finite damping frequency");
    TORCH_CHECK(std::isfinite(amp_norm), "non-finite amplitude norm");
    TORCH_CHECK(std::isfinite(mixing_322.real()), "non-finite mixing real");
    TORCH_CHECK(std::isfinite(mixing_322.imag()), "non-finite mixing imag");
    TORCH_CHECK(std::isfinite(mixing_323.real()), "non-finite mixing real");
    TORCH_CHECK(std::isfinite(mixing_323.imag()), "non-finite mixing imag");
    validate_packed(tensor_real, 22);
    validate_packed(auxiliary, 4);

    const double* tensor_values = tensor_real.data_ptr<double>();
    const double* auxiliary_values = auxiliary.data_ptr<double>();
    for (int index = 0; index != 22; ++index) {
        TORCH_CHECK(std::isfinite(tensor_values[index]), "non-finite scalar");
    }
    for (int index = 0; index != 4; ++index) {
        TORCH_CHECK(std::isfinite(auxiliary_values[index]), "non-finite auxiliary");
    }

    double values[5];
    values[0] = component(
        point, tensor_values, total_mass_seconds, f_ring_32, f_damp_32,
        amp_norm, mixing_322, mixing_323, auxiliary_values
    );
    const double step = 1.0e-9;
    values[1] = component(
        point + 2.0 * step, tensor_values, total_mass_seconds, f_ring_32,
        f_damp_32, amp_norm, mixing_322, mixing_323, auxiliary_values
    );
    values[2] = component(
        point + step, tensor_values, total_mass_seconds, f_ring_32,
        f_damp_32, amp_norm, mixing_322, mixing_323, auxiliary_values
    );
    values[3] = component(
        point - step, tensor_values, total_mass_seconds, f_ring_32,
        f_damp_32, amp_norm, mixing_322, mixing_323, auxiliary_values
    );
    values[4] = component(
        point - 2.0 * step, tensor_values, total_mass_seconds, f_ring_32,
        f_damp_32, amp_norm, mixing_322, mixing_323, auxiliary_values
    );

    double derivative = -values[1];
    derivative = derivative + 8.0 * values[2];
    derivative = derivative - 8.0 * values[3];
    derivative = derivative + values[4];
    derivative = derivative / (12.0 * step);
    TORCH_CHECK(std::isfinite(values[0]), "non-finite value");
    TORCH_CHECK(std::isfinite(derivative), "non-finite derivative");

    auto output = torch::empty({2}, tensor_real.options());
    double* output_values = output.data_ptr<double>();
    output_values[0] = values[0];
    output_values[1] = derivative;
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("evaluate_packed", &evaluate_packed);
}
