// Copyright (C) 2026 PyCBC contributors
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.

#include <torch/extension.h>

#include <c10/util/complex.h>

#include <cmath>

namespace {

void validate_inputs(
    const torch::Tensor& coefficients,
    const torch::Tensor& pseudo
) {
    TORCH_CHECK(
        coefficients.device().is_cpu() && pseudo.device().is_cpu(),
        "mode-44 native boundary is CPU-only"
    );
    TORCH_CHECK(
        coefficients.scalar_type() == torch::kComplexDouble,
        "mode-44 coefficients must be complex128"
    );
    TORCH_CHECK(
        pseudo.scalar_type() == torch::kFloat64,
        "mode-44 pseudo coefficients must be float64"
    );
    TORCH_CHECK(
        coefficients.dim() == 1 && coefficients.size(0) == 7,
        "mode-44 native boundary expects seven PN coefficients"
    );
    TORCH_CHECK(
        pseudo.dim() == 1 && pseudo.size(0) == 3,
        "mode-44 native boundary expects three pseudo coefficients"
    );
    TORCH_CHECK(
        coefficients.is_contiguous() && pseudo.is_contiguous(),
        "mode-44 native boundary expects contiguous inputs"
    );
    TORCH_CHECK(
        coefficients.storage_offset() == 0 && pseudo.storage_offset() == 0,
        "mode-44 native boundary expects zero-offset inputs"
    );
}

double component(
    double frequency,
    const c10::complex<double>* coefficients,
    const double* pseudo,
    double amp_norm,
    double fcut,
    double pn_dominant,
    double global_factor
) {
    // Keep each multiply and add separate and in the eager Horner order.
    double frequency_power = std::pow(frequency, 1.0 / 3.0);
    double real = coefficients[6].real();
    double imag = coefficients[6].imag();
    for (int index = 5; index >= 0; --index) {
        real = real * frequency_power;
        imag = imag * frequency_power;
        real = real + coefficients[index].real();
        imag = imag + coefficients[index].imag();
    }

    // The eager expression evaluates this leading power independently in
    // the PN and pseudo-PN terms, so deliberately compute it twice here.
    double leading_power = std::pow(frequency, -7.0 / 6.0);
    double pn = std::hypot(real, imag);
    pn = pn * global_factor;
    pn = pn * leading_power;
    pn = pn * amp_norm;

    double ratio = frequency / fcut;
    double pseudo_terms = pseudo[0] * std::pow(ratio, 7.0 / 3.0);
    pseudo_terms =
        pseudo_terms + pseudo[1] * std::pow(ratio, 8.0 / 3.0);
    // Torch's scalar integer-power kernel uses this ordered multiplication
    // tree for ratio**3.
    double ratio_cubed = ratio * ratio;
    ratio_cubed = ratio_cubed * ratio;
    pseudo_terms = pseudo_terms + pseudo[2] * ratio_cubed;
    double pseudo_value =
        pn_dominant * std::pow(frequency, -7.0 / 6.0);
    pseudo_value = pseudo_value * pseudo_terms;
    return pn + pseudo_value;
}

}  // namespace

torch::Tensor evaluate(
    double point,
    const torch::Tensor& coefficients,
    const torch::Tensor& pseudo,
    double amp_norm,
    double fcut,
    double pn_dominant,
    double global_factor
) {
    validate_inputs(coefficients, pseudo);
    TORCH_CHECK(std::isfinite(point), "mode-44 point must be finite");
    TORCH_CHECK(std::isfinite(amp_norm), "mode-44 amp norm must be finite");
    TORCH_CHECK(std::isfinite(fcut), "mode-44 cutoff must be finite");
    TORCH_CHECK(
        std::isfinite(pn_dominant),
        "mode-44 dominant amplitude must be finite"
    );
    TORCH_CHECK(
        std::isfinite(global_factor),
        "mode-44 global factor must be finite"
    );

    const auto* coefficient_values =
        coefficients.data_ptr<c10::complex<double>>();
    const auto* pseudo_values = pseudo.data_ptr<double>();
    for (int index = 0; index != 7; ++index) {
        TORCH_CHECK(
            std::isfinite(coefficient_values[index].real())
                && std::isfinite(coefficient_values[index].imag()),
            "mode-44 PN coefficients must be finite"
        );
    }
    for (int index = 0; index != 3; ++index) {
        TORCH_CHECK(
            std::isfinite(pseudo_values[index]),
            "mode-44 pseudo coefficients must be finite"
        );
    }

    const double step = 1.0e-9;
    double values[5];
    values[0] = component(
        point,
        coefficient_values,
        pseudo_values,
        amp_norm,
        fcut,
        pn_dominant,
        global_factor
    );
    values[1] = component(
        point + 2.0 * step,
        coefficient_values,
        pseudo_values,
        amp_norm,
        fcut,
        pn_dominant,
        global_factor
    );
    values[2] = component(
        point + step,
        coefficient_values,
        pseudo_values,
        amp_norm,
        fcut,
        pn_dominant,
        global_factor
    );
    values[3] = component(
        point - step,
        coefficient_values,
        pseudo_values,
        amp_norm,
        fcut,
        pn_dominant,
        global_factor
    );
    values[4] = component(
        point - 2.0 * step,
        coefficient_values,
        pseudo_values,
        amp_norm,
        fcut,
        pn_dominant,
        global_factor
    );

    double derivative = -values[1];
    derivative = derivative + 8.0 * values[2];
    derivative = derivative - 8.0 * values[3];
    derivative = derivative + values[4];
    derivative = derivative / (12.0 * step);

    auto output = torch::empty({2}, pseudo.options());
    double* output_values = output.data_ptr<double>();
    output_values[0] = values[0];
    output_values[1] = derivative;
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("evaluate", &evaluate);
}
