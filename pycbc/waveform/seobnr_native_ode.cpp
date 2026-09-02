// Copyright (C) 2026 PyCBC contributors
//
// This program is free software; you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3 of the License, or (at your
// option) any later version.

#include <torch/extension.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

constexpr int64_t STACK_DIM_LIMIT = 32;

// GSL RKF45 (Fehlberg 4(5)) Butcher Tableau Coefficients
constexpr double B1 = 902880.0 / 7618050.0;
constexpr double B3 = 3953664.0 / 7618050.0;
constexpr double B4 = 3855735.0 / 7618050.0;
constexpr double B5 = -1371249.0 / 7618050.0;
constexpr double B6 = 277020.0 / 7618050.0;

constexpr double E1 = 1.0 / 360.0;
constexpr double E3 = -128.0 / 4275.0;
constexpr double E4 = -2197.0 / 75240.0;
constexpr double E5 = 1.0 / 50.0;
constexpr double E6 = 2.0 / 55.0;

constexpr double A21 = 1.0 / 4.0;

constexpr double A31 = 3.0 / 32.0;
constexpr double A32 = 9.0 / 32.0;

constexpr double A41 = 1932.0 / 2197.0;
constexpr double A42 = -7200.0 / 2197.0;
constexpr double A43 = 7296.0 / 2197.0;

constexpr double A51 = 8341.0 / 4104.0;
constexpr double A52 = -32832.0 / 4104.0;
constexpr double A53 = 29440.0 / 4104.0;
constexpr double A54 = -845.0 / 4104.0;

constexpr double A61 = -6080.0 / 20520.0;
constexpr double A62 = 41040.0 / 20520.0;
constexpr double A63 = -28352.0 / 20520.0;
constexpr double A64 = 9295.0 / 20520.0;
constexpr double A65 = -5643.0 / 20520.0;

enum StopMode : int {
    STOP_MODE_NONE = 0,
    STOP_MODE_CARTESIAN = 1,
    STOP_MODE_ALIGNED_OMEGA_PEAK = 2,
    STOP_MODE_ALIGNED_HIS = 3,
};

inline void validate_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat64, name, " must be float64");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

// Call Python RHS function with zero tensor copying via from_blob views
inline bool eval_rhs_py(
    const py::function& rhs_fn,
    double t_val,
    const double* y_ptr,
    int64_t dim,
    double* out_dydt,
    const torch::TensorOptions& options
) {
    auto t_tensor = torch::tensor(t_val, options);
    auto y_tensor = torch::from_blob(const_cast<double*>(y_ptr), {dim}, options);
    py::object res_obj = rhs_fn(t_tensor, y_tensor);
    auto res_tensor = res_obj.cast<torch::Tensor>();
    if (!res_tensor.is_contiguous() || res_tensor.scalar_type() != torch::kFloat64) {
        res_tensor = res_tensor.to(torch::kCPU, torch::kFloat64).contiguous();
    }
    const double* data = res_tensor.data_ptr<double>();
    for (int64_t i = 0; i < dim; ++i) {
        out_dydt[i] = data[i];
    }
    return true;
}

// Fast single RKF45 trial step on raw double buffers
bool rkf45_step_kernel(
    const std::function<void(double, const double*, double*)>& rhs_fn,
    double t,
    const double* y,
    double h,
    int64_t dim,
    const double* k1_in,
    double* y_next,
    double* err,
    double* k1_out,
    double* k6_out,
    double* dydt_out,
    bool compute_dydt_out
) {
    double stack_mem[7 * STACK_DIM_LIMIT];
    std::vector<double> heap_mem;
    double* buf = stack_mem;
    if (dim > STACK_DIM_LIMIT) {
        heap_mem.resize(7 * dim);
        buf = heap_mem.data();
    }

    double* k1 = buf;
    double* k2 = buf + dim;
    double* k3 = buf + 2 * dim;
    double* k4 = buf + 3 * dim;
    double* k5 = buf + 4 * dim;
    double* k6 = buf + 5 * dim;
    double* y_stage = buf + 6 * dim;

    // Stage 1
    if (k1_in != nullptr) {
        for (int64_t i = 0; i < dim; ++i) k1[i] = k1_in[i];
    } else {
        rhs_fn(t, y, k1);
    }
    if (k1_out != nullptr) {
        for (int64_t i = 0; i < dim; ++i) k1_out[i] = k1[i];
    }

    // Stage 2: t + 1/4 h, y + 1/4 h k1
    for (int64_t i = 0; i < dim; ++i) {
        y_stage[i] = y[i] + (A21 * h) * k1[i];
    }
    rhs_fn(t + A21 * h, y_stage, k2);

    // Stage 3: t + 3/8 h, y + h (3/32 k1 + 9/32 k2)
    for (int64_t i = 0; i < dim; ++i) {
        y_stage[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
    }
    rhs_fn(t + 3.0 / 8.0 * h, y_stage, k3);

    // Stage 4: t + 12/13 h, y + h (1932/2197 k1 - 7200/2197 k2 + 7296/2197 k3)
    for (int64_t i = 0; i < dim; ++i) {
        y_stage[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
    }
    rhs_fn(t + 12.0 / 13.0 * h, y_stage, k4);

    // Stage 5: t + h, y + h (8341/4104 k1 - 32832/4104 k2 + 29440/4104 k3 - 845/4104 k4)
    for (int64_t i = 0; i < dim; ++i) {
        y_stage[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
    }
    rhs_fn(t + h, y_stage, k5);

    // Stage 6: t + 1/2 h, y + h (-6080/20520 k1 + 41040/20520 k2 - 28352/20520 k3 + 9295/20520 k4 - 5643/20520 k5)
    for (int64_t i = 0; i < dim; ++i) {
        y_stage[i] = y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
    }
    rhs_fn(t + 0.5 * h, y_stage, k6);
    if (k6_out != nullptr) {
        for (int64_t i = 0; i < dim; ++i) k6_out[i] = k6[i];
    }

    // 4th order candidate update
    for (int64_t i = 0; i < dim; ++i) {
        double deriv = B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i];
        y_next[i] = y[i] + h * deriv;
    }

    // Truncation error estimate (5th - 4th difference)
    for (int64_t i = 0; i < dim; ++i) {
        err[i] = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i]);
    }

    // Optional derivative at proposed state
    if (compute_dydt_out && dydt_out != nullptr) {
        rhs_fn(t + h, y_next, dydt_out);
    }
    return true;
}

// Evaluate Cartesian LAL stop quantities and conditions
bool check_cartesian_stop(
    const double* y,
    const double* dy,
    double& prev_omega,
    double& prev_dr,
    int& omega_peaked,
    int& stop_reason
) {
    const double rx = y[0], ry = y[1], rz = y[2];
    const double px = y[3], py = y[4], pz = y[5];
    const double rdotx = dy[0], rdoty = dy[1], rdotz = dy[2];
    const double pdotx = dy[3], pdoty = dy[4], pdotz = dy[5];

    double r2 = rx * rx + ry * ry + rz * rz;
    if (r2 < 1.0e-24) r2 = 1.0e-24;
    double r = std::sqrt(r2);
    double r_clamp = (r < 1.0e-12) ? 1.0e-12 : r;

    // Cross product r x rdot
    double cx = ry * rdotz - rz * rdoty;
    double cy = rz * rdotx - rx * rdotz;
    double cz = rx * rdoty - ry * rdotx;
    double omega = std::sqrt(cx * cx + cy * cy + cz * cz) / r2;

    double p_dot_r = (px * rx + py * ry + pz * rz) / r_clamp;
    double drdt = (rdotx * rx + rdoty * ry + rdotz * rz) / r_clamp;

    double pr_dot = -(px * rx + py * ry + pz * rz) * drdt / r2
                    + (pdotx * rx + pdoty * ry + pdotz * rz) / r_clamp
                    + (rdotx * px + rdoty * py + rdotz * pz) / r_clamp;

    double p_norm = std::sqrt(px * px + py * py + pz * pz);
    bool dpdt_large = (std::fabs(pdotx) > 10.0 || std::fabs(pdoty) > 10.0 || std::fabs(pdotz) > 10.0);
    bool pphi_large = (pz > 10.0);

    if (r2 < 16.0 && (p_dot_r >= 0.0 || drdt >= 0.0)) {
        stop_reason = (p_dot_r >= 0.0) ? 0 : 1;
        return true;
    }
    if (r2 < 4.0 && pr_dot > 0.0) {
        stop_reason = 2;
        return true;
    }
    if (r2 < 16.0 && (p_norm > 10.0 || p_norm < 1.0e-10)) {
        stop_reason = (p_norm > 10.0) ? 3 : 4;
        return true;
    }
    if (r2 < 16.0 && omega < prev_omega) {
        omega_peaked = 1;
    }
    if (r2 < 4.0 && omega_peaked == 1 && omega > prev_omega) {
        stop_reason = 5;
        return true;
    }
    if ((r2 < 16.0 && omega < 0.04) || (r2 < 4.0 && omega < 0.14 && omega_peaked == 1)) {
        stop_reason = 6;
        return true;
    }
    if (r2 < 16.0 && omega > 1.0) {
        stop_reason = 7;
        return true;
    }
    prev_omega = omega;
    if (r2 < 25.0 && dpdt_large) {
        stop_reason = 8;
        return true;
    }
    if (r2 < 16.0 && pphi_large) {
        stop_reason = 9;
        return true;
    }
    if (r2 < 9.0 && drdt > prev_dr) {
        prev_dr = drdt;
        stop_reason = 10;
        return true;
    }
    prev_dr = drdt;
    return false;
}

}  // namespace

// Single RKF45 step exposed to Python
py::tuple rkf45_step_native(
    const py::function& rhs_fn,
    double t,
    const torch::Tensor& y_tensor,
    double h,
    const py::object& k1_obj,
    bool compute_final_derivative
) {
    validate_tensor(y_tensor, "y");
    int64_t dim = y_tensor.numel();
    const double* y = y_tensor.data_ptr<double>();
    auto options = y_tensor.options();

    std::vector<double> k1_buf;
    const double* k1_ptr = nullptr;
    if (!k1_obj.is_none()) {
        auto k1_tensor = k1_obj.cast<torch::Tensor>();
        validate_tensor(k1_tensor, "k1");
        k1_ptr = k1_tensor.data_ptr<double>();
    }

    auto rhs_wrapper = [&](double t_val, const double* y_val, double* dydt_val) {
        eval_rhs_py(rhs_fn, t_val, y_val, dim, dydt_val, options);
    };

    auto y_next_t = torch::empty({dim}, options);
    auto err_t = torch::empty({dim}, options);
    auto k1_out_t = torch::empty({dim}, options);
    auto k6_out_t = torch::empty({dim}, options);
    torch::Tensor dydt_out_t;
    double* dydt_out_ptr = nullptr;
    if (compute_final_derivative) {
        dydt_out_t = torch::empty({dim}, options);
        dydt_out_ptr = dydt_out_t.data_ptr<double>();
    }

    rkf45_step_kernel(
        rhs_wrapper,
        t,
        y,
        h,
        dim,
        k1_ptr,
        y_next_t.data_ptr<double>(),
        err_t.data_ptr<double>(),
        k1_out_t.data_ptr<double>(),
        k6_out_t.data_ptr<double>(),
        dydt_out_ptr,
        compute_final_derivative
    );

    return py::make_tuple(
        y_next_t,
        err_t,
        k1_out_t,
        k6_out_t,
        compute_final_derivative ? py::cast(dydt_out_t) : py::none()
    );
}

// Adaptive RKF45 trajectory integrator in native C++
py::tuple integrate_native(
    const py::function& rhs_fn,
    const torch::Tensor& y0_tensor,
    double t0,
    double t1,
    double h0,
    double rtol,
    double atol,
    int64_t max_steps,
    double h_min,
    double h_max,
    const py::object& stop_fn_obj,
    int stop_mode,
    double initial_prev_omega,
    double initial_prev_dr,
    int initial_omega_peaked,
    bool return_diagnostics
) {
    validate_tensor(y0_tensor, "y0");
    int64_t dim = y0_tensor.numel();
    auto options = y0_tensor.options();

    auto rhs_wrapper = [&](double t_val, const double* y_val, double* dydt_val) {
        eval_rhs_py(rhs_fn, t_val, y_val, dim, dydt_val, options);
    };

    double t = t0;
    double h = h0;
    double prev_omega = initial_prev_omega;
    double prev_dr = initial_prev_dr;
    int omega_peaked = initial_omega_peaked;
    int stop_reason_int = -1;
    std::string stop_reason_str = "";

    const double finfo_tiny = std::numeric_limits<double>::min();
    const int max_retries = 1;
    int retries_left = max_retries;
    int64_t accepted_steps = 0;
    int64_t rejected_steps = 0;
    int64_t attempted_steps = 0;
    bool hit_max_steps = false;
    bool stopped_or_finished = false;

    std::vector<double> t_buf;
    std::vector<double> y_buf;
    t_buf.reserve(std::min<int64_t>(max_steps, 4096));
    y_buf.reserve(std::min<int64_t>(max_steps * dim, 4096 * dim));

    double current_y[STACK_DIM_LIMIT];
    double dydt_in[STACK_DIM_LIMIT];
    double dydt_out[STACK_DIM_LIMIT];
    double y_trial[STACK_DIM_LIMIT];
    double err[STACK_DIM_LIMIT];

    std::vector<double> current_y_heap, dydt_in_heap, dydt_out_heap, y_trial_heap, err_heap;
    double* y_curr = current_y;
    double* d_in = dydt_in;
    double* d_out = dydt_out;
    double* y_try = y_trial;
    double* e_try = err;

    if (dim > STACK_DIM_LIMIT) {
        current_y_heap.resize(dim);
        dydt_in_heap.resize(dim);
        dydt_out_heap.resize(dim);
        y_trial_heap.resize(dim);
        err_heap.resize(dim);
        y_curr = current_y_heap.data();
        d_in = dydt_in_heap.data();
        d_out = dydt_out_heap.data();
        y_try = y_trial_heap.data();
        e_try = err_heap.data();
    }

    const double* y0_data = y0_tensor.data_ptr<double>();
    for (int64_t i = 0; i < dim; ++i) {
        y_curr[i] = y0_data[i];
    }

    // Initial derivative evaluation before the loop
    rhs_wrapper(t, y_curr, d_in);

    bool has_custom_stop_fn = !stop_fn_obj.is_none();
    py::function custom_stop_fn;
    if (has_custom_stop_fn) {
        custom_stop_fn = stop_fn_obj.cast<py::function>();
    }

    for (int64_t step = 0; step < max_steps; ++step) {
        attempted_steps = step + 1;
        if ((t + h) > t1) {
            h = t1 - t;
        }
        double h_trial = h;

        bool trial_failed = false;
        try {
            rkf45_step_kernel(
                rhs_wrapper,
                t,
                y_curr,
                h,
                dim,
                d_in,
                y_try,
                e_try,
                nullptr,
                nullptr,
                d_out,
                true
            );
            for (int64_t i = 0; i < dim; ++i) {
                if (!std::isfinite(y_try[i]) || !std::isfinite(d_out[i]) || !std::isfinite(e_try[i])) {
                    trial_failed = true;
                    break;
                }
            }
        } catch (const std::exception&) {
            trial_failed = true;
        }

        if (trial_failed) {
            rejected_steps++;
            retries_left--;
            h = h / 10.0;
            if (retries_left < 0 || (h_min > 0.0 && h < h_min)) {
                if (accepted_steps > 0) {
                    break;
                }
                throw std::runtime_error("NaN/inf encountered in ODE trial step");
            }
            continue;
        }

        // Error ratio & GSL step adjustment
        double worst_err_norm = 0.0;
        for (int64_t i = 0; i < dim; ++i) {
            double scale = atol + rtol * std::fabs(y_try[i]);
            double ratio_i = std::fabs(e_try[i]) / (scale > 0.0 ? scale : finfo_tiny);
            if (!std::isfinite(ratio_i)) ratio_i = std::numeric_limits<double>::infinity();
            if (ratio_i > worst_err_norm) worst_err_norm = ratio_i;
        }

        double h_next = h;
        bool accepted = true;

        if (worst_err_norm > 1.1 && h > finfo_tiny) {
            double ratio = 0.9 * std::pow(std::max(worst_err_norm, finfo_tiny), -1.0 / 5.0);
            if (ratio < 0.2) ratio = 0.2;
            double h_new = h * ratio;
            if (h_min > 0.0 && h_new < h_min) h_new = h_min;
            if (h_new < h) {
                h = h_new;
                accepted = false;
                rejected_steps++;
                continue;
            }
            h_next = h_new;
        } else if (worst_err_norm < 0.5) {
            double ratio = 0.9 * std::pow(std::max(worst_err_norm, finfo_tiny), -1.0 / 6.0);
            if (ratio < 1.0) ratio = 1.0;
            if (ratio > 5.0) ratio = 5.0;
            h_next = h * ratio;
        }

        if (accepted) {
            t += h;
            for (int64_t i = 0; i < dim; ++i) {
                y_curr[i] = y_try[i];
                d_in[i] = d_out[i];
            }

            t_buf.push_back(t);
            for (int64_t i = 0; i < dim; ++i) {
                y_buf.push_back(y_curr[i]);
            }
            accepted_steps++;

            bool should_stop = false;

            // 1. Built-in fast C++ stop condition
            if (stop_mode == STOP_MODE_CARTESIAN && dim >= 14) {
                if (check_cartesian_stop(y_curr, d_in, prev_omega, prev_dr, omega_peaked, stop_reason_int)) {
                    should_stop = true;
                }
            } else if (stop_mode == STOP_MODE_ALIGNED_HIS) {
                double r_val = y_curr[0];
                double omega_val = d_in[1];
                if (r_val < 6.0 && omega_val < prev_omega) omega_peaked++;
                if (d_in[2] >= 0.0 || omega_peaked == 5) {
                    stop_reason_str = "aligned_his";
                    should_stop = true;
                }
                prev_omega = omega_val;
            } else if (stop_mode == STOP_MODE_ALIGNED_OMEGA_PEAK) {
                double r_val = y_curr[0];
                double omega_val = d_in[1];
                if (r_val < 6.0 && (omega_val < prev_omega || d_in[2] >= 0.0)) {
                    stop_reason_str = "aligned_omega_peak";
                    should_stop = true;
                }
                prev_omega = omega_val;
            }

            // 2. Custom Python stop callback if provided and no built-in stop mode
            if (!should_stop && stop_mode == STOP_MODE_NONE && has_custom_stop_fn) {
                try {
                    auto t_t = torch::tensor(t, options);
                    auto y_t = torch::from_blob(y_curr, {dim}, options);
                    auto dy_t = torch::from_blob(d_in, {dim}, options);
                    py::object res;
                    try {
                        res = custom_stop_fn(t_t, y_t, dy_t);
                    } catch (py::error_already_set& err_set) {
                        if (err_set.matches(PyExc_TypeError)) {
                            err_set.restore();
                            PyErr_Clear();
                            res = custom_stop_fn(t_t, y_t);
                        } else {
                            throw;
                        }
                    }
                    if (py::isinstance<torch::Tensor>(res)) {
                        auto res_t = res.cast<torch::Tensor>();
                        should_stop = res_t.all().item<bool>();
                    } else {
                        should_stop = py::bool_(res);
                    }
                } catch (const std::exception&) {
                    // if stop function fails, continue
                }
            }

            if (should_stop || t >= t1) {
                stopped_or_finished = true;
                break;
            }
            retries_left = max_retries;
        }

        h = h_next;
        if (h_max > 0.0 && h > h_max) h = h_max;
    }

    if (!stopped_or_finished && max_steps > 0) {
        hit_max_steps = true;
    }

    // Allocate output torch::Tensors
    auto t_out = torch::empty({accepted_steps}, options);
    auto y_out = torch::empty({accepted_steps, dim}, options);
    if (accepted_steps > 0) {
        std::copy(t_buf.begin(), t_buf.end(), t_out.data_ptr<double>());
        std::copy(y_buf.begin(), y_buf.end(), y_out.data_ptr<double>());
    }

    py::tuple traj = py::make_tuple(t_out, y_out);
    if (!return_diagnostics) {
        return traj;
    }

    py::dict diag;
    diag["accepted_steps"] = accepted_steps;
    diag["rejected_steps"] = rejected_steps;
    diag["attempted_steps"] = attempted_steps;
    diag["max_steps"] = max_steps;
    diag["hit_max_steps"] = hit_max_steps;
    diag["t_end"] = t;
    diag["h_final"] = h;
    diag["prev_omega"] = prev_omega;
    diag["prev_dr"] = prev_dr;
    diag["omega_peaked"] = omega_peaked;
    if (stop_reason_int >= 0) {
        diag["stop_reason"] = stop_reason_int;
    } else if (!stop_reason_str.empty()) {
        diag["stop_reason"] = stop_reason_str;
    } else {
        diag["stop_reason"] = py::none();
    }

    return py::make_tuple(traj, diag);
}

// Pure C++ RHS benchmark on 14D nonlinear coupled system
void benchmark_14d_rhs(double t, const double* y, double* dydt) {
    for (int64_t i = 0; i < 14; i += 2) {
        double freq = 1.0 + 0.2 * i;
        dydt[i] = y[i + 1];
        dydt[i + 1] = -(freq * freq) * y[i] - 0.05 * std::sin(y[i]);
    }
}

py::tuple integrate_cpp_benchmark(
    const torch::Tensor& y0_tensor,
    double t0,
    double t1,
    double h0,
    double rtol,
    double atol,
    int64_t max_steps
) {
    validate_tensor(y0_tensor, "y0");
    int64_t dim = y0_tensor.numel();
    auto options = y0_tensor.options();

    double t = t0;
    double h = h0;
    const double finfo_tiny = std::numeric_limits<double>::min();
    int64_t accepted_steps = 0;
    int64_t rejected_steps = 0;
    int64_t attempted_steps = 0;

    std::vector<double> t_buf;
    std::vector<double> y_buf;
    t_buf.reserve(std::min<int64_t>(max_steps, 4096));
    y_buf.reserve(std::min<int64_t>(max_steps * dim, 4096 * dim));

    double y_curr[14];
    double d_in[14];
    double d_out[14];
    double y_try[14];
    double e_try[14];

    const double* y0_data = y0_tensor.data_ptr<double>();
    for (int64_t i = 0; i < 14; ++i) y_curr[i] = y0_data[i];
    benchmark_14d_rhs(t, y_curr, d_in);

    for (int64_t step = 0; step < max_steps; ++step) {
        attempted_steps = step + 1;
        if ((t + h) > t1) h = t1 - t;

        rkf45_step_kernel(
            benchmark_14d_rhs,
            t, y_curr, h, 14, d_in, y_try, e_try, nullptr, nullptr, d_out, true
        );

        double worst_err_norm = 0.0;
        for (int64_t i = 0; i < 14; ++i) {
            double scale = atol + rtol * std::fabs(y_try[i]);
            double ratio_i = std::fabs(e_try[i]) / (scale > 0.0 ? scale : finfo_tiny);
            if (ratio_i > worst_err_norm) worst_err_norm = ratio_i;
        }

        double h_next = h;
        if (worst_err_norm > 1.1 && h > finfo_tiny) {
            double ratio = 0.9 * std::pow(std::max(worst_err_norm, finfo_tiny), -0.2);
            if (ratio < 0.2) ratio = 0.2;
            h = h * ratio;
            rejected_steps++;
            continue;
        } else if (worst_err_norm < 0.5) {
            double ratio = 0.9 * std::pow(std::max(worst_err_norm, finfo_tiny), -1.0 / 6.0);
            if (ratio < 1.0) ratio = 1.0;
            if (ratio > 5.0) ratio = 5.0;
            h_next = h * ratio;
        }

        t += h;
        for (int64_t i = 0; i < 14; ++i) {
            y_curr[i] = y_try[i];
            d_in[i] = d_out[i];
        }
        t_buf.push_back(t);
        for (int64_t i = 0; i < 14; ++i) y_buf.push_back(y_curr[i]);
        accepted_steps++;

        if (t >= t1) break;
        h = h_next;
    }

    auto t_out = torch::empty({accepted_steps}, options);
    auto y_out = torch::empty({accepted_steps, dim}, options);
    std::copy(t_buf.begin(), t_buf.end(), t_out.data_ptr<double>());
    std::copy(y_buf.begin(), y_buf.end(), y_out.data_ptr<double>());

    py::dict diag;
    diag["accepted_steps"] = accepted_steps;
    diag["rejected_steps"] = rejected_steps;
    diag["attempted_steps"] = attempted_steps;
    return py::make_tuple(py::make_tuple(t_out, y_out), diag);
}

// ---------------------------------------------------------------------------
// Native C++ Natural Cubic Spline Solver & Multi-Derivative Evaluator
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> natural_spline_coeffs_native(
    const torch::Tensor& x_tensor,
    const torch::Tensor& y_tensor
) {
    validate_tensor(x_tensor, "x");
    validate_tensor(y_tensor, "y");
    int64_t n = x_tensor.numel();
    TORCH_CHECK(y_tensor.numel() == n, "x and y must have same length");
    TORCH_CHECK(n >= 2, "need at least 2 points");

    auto options = x_tensor.options();
    auto b_tensor = torch::empty({n - 1}, options);
    auto c_tensor = torch::empty({n - 1}, options);
    auto d_tensor = torch::empty({n - 1}, options);

    const double* x = x_tensor.data_ptr<double>();
    const double* y = y_tensor.data_ptr<double>();
    double* b = b_tensor.data_ptr<double>();
    double* c = c_tensor.data_ptr<double>();
    double* d = d_tensor.data_ptr<double>();

    std::vector<double> h(n - 1);
    std::vector<double> alpha(n);
    std::vector<double> l(n);
    std::vector<double> mu(n);
    std::vector<double> z(n);
    std::vector<double> c_full(n);

    for (int64_t i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
    }
    for (int64_t i = 1; i < n - 1; ++i) {
        alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1]);
    }
    l[0] = 1.0;
    mu[0] = 0.0;
    z[0] = 0.0;
    for (int64_t i = 1; i < n - 1; ++i) {
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    l[n - 1] = 1.0;
    z[n - 1] = 0.0;
    c_full[n - 1] = 0.0;

    for (int64_t j = n - 2; j >= 0; --j) {
        c_full[j] = z[j] - mu[j] * c_full[j + 1];
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c_full[j + 1] + 2.0 * c_full[j]) / 3.0;
        d[j] = (c_full[j + 1] - c_full[j]) / (3.0 * h[j]);
        c[j] = c_full[j];
    }

    return std::make_tuple(b_tensor, c_tensor, d_tensor);
}

std::tuple<double, double, double> natural_spline_eval_derivs_native(
    const torch::Tensor& x_tensor,
    const torch::Tensor& y_tensor,
    const torch::Tensor& b_tensor,
    const torch::Tensor& c_tensor,
    const torch::Tensor& d_tensor,
    double xq
) {
    int64_t n = x_tensor.numel();
    const double* x = x_tensor.data_ptr<double>();
    const double* y = y_tensor.data_ptr<double>();
    const double* b = b_tensor.data_ptr<double>();
    const double* c = c_tensor.data_ptr<double>();
    const double* d = d_tensor.data_ptr<double>();

    int64_t low = 0, high = n - 1;
    while (low < high) {
        int64_t mid = low + (high - low) / 2;
        if (x[mid] < xq) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    int64_t idx = low - 1;
    if (idx < 0) idx = 0;
    if (idx > n - 2) idx = n - 2;

    double dx = xq - x[idx];
    double val = y[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx]));
    double d1 = b[idx] + 2.0 * c[idx] * dx + 3.0 * d[idx] * dx * dx;
    double d2 = 2.0 * c[idx] + 6.0 * d[idx] * dx;

    return std::make_tuple(val, d1, d2);
}

torch::Tensor natural_spline_interpolate_native(
    const torch::Tensor& query_tensor,
    const torch::Tensor& x_tensor,
    const torch::Tensor& y_tensor,
    int64_t derivative,
    bool extrapolate
) {
    validate_tensor(query_tensor, "query");
    validate_tensor(x_tensor, "x");
    TORCH_CHECK(y_tensor.device().is_cpu(), "y must be a CPU tensor");
    TORCH_CHECK(y_tensor.scalar_type() == torch::kFloat64, "y must be float64");
    TORCH_CHECK(y_tensor.is_contiguous(), "y must be contiguous");

    int64_t n = x_tensor.numel();
    int64_t nq = query_tensor.numel();
    TORCH_CHECK(n >= 2, "need at least 2 points");

    const double* x = x_tensor.data_ptr<double>();
    const double* xq = query_tensor.data_ptr<double>();
    const double x_min = x[0];
    const double x_max = x[n - 1];

    std::vector<double> h(n - 1);
    std::vector<double> l(n);
    std::vector<double> mu(n);
    for (int64_t i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
    }
    l[0] = 1.0;
    mu[0] = 0.0;
    for (int64_t i = 1; i < n - 1; ++i) {
        l[i] = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
    }
    l[n - 1] = 1.0;

    std::vector<int64_t> query_idx(nq);
    std::vector<double> query_dx(nq);
    std::vector<bool> query_valid(nq, true);
    for (int64_t i = 0; i < nq; ++i) {
        double q = xq[i];
        if (!extrapolate && (q < x_min || q > x_max)) {
            query_valid[i] = false;
            continue;
        }
        int64_t low = 0, high = n - 1;
        while (low < high) {
            int64_t mid = low + (high - low) / 2;
            if (x[mid] < q) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        int64_t idx = low - 1;
        if (idx < 0) idx = 0;
        if (idx > n - 2) idx = n - 2;
        query_idx[i] = idx;
        query_dx[i] = q - x[idx];
    }

    if (y_tensor.dim() == 1) {
        TORCH_CHECK(y_tensor.numel() == n, "x and y must have same length");
        auto out_tensor = torch::empty_like(query_tensor);
        const double* y = y_tensor.data_ptr<double>();
        double* out = out_tensor.data_ptr<double>();

        std::vector<double> alpha(n), z(n), c_full(n), b(n - 1), d(n - 1);
        for (int64_t i = 1; i < n - 1; ++i) {
            alpha[i] = (3.0 / h[i]) * (y[i + 1] - y[i]) - (3.0 / h[i - 1]) * (y[i] - y[i - 1]);
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        z[0] = 0.0;
        z[n - 1] = 0.0;
        c_full[n - 1] = 0.0;

        for (int64_t j = n - 2; j >= 0; --j) {
            c_full[j] = z[j] - mu[j] * c_full[j + 1];
            b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c_full[j + 1] + 2.0 * c_full[j]) / 3.0;
            d[j] = (c_full[j + 1] - c_full[j]) / (3.0 * h[j]);
        }

        for (int64_t i = 0; i < nq; ++i) {
            if (!query_valid[i]) {
                out[i] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }
            int64_t idx = query_idx[i];
            double dx = query_dx[i];
            if (derivative == 0) {
                out[i] = y[idx] + dx * (b[idx] + dx * (c_full[idx] + dx * d[idx]));
            } else if (derivative == 1) {
                out[i] = b[idx] + 2.0 * c_full[idx] * dx + 3.0 * d[idx] * dx * dx;
            } else if (derivative == 2) {
                out[i] = 2.0 * c_full[idx] + 6.0 * d[idx] * dx;
            } else if (derivative == 3) {
                out[i] = 6.0 * d[idx];
            }
        }
        return out_tensor;
    } else if (y_tensor.dim() == 2) {
        TORCH_CHECK(y_tensor.size(0) == n, "x and y.size(0) must have same length");
        int64_t cols = y_tensor.size(1);
        auto out_tensor = torch::empty({nq, cols}, query_tensor.options());
        const double* y_base = y_tensor.data_ptr<double>();
        double* out_base = out_tensor.data_ptr<double>();

        std::vector<double> alpha(n), z(n), c_full(n), b(n - 1), d(n - 1);
        for (int64_t col = 0; col < cols; ++col) {
            auto get_y = [&](int64_t row) { return y_base[row * cols + col]; };
            for (int64_t i = 1; i < n - 1; ++i) {
                alpha[i] = (3.0 / h[i]) * (get_y(i + 1) - get_y(i)) - (3.0 / h[i - 1]) * (get_y(i) - get_y(i - 1));
                z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
            }
            z[0] = 0.0;
            z[n - 1] = 0.0;
            c_full[n - 1] = 0.0;

            for (int64_t j = n - 2; j >= 0; --j) {
                c_full[j] = z[j] - mu[j] * c_full[j + 1];
                b[j] = (get_y(j + 1) - get_y(j)) / h[j] - h[j] * (c_full[j + 1] + 2.0 * c_full[j]) / 3.0;
                d[j] = (c_full[j + 1] - c_full[j]) / (3.0 * h[j]);
            }

            for (int64_t i = 0; i < nq; ++i) {
                if (!query_valid[i]) {
                    out_base[i * cols + col] = std::numeric_limits<double>::quiet_NaN();
                    continue;
                }
                int64_t idx = query_idx[i];
                double dx = query_dx[i];
                if (derivative == 0) {
                    out_base[i * cols + col] = get_y(idx) + dx * (b[idx] + dx * (c_full[idx] + dx * d[idx]));
                } else if (derivative == 1) {
                    out_base[i * cols + col] = b[idx] + 2.0 * c_full[idx] * dx + 3.0 * d[idx] * dx * dx;
                } else if (derivative == 2) {
                    out_base[i * cols + col] = 2.0 * c_full[idx] + 6.0 * d[idx] * dx;
                } else if (derivative == 3) {
                    out_base[i * cols + col] = 6.0 * d[idx];
                }
            }
        }
        return out_tensor;
    } else {
        TORCH_CHECK(false, "y_tensor must be 1D or 2D");
    }
}


// GSL 5-point central derivative
inline double gsl_deriv_central_c(const std::function<double(double)>& f, double x, double h) {
    constexpr double eps_f = 2.2204460492503131e-16;
    double fm1 = f(x - h);
    double fp1 = f(x + h);
    double fmh = f(x - h * 0.5);
    double fph = f(x + h * 0.5);
    double r3 = 0.5 * (fp1 - fm1);
    double r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;
    double e3 = (std::fabs(fp1) + std::fabs(fm1)) * eps_f;
    double e5 = 2.0 * (std::fabs(fph) + std::fabs(fmh)) * eps_f + e3;
    double dy = std::max(std::fabs(r3 / h), std::fabs(r5 / h)) * (std::fabs(x) / h) * eps_f;
    double result = r5 / h;
    double trunc = std::fabs((r5 - r3) / h);
    double round_err = std::fabs(e5 / h) + dy;
    double error = round_err + trunc;
    if (round_err < trunc && round_err > 0.0 && trunc > 0.0) {
        double h_opt = h * std::cbrt(round_err / (2.0 * trunc));
        double fm1_opt = f(x - h_opt);
        double fp1_opt = f(x + h_opt);
        double fmh_opt = f(x - h_opt * 0.5);
        double fph_opt = f(x + h_opt * 0.5);
        double r3_opt = 0.5 * (fp1_opt - fm1_opt);
        double r5_opt = (4.0 / 3.0) * (fph_opt - fmh_opt) - (1.0 / 3.0) * r3_opt;
        double e3_opt = (std::fabs(fp1_opt) + std::fabs(fm1_opt)) * eps_f;
        double e5_opt = 2.0 * (std::fabs(fph_opt) + std::fabs(fmh_opt)) * eps_f + e3_opt;
        double dy_opt = std::max(std::fabs(r3_opt / h_opt), std::fabs(r5_opt / h_opt)) * (std::fabs(x) / h_opt) * eps_f;
        double r_opt = r5_opt / h_opt;
        double trunc_opt = std::fabs((r5_opt - r3_opt) / h_opt);
        double round_opt = std::fabs(e5_opt / h_opt) + dy_opt;
        double error_opt = round_opt + trunc_opt;
        if (error_opt < error && std::fabs(r_opt - result) < 4.0 * error) {
            result = r_opt;
        }
    }
    return result;
}

inline double gsl_deriv_central_with_err_c(const std::function<double(double)>& f, double x, double h, double* abs_err) {
    constexpr double eps_f = 2.2204460492503131e-16;
    double fm1 = f(x - h);
    double fp1 = f(x + h);
    double fmh = f(x - h * 0.5);
    double fph = f(x + h * 0.5);
    double r3 = 0.5 * (fp1 - fm1);
    double r5 = (4.0 / 3.0) * (fph - fmh) - (1.0 / 3.0) * r3;
    double e3 = (std::fabs(fp1) + std::fabs(fm1)) * eps_f;
    double e5 = 2.0 * (std::fabs(fph) + std::fabs(fmh)) * eps_f + e3;
    double dy = std::max(std::fabs(r3 / h), std::fabs(r5 / h)) * (std::fabs(x) / h) * eps_f;
    double result = r5 / h;
    double trunc = std::fabs((r5 - r3) / h);
    double round_err = std::fabs(e5 / h) + dy;
    double error = round_err + trunc;
    if (round_err < trunc && round_err > 0.0 && trunc > 0.0) {
        double h_opt = h * std::cbrt(round_err / (2.0 * trunc));
        double fm1_opt = f(x - h_opt);
        double fp1_opt = f(x + h_opt);
        double fmh_opt = f(x - h_opt * 0.5);
        double fph_opt = f(x + h_opt * 0.5);
        double r3_opt = 0.5 * (fp1_opt - fm1_opt);
        double r5_opt = (4.0 / 3.0) * (fph_opt - fmh_opt) - (1.0 / 3.0) * r3_opt;
        double e3_opt = (std::fabs(fp1_opt) + std::fabs(fm1_opt)) * eps_f;
        double e5_opt = 2.0 * (std::fabs(fph_opt) + std::fabs(fmh_opt)) * eps_f + e3_opt;
        double dy_opt = std::max(std::fabs(r3_opt / h_opt), std::fabs(r5_opt / h_opt)) * (std::fabs(x) / h_opt) * eps_f;
        double r_opt = r5_opt / h_opt;
        double trunc_opt = std::fabs((r5_opt - r3_opt) / h_opt);
        double round_opt = std::fabs(e5_opt / h_opt) + dy_opt;
        double error_opt = round_opt + trunc_opt;
        if (error_opt < error && std::fabs(r_opt - result) < 4.0 * error) {
            result = r_opt;
            error = error_opt;
        }
    }
    if (abs_err) *abs_err = error;
    return result;
}

inline double robust_gsl_derivative_c(const std::function<double(double)>& f, double x, double h) {
    double abs_err = 0.0;
    double result = gsl_deriv_central_with_err_c(f, x, h, &abs_err);
    constexpr double frac = 0.01;
    if (abs_err <= frac * std::fabs(result)) {
        return result;
    }
    for (int n = 1; n <= 10; ++n) {
        double h1 = h * double(2 * n);
        double h2 = h / double(2 * n);
        double abs_err1 = 0.0, abs_err2 = 0.0;
        double temp1 = gsl_deriv_central_with_err_c(f, x, h1, &abs_err1);
        double temp2 = gsl_deriv_central_with_err_c(f, x, h2, &abs_err2);
        double t1 = std::fabs(temp1);
        double t2 = std::fabs(temp2);
        double e1 = std::fabs(abs_err1);
        double e2 = std::fabs(abs_err2);
        double rel1 = (t1 == 0.0) ? std::numeric_limits<double>::infinity() : e1 / t1;
        double rel2 = (t2 == 0.0) ? std::numeric_limits<double>::infinity() : e2 / t2;
        if (rel1 < rel2 && e1 < frac * t1) return temp1;
        if (rel1 > rel2 && e2 < frac * t2) return temp2;
    }
    return result;
}

inline double calcomega_polar_derivative_core_c(
    double pphi,
    double r_polar,
    double theta_polar,
    double phi_polar,
    double ptheta_polar,
    const double s1_m2[3],
    const double s2_m2[3],
    double mass1,
    double mass2,
    double eta,
    double M,
    double h_k0, double h_k1, double h_k2, double h_k3, double h_k4, double h_k5, double h_k5l,
    double h_KK, double h_d1, double h_d1v2, double h_dheffSS, double h_dheffSSv2,
    double h_b3, double h_bb3
) {
    double mass1_norm = mass1 / M;
    double mass2_norm = mass2 / M;

    double norm_total_mass2 = (mass1_norm + mass2_norm) * (mass1_norm + mass2_norm);
    double sigmaKerr_vec[3] = {
        (s1_m2[0] + s2_m2[0]) / norm_total_mass2,
        (s1_m2[1] + s2_m2[1]) / norm_total_mass2,
        (s1_m2[2] + s2_m2[2]) / norm_total_mass2
    };
    double sigmaStar_vec[3] = {
        ((mass2_norm / mass1_norm) * s1_m2[0] + (mass1_norm / mass2_norm) * s2_m2[0]) / norm_total_mass2,
        ((mass2_norm / mass1_norm) * s1_m2[1] + (mass1_norm / mass2_norm) * s2_m2[1]) / norm_total_mass2,
        ((mass2_norm / mass1_norm) * s1_m2[2] + (mass1_norm / mass2_norm) * s2_m2[2]) / norm_total_mass2
    };
    double a2 = sigmaKerr_vec[0]*sigmaKerr_vec[0] + sigmaKerr_vec[1]*sigmaKerr_vec[1] + sigmaKerr_vec[2]*sigmaKerr_vec[2];
    if (a2 < 1e-16) a2 = 1e-16;
    double a = std::sqrt(a2);
    double a_mag = a;
    double a_mag_clamp = (a_mag < 1.0e-15) ? 1.0e-15 : a_mag;
    double e3_hat[3] = {sigmaKerr_vec[0] / a_mag_clamp, sigmaKerr_vec[1] / a_mag_clamp, sigmaKerr_vec[2] / a_mag_clamp};
    if (a_mag < 1.0e-12) {
        e3_hat[0] = 1.0 / std::sqrt(3.0);
        e3_hat[1] = 1.0 / std::sqrt(3.0);
        e3_hat[2] = 1.0 / std::sqrt(3.0);
    }
    double sin_th = std::sin(theta_polar), cos_th = std::cos(theta_polar);
    double sin_ph = std::sin(phi_polar), cos_ph = std::cos(phi_polar);
    double sin_s = (sin_th < 1.0e-12) ? 1.0e-12 : sin_th;
    double r_s = (r_polar < 1.0e-12) ? 1.0e-12 : r_polar;
    double rcart[3] = {r_polar * cos_th, -r_polar * sin_th * sin_ph, r_polar * sin_th * cos_ph};
    double pcart[3] = {
        -ptheta_polar / r_s * sin_th,
        -ptheta_polar / r_s * cos_th * sin_ph - pphi / (r_s * sin_s) * cos_ph,
        ptheta_polar / r_s * cos_th * cos_ph - pphi / (r_s * sin_s) * sin_ph
    };
    double u_pphi[3] = {0.0, -cos_ph / (r_s * sin_s), -sin_ph / (r_s * sin_s)};
    double L_eval[3] = {
        rcart[1]*pcart[2] - rcart[2]*pcart[1],
        rcart[2]*pcart[0] - rcart[0]*pcart[2],
        rcart[0]*pcart[1] - rcart[1]*pcart[0]
    };
    double rcart_norm = std::sqrt(rcart[0]*rcart[0] + rcart[1]*rcart[1] + rcart[2]*rcart[2]);
    if (rcart_norm < 1.0e-15) rcart_norm = 1.0e-15;
    double n_hat[3] = {rcart[0] / rcart_norm, rcart[1] / rcart_norm, rcart[2] / rcart_norm};
    double L_norm = std::sqrt(L_eval[0]*L_eval[0] + L_eval[1]*L_eval[1] + L_eval[2]*L_eval[2]);
    if (L_norm < 1.0e-15) L_norm = 1.0e-15;
    double Lhat[3] = {L_eval[0] / L_norm, L_eval[1] / L_norm, L_eval[2] / L_norm};
    double lambda_hat_raw[3] = {
        Lhat[1]*n_hat[2] - Lhat[2]*n_hat[1],
        Lhat[2]*n_hat[0] - Lhat[0]*n_hat[2],
        Lhat[0]*n_hat[1] - Lhat[1]*n_hat[0]
    };
    double lh_norm = std::sqrt(lambda_hat_raw[0]*lambda_hat_raw[0] + lambda_hat_raw[1]*lambda_hat_raw[1] + lambda_hat_raw[2]*lambda_hat_raw[2]);
    if (lh_norm < 1.0e-15) lh_norm = 1.0e-15;
    double lambda_hat[3] = {lambda_hat_raw[0] / lh_norm, lambda_hat_raw[1] / lh_norm, lambda_hat_raw[2] / lh_norm};

    double costheta = e3_hat[0]*n_hat[0] + e3_hat[1]*n_hat[1] + e3_hat[2]*n_hat[2];
    double xi_vec[3] = {
        e3_hat[1]*n_hat[2] - e3_hat[2]*n_hat[1],
        e3_hat[2]*n_hat[0] - e3_hat[0]*n_hat[2],
        e3_hat[0]*n_hat[1] - e3_hat[1]*n_hat[0]
    };
    double xi2 = 1.0 - costheta * costheta;
    if (1.0 - std::fabs(costheta) <= 1.0e-8) {
        double angle = 1.8e-3;
        double cos_a = std::cos(angle), sin_a = std::sin(angle);
        double kcrossv[3] = {
            lambda_hat[1]*e3_hat[2] - lambda_hat[2]*e3_hat[1],
            lambda_hat[2]*e3_hat[0] - lambda_hat[0]*e3_hat[2],
            lambda_hat[0]*e3_hat[1] - lambda_hat[1]*e3_hat[0]
        };
        double kdotv = lambda_hat[0]*e3_hat[0] + lambda_hat[1]*e3_hat[1] + lambda_hat[2]*e3_hat[2];
        for (int i = 0; i < 3; ++i) {
            e3_hat[i] = e3_hat[i]*cos_a + kcrossv[i]*sin_a + lambda_hat[i]*kdotv*(1.0 - cos_a);
        }
        xi_vec[0] = e3_hat[1]*n_hat[2] - e3_hat[2]*n_hat[1];
        xi_vec[1] = e3_hat[2]*n_hat[0] - e3_hat[0]*n_hat[2];
        xi_vec[2] = e3_hat[0]*n_hat[1] - e3_hat[1]*n_hat[0];
        costheta = e3_hat[0]*n_hat[0] + e3_hat[1]*n_hat[1] + e3_hat[2]*n_hat[2];
        xi2 = 1.0 - costheta * costheta;
    }
    double v_vec[3] = {
        n_hat[1]*xi_vec[2] - n_hat[2]*xi_vec[1],
        n_hat[2]*xi_vec[0] - n_hat[0]*xi_vec[2],
        n_hat[0]*xi_vec[1] - n_hat[1]*xi_vec[0]
    };

    double r_clamp_var = (r_polar < 1e-9) ? 1e-9 : r_polar;
    double u_b = 1.0 / r_clamp_var;
    if (u_b < 1e-9) u_b = 1e-9;
    double u2_b = u_b * u_b;
    double u3_b = u2_b * u_b;
    double u4_b = u2_b * u2_b;
    double u5_b = u4_b * u_b;

    const double invlog_2e = 0.69314718055994530941723212145817656807550013436026;
    double logu_b = std::log2(u_b) * invlog_2e;
    double denom_KK = -1.0 + eta * h_KK;
    if (std::fabs(denom_KK) < 1.0e-14) denom_KK = (denom_KK >= 0 ? 1.0 : -1.0) * 1.0e-14;
    double invm1PlusEtaKK = 1.0 / denom_KK;

    double logarg_b = h_k1 * u_b + h_k2 * u2_b + h_k3 * u3_b + h_k4 * u4_b + h_k5 * u5_b + h_k5l * u5_b * logu_b;
    double logTerms_b = 1.0 + eta * h_k0 + eta * std::log1p(std::fabs(1.0 + logarg_b) - 1.0);
    double bulk_b = invm1PlusEtaKK * (invm1PlusEtaKK + 2.0 * u_b) + a2 * u2_b;
    double deltaU_b = std::fabs(bulk_b * logTerms_b);
    double r2_b = r_polar * r_polar;
    double deltaT_b = r2_b * deltaU_b;
    double deltaU_u_b = 2.0 * (invm1PlusEtaKK + a2 * u_b) * logTerms_b + bulk_b * (eta * (h_k1 + u_b * (2.0 * h_k2 + u_b * (3.0 * h_k3 + u_b * (4.0 * h_k4 + 5.0 * (h_k5 + h_k5l * logu_b) * u_b))))) / (1.0 + logarg_b);
    double deltaT_r_b = 2.0 * r_polar * deltaU_b - deltaU_u_b;
    double D_b_arg = 6.0 * eta * u2_b + 2.0 * (26.0 - 3.0 * eta) * eta * u3_b;
    double D_b = 1.0 + std::log1p(D_b_arg);
    double deltaR_b = deltaT_b * D_b;
    double w2_b = r2_b + a2;
    double rho2_b = r2_b + a2 * costheta * costheta;
    double Lambda_b = std::fabs(w2_b * w2_b - a2 * deltaT_b * xi2);
    double invrho2xi2Lambda_b = 1.0 / (rho2_b * xi2 * Lambda_b);
    double invrho2 = xi2 * (Lambda_b * invrho2xi2Lambda_b);
    double invxi2 = rho2_b * (Lambda_b * invrho2xi2Lambda_b);
    double invLambda_b = xi2 * rho2_b * invrho2xi2Lambda_b;

    double pvr = (pcart[0]*v_vec[0] + pcart[1]*v_vec[1] + pcart[2]*v_vec[2]) * r_polar;
    double dpvr_dpphi = (u_pphi[0]*v_vec[0] + u_pphi[1]*v_vec[1] + u_pphi[2]*v_vec[2]) * r_polar;
    double pf = (pcart[0]*xi_vec[0] + pcart[1]*xi_vec[1] + pcart[2]*xi_vec[2]) * r_polar;
    double dpf_dpphi = (u_pphi[0]*xi_vec[0] + u_pphi[1]*xi_vec[1] + u_pphi[2]*xi_vec[2]) * r_polar;

    double ww = 2.0 * a * r_polar + h_b3 * eta * a2 * a * u_b + h_bb3 * eta * a * u_b;
    double pf2 = pf * pf;
    double Q = 1.0 + pvr * pvr * invrho2 * invxi2 + pf2 * rho2_b * invLambda_b * invxi2;
    double dQ_dpphi = 2.0 * pvr * dpvr_dpphi * invrho2 * invxi2 + 2.0 * pf * dpf_dpphi * rho2_b * invLambda_b * invxi2;
    double pp = Q - 1.0;
    double dpp_dpphi = dQ_dpphi;

    double expnu2 = (rho2_b * deltaT_b) * invLambda_b;
    if (expnu2 < 1.0e-16) expnu2 = 1.0e-16;
    double expnu = std::sqrt(expnu2);
    double sqrtQ = std::sqrt((Q < 1.0e-16) ? 1.0e-16 : Q);
    double Hns = sqrtQ * expnu + pf * ww * invLambda_b;
    double dHns_dpphi = (0.5 * expnu / sqrtQ) * dQ_dpphi + dpf_dpphi * ww * invLambda_b;

    double d_sM1_dpp = (206.0 * r_polar + 46.0 * pp * r_polar * r_polar + (-120.0 * r_polar + 6.0 * pp * r_polar * r_polar) * eta) * eta * u2_b * (-1.0 / 72.0);
    double d_sM2_dpp = (-109.0 / 36.0 * u2_b * r_polar - 10.0 / 16.0 * pp * u2_b * r_polar * r_polar + 17.0 / 12.0 * u2_b * r_polar * eta) * eta;
    double sMultiplier1 = (-706.0 + (206.0 * pp + 23.0 * pp * pp * r_polar) * r_polar + (54.0 + (-120.0 * pp + 3.0 * pp * pp * r_polar) * r_polar) * eta) * eta * u2_b * (-1.0 / 72.0);
    double sMultiplier2 = (-56.0 / 9.0 * u2_b + (-109.0 / 36.0 * pp * u2_b - 5.0 / 16.0 * pp * pp * u2_b * r_polar) * r_polar + (-7.0 / 3.0 * u2_b + 17.0 / 12.0 * pp * u2_b * r_polar) * eta) * eta;

    double deltaSigmaStar[3], d_deltaSigmaStar_dpp[3], s_vec[3], ds_vec_dpphi[3];
    for (int i = 0; i < 3; ++i) {
        deltaSigmaStar[i] = eta * ((-8.0 + 3.0 * r_polar * pp) * sigmaKerr_vec[i] + (14.0 + 4.0 * pp * r_polar) * sigmaStar_vec[i]) * (1.0 / 12.0) * u_b
                           + sMultiplier1 * sigmaStar_vec[i] + sMultiplier2 * sigmaKerr_vec[i]
                           + h_d1 * eta * sigmaStar_vec[i] * u3_b + h_d1v2 * eta * sigmaKerr_vec[i] * u3_b;
        d_deltaSigmaStar_dpp[i] = eta * (3.0 * r_polar * sigmaKerr_vec[i] + 4.0 * r_polar * sigmaStar_vec[i]) * (1.0 / 12.0) * u_b
                                 + d_sM1_dpp * sigmaStar_vec[i] + d_sM2_dpp * sigmaKerr_vec[i];
        s_vec[i] = sigmaStar_vec[i] + deltaSigmaStar[i];
        ds_vec_dpphi[i] = d_deltaSigmaStar_dpp[i] * dpp_dpphi;
    }
    double sxi = s_vec[0]*xi_vec[0] + s_vec[1]*xi_vec[1] + s_vec[2]*xi_vec[2];
    double sv = s_vec[0]*v_vec[0] + s_vec[1]*v_vec[1] + s_vec[2]*v_vec[2];
    double sn = s_vec[0]*n_hat[0] + s_vec[1]*n_hat[1] + s_vec[2]*n_hat[2];
    double s3 = s_vec[0]*e3_hat[0] + s_vec[1]*e3_hat[1] + s_vec[2]*e3_hat[2];

    double dsxi_dpphi = ds_vec_dpphi[0]*xi_vec[0] + ds_vec_dpphi[1]*xi_vec[1] + ds_vec_dpphi[2]*xi_vec[2];
    double dsv_dpphi = ds_vec_dpphi[0]*v_vec[0] + ds_vec_dpphi[1]*v_vec[1] + ds_vec_dpphi[2]*v_vec[2];
    double dsn_dpphi = ds_vec_dpphi[0]*n_hat[0] + ds_vec_dpphi[1]*n_hat[1] + ds_vec_dpphi[2]*n_hat[2];
    double ds3_dpphi = ds_vec_dpphi[0]*e3_hat[0] + ds_vec_dpphi[1]*e3_hat[1] + ds_vec_dpphi[2]*e3_hat[2];

    double sqrtdeltaT = std::sqrt((deltaT_b < 1.0e-16) ? 1.0e-16 : deltaT_b);
    double sqrtdeltaR = std::sqrt((deltaR_b < 1.0e-16) ? 1.0e-16 : deltaR_b);
    double invsqrtdeltaT = 1.0 / sqrtdeltaT;
    double invsqrtdeltaR = 1.0 / sqrtdeltaR;
    double invdeltaT = 1.0 / deltaT_b;
    double w = ww * invLambda_b;
    double expMU = std::sqrt((rho2_b < 1.0e-16) ? 1.0e-16 : rho2_b);
    double invexpnu = 1.0 / expnu;
    double invexpMU = 1.0 / expMU;

    double Lambda_r = 4.0 * r_polar * w2_b - a2 * deltaT_r_b * xi2;
    double ww_r = 2.0 * a - (a2 * a * h_b3 * eta) * u2_b - h_bb3 * eta * a * u2_b;
    double BR = (-deltaT_b * invsqrtdeltaR + deltaT_r_b * 0.5) * invsqrtdeltaT;
    double wr = (-Lambda_r * ww + Lambda_b * ww_r) * (invLambda_b * invLambda_b);
    double nur = (r_polar * invrho2 + (w2_b * (-4.0 * r_polar * deltaT_b + w2_b * deltaT_r_b)) * 0.5 * invdeltaT * invLambda_b);
    double mur = (r_polar * invrho2 - invsqrtdeltaR);
    double wcos = -2.0 * (a2 * costheta) * deltaT_b * ww * (invLambda_b * invLambda_b);
    double nucos = (a2 * costheta) * w2_b * (w2_b - deltaT_b) * (invrho2 * invLambda_b);
    double mucos = (a2 * costheta) * invrho2;

    double dHs_dpphi = w * ds3_dpphi;
    double denom_Q = 2.0 * sqrtdeltaT * (1.0 + sqrtQ) * sqrtQ * xi2;
    double d_denom_Q_dpphi = 2.0 * sqrtdeltaT * xi2 * (1.0 + 2.0 * sqrtQ) / (2.0 * sqrtQ) * dQ_dpphi;

    double term1 = (expMU * expMU) * (expnu * expnu) * (pf * pf) * sv;
    double d_term1 = (expMU * expMU) * (expnu * expnu) * (2.0 * pf * dpf_dpphi * sv + pf * pf * dsv_dpphi);
    double term2 = sqrtdeltaT * (expMU * expnu) * pf * pvr * sxi;
    double d_term2 = sqrtdeltaT * (expMU * expnu) * (dpf_dpphi * pvr * sxi + pf * dpvr_dpphi * sxi + pf * pvr * dsxi_dpphi);
    double term3 = (sqrtdeltaT * sqrtdeltaT) * xi2 * (expMU * expMU) * (sqrtQ + Q) * sv;
    double d_sqrtQ_plus_Q = (1.0 / (2.0 * sqrtQ) + 1.0) * dQ_dpphi;
    double d_term3 = (sqrtdeltaT * sqrtdeltaT) * xi2 * (expMU * expMU) * (d_sqrtQ_plus_Q * sv + (sqrtQ + Q) * dsv_dpphi);

    double Hwr_num = (invexpMU * invexpMU * invexpMU * invexpnu) * sqrtdeltaR * (term1 - term2 + term3);
    double d_Hwr_num = (invexpMU * invexpMU * invexpMU * invexpnu) * sqrtdeltaR * (d_term1 - d_term2 + d_term3);
    double dHwr_dpphi = (d_Hwr_num * denom_Q - Hwr_num * d_denom_Q_dpphi) / (denom_Q * denom_Q);
    dHs_dpphi += dHwr_dpphi * wr;

    double denom_wcos = 2.0 * sqrtdeltaT * (1.0 + sqrtQ) * sqrtQ;
    double d_denom_wcos_dpphi = 2.0 * sqrtdeltaT * (1.0 + 2.0 * sqrtQ) / (2.0 * sqrtQ) * dQ_dpphi;
    double wcos_inner = -(expMU * expMU) * (expnu * expnu) * (pf * pf) + (sqrtdeltaT * sqrtdeltaT) * (pvr * pvr - (expMU * expMU) * (sqrtQ + Q) * xi2);
    double d_wcos_inner = -(expMU * expMU) * (expnu * expnu) * (2.0 * pf * dpf_dpphi) + (sqrtdeltaT * sqrtdeltaT) * (2.0 * pvr * dpvr_dpphi - (expMU * expMU) * d_sqrtQ_plus_Q * xi2);
    double Hwcos_num = (invexpMU * invexpMU * invexpMU * invexpnu) * sn * wcos_inner;
    double d_Hwcos_num = (invexpMU * invexpMU * invexpMU * invexpnu) * (dsn_dpphi * wcos_inner + sn * d_wcos_inner);
    double dHwcos_dpphi = (d_Hwcos_num * denom_wcos - Hwcos_num * d_denom_wcos_dpphi) / (denom_wcos * denom_wcos);
    dHs_dpphi += dHwcos_dpphi * wcos;

    double fac_SOL = (expnu * expnu * invexpMU) * (-sqrtdeltaT + (expMU * expnu)) / (deltaT_b * xi2);
    double HSOL_num = pf * s3;
    double d_HSOL_num = dpf_dpphi * s3 + pf * ds3_dpphi;
    double dHSOL_dpphi = fac_SOL * (d_HSOL_num * sqrtQ - HSOL_num * (0.5 / sqrtQ * dQ_dpphi)) / Q;
    dHs_dpphi += dHSOL_dpphi;

    double denom_SONL = deltaT_b * (sqrtQ + Q) * xi2;
    double d_denom_SONL = deltaT_b * xi2 * d_sqrtQ_plus_Q;
    double p_SONL_1 = -(sqrtdeltaT * expMU * expnu * nucos * xi2) * pf * (1.0 + 2.0 * sqrtQ) * sn;
    double d_p_SONL_1 = -(sqrtdeltaT * expMU * expnu * nucos * xi2) * ((dpf_dpphi * (1.0 + 2.0 * sqrtQ) + pf * (1.0 / sqrtQ * dQ_dpphi)) * sn + pf * (1.0 + 2.0 * sqrtQ) * dsn_dpphi);
    double p_SONL_2a = -(BR * expMU * expnu) * pf * (1.0 + sqrtQ) * sv;
    double d_p_SONL_2a = -(BR * expMU * expnu) * ((dpf_dpphi * (1.0 + sqrtQ) + pf * (0.5 / sqrtQ * dQ_dpphi)) * sv + pf * (1.0 + sqrtQ) * dsv_dpphi);
    double p_SONL_2b1 = (expMU * expnu * nur) * pf * (1.0 + 2.0 * sqrtQ) * sv;
    double d_p_SONL_2b1 = (expMU * expnu * nur) * ((dpf_dpphi * (1.0 + 2.0 * sqrtQ) + pf * (1.0 / sqrtQ * dQ_dpphi)) * sv + pf * (1.0 + 2.0 * sqrtQ) * dsv_dpphi);
    double p_SONL_2b2 = sqrtdeltaT * mur * pvr * sxi;
    double d_p_SONL_2b2 = sqrtdeltaT * mur * (dpvr_dpphi * sxi + pvr * dsxi_dpphi);
    double p_SONL_2b3 = sqrtdeltaT * sxi * sqrtQ * (mur - nur) * pvr;
    double d_p_SONL_2b3 = sqrtdeltaT * (mur - nur) * ((dsxi_dpphi * sqrtQ + sxi * (0.5 / sqrtQ * dQ_dpphi)) * pvr + sxi * sqrtQ * dpvr_dpphi);
    double p_SONL_2 = (p_SONL_2a + sqrtdeltaT * (p_SONL_2b1 + p_SONL_2b2 + p_SONL_2b3)) * sqrtdeltaR;
    double d_p_SONL_2 = (d_p_SONL_2a + sqrtdeltaT * (d_p_SONL_2b1 + d_p_SONL_2b2 + d_p_SONL_2b3)) * sqrtdeltaR;
    double HSONL_bracket = p_SONL_1 + p_SONL_2;
    double d_HSONL_bracket = d_p_SONL_1 + d_p_SONL_2;
    double fac_SONL = expnu * (invexpMU * invexpMU);
    double HSONL_num = fac_SONL * HSONL_bracket;
    double d_HSONL_num = fac_SONL * d_HSONL_bracket;
    double dHSONL_dpphi = (d_HSONL_num * denom_SONL - HSONL_num * d_denom_SONL) / (denom_SONL * denom_SONL);
    dHs_dpphi += dHSONL_dpphi;

    double s_dot_s = s_vec[0]*s_vec[0] + s_vec[1]*s_vec[1] + s_vec[2]*s_vec[2];
    double s_dot_ds = s_vec[0]*ds_vec_dpphi[0] + s_vec[1]*ds_vec_dpphi[1] + s_vec[2]*ds_vec_dpphi[2];
    double dHss_dpphi = -0.5 * u3_b * (2.0 * s_dot_ds - 6.0 * sn * dsn_dpphi);
    double dHeff_dpphi = dHns_dpphi + dHs_dpphi + dHss_dpphi;

    double Hs_val = w * s3 + (Hwr_num / denom_Q) * wr + (Hwcos_num / denom_wcos) * wcos + (HSOL_num / (deltaT_b * sqrtQ * xi2)) * (expnu * expnu * invexpMU) * (-sqrtdeltaT + (expMU * expnu)) + (HSONL_num / denom_SONL);
    double Hss_val = -0.5 * u3_b * (s_dot_s - 3.0 * sn * sn);
    double sigmaKerr_dot_star = sigmaKerr_vec[0]*sigmaStar_vec[0] + sigmaKerr_vec[1]*sigmaStar_vec[1] + sigmaKerr_vec[2]*sigmaStar_vec[2];
    double s1_sq = s1_m2[0]*s1_m2[0] + s1_m2[1]*s1_m2[1] + s1_m2[2]*s1_m2[2];
    double s2_sq = s2_m2[0]*s2_m2[0] + s2_m2[1]*s2_m2[1] + s2_m2[2]*s2_m2[2];
    double Heff_val = Hns + Hs_val + Hss_val + h_dheffSS * eta * sigmaKerr_dot_star * u4_b + h_dheffSSv2 * eta * u4_b * (s1_sq + s2_sq);
    double Hreal_arg = 1.0 + 2.0 * eta * (std::fabs(Heff_val) - 1.0);
    if (Hreal_arg < 1.0e-16) Hreal_arg = 1.0e-16;
    double Hreal = std::sqrt(Hreal_arg);
    return dHeff_dpphi / Hreal;
}

double calcomega_polar_derivative_core_native(
    double pphi,
    double r_polar,
    double theta_polar,
    double phi_polar,
    double ptheta_polar,
    const torch::Tensor& s1_m2_t,
    const torch::Tensor& s2_m2_t,
    double mass1,
    double mass2,
    double eta,
    double M,
    double h_k0, double h_k1, double h_k2, double h_k3, double h_k4, double h_k5, double h_k5l,
    double h_KK, double h_d1, double h_d1v2, double h_dheffSS, double h_dheffSSv2,
    double h_b3, double h_bb3
) {
    const double* s1_m2 = s1_m2_t.data_ptr<double>();
    const double* s2_m2 = s2_m2_t.data_ptr<double>();
    return calcomega_polar_derivative_core_c(
        pphi, r_polar, theta_polar, phi_polar, ptheta_polar,
        s1_m2, s2_m2, mass1, mass2, eta, M,
        h_k0, h_k1, h_k2, h_k3, h_k4, h_k5, h_k5l,
        h_KK, h_d1, h_d1v2, h_dheffSS, h_dheffSSv2,
        h_b3, h_bb3
    );
}

struct HCoeffs {

    double KK, k0, k1, k2, k3, k4, k5, k5l, b3, bb3, d1, d1v2, dheffSS, dheffSSv2;
};

inline double eval_hcoeff_poly_c(const double* c, double eta, double chi) {
    double chi2 = chi * chi;
    double chi3 = chi2 * chi;
    double eta2 = eta * eta;
    double eta3 = eta2 * eta;
    return c[0] + c[1]*chi + c[2]*chi2 + c[3]*chi3
         + c[4]*eta + c[5]*eta*chi + c[6]*eta*chi2 + c[7]*eta*chi3
         + c[8]*eta2 + c[9]*eta2*chi + c[10]*eta2*chi2 + c[11]*eta2*chi3
         + c[12]*eta3 + c[13]*eta3*chi + c[14]*eta3*chi2 + c[15]*eta3*chi3;
}

inline void compute_spin_aligned_hcoeffs_c(double eta, double a, double chi_eff, HCoeffs* h) {
    constexpr double COEFFS_K[16] = {
        1.7336, -1.62045, -1.38086, 1.43659,
        10.2573, 2.26831, 0.0, -0.426958,
        -126.687, 17.3736, 6.16466, 0.0,
        267.788, -27.5201, 31.1746, -59.1658
    };
    constexpr double COEFFS_DSO[16] = {
        -44.5324, 0.0, 0.0, 66.1987,
        0.0, 0.0, -343.313, -568.651,
        0.0, 2495.29, 0.0, 147.481,
        0.0, 0.0, 0.0, 0.0
    };
    constexpr double COEFFS_DSS[16] = {
        6.06807, 0.0, 0.0, 0.0,
        -36.0272, 37.1964, 0.0, -41.0003,
        0.0, 0.0, -326.325, 528.511,
        706.958, 0.0, 1161.78, 0.0
    };
    constexpr double PI_VAL = 3.14159265358979323846;
    constexpr double EULER_GAMMA = 0.577215664901532860606512;
    constexpr double LN2 = 0.693147180559945309417232121458176568;

    double KK = eval_hcoeff_poly_c(COEFFS_K, eta, chi_eff);
    double m1PlusEtaKK = -1.0 + eta * KK;
    double invm1PlusEtaKK = 1.0 / m1PlusEtaKK;

    double k0 = KK * (m1PlusEtaKK - 1.0);
    double k1 = -2.0 * (k0 + KK) * m1PlusEtaKK;
    double k1p2 = k1 * k1;
    double k1p3 = k1 * k1p2;

    double k2 = (k1 * (k1 - 4.0 * m1PlusEtaKK)) * 0.5 - a * a * k0 * m1PlusEtaKK * m1PlusEtaKK;
    double k3 = -(k1 * k1) * k1 * (1.0/3.0) + k1 * k2 + (k1 * k1) * m1PlusEtaKK - 2.0 * (k2 - m1PlusEtaKK) * m1PlusEtaKK - a * a * k1 * (m1PlusEtaKK * m1PlusEtaKK);
    double k4 = (24.0 / 96.0) * (k1 * k1) * (k1 * k1)
        - (96.0 / 96.0) * (k1 * k1) * k2
        + (48.0 / 96.0) * k2 * k2
        - (64.0 / 96.0) * (k1 * k1) * k1 * m1PlusEtaKK
        + (48.0 / 96.0) * (a * a) * (k1 * k1 - 2.0 * k2) * (m1PlusEtaKK * m1PlusEtaKK)
        + (96.0 / 96.0) * k1 * (k3 + 2.0 * k2 * m1PlusEtaKK)
        - m1PlusEtaKK * ((192.0 / 96.0) * k3 + m1PlusEtaKK * (-(3008.0 / 96.0) + (123.0 / 96.0) * PI_VAL * PI_VAL));

    double k5 = m1PlusEtaKK * m1PlusEtaKK * (
        -4237.0 / 60.0
        + 128.0 / 5.0 * EULER_GAMMA
        + 2275.0 * PI_VAL * PI_VAL / 512.0
        - (1.0/3.0) * (a * a) * (k1p3 - 3.0 * (k1 * k2) + 3.0 * k3)
        - ((k1p3 * k1p2) - 5.0 * (k1p3 * k2) + 5.0 * k1 * k2 * k2 + 5.0 * k1p2 * k3 - 5.0 * k2 * k3 - 5.0 * k1 * k4)
        * 0.2 * invm1PlusEtaKK * invm1PlusEtaKK
        + ((k1p2 * k1p2) - 4.0 * (k1p2 * k2) + 2.0 * k2 * k2 + 4.0 * k1 * k3 - 4.0 * k4)
        * 0.5 * invm1PlusEtaKK
        + (256.0 / 5.0) * LN2
        + (41.0 * PI_VAL * PI_VAL / 32.0 - 221.0 / 6.0) * eta
    );
    double k5l = (m1PlusEtaKK * m1PlusEtaKK) * (64.0 / 5.0);

    h->KK = KK;
    h->k0 = k0;
    h->k1 = k1;
    h->k2 = k2;
    h->k3 = k3;
    h->k4 = k4;
    h->k5 = k5;
    h->k5l = k5l;
    h->b3 = 0.0;
    h->bb3 = 0.0;
    h->d1 = 0.0;
    h->d1v2 = eval_hcoeff_poly_c(COEFFS_DSO, eta, chi_eff);
    h->dheffSS = 0.0;
    h->dheffSSv2 = eval_hcoeff_poly_c(COEFFS_DSS, eta, chi_eff);
}

inline void instantaneous_hcoeffs_c(
    double eta,
    const double L_vec[3],
    const double s1_m2[3],
    const double s2_m2[3],
    HCoeffs* h
) {
    double sigma_vec[3] = {s1_m2[0] + s2_m2[0], s1_m2[1] + s2_m2[1], s1_m2[2] + s2_m2[2]};
    double sigma_norm = std::sqrt(sigma_vec[0]*sigma_vec[0] + sigma_vec[1]*sigma_vec[1] + sigma_vec[2]*sigma_vec[2]);

    double L_mag = std::sqrt(L_vec[0]*L_vec[0] + L_vec[1]*L_vec[1] + L_vec[2]*L_vec[2]);
    double L_mag_clamp = (L_mag < 1.0e-15) ? 1.0e-15 : L_mag;
    double Lhat[3] = {L_vec[0] / L_mag_clamp, L_vec[1] / L_mag_clamp, L_vec[2] / L_mag_clamp};

    double S1_dot_L = s1_m2[0]*Lhat[0] + s1_m2[1]*Lhat[1] + s1_m2[2]*Lhat[2];
    double S2_dot_L = s2_m2[0]*Lhat[0] + s2_m2[1]*Lhat[1] + s2_m2[2]*Lhat[2];
    double S1_perp[3] = {s1_m2[0] - S1_dot_L * Lhat[0], s1_m2[1] - S1_dot_L * Lhat[1], s1_m2[2] - S1_dot_L * Lhat[2]};
    double S2_perp[3] = {s2_m2[0] - S2_dot_L * Lhat[0], s2_m2[1] - S2_dot_L * Lhat[1], s2_m2[2] - S2_dot_L * Lhat[2]};

    double denom = 1.0 - 2.0 * eta;
    if (std::fabs(denom) < 1.0e-12) denom = (denom >= 0 ? 1.0 : -1.0) * 1.0e-12;
    double chi_raw = (sigma_vec[0]*Lhat[0] + sigma_vec[1]*Lhat[1] + sigma_vec[2]*Lhat[2]) / denom;

    double S_perp_sum[3] = {S1_perp[0] + S2_perp[0], S1_perp[1] + S2_perp[1], S1_perp[2] + S2_perp[2]};
    double perpendicular_projection = S_perp_sum[0]*sigma_vec[0] + S_perp_sum[1]*sigma_vec[1] + S_perp_sum[2]*sigma_vec[2];
    chi_raw += perpendicular_projection / sigma_norm / denom / 2.0;

    double chi_aug = (sigma_norm > 1.0e-6) ? chi_raw : 0.0;
    compute_spin_aligned_hcoeffs_c(eta, sigma_norm, chi_aug, h);
}

inline double eob_hamiltonian_c(
    const double r_vec[3],
    const double p_vec[3],
    const double S1_weighted[3],
    const double S2_weighted[3],
    double mass1,
    double mass2,
    double eta,
    double M,
    double h_k0, double h_k1, double h_k2, double h_k3, double h_k4, double h_k5, double h_k5l,
    double h_KK, double h_d1, double h_d1v2, double h_dheffSS, double h_dheffSSv2,
    double h_b3, double h_bb3,
    int tortoise,
    bool p_is_tortoise
) {
    double mass1_norm = mass1 / M;
    double mass2_norm = mass2 / M;
    double s1_m2[3] = {S1_weighted[0], S1_weighted[1], S1_weighted[2]};
    double s2_m2[3] = {S2_weighted[0], S2_weighted[1], S2_weighted[2]};
    double sigmaKerr_vec[3] = {s1_m2[0] + s2_m2[0], s1_m2[1] + s2_m2[1], s1_m2[2] + s2_m2[2]};
    double sigmaStar_vec[3] = {
        (mass2_norm / mass1_norm) * s1_m2[0] + (mass1_norm / mass2_norm) * s2_m2[0],
        (mass2_norm / mass1_norm) * s1_m2[1] + (mass1_norm / mass2_norm) * s2_m2[1],
        (mass2_norm / mass1_norm) * s1_m2[2] + (mass1_norm / mass2_norm) * s2_m2[2]
    };
    double a2 = sigmaKerr_vec[0]*sigmaKerr_vec[0] + sigmaKerr_vec[1]*sigmaKerr_vec[1] + sigmaKerr_vec[2]*sigmaKerr_vec[2];
    if (a2 < 1e-16) a2 = 1e-16;
    double a = std::sqrt(a2);
    double a_mag = a;
    double a_mag_clamp = (a_mag < 1e-15) ? 1e-15 : a_mag;
    double e3_hat[3] = {sigmaKerr_vec[0] / a_mag_clamp, sigmaKerr_vec[1] / a_mag_clamp, sigmaKerr_vec[2] / a_mag_clamp};
    if (a_mag < 1.0e-12) {
        e3_hat[0] = 1.0 / std::sqrt(3.0);
        e3_hat[1] = 1.0 / std::sqrt(3.0);
        e3_hat[2] = 1.0 / std::sqrt(3.0);
    }

    double r2_source = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
    double r = std::sqrt(r2_source);
    double r_clamp = (r < 1e-15) ? 1e-15 : r;
    double n_hat[3] = {r_vec[0] / r_clamp, r_vec[1] / r_clamp, r_vec[2] / r_clamp};

    double L_vec[3] = {
        r_vec[1]*p_vec[2] - r_vec[2]*p_vec[1],
        r_vec[2]*p_vec[0] - r_vec[0]*p_vec[2],
        r_vec[0]*p_vec[1] - r_vec[1]*p_vec[0]
    };
    double L_mag = std::sqrt(L_vec[0]*L_vec[0] + L_vec[1]*L_vec[1] + L_vec[2]*L_vec[2]);
    double L_mag_clamp = (L_mag < 1e-15) ? 1e-15 : L_mag;
    double Lhat[3] = {L_vec[0] / L_mag_clamp, L_vec[1] / L_mag_clamp, L_vec[2] / L_mag_clamp};

    HCoeffs h_inst;
    instantaneous_hcoeffs_c(eta, L_vec, s1_m2, s2_m2, &h_inst);
    h_k0 = h_inst.k0; h_k1 = h_inst.k1; h_k2 = h_inst.k2; h_k3 = h_inst.k3;
    h_k4 = h_inst.k4; h_k5 = h_inst.k5; h_k5l = h_inst.k5l; h_KK = h_inst.KK;
    h_d1 = h_inst.d1; h_d1v2 = h_inst.d1v2; h_dheffSS = h_inst.dheffSS; h_dheffSSv2 = h_inst.dheffSSv2;

    double lambda_hat_raw[3] = {
        Lhat[1]*n_hat[2] - Lhat[2]*n_hat[1],
        Lhat[2]*n_hat[0] - Lhat[0]*n_hat[2],
        Lhat[0]*n_hat[1] - Lhat[1]*n_hat[0]
    };
    double lh_norm = std::sqrt(lambda_hat_raw[0]*lambda_hat_raw[0] + lambda_hat_raw[1]*lambda_hat_raw[1] + lambda_hat_raw[2]*lambda_hat_raw[2]);
    double lh_clamp = (lh_norm < 1e-15) ? 1e-15 : lh_norm;
    double lambda_hat[3] = {lambda_hat_raw[0] / lh_clamp, lambda_hat_raw[1] / lh_clamp, lambda_hat_raw[2] / lh_clamp};

    double costheta = e3_hat[0]*n_hat[0] + e3_hat[1]*n_hat[1] + e3_hat[2]*n_hat[2];
    double xi_vec[3] = {
        e3_hat[1]*n_hat[2] - e3_hat[2]*n_hat[1],
        e3_hat[2]*n_hat[0] - e3_hat[0]*n_hat[2],
        e3_hat[0]*n_hat[1] - e3_hat[1]*n_hat[0]
    };
    double xi2 = 1.0 - costheta * costheta;
    if (1.0 - std::fabs(costheta) <= 1.0e-8) {
        double angle = 1.8e-3;
        double cos_a = std::cos(angle), sin_a = std::sin(angle);
        double kcrossv[3] = {
            lambda_hat[1]*e3_hat[2] - lambda_hat[2]*e3_hat[1],
            lambda_hat[2]*e3_hat[0] - lambda_hat[0]*e3_hat[2],
            lambda_hat[0]*e3_hat[1] - lambda_hat[1]*e3_hat[0]
        };
        double kdotv = lambda_hat[0]*e3_hat[0] + lambda_hat[1]*e3_hat[1] + lambda_hat[2]*e3_hat[2];
        for (int i = 0; i < 3; ++i) {
            e3_hat[i] = e3_hat[i]*cos_a + kcrossv[i]*sin_a + lambda_hat[i]*kdotv*(1.0 - cos_a);
        }
        xi_vec[0] = e3_hat[1]*n_hat[2] - e3_hat[2]*n_hat[1];
        xi_vec[1] = e3_hat[2]*n_hat[0] - e3_hat[0]*n_hat[2];
        xi_vec[2] = e3_hat[0]*n_hat[1] - e3_hat[1]*n_hat[0];
        costheta = e3_hat[0]*n_hat[0] + e3_hat[1]*n_hat[1] + e3_hat[2]*n_hat[2];
        xi2 = 1.0 - costheta * costheta;
    }
    double v_vec[3] = {
        n_hat[1]*xi_vec[2] - n_hat[2]*xi_vec[1],
        n_hat[2]*xi_vec[0] - n_hat[0]*xi_vec[2],
        n_hat[0]*xi_vec[1] - n_hat[1]*xi_vec[0]
    };

    double r_var = r;
    double r_clamp_var = (r_var < 1e-9) ? 1e-9 : r_var;
    double u_b = 1.0 / r_clamp_var;
    if (u_b < 1e-9) u_b = 1e-9;
    double u2_b = u_b * u_b;
    double u3_b = u2_b * u_b;
    double u4_b = u2_b * u2_b;
    double u5_b = u4_b * u_b;

    const double invlog_2e = 0.69314718055994530941723212145817656807550013436026;
    double logu_b = std::log2(u_b) * invlog_2e;
    double denom_KK = -1.0 + eta * h_KK;
    if (std::fabs(denom_KK) < 1.0e-14) denom_KK = (denom_KK >= 0 ? 1.0 : -1.0) * 1.0e-14;
    double invm1PlusEtaKK = 1.0 / denom_KK;

    double logarg_b = h_k1 * u_b + h_k2 * u2_b + h_k3 * u3_b + h_k4 * u4_b + h_k5 * u5_b + h_k5l * u5_b * logu_b;
    double logTerms_b = 1.0 + eta * h_k0 + eta * std::log1p(std::fabs(1.0 + logarg_b) - 1.0);
    double bulk_b = invm1PlusEtaKK * (invm1PlusEtaKK + 2.0 * u_b) + a2 * u2_b;
    double deltaU_b = std::fabs(bulk_b * logTerms_b);
    double r2_b = r2_source;
    double deltaT_b = r2_b * deltaU_b;
    double deltaU_u_b = 2.0 * (invm1PlusEtaKK + a2 * u_b) * logTerms_b + bulk_b * (eta * (h_k1 + u_b * (2.0 * h_k2 + u_b * (3.0 * h_k3 + u_b * (4.0 * h_k4 + 5.0 * (h_k5 + h_k5l * logu_b) * u_b))))) / (1.0 + logarg_b);
    double deltaT_r_b = 2.0 * r_var * deltaU_b - deltaU_u_b;
    double D_b_arg = 6.0 * eta * u2_b + 2.0 * (26.0 - 3.0 * eta) * eta * u3_b;
    double D_b = 1.0 + std::log1p(D_b_arg);
    double deltaR_b = deltaT_b * D_b;
    double w2_b = r2_b + a2;
    double rho2_b = r2_b + a2 * costheta * costheta;
    double Lambda_b = std::fabs(w2_b * w2_b - a2 * deltaT_b * xi2);
    double invrho2xi2Lambda_b = 1.0 / (rho2_b * xi2 * Lambda_b);
    double invrho2 = xi2 * (Lambda_b * invrho2xi2Lambda_b);
    double invxi2 = rho2_b * (Lambda_b * invrho2xi2Lambda_b);
    double invLambda_b = xi2 * rho2_b * invrho2xi2Lambda_b;

    double csi_b = std::sqrt(std::fabs(deltaT_b * deltaR_b)) / w2_b;
    double csi1_b = 1.0, csi2_b = 1.0, csi_out = csi_b;
    if (!p_is_tortoise && tortoise != 2) {
        csi1_b = 1.0; csi2_b = 1.0; csi_out = 1.0;
    } else if (tortoise == 1) {
        csi1_b = csi_b; csi2_b = 1.0; csi_out = csi_b;
    } else if (tortoise == 2) {
        csi1_b = 1.0; csi2_b = csi_b; csi_out = csi_b;
    }

    double pr_proj = p_vec[0]*n_hat[0] + p_vec[1]*n_hat[1] + p_vec[2]*n_hat[2];
    double p_vec_use[3];
    double prT;
    if (p_is_tortoise) {
        prT = pr_proj * csi2_b;
        double radial_scale = 1.0 - 1.0 / ((csi1_b < 1.0e-15) ? 1.0e-15 : csi1_b);
        for (int i = 0; i < 3; ++i) {
            p_vec_use[i] = p_vec[i] - n_hat[i] * prT * radial_scale;
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            p_vec_use[i] = p_vec[i];
        }
        double csi_fac = (csi_out < 1.0e-15) ? 1.0e-15 : csi_out;
        prT = pr_proj * csi_fac;
    }

    double tmpP[3] = {p_vec_use[0], p_vec_use[1], p_vec_use[2]};
    double pn = tmpP[0]*n_hat[0] + tmpP[1]*n_hat[1] + tmpP[2]*n_hat[2];
    double pvr = (tmpP[0]*v_vec[0] + tmpP[1]*v_vec[1] + tmpP[2]*v_vec[2]) * r_var;
    double ptheta2 = pvr * pvr * invxi2;
    double qq = 2.0 * eta * (4.0 - 3.0 * eta);
    double ww = 2.0 * a * r_var + h_b3 * eta * a2 * a * u_b + h_bb3 * eta * a * u_b;
    double pf0 = (tmpP[0]*xi_vec[0] + tmpP[1]*xi_vec[1] + tmpP[2]*xi_vec[2]) * r_var;
    double pf_use = pf0;
    double prT2 = prT * prT;
    double pf2 = pf_use * pf_use;
    double pn2_raw = pn * pn;

    double base = 1.0 + (prT2 * prT2) * qq * u2_b + ptheta2 * invrho2 + pf2 * rho2_b * invLambda_b * invxi2 + pn2_raw * deltaR_b * invrho2;
    double Hns = std::sqrt((base * (rho2_b * deltaT_b) * invLambda_b < 1.0e-16) ? 1.0e-16 : base * (rho2_b * deltaT_b) * invLambda_b) + pf_use * ww * invLambda_b;
    double Q = 1.0 + pvr * pvr * invrho2 * invxi2 + pf2 * rho2_b * invLambda_b * invxi2 + pn2_raw * deltaR_b * invrho2;
    double pn2 = pn2_raw * deltaR_b * invrho2;
    double pp = Q - 1.0;

    double sMultiplier1 = (-706.0 + (206.0 * pp - 282.0 * pn2 + (-96.0 * pn2 * pp + 23.0 * pp * pp) * r_var) * r_var
        + (54.0 + (-120.0 * pp + 324.0 * pn2 + (-360.0 * pn2 * pn2 + 126.0 * pn2 * pp + 3.0 * pp * pp) * r_var) * r_var) * eta) * eta * u2_b * (-1.0 / 72.0);
    double sMultiplier2 = (-56.0 / 9.0 * u2_b + (-2.0 / 3.0 * pn2 * u2_b - 109.0 / 36.0 * pp * u2_b + (pn2 * pp * u2_b / 4.0 - 5.0 / 16.0 * pp * pp * u2_b) * r_var) * r_var
        + (-7.0 / 3.0 * u2_b + (-49.0 / 8.0 * pn2 * u2_b + 17.0 / 12.0 * pp * u2_b + (45.0 / 8.0 * pn2 * pn2 * u2_b - 13.0 / 8.0 * pn2 * pp * u2_b) * r_var) * r_var) * eta) * eta;

    double deltaSigmaStar[3], s_vec[3];
    for (int i = 0; i < 3; ++i) {
        deltaSigmaStar[i] = eta * ((-8.0 - 3.0 * r_var * (12.0 * pn2 - pp)) * sigmaKerr_vec[i] + (14.0 + (-30.0 * pn2 + 4.0 * pp) * r_var) * sigmaStar_vec[i]) * (1.0 / 12.0) * u_b
            + sMultiplier1 * sigmaStar_vec[i] + sMultiplier2 * sigmaKerr_vec[i]
            + h_d1 * eta * sigmaStar_vec[i] * u3_b + h_d1v2 * eta * sigmaKerr_vec[i] * u3_b;
        s_vec[i] = sigmaStar_vec[i] + deltaSigmaStar[i];
    }
    double sx = s_vec[0], sy = s_vec[1], sz = s_vec[2];
    double sxi = s_vec[0]*xi_vec[0] + s_vec[1]*xi_vec[1] + s_vec[2]*xi_vec[2];
    double sv = s_vec[0]*v_vec[0] + s_vec[1]*v_vec[1] + s_vec[2]*v_vec[2];
    double sn = s_vec[0]*n_hat[0] + s_vec[1]*n_hat[1] + s_vec[2]*n_hat[2];
    double s3 = s_vec[0]*e3_hat[0] + s_vec[1]*e3_hat[1] + s_vec[2]*e3_hat[2];

    double sqrtdeltaT = std::sqrt((deltaT_b < 1.0e-16) ? 1.0e-16 : deltaT_b);
    double sqrtdeltaR = std::sqrt((deltaR_b < 1.0e-16) ? 1.0e-16 : deltaR_b);
    double invdeltaTsqrtdeltaTsqrtdeltaR = 1.0 / ((sqrtdeltaT * deltaT_b * sqrtdeltaR < 1.0e-16) ? 1.0e-16 : sqrtdeltaT * deltaT_b * sqrtdeltaR);
    double invsqrtdeltaT = deltaT_b * (sqrtdeltaR * invdeltaTsqrtdeltaTsqrtdeltaR);
    double invsqrtdeltaR = deltaT_b * sqrtdeltaT * invdeltaTsqrtdeltaTsqrtdeltaR;
    double invdeltaT = sqrtdeltaT * (sqrtdeltaR * invdeltaTsqrtdeltaTsqrtdeltaR);

    double w = ww * invLambda_b;
    double expnu = std::sqrt((deltaT_b * rho2_b * invLambda_b < 1.0e-16) ? 1.0e-16 : deltaT_b * rho2_b * invLambda_b);
    double expMU = std::sqrt((rho2_b < 1.0e-16) ? 1.0e-16 : rho2_b);
    double invexpnuexpMU = 1.0 / ((expnu * expMU < 1.0e-16) ? 1.0e-16 : expnu * expMU);
    double invexpnu = expMU * invexpnuexpMU;
    double invexpMU = expnu * invexpnuexpMU;

    double Lambda_r = 4.0 * r_var * w2_b - a2 * deltaT_r_b * xi2;
    double ww_r = 2.0 * a - (a2 * a * h_b3 * eta) * u2_b - h_bb3 * eta * a * u2_b;
    double BR = (-deltaT_b * invsqrtdeltaR + deltaT_r_b * 0.5) * invsqrtdeltaT;
    double wr = (-Lambda_r * ww + Lambda_b * ww_r) * (invLambda_b * invLambda_b);
    double nur = (r_var * invrho2 + (w2_b * (-4.0 * r_var * deltaT_b + w2_b * deltaT_r_b)) * 0.5 * invdeltaT * invLambda_b);
    double mur = (r_var * invrho2 - invsqrtdeltaR);
    double wcos = -2.0 * (a2 * costheta) * deltaT_b * ww * (invLambda_b * invLambda_b);
    double nucos = (a2 * costheta) * w2_b * (w2_b - deltaT_b) * (invrho2 * invLambda_b);
    double mucos = (a2 * costheta) * invrho2;

    double sqrtQ = std::sqrt((Q < 1.0e-16) ? 1.0e-16 : Q);
    double inv2B1psqrtQsqrtQ = 1.0 / ((2.0 * sqrtdeltaT * (1.0 + sqrtQ) * sqrtQ < 1.0e-16) ? 1.0e-16 : 2.0 * sqrtdeltaT * (1.0 + sqrtQ) * sqrtQ);

    double Hwr = ((invexpMU * invexpMU * invexpMU * invexpnu) * sqrtdeltaR * (
        (expMU * expMU) * (expnu * expnu) * (pf_use * pf_use) * sv
        - sqrtdeltaT * (expMU * expnu) * pf_use * pvr * sxi
        + sqrtdeltaT * sqrtdeltaT * xi2 * ((expMU * expMU) * (sqrtQ + Q) * sv + pn * pvr * sn * sqrtdeltaR - pn * pn * sv * deltaR_b)
    )) * inv2B1psqrtQsqrtQ * invxi2;

    double Hwcos = ((invexpMU * invexpMU * invexpMU * invexpnu) * (
        sn * (-(expMU * expMU) * (expnu * expnu) * (pf_use * pf_use) + sqrtdeltaT * sqrtdeltaT * (pvr * pvr - (expMU * expMU) * (sqrtQ + Q) * xi2))
        - sqrtdeltaT * pn * (sqrtdeltaT * pvr * sv - (expMU * expnu) * pf_use * sxi) * sqrtdeltaR
    )) * inv2B1psqrtQsqrtQ;

    double HSOL = ((expnu * expnu * invexpMU) * (-sqrtdeltaT + (expMU * expnu)) * pf_use * s3) / (deltaT_b * sqrtQ) * invxi2;

    double HSONL = ((expnu * (invexpMU * invexpMU)) * (
        -(sqrtdeltaT * expMU * expnu * nucos * pf_use * (1.0 + 2.0 * sqrtQ) * sn * xi2)
        + (-(BR * (expMU * expnu) * pf_use * (1.0 + sqrtQ) * sv)
           + sqrtdeltaT * ((expMU * expnu) * nur * pf_use * (1.0 + 2.0 * sqrtQ) * sv + sqrtdeltaT * mur * pvr * sxi + sqrtdeltaT * sxi * (-(mucos * pn * xi2) + sqrtQ * (mur * pvr - nur * pvr + (-mucos + nucos) * pn * xi2)))) * sqrtdeltaR
    )) * invxi2 / (deltaT_b * (sqrtQ + Q));

    double Hs = w * s3 + Hwr * wr + Hwcos * wcos + HSOL + HSONL;
    double Hss = -0.5 * u3_b * (sx * sx + sy * sy + sz * sz - 3.0 * sn * sn);

    double sigmaKerr_dot_star = sigmaKerr_vec[0]*sigmaStar_vec[0] + sigmaKerr_vec[1]*sigmaStar_vec[1] + sigmaKerr_vec[2]*sigmaStar_vec[2];
    double s1s2_square_sum = s1_m2[0]*s1_m2[0] + s1_m2[1]*s1_m2[1] + s1_m2[2]*s1_m2[2] + s2_m2[0]*s2_m2[0] + s2_m2[1]*s2_m2[1] + s2_m2[2]*s2_m2[2];
    double H_eff = Hns + Hs + Hss + h_dheffSS * eta * sigmaKerr_dot_star * u4_b + h_dheffSSv2 * eta * u4_b * s1s2_square_sum;

    double Hreal_arg = 1.0 + 2.0 * eta * (std::fabs(H_eff) - 1.0);
    double Hreal = std::sqrt((Hreal_arg < 1.0e-16) ? 1.0e-16 : Hreal_arg);
    double r_newton = (r < 1.0e-12) ? 1.0e-12 : r;
    double Hreal_fallback = std::sqrt((1.0 + (pf0 * pf0) / (r_newton * r_newton) < 1.0e-12) ? 1.0e-12 : 1.0 + (pf0 * pf0) / (r_newton * r_newton));
    if (Hreal <= 1.0e-8) Hreal = Hreal_fallback;

    return Hreal;
}

double eob_hamiltonian_native(
    const torch::Tensor& r_vec_t,
    const torch::Tensor& p_vec_t,
    const torch::Tensor& s1_m2_t,
    const torch::Tensor& s2_m2_t,
    double mass1,
    double mass2,
    double eta,
    double M,
    double h_k0, double h_k1, double h_k2, double h_k3, double h_k4, double h_k5, double h_k5l,
    double h_KK, double h_d1, double h_d1v2, double h_dheffSS, double h_dheffSSv2,
    double h_b3, double h_bb3,
    int tortoise,
    bool p_is_tortoise
) {
    const double* r_vec = r_vec_t.data_ptr<double>();
    const double* p_vec = p_vec_t.data_ptr<double>();
    const double* s1_m2 = s1_m2_t.data_ptr<double>();
    const double* s2_m2 = s2_m2_t.data_ptr<double>();
    return eob_hamiltonian_c(
        r_vec, p_vec, s1_m2, s2_m2,
        mass1, mass2, eta, M,
        h_k0, h_k1, h_k2, h_k3, h_k4, h_k5, h_k5l,
        h_KK, h_d1, h_d1v2, h_dheffSS, h_dheffSSv2,
        h_b3, h_bb3,
        tortoise, p_is_tortoise
    );
}

torch::Tensor dH_dx_cartesian_native(
    const torch::Tensor& r_vec_t,
    const torch::Tensor& p_vec_t,
    const torch::Tensor& s1_m2_t,
    const torch::Tensor& s2_m2_t,
    double mass1,
    double mass2,
    double eta,
    double M,
    double h_k0, double h_k1, double h_k2, double h_k3, double h_k4, double h_k5, double h_k5l,
    double h_KK, double h_d1, double h_d1v2, double h_dheffSS, double h_dheffSSv2,
    double h_b3, double h_bb3,
    double step
) {
    const double* r_in = r_vec_t.data_ptr<double>();
    const double* p_vec = p_vec_t.data_ptr<double>();
    const double* s1_m2 = s1_m2_t.data_ptr<double>();
    const double* s2_m2 = s2_m2_t.data_ptr<double>();

    auto grad_out = torch::empty({3}, r_vec_t.options());
    double* grad_data = grad_out.data_ptr<double>();

    double r_tmp[3] = {r_in[0], r_in[1], r_in[2]};

    for (int axis = 0; axis < 3; ++axis) {
        auto f = [&](double x_val) -> double {
            double orig = r_tmp[axis];
            r_tmp[axis] = x_val;
            double h_val = eob_hamiltonian_c(
                r_tmp, p_vec, s1_m2, s2_m2,
                mass1, mass2, eta, M,
                h_k0, h_k1, h_k2, h_k3, h_k4, h_k5, h_k5l,
                h_KK, h_d1, h_d1v2, h_dheffSS, h_dheffSSv2,
                h_b3, h_bb3,
                2, false
            ) / eta;


            r_tmp[axis] = orig;
            return h_val;
        };
        grad_data[axis] = gsl_deriv_central_c(f, r_in[axis], step);
    }
    return grad_out;
}

torch::Tensor dH_dp_cartesian_native(
    const torch::Tensor& r_vec_t,
    const torch::Tensor& p_vec_t,
    const torch::Tensor& s1_m2_t,
    const torch::Tensor& s2_m2_t,
    double mass1,
    double mass2,
    double eta,
    double M,
    double h_k0, double h_k1, double h_k2, double h_k3, double h_k4, double h_k5, double h_k5l,
    double h_KK, double h_d1, double h_d1v2, double h_dheffSS, double h_dheffSSv2,
    double h_b3, double h_bb3,
    double step
) {
    const double* r_vec = r_vec_t.data_ptr<double>();
    const double* p_in = p_vec_t.data_ptr<double>();
    const double* s1_m2 = s1_m2_t.data_ptr<double>();
    const double* s2_m2 = s2_m2_t.data_ptr<double>();

    auto grad_out = torch::empty({3}, p_vec_t.options());
    double* grad_data = grad_out.data_ptr<double>();

    double p_tmp[3] = {p_in[0], p_in[1], p_in[2]};

    for (int axis = 0; axis < 3; ++axis) {
        auto f = [&](double p_val) -> double {
            double orig = p_tmp[axis];
            p_tmp[axis] = p_val;
            double h_val = eob_hamiltonian_c(
                r_vec, p_tmp, s1_m2, s2_m2,
                mass1, mass2, eta, M,
                h_k0, h_k1, h_k2, h_k3, h_k4, h_k5, h_k5l,
                h_KK, h_d1, h_d1v2, h_dheffSS, h_dheffSSv2,
                h_b3, h_bb3,
                1, true
            ) / eta;
            p_tmp[axis] = orig;
            return h_val;
        };
        grad_data[axis] = gsl_deriv_central_c(f, p_in[axis], step);
    }
    return grad_out;
}

std::tuple<torch::Tensor, torch::Tensor> dH_dspin_cartesian_native(
    const torch::Tensor& r_vec_t,
    const torch::Tensor& p_vec_t,
    const torch::Tensor& s1_m2_t,
    const torch::Tensor& s2_m2_t,
    double mass1,
    double mass2,
    double eta,
    double M,
    double h_k0, double h_k1, double h_k2, double h_k3, double h_k4, double h_k5, double h_k5l,
    double h_KK, double h_d1, double h_d1v2, double h_dheffSS, double h_dheffSSv2,
    double h_b3, double h_bb3,
    double step
) {
    const double* r_vec = r_vec_t.data_ptr<double>();
    const double* p_vec = p_vec_t.data_ptr<double>();
    const double* s1_in = s1_m2_t.data_ptr<double>();
    const double* s2_in = s2_m2_t.data_ptr<double>();

    auto grad_s1 = torch::empty({3}, s1_m2_t.options());
    auto grad_s2 = torch::empty({3}, s2_m2_t.options());
    double* d1_data = grad_s1.data_ptr<double>();
    double* d2_data = grad_s2.data_ptr<double>();

    double s1_tmp[3] = {s1_in[0], s1_in[1], s1_in[2]};
    double s2_tmp[3] = {s2_in[0], s2_in[1], s2_in[2]};

    for (int axis = 0; axis < 3; ++axis) {
        auto f1 = [&](double s_val) -> double {
            double orig = s1_tmp[axis];
            s1_tmp[axis] = s_val;
            double h_val = eob_hamiltonian_c(
                r_vec, p_vec, s1_tmp, s2_tmp,
                mass1, mass2, eta, M,
                h_k0, h_k1, h_k2, h_k3, h_k4, h_k5, h_k5l,
                h_KK, h_d1, h_d1v2, h_dheffSS, h_dheffSSv2,
                h_b3, h_bb3,
                1, true
            ) / eta;
            s1_tmp[axis] = orig;
            return h_val;
        };
        d1_data[axis] = gsl_deriv_central_c(f1, s1_in[axis], step);

        auto f2 = [&](double s_val) -> double {
            double orig = s2_tmp[axis];
            s2_tmp[axis] = s_val;
            double h_val = eob_hamiltonian_c(
                r_vec, p_vec, s1_tmp, s2_tmp,
                mass1, mass2, eta, M,
                h_k0, h_k1, h_k2, h_k3, h_k4, h_k5, h_k5l,
                h_KK, h_d1, h_d1v2, h_dheffSS, h_dheffSSv2,
                h_b3, h_bb3,
                1, true
            ) / eta;
            s2_tmp[axis] = orig;
            return h_val;
        };
        d2_data[axis] = gsl_deriv_central_c(f2, s2_in[axis], step);
    }
    return std::make_tuple(grad_s1, grad_s2);
}

inline int double_factorial_c(int n) {
    if (n <= 0) return 1;
    int res = 1;
    for (int k = n; k > 0; k -= 2) {
        res *= k;
    }
    return res;
}

inline double factorial_c(int n) {
    double res = 1.0;
    for (int k = 2; k <= n; ++k) {
        res *= (double)k;
    }
    return res;
}

inline double abs_scalar_sph_pi_over2_c(int l, int m) {
    int abs_m = std::abs(m);
    if (l < 0 || abs_m < 0 || abs_m > l || (l + abs_m) % 2 != 0) {
        return 0.0;
    }
    double sign = ((l + abs_m) / 2) % 2 ? -1.0 : 1.0;
    double leg = sign * ((double)double_factorial_c(l + abs_m - 1) / (double)double_factorial_c(l - abs_m));
    if (m < 0 && (abs_m % 2 == 1)) {
        leg *= -1.0;
    }
    double fact_ratio = 1.0;
    for (int k = l - abs_m + 1; k <= l + abs_m; ++k) {
        fact_ratio /= (double)k;
    }
    constexpr double PI_VAL = 3.14159265358979323846;
    double norm = std::sqrt((2.0 * (double)l + 1.0) / (4.0 * PI_VAL) * fact_ratio);
    return std::fabs(norm * leg);
}

inline double calc_prefix_abs_c(int l, int m, double mass1, double mass2, double eta) {
    int epsilon = (l + m) % 2;
    int sign = (m % 2 == 0) ? 1 : -1;
    double total = mass1 + mass2;
    double x1 = mass1 / total;
    double x2 = mass2 / total;
    double c;
    if (mass1 != mass2 || sign == 1) {
        c = std::pow(x2, l + epsilon - 1) + (double)sign * std::pow(x1, l + epsilon - 1);
    } else {
        if (l == 2 || l == 3) c = -1.0;
        else if (l == 4 || l == 5) c = -0.5;
        else c = 0.0;
    }

    constexpr double PI_VAL = 3.14159265358979323846;
    double mult1, mult2;
    if (epsilon == 0) {
        mult1 = 8.0 * PI_VAL / (double)double_factorial_c(2 * l + 1);
        mult2 = std::sqrt((double)((l + 1) * (l + 2)) / (double)(l * (l - 1)));
    } else {
        mult1 = 16.0 * PI_VAL / (double)double_factorial_c(2 * l + 1);
        mult2 = std::sqrt((double)((2 * l + 1) * (l + 2) * (l * l - m * m)) / (double)((2 * l - 1) * (l + 1) * l * (l - 1)));
    }
    return std::fabs(std::pow((double)m, l) * mult1 * mult2 * eta * c);
}

inline void rho_aux_flux_c(
    int l, int m, double v, double eta, double chi1z, double chi2z,
    double mass1, double mass2, double tplspin, double* rho_out, double* aux_out
) {
    double M = mass1 + mass2;
    double dM = (mass1 - mass2) / M;
    double dM2 = dM * dM;
    double chiS = 0.5 * (chi1z + chi2z);
    double chiA = 0.5 * (chi1z - chi2z);
    double a_delta = chiS + chiA * dM;
    double a = tplspin;
    double a2 = a * a;
    double a3 = a2 * a;
    double eta2 = eta * eta;
    double eta3 = eta2 * eta;
    constexpr double EULER_GAMMA = 0.577215664901532860606512;
    constexpr double PI_VAL = 3.14159265358979323846;
    double eulerlog = EULER_GAMMA + std::log(2.0 * (double)std::abs(m) * (v < 1.0e-16 ? 1.0e-16 : v));

    double rho = 1.0;
    double aux = 0.0;

    if (l == 2 && m == 2) {
        double rho22v2 = -43.0 / 42.0 + (55.0 * eta) / 84.0;
        double rho22v3 = (-2.0 * (chiS + chiA * dM - chiS * eta)) / 3.0;
        double rho22v4 = -20555.0 / 10584.0 + 0.5 * a_delta * a_delta - (33025.0 * eta) / 21168.0 + (19583.0 * eta2) / 42336.0;
        double rho22v5 = ((-34.0 / 21.0 + 49.0 * eta / 18.0 + 209.0 * eta2 / 126.0) * chiS) + ((-34.0 / 21.0 - 19.0 * eta / 42.0) * dM * chiA);
        double rho22v6 = 1556919113.0 / 122245200.0 + (89.0 * a2) / 252.0 - (48993925.0 * eta) / 9779616.0 - (6292061.0 * eta2) / 3259872.0 + (10620745.0 * eta3) / 39118464.0 + (41.0 * eta * PI_VAL * PI_VAL) / 192.0;
        double rho22v6l = -428.0 / 105.0;
        double rho22v7 = a3 / 3.0 + chiA * dM * (18733.0 / 15876.0 + (50140.0 * eta) / 3969.0 + (97865.0 * eta2) / 63504.0) + chiS * (18733.0 / 15876.0 + (74749.0 * eta) / 5292.0 - (245717.0 * eta2) / 63504.0 + (50803.0 * eta3) / 63504.0);
        double rho22v8 = -387216563023.0 / 160190110080.0 + (18353.0 * a2) / 21168.0 - (a2 * a2) / 8.0;
        double rho22v8l = 9202.0 / 2205.0;
        double rho22v10 = -16094530514677.0 / 533967033600.0;
        double rho22v10l = 439877.0 / 55566.0;
        rho = 1.0 + v * v * (
            rho22v2
            + v * (rho22v3 + v * (rho22v4 + v * (rho22v5 + v * (rho22v6 + rho22v6l * eulerlog + v * (rho22v7 + v * (rho22v8 + rho22v8l * eulerlog + (rho22v10 + rho22v10l * eulerlog) * v * v))))))
        );
    } else if (l == 2 && m == 1) {
        double rho21v1 = 0.0;
        double rho21v2 = -59.0 / 56.0 + (23.0 * eta) / 84.0;
        double rho21v3 = 0.0;
        double rho21v4 = -47009.0 / 56448.0 - (865.0 * a2) / 1792.0 - (405.0 * a2 * a2) / 2048.0 - (10993.0 * eta) / 14112.0 + (617.0 * eta2) / 4704.0;
        double rho21v5 = (-98635.0 * a) / 75264.0 + (2031.0 * a * a2) / 7168.0 - (1701.0 * a2 * a3) / 8192.0;
        double rho21v6 = 7613184941.0 / 2607897600.0 + (9032393.0 * a2) / 1806336.0 + (3897.0 * a2 * a2) / 16384.0 - (15309.0 * a3 * a3) / 65536.0;
        double rho21v6l = -107.0 / 105.0;
        double rho21v7 = (-3859374457.0 * a) / 1159065600.0 - (55169.0 * a3) / 16384.0 + (18603.0 * a2 * a3) / 65536.0 - (72171.0 * a2 * a2 * a3) / 262144.0;
        double rho21v7l = 107.0 * a / 140.0;
        double rho21v8 = -1168617463883.0 / 911303737344.0;
        double rho21v8l = 6313.0 / 5880.0;
        double rho21v10 = -63735873771463.0 / 16569158860800.0;
        double rho21v10l = 5029963.0 / 5927040.0;
        rho = 1.0 + v * (
            rho21v1
            + v * (
                rho21v2
                + v * (
                    rho21v3
                    + v * (
                        rho21v4
                        + v * (
                            rho21v5
                            + v * (
                                rho21v6
                                + rho21v6l * eulerlog
                                + v * (
                                    rho21v7
                                    + rho21v7l * eulerlog
                                    + v * (rho21v8 + rho21v8l * eulerlog + (rho21v10 + rho21v10l * eulerlog) * v * v)
                                )
                            )
                        )
                    )
                )
            )
        );
        double inv_dM = (std::fabs(dM) >= 1e-12) ? (1.0 / dM) : 0.0;
        double f21v1 = (dM2 != 0.0) ? ((-3.0 * (chiS + chiA * inv_dM)) / 2.0) : (-1.5 * chiA);
        double f21v3 = (dM2 != 0.0) ? ((chiS * dM * (427.0 + 79.0 * eta) + chiA * (147.0 + 280.0 * dM2 + 1251.0 * eta)) / (84.0 * dM)) : (0.375 * chiA);
        aux = v * f21v1 + v * v * v * f21v3;
    } else if (l == 3 && m == 3) {
        double rho33v2 = -7.0 / 6.0 + (2.0 * eta) / 3.0;
        double rho33v3 = 0.0;
        double rho33v4 = -6719.0 / 3960.0 + a2 / 2.0 - (1861.0 * eta) / 990.0 + (149.0 * eta2) / 330.0;
        double rho33v5 = (-4.0 * a) / 3.0;
        double rho33v6 = 3203101567.0 / 227026800.0 + (5.0 * a2) / 36.0;
        double rho33v6l = -26.0 / 7.0;
        double rho33v7 = (5297.0 * a) / 2970.0 + a3 / 3.0;
        double rho33v8 = -57566572157.0 / 8562153600.0;
        double rho33v8l = 13.0 / 3.0;
        double rho33v10 = 0.0;
        double rho33v10l = 0.0;
        rho = 1.0 + v * v * (
            rho33v2
            + v * (
                rho33v3
                + v * (
                    rho33v4
                    + v * (
                        rho33v5
                        + v * (
                            rho33v6
                            + rho33v6l * eulerlog
                            + v * (
                                rho33v7
                                + v * (rho33v8 + rho33v8l * eulerlog + (rho33v10 + rho33v10l * eulerlog) * v * v)
                            )
                        )
                    )
                )
            )
        );
        double f33v3 = (dM2 != 0.0) ? ((chiS * dM * (-4.0 + 5.0 * eta) + chiA * (-4.0 + 19.0 * eta)) / (2.0 * dM)) : (0.375 * chiA);
        aux = v * v * v * f33v3;
    } else if (l == 3 && m == 2) {
        double m1Plus3eta = -1.0 + 3.0 * eta;
        double m1Plus3eta2 = m1Plus3eta * m1Plus3eta;
        double rho32v = (4.0 * chiS * eta) / (-3.0 * m1Plus3eta);
        double rho32v2 = (328.0 - 1115.0 * eta + 320.0 * eta2) / (270.0 * m1Plus3eta);
        double rho32v3 = (2.0 * a) / 9.0;
        double rho32v4 = a2 / 3.0 + (-1444528.0 + 8050045.0 * eta - 4725605.0 * eta2 - 20338960.0 * eta3 + 3085640.0 * eta2 * eta2) / (1603800.0 * m1Plus3eta2);
        double rho32v5 = (-2788.0 * a) / 1215.0;
        double rho32v6 = 5849948554.0 / 940355325.0 + (488.0 * a2) / 405.0;
        double rho32v6l = -104.0 / 63.0;
        double rho32v8 = -10607269449358.0 / 3072140846775.0;
        double rho32v8l = 17056.0 / 8505.0;
        rho = 1.0 + v * (
            rho32v
            + v * (
                rho32v2
                + v * (
                    rho32v3
                    + v * (
                        rho32v4
                        + v * (
                            rho32v5
                            + v * (rho32v6 + rho32v6l * eulerlog + (rho32v8 + rho32v8l * eulerlog) * v * v)
                        )
                    )
                )
            )
        );
    } else if (l == 3 && m == 1) {
        if (dM2 != 0.0) {
            double rho31v2 = -13.0 / 18.0 - (2.0 * eta) / 9.0;
            double rho31v3 = 0.0;
            double rho31v4 = 101.0 / 7128.0 - (5.0 * a2) / 6.0 - (1685.0 * eta) / 1782.0 - (829.0 * eta2) / 1782.0;
            double rho31v5 = (4.0 * a) / 9.0;
            double rho31v6 = 11706720301.0 / 6129723600.0 - (49.0 * a2) / 108.0;
            double rho31v6l = -26.0 / 63.0;
            double rho31v7 = (-2579.0 * a) / 5346.0 + a3 / 9.0;
            double rho31v8 = 2606097992581.0 / 4854741091200.0;
            double rho31v8l = 169.0 / 567.0;
            rho = 1.0 + v * v * (
                rho31v2
                + v * (
                    rho31v3
                    + v * (
                        rho31v4
                        + v * (
                            rho31v5
                            + v * (rho31v6 + rho31v6l * eulerlog + v * (rho31v7 + (rho31v8 + rho31v8l * eulerlog) * v))
                        )
                    )
                )
            );
            double f31v3 = (chiA * (-4.0 + 11.0 * eta) + chiS * dM * (-4.0 + 13.0 * eta)) / (2.0 * dM);
            aux = v * v * v * f31v3;
        } else {
            rho = 1.0;
            double f31v3 = -5.0 * chiA / 8.0;
            aux = v * v * v * f31v3;
        }
    } else if (l == 4 && m == 4) {
        double m1Plus3eta = -1.0 + 3.0 * eta;
        double m1Plus3eta2 = m1Plus3eta * m1Plus3eta;
        double rho44v2 = (1614.0 - 5870.0 * eta + 2625.0 * eta2) / (1320.0 * m1Plus3eta);
        double rho44v3 = (chiA * (10.0 - 39.0 * eta) * dM + chiS * (10.0 - 41.0 * eta + 42.0 * eta2)) / (15.0 * m1Plus3eta);
        double rho44v4 = a2 / 2.0 + (-511573572.0 + 2338945704.0 * eta - 313857376.0 * eta2 - 6733146000.0 * eta * eta2 + 1252563795.0 * eta2 * eta2) / (317116800.0 * m1Plus3eta2);
        double rho44v5 = (-69.0 * a) / 55.0;
        double rho44v6 = 16600939332793.0 / 1098809712000.0 + (217.0 * a2) / 3960.0;
        double rho44v6l = -12568.0 / 3465.0;
        rho = 1.0 + v * v * (
            rho44v2
            + v * (rho44v3 + v * (rho44v4 + v * (rho44v5 + (rho44v6 + rho44v6l * eulerlog) * v)))
        );
    } else if (l == 4 && m == 3) {
        if (dM2 != 0.0) {
            double rho43v = 0.0;
            double rho43v2 = (222.0 - 547.0 * eta + 160.0 * eta2) / (176.0 * (-1.0 + 2.0 * eta));
            double rho43v4 = -6894273.0 / 7047040.0 + (3.0 * a2) / 8.0;
            double rho43v5 = (-12113.0 * a) / 6160.0;
            double rho43v6 = 1664224207351.0 / 195343948800.0;
            double rho43v6l = -1571.0 / 770.0;
            rho = 1.0 + v * (
                rho43v
                + v * (
                    rho43v2
                    + v * v * (rho43v4 + v * (rho43v5 + (rho43v6 + rho43v6l * eulerlog) * v))
                )
            );
            double f43v = (5.0 * (chiA - chiS * dM) * eta) / (2.0 * dM * (-1.0 + 2.0 * eta));
            aux = v * f43v;
        } else {
            rho = 1.0;
            double f43v = -5.0 * chiA / 4.0;
            aux = v * f43v;
        }
    } else if (l == 4 && m == 2) {
        double m1Plus3eta = -1.0 + 3.0 * eta;
        double m1Plus3eta2 = m1Plus3eta * m1Plus3eta;
        double rho42v2 = (1146.0 - 3530.0 * eta + 285.0 * eta2) / (1320.0 * m1Plus3eta);
        double rho42v3 = (chiA * (10.0 - 21.0 * eta) * dM + chiS * (10.0 - 59.0 * eta + 78.0 * eta2)) / (15.0 * m1Plus3eta);
        double rho42v4 = a2 / 2.0 + (-114859044.0 + 295834536.0 * eta + 1204388696.0 * eta2 - 3047981160.0 * eta3 - 379526805.0 * eta2 * eta2) / (317116800.0 * m1Plus3eta2);
        double rho42v5 = (-7.0 * a) / 110.0;
        double rho42v6 = 848238724511.0 / 219761942400.0 + (2323.0 * a2) / 3960.0;
        double rho42v6l = -3142.0 / 3465.0;
        rho = 1.0 + v * v * (
            rho42v2 + v * (rho42v3 + v * (rho42v4 + v * (rho42v5 + (rho42v6 + rho42v6l * eulerlog) * v)))
        );
    } else if (l == 4 && m == 1) {
        if (dM2 != 0.0) {
            double rho41v = 0.0;
            double rho41v2 = (602.0 - 1385.0 * eta + 288.0 * eta2) / (528.0 * (-1.0 + 2.0 * eta));
            double rho41v4 = -7775491.0 / 21141120.0 + (3.0 * a2) / 8.0;
            double rho41v5 = (-20033.0 * a) / 55440.0 - (5.0 * a3) / 6.0;
            double rho41v6 = 1227423222031.0 / 1758095539200.0;
            double rho41v6l = -1571.0 / 6930.0;
            rho = 1.0 + v * (
                rho41v
                + v * (
                    rho41v2
                    + v * v * (rho41v4 + v * (rho41v5 + (rho41v6 + rho41v6l * eulerlog) * v))
                )
            );
            double f41v = (5.0 * (chiA - chiS * dM) * eta) / (2.0 * dM * (-1.0 + 2.0 * eta));
            aux = v * f41v;
        } else {
            rho = 1.0;
            double f41v = -5.0 * chiA / 4.0;
            aux = v * f41v;
        }
    } else if (l == 5 && m == 5) {
        double denom = (-1.0 + 2.0 * eta);
        double rho55v2 = (487.0 - 1298.0 * eta + 512.0 * eta2) / (390.0 * denom);
        double rho55v3 = (-2.0 * a) / 3.0;
        double rho55v4 = -3353747.0 / 2129400.0 + a2 / 2.0;
        double rho55v5 = -241.0 * a / 195.0;
        rho = 1.0 + v * v * (rho55v2 + v * (rho55v3 + v * (rho55v4 + rho55v5 * v)));
    } else if (l == 5 && m == 4) {
        double den = 1.0 - 5.0 * eta + 5.0 * eta2;
        double rho54v2 = (-17448.0 + 96019.0 * eta - 127610.0 * eta2 + 33320.0 * eta3) / (13650.0 * den);
        double rho54v3 = (-2.0 * a) / 15.0;
        double rho54v4 = -16213384.0 / 15526875.0 + (2.0 * a2) / 5.0;
        rho = 1.0 + v * v * (rho54v2 + v * (rho54v3 + rho54v4 * v));
    } else if (l == 5 && m == 3) {
        if (dM2 != 0.0) {
            double rho53v2 = (375.0 - 850.0 * eta + 176.0 * eta2) / (390.0 * (-1.0 + 2.0 * eta));
            double rho53v3 = (-2.0 * a) / 3.0;
            double rho53v4 = -410833.0 / 709800.0 + a2 / 2.0;
            double rho53v5 = -103.0 * a / 325.0;
            rho = 1.0 + v * v * (rho53v2 + v * (rho53v3 + v * (rho53v4 + rho53v5 * v)));
        } else {
            rho = 1.0;
        }
    } else if (l == 5 && m == 2) {
        double den = 1.0 - 5.0 * eta + 5.0 * eta2;
        double rho52v2 = (-15828.0 + 84679.0 * eta - 104930.0 * eta2 + 21980.0 * eta3) / (13650.0 * den);
        double rho52v3 = (-2.0 * a) / 15.0;
        double rho52v4 = -7187914.0 / 15526875.0 + (2.0 * a2) / 5.0;
        rho = 1.0 + v * v * (rho52v2 + v * (rho52v3 + rho52v4 * v));
    } else if (l == 5 && m == 1) {
        if (dM2 != 0.0) {
            double rho51v2 = (319.0 - 626.0 * eta + 8.0 * eta2) / (390.0 * (-1.0 + 2.0 * eta));
            double rho51v3 = (-2.0 * a) / 3.0;
            double rho51v4 = -31877.0 / 304200.0 + a2 / 2.0;
            double rho51v5 = 139.0 * a / 975.0;
            rho = 1.0 + v * v * (rho51v2 + v * (rho51v3 + v * (rho51v4 + rho51v5 * v)));
        } else {
            rho = 1.0;
        }
    } else if (l == 6) {
        double den_even = 1.0 - 5.0 * eta + 5.0 * eta2;
        double den_odd = dM2 + 3.0 * eta2;
        if (m == 6) {
            double rho66v2 = (-106.0 + 602.0 * eta - 861.0 * eta2 + 273.0 * eta3) / (84.0 * den_even);
            double rho66v3 = (-2.0 * a) / 3.0;
            double rho66v4 = -1025435.0 / 659736.0 + a2 / 2.0;
            rho = 1.0 + v * v * (rho66v2 + v * (rho66v3 + rho66v4 * v));
        } else if (m == 5) {
            if (dM2 != 0.0) {
                double rho65v2 = (-185.0 + 838.0 * eta - 910.0 * eta2 + 220.0 * eta3) / (144.0 * den_odd);
                double rho65v3 = -2.0 * a / 9.0;
                rho = 1.0 + v * v * (rho65v2 + rho65v3 * v);
            } else {
                rho = 1.0;
            }
        } else if (m == 4) {
            double rho64v2 = (-86.0 + 462.0 * eta - 581.0 * eta2 + 133.0 * eta3) / (84.0 * den_even);
            double rho64v3 = (-2.0 * a) / 3.0;
            double rho64v4 = -476887.0 / 659736.0 + a2 / 2.0;
            rho = 1.0 + v * v * (rho64v2 + v * (rho64v3 + rho64v4 * v));
        } else if (m == 3) {
            if (dM2 != 0.0) {
                double rho63v2 = (-169.0 + 742.0 * eta - 750.0 * eta2 + 156.0 * eta3) / (144.0 * den_odd);
                double rho63v3 = -2.0 * a / 9.0;
                rho = 1.0 + v * v * (rho63v2 + rho63v3 * v);
            } else {
                rho = 1.0;
            }
        } else if (m == 2) {
            double rho62v2 = (-74.0 + 378.0 * eta - 413.0 * eta2 + 49.0 * eta3) / (84.0 * den_even);
            double rho62v3 = (-2.0 * a) / 3.0;
            double rho62v4 = -817991.0 / 3298680.0 + a2 / 2.0;
            rho = 1.0 + v * v * (rho62v2 + v * (rho62v3 + rho62v4 * v));
        } else if (m == 1) {
            if (dM2 != 0.0) {
                double rho61v2 = (-161.0 + 694.0 * eta - 670.0 * eta2 + 124.0 * eta3) / (144.0 * den_odd);
                double rho61v3 = -2.0 * a / 9.0;
                rho = 1.0 + v * v * (rho61v2 + rho61v3 * v);
            } else {
                rho = 1.0;
            }
        }
    } else if (l == 7) {
        double den_even = -1.0 + 7.0 * eta - 14.0 * eta2 + 7.0 * eta3;
        double den_odd = dM2 + 3.0 * eta2;
        if (m == 7 || m == 5 || m == 3 || m == 1) {
            if (dM2 != 0.0) {
                double rho2;
                if (m == 7) rho2 = (-906.0 + 4246.0 * eta - 4963.0 * eta2 + 1380.0 * eta3) / (714.0 * den_odd);
                else if (m == 5) rho2 = (-762.0 + 3382.0 * eta - 3523.0 * eta2 + 804.0 * eta3) / (714.0 * den_odd);
                else if (m == 3) rho2 = (-666.0 + 2806.0 * eta - 2563.0 * eta2 + 420.0 * eta3) / (714.0 * den_odd);
                else rho2 = (-618.0 + 2518.0 * eta - 2083.0 * eta2 + 228.0 * eta3) / (714.0 * den_odd);
                rho = 1.0 + v * v * (rho2 - (2.0 * a / 3.0) * v);
            } else {
                rho = 1.0;
            }
        } else if (m == 6) {
            double rho76v2 = (2144.0 - 16185.0 * eta + 37828.0 * eta2 - 29351.0 * eta3 + 6104.0 * eta2 * eta2) / (1666.0 * den_even);
            rho = 1.0 + rho76v2 * v * v;
        } else if (m == 4) {
            double rho74v2 = (17756.0 - 131805.0 * eta + 298872.0 * eta2 - 217959.0 * eta3 + 41076.0 * eta2 * eta2) / (14994.0 * den_even);
            rho = 1.0 + rho74v2 * v * v;
        } else if (m == 2) {
            double rho72v2 = (16832.0 - 123489.0 * eta + 273924.0 * eta2 - 190239.0 * eta3 + 32760.0 * eta2 * eta2) / (14994.0 * den_even);
            rho = 1.0 + rho72v2 * v * v;
        }
    } else if (l == 8) {
        double den_even = -1.0 + 7.0 * eta - 14.0 * eta2 + 7.0 * eta3;
        double den_odd = -1.0 + 6.0 * eta - 10.0 * eta2 + 4.0 * eta3;
        double rho2 = 0.0;
        bool active = false;
        if (m == 8) {
            rho2 = (3482.0 - 26778.0 * eta + 64659.0 * eta2 - 53445.0 * eta3 + 12243.0 * eta2 * eta2) / (2736.0 * den_even);
            active = true;
        } else if (m == 7 && dM2 != 0.0) {
            rho2 = (23478.0 - 154099.0 * eta + 309498.0 * eta2 - 207550.0 * eta3 + 38920.0 * eta2 * eta2) / (18240.0 * den_odd);
            active = true;
        } else if (m == 6) {
            rho2 = (1002.0 - 7498.0 * eta + 17269.0 * eta2 - 13055.0 * eta3 + 2653.0 * eta2 * eta2) / (912.0 * den_even);
            active = true;
        } else if (m == 5 && dM2 != 0.0) {
            rho2 = (4350.0 - 28055.0 * eta + 54642.0 * eta2 - 34598.0 * eta3 + 6056.0 * eta2 * eta2) / (3648.0 * den_odd);
            active = true;
        } else if (m == 4) {
            rho2 = (2666.0 - 19434.0 * eta + 42627.0 * eta2 - 28965.0 * eta3 + 4899.0 * eta2 * eta2) / (2736.0 * den_even);
            active = true;
        } else if (m == 3 && dM2 != 0.0) {
            rho2 = (20598.0 - 131059.0 * eta + 249018.0 * eta2 - 149950.0 * eta3 + 24520.0 * eta2 * eta2) / (18240.0 * den_odd);
            active = true;
        } else if (m == 2) {
            rho2 = (2462.0 - 17598.0 * eta + 37119.0 * eta2 - 22845.0 * eta3 + 3063.0 * eta2 * eta2) / (2736.0 * den_even);
            active = true;
        } else if (m == 1 && dM2 != 0.0) {
            rho2 = (20022.0 - 126451.0 * eta + 236922.0 * eta2 - 138430.0 * eta3 + 21640.0 * eta2 * eta2) / (18240.0 * den_odd);
            active = true;
        }
        if (active) {
            rho = 1.0 + rho2 * v * v;
        } else {
            rho = 1.0;
        }
    }

    *rho_out = rho;
    *aux_out = aux;
}

inline void calculate_rdot_c(
    const double r_vec[3],
    const double p_vec[3],
    const double s1_w[3],
    const double s2_w[3],
    double mass1,
    double mass2,
    double eta,
    double M,
    double rdot_out[3]
) {
    double L_vec[3] = {
        r_vec[1] * p_vec[2] - r_vec[2] * p_vec[1],
        r_vec[2] * p_vec[0] - r_vec[0] * p_vec[2],
        r_vec[0] * p_vec[1] - r_vec[1] * p_vec[0]
    };
    HCoeffs h_inst;
    instantaneous_hcoeffs_c(eta, L_vec, s1_w, s2_w, &h_inst);

    double r_clamp = std::sqrt(r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2]);
    if (r_clamp < 1.0e-15) r_clamp = 1.0e-15;
    double r2 = r_clamp * r_clamp;
    double u = 1.0 / r_clamp;
    double u2 = u * u;
    double u3 = u2 * u;
    double u4 = u2 * u2;
    double u5 = u4 * u;
    double logu = std::log(u);
    double sigma_vec[3] = {s1_w[0] + s2_w[0], s1_w[1] + s2_w[1], s1_w[2] + s2_w[2]};
    double a2 = sigma_vec[0]*sigma_vec[0] + sigma_vec[1]*sigma_vec[1] + sigma_vec[2]*sigma_vec[2];
    double w2 = r2 + a2;

    double D_term = 1.0 + 6.0 * eta * u2 + 2.0 * (26.0 - 3.0 * eta) * eta * u3;
    double D = 1.0 + std::log(D_term);
    double m1PlusetaKK = -1.0 + eta * h_inst.KK;
    double bulk = 1.0 / (m1PlusetaKK * m1PlusetaKK) + (2.0 * u) / m1PlusetaKK + a2 * u2;
    double log_arg = 1.0 + h_inst.k1 * u + h_inst.k2 * u2 + h_inst.k3 * u3 + h_inst.k4 * u4 + h_inst.k5 * u5 + h_inst.k5l * u5 * logu;
    double logTerms = 1.0 + eta * h_inst.k0 + eta * std::log(std::fabs(log_arg));
    double deltaU = bulk * logTerms;
    double deltaT = r2 * deltaU;
    double deltaR = deltaT * D;
    double csi = std::sqrt(std::max(deltaT * deltaR, 0.0)) / w2;
    double csi_fac = (csi < 1.0e-15) ? 1.0e-15 : csi;

    double Tmat[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            double tij = (r_vec[i] * r_vec[j] / r2) * (csi_fac - 1.0);
            if (i == j) {
                tij += 1.0;
            }
            Tmat[i][j] = tij;
            Tmat[j][i] = tij;
        }
    }

    double dH_dpvec[3];
    double p_tmp[3] = {p_vec[0], p_vec[1], p_vec[2]};
    constexpr double STEP_DERIV = 2.0e-3;
    for (int axis = 0; axis < 3; ++axis) {
        auto f = [&](double p_val) -> double {
            double orig = p_tmp[axis];
            p_tmp[axis] = p_val;
            double h_val = eob_hamiltonian_c(
                r_vec, p_tmp, s1_w, s2_w,
                mass1, mass2, eta, M,
                h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
                h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
                h_inst.b3, h_inst.bb3,
                1, true
            ) / eta;
            p_tmp[axis] = orig;
            return h_val;
        };
        dH_dpvec[axis] = gsl_deriv_central_c(f, p_vec[axis], STEP_DERIV);
    }

    rdot_out[0] = Tmat[0][0]*dH_dpvec[0] + Tmat[0][1]*dH_dpvec[1] + Tmat[0][2]*dH_dpvec[2];
    rdot_out[1] = Tmat[1][0]*dH_dpvec[0] + Tmat[1][1]*dH_dpvec[1] + Tmat[1][2]*dH_dpvec[2];
    rdot_out[2] = Tmat[2][0]*dH_dpvec[0] + Tmat[2][1]*dH_dpvec[1] + Tmat[2][2]*dH_dpvec[2];
}

inline double calcomega_lal_polar_derivative_point_c(
    const double r_vec[3],
    const double p_vec[3],
    const double s1[3],
    const double s2[3],
    double mass1,
    double mass2,
    double eta,
    double M,
    const double* rdot_in = nullptr,
    const double* s1_weighted_in = nullptr,
    const double* s2_weighted_in = nullptr
) {
    double s1_scale = (mass1 * mass1) / (M * M);
    double s2_scale = (mass2 * mass2) / (M * M);

    double s1_w[3], s2_w[3];
    if (s1_weighted_in) {
        s1_w[0] = s1_weighted_in[0]; s1_w[1] = s1_weighted_in[1]; s1_w[2] = s1_weighted_in[2];
    } else {
        s1_w[0] = s1[0] * s1_scale; s1_w[1] = s1[1] * s1_scale; s1_w[2] = s1[2] * s1_scale;
    }
    if (s2_weighted_in) {
        s2_w[0] = s2_weighted_in[0]; s2_w[1] = s2_weighted_in[1]; s2_w[2] = s2_weighted_in[2];
    } else {
        s2_w[0] = s2[0] * s2_scale; s2_w[1] = s2[1] * s2_scale; s2_w[2] = s2[2] * s2_scale;
    }

    double rdot[3];
    if (rdot_in) {
        rdot[0] = rdot_in[0]; rdot[1] = rdot_in[1]; rdot[2] = rdot_in[2];
    } else {
        calculate_rdot_c(r_vec, p_vec, s1_w, s2_w, mass1, mass2, eta, M, rdot);
    }

    double LN[3] = {
        r_vec[1] * rdot[2] - r_vec[2] * rdot[1],
        r_vec[2] * rdot[0] - r_vec[0] * rdot[2],
        r_vec[0] * rdot[1] - r_vec[1] * rdot[0]
    };
    double LN_norm = std::sqrt(LN[0]*LN[0] + LN[1]*LN[1] + LN[2]*LN[2]);
    double LN_clamp = (LN_norm < 1.0e-15) ? 1.0e-15 : LN_norm;
    double LNhat[3] = {LN[0] / LN_clamp, LN[1] / LN_clamp, LN[2] / LN_clamp};

    bool use_rot1 = (LNhat[0] >= 0.9);
    constexpr double invsqrt2 = 0.707106781186547524400844362104849039;
    double Rot1[3][3];
    double Xprime[3];
    if (use_rot1) {
        Rot1[0][0] = invsqrt2; Rot1[0][1] = -invsqrt2; Rot1[0][2] = 0.0;
        Rot1[1][0] = invsqrt2; Rot1[1][1] = invsqrt2;  Rot1[1][2] = 0.0;
        Rot1[2][0] = 0.0;      Rot1[2][1] = 0.0;       Rot1[2][2] = 1.0;
        for (int i = 0; i < 3; ++i) {
            Xprime[i] = Rot1[i][0] * LNhat[0] + Rot1[i][1] * LNhat[1] + Rot1[i][2] * LNhat[2];
        }
    } else {
        Rot1[0][0] = 1.0; Rot1[0][1] = 0.0; Rot1[0][2] = 0.0;
        Rot1[1][0] = 0.0; Rot1[1][1] = 1.0; Rot1[1][2] = 0.0;
        Rot1[2][0] = 0.0; Rot1[2][1] = 0.0; Rot1[2][2] = 1.0;
        Xprime[0] = LNhat[0]; Xprime[1] = LNhat[1]; Xprime[2] = LNhat[2];
    }

    double Yprime[3] = {0.0, Xprime[2], -Xprime[1]};
    double Y_norm = std::sqrt(Yprime[0]*Yprime[0] + Yprime[1]*Yprime[1] + Yprime[2]*Yprime[2]);
    double Y_clamp = (Y_norm < 1.0e-15) ? 1.0e-15 : Y_norm;
    Yprime[0] /= Y_clamp; Yprime[1] /= Y_clamp; Yprime[2] /= Y_clamp;

    double Zprime[3] = {
        Xprime[1]*Yprime[2] - Xprime[2]*Yprime[1],
        Xprime[2]*Yprime[0] - Xprime[0]*Yprime[2],
        Xprime[0]*Yprime[1] - Xprime[1]*Yprime[0]
    };
    double Z_norm = std::sqrt(Zprime[0]*Zprime[0] + Zprime[1]*Zprime[1] + Zprime[2]*Zprime[2]);
    double Z_clamp = (Z_norm < 1.0e-15) ? 1.0e-15 : Z_norm;
    Zprime[0] /= Z_clamp; Zprime[1] /= Z_clamp; Zprime[2] /= Z_clamp;

    double Rot2[3][3] = {
        {Xprime[0], Xprime[1], Xprime[2]},
        {Yprime[0], Yprime[1], Yprime[2]},
        {Zprime[0], Zprime[1], Zprime[2]}
    };

    auto rotate = [&](const double in[3], double out[3]) {
        double tmp[3];
        for (int i = 0; i < 3; ++i) {
            tmp[i] = Rot1[i][0] * in[0] + Rot1[i][1] * in[1] + Rot1[i][2] * in[2];
        }
        for (int i = 0; i < 3; ++i) {
            out[i] = Rot2[i][0] * tmp[0] + Rot2[i][1] * tmp[1] + Rot2[i][2] * tmp[2];
        }
    };

    double r_prime[3], p_prime[3], s1_w_prime[3], s2_w_prime[3];
    rotate(r_vec, r_prime);
    rotate(p_vec, p_prime);
    rotate(s1_w, s1_w_prime);
    rotate(s2_w, s2_w_prime);

    double r_polar = std::sqrt(r_prime[0]*r_prime[0] + r_prime[1]*r_prime[1] + r_prime[2]*r_prime[2]);
    double r_polar_clamp = (r_polar < 1.0e-15) ? 1.0e-15 : r_polar;
    double r_over_norm = r_prime[0] / r_polar_clamp;
    if (r_over_norm < -1.0) r_over_norm = -1.0;
    if (r_over_norm > 1.0) r_over_norm = 1.0;
    double theta_polar = std::acos(r_over_norm);
    double phi_polar = std::atan2(-r_prime[1], r_prime[2]);

    double r_cross_x[3] = {0.0, r_prime[2], -r_prime[1]};
    double r_cross_x_cross_r[3] = {
        r_cross_x[1] * r_prime[2] - r_cross_x[2] * r_prime[1],
        r_cross_x[2] * r_prime[0] - r_cross_x[0] * r_prime[2],
        r_cross_x[0] * r_prime[1] - r_cross_x[1] * r_prime[0]
    };
    double sin_theta = std::sin(theta_polar);
    if (sin_theta < 1.0e-12) sin_theta = 1.0e-12;
    double ptheta_polar = -(r_cross_x_cross_r[0]*p_prime[0] + r_cross_x_cross_r[1]*p_prime[1] + r_cross_x_cross_r[2]*p_prime[2]) / (r_polar_clamp * sin_theta);
    double pphi_polar = -(r_cross_x[0]*p_prime[0] + r_cross_x[1]*p_prime[1] + r_cross_x[2]*p_prime[2]);

    double sin_th = std::sin(theta_polar);
    double cos_th = std::cos(theta_polar);
    double sin_ph = std::sin(phi_polar);
    double cos_ph = std::cos(phi_polar);
    double sin_s = (sin_th < 1.0e-12) ? 1.0e-12 : sin_th;
    double r_s = (r_polar < 1.0e-12) ? 1.0e-12 : r_polar;

    double rcart[3] = {
        r_polar * cos_th,
        -r_polar * sin_th * sin_ph,
        r_polar * sin_th * cos_ph
    };
    double pcart[3] = {
        -ptheta_polar / r_s * sin_th,
        -ptheta_polar / r_s * cos_th * sin_ph - pphi_polar / (r_s * sin_s) * cos_ph,
        ptheta_polar / r_s * cos_th * cos_ph - pphi_polar / (r_s * sin_s) * sin_ph
    };

    double L_eval[3] = {
        rcart[1]*pcart[2] - rcart[2]*pcart[1],
        rcart[2]*pcart[0] - rcart[0]*pcart[2],
        rcart[0]*pcart[1] - rcart[1]*pcart[0]
    };

    HCoeffs h_inst;
    instantaneous_hcoeffs_c(eta, L_eval, s1_w_prime, s2_w_prime, &h_inst);

    return calcomega_polar_derivative_core_c(
        pphi_polar, r_polar, theta_polar, phi_polar, ptheta_polar,
        s1_w_prime, s2_w_prime, mass1, mass2, eta, M,
        h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
        h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
        h_inst.b3, h_inst.bb3
    );
}

inline double aligned_non_keplerian_omega_c(
    double r,
    double pphi,
    const double S1_weighted[3],
    const double S2_weighted[3],
    double mass1,
    double mass2,
    double eta,
    double M,
    const HCoeffs& h
) {
    double r_clamp = (r < 1.0e-15) ? 1.0e-15 : r;
    double py0 = pphi / r_clamp;
    double r_vec[3] = {r, 0.0, 0.0};
    auto f = [&](double py_eval) -> double {
        double p_vec[3] = {0.0, py_eval, 0.0};
        return eob_hamiltonian_c(
            r_vec, p_vec, S1_weighted, S2_weighted,
            mass1, mass2, eta, M,
            h.k0, h.k1, h.k2, h.k3, h.k4, h.k5, h.k5l,
            h.KK, h.d1, h.d1v2, h.dheffSS, h.dheffSSv2,
            h.b3, h.bb3,
            1, true
        ) / eta;
    };
    constexpr double STEP_DERIV = 2.0e-3;
    double dH_dpy = gsl_deriv_central_c(f, py0, STEP_DERIV);
    return std::fabs(dH_dpy / r_clamp);
}

inline double non_keplerian_vphi_point_c(
    double r,
    double omega,
    double phi,
    const double L_vec[3],
    const double s1[3],
    const double s2[3],
    double mass1,
    double mass2,
    double eta,
    double M,
    bool aligned_spins,
    const double* r_vec_in = nullptr,
    const double* p_vec_in = nullptr,
    const double* rdot_in = nullptr,
    const double* s1_weighted_in = nullptr,
    const double* s2_weighted_in = nullptr
) {
    double omega_circ = 0.0;
    if (aligned_spins) {
        double s1_scale = (mass1 * mass1) / (M * M);
        double s2_scale = (mass2 * mass2) / (M * M);
        double s1_w[3], s2_w[3];
        if (s1_weighted_in) {
            s1_w[0] = s1_weighted_in[0]; s1_w[1] = s1_weighted_in[1]; s1_w[2] = s1_weighted_in[2];
        } else {
            s1_w[0] = s1[0] * s1_scale; s1_w[1] = s1[1] * s1_scale; s1_w[2] = s1[2] * s1_scale;
        }
        if (s2_weighted_in) {
            s2_w[0] = s2_weighted_in[0]; s2_w[1] = s2_weighted_in[1]; s2_w[2] = s2_weighted_in[2];
        } else {
            s2_w[0] = s2[0] * s2_scale; s2_w[1] = s2[1] * s2_scale; s2_w[2] = s2[2] * s2_scale;
        }
        double pphi = std::sqrt(L_vec[0]*L_vec[0] + L_vec[1]*L_vec[1] + L_vec[2]*L_vec[2]);
        HCoeffs h_inst;
        instantaneous_hcoeffs_c(eta, L_vec, s1_w, s2_w, &h_inst);
        omega_circ = aligned_non_keplerian_omega_c(r, pphi, s1_w, s2_w, mass1, mass2, eta, M, h_inst);
    } else {
        double r_eval[3], p_eval[3];
        if (r_vec_in && p_vec_in) {
            r_eval[0] = r_vec_in[0]; r_eval[1] = r_vec_in[1]; r_eval[2] = r_vec_in[2];
            p_eval[0] = p_vec_in[0]; p_eval[1] = p_vec_in[1]; p_eval[2] = p_vec_in[2];
        } else {
            bool aligned_gauge = (
                (std::fabs(L_vec[0]) + std::fabs(L_vec[1]) < 1.0e-14) &&
                (std::fabs(s1[0]) + std::fabs(s1[1]) + std::fabs(s2[0]) + std::fabs(s2[1]) < 1.0e-14)
            );
            double phi_eval = aligned_gauge ? 0.0 : phi;
            double L_mag = std::sqrt(L_vec[0]*L_vec[0] + L_vec[1]*L_vec[1] + L_vec[2]*L_vec[2]);
            double L_clamp = (L_mag < 1.0e-15) ? 1.0e-15 : L_mag;
            double Lhat[3] = {L_vec[0] / L_clamp, L_vec[1] / L_clamp, L_vec[2] / L_clamp};
            double xhat[3] = {1.0, 0.0, 0.0};
            double yhat[3] = {0.0, 1.0, 0.0};
            double dot_xL = xhat[0]*Lhat[0] + xhat[1]*Lhat[1] + xhat[2]*Lhat[2];
            double e1[3] = {xhat[0] - dot_xL * Lhat[0], xhat[1] - dot_xL * Lhat[1], xhat[2] - dot_xL * Lhat[2]};
            double e1_norm = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
            if (e1_norm < 1.0e-14) {
                double dot_yL = yhat[0]*Lhat[0] + yhat[1]*Lhat[1] + yhat[2]*Lhat[2];
                e1[0] = yhat[0] - dot_yL * Lhat[0];
                e1[1] = yhat[1] - dot_yL * Lhat[1];
                e1[2] = yhat[2] - dot_yL * Lhat[2];
                e1_norm = std::sqrt(e1[0]*e1[0] + e1[1]*e1[1] + e1[2]*e1[2]);
            }
            double e1_clamp = (e1_norm < 1.0e-15) ? 1.0e-15 : e1_norm;
            e1[0] /= e1_clamp; e1[1] /= e1_clamp; e1[2] /= e1_clamp;

            double e2[3] = {
                Lhat[1]*e1[2] - Lhat[2]*e1[1],
                Lhat[2]*e1[0] - Lhat[0]*e1[2],
                Lhat[0]*e1[1] - Lhat[1]*e1[0]
            };
            double cos_p = std::cos(phi_eval), sin_p = std::sin(phi_eval);
            double n_hat[3] = {
                cos_p * e1[0] + sin_p * e2[0],
                cos_p * e1[1] + sin_p * e2[1],
                cos_p * e1[2] + sin_p * e2[2]
            };
            double n_norm = std::sqrt(n_hat[0]*n_hat[0] + n_hat[1]*n_hat[1] + n_hat[2]*n_hat[2]);
            double n_clamp = (n_norm < 1.0e-15) ? 1.0e-15 : n_norm;
            n_hat[0] /= n_clamp; n_hat[1] /= n_clamp; n_hat[2] /= n_clamp;

            r_eval[0] = r * n_hat[0];
            r_eval[1] = r * n_hat[1];
            r_eval[2] = r * n_hat[2];

            double n_cross_L[3] = {
                n_hat[1]*L_vec[2] - n_hat[2]*L_vec[1],
                n_hat[2]*L_vec[0] - n_hat[0]*L_vec[2],
                n_hat[0]*L_vec[1] - n_hat[1]*L_vec[0]
            };
            double r_denom = (r < 1.0e-12) ? 1.0e-12 : r;
            p_eval[0] = -n_cross_L[0] / r_denom;
            p_eval[1] = -n_cross_L[1] / r_denom;
            p_eval[2] = -n_cross_L[2] / r_denom;
        }

        double val = calcomega_lal_polar_derivative_point_c(
            r_eval, p_eval, s1, s2, mass1, mass2, eta, M,
            rdot_in, s1_weighted_in, s2_weighted_in
        );
        omega_circ = std::fabs(val);
    }

    double om_clamp = (omega_circ < 1.0e-12) ? 1.0e-12 : omega_circ;
    double r_clamp = (r < 1.0e-12) ? 1.0e-12 : r;
    double denom_coeff = om_clamp * om_clamp * r_clamp * r_clamp * r_clamp;
    double coeff = 1.0 / denom_coeff;
    if (coeff < 1.0e-12) coeff = 1.0e-12;
    double vphi = r * std::cbrt(coeff) * std::fabs(omega);
    return (vphi < 1.0e-12) ? 1.0e-12 : vphi;
}

inline double non_keplerian_vphi_c(
    const double r_vec[3],
    const double p_vec[3],
    const double dxdt[3],
    const double S1_weighted[3],
    const double S2_weighted[3],
    double mass1,
    double mass2,
    double eta,
    double M,
    double omega,
    double r
) {
    double omega_circ_val = calcomega_lal_polar_derivative_point_c(
        r_vec, p_vec, S1_weighted, S2_weighted,
        mass1, mass2, eta, M,
        dxdt, S1_weighted, S2_weighted
    );
    double omega_circ = std::fabs(omega_circ_val);
    double om_clamp = (omega_circ < 1.0e-12) ? 1.0e-12 : omega_circ;
    double r_clamp = (r < 1.0e-12) ? 1.0e-12 : r;
    double denom_coeff = om_clamp * om_clamp * r_clamp * r_clamp * r_clamp;
    double coeff = 1.0 / denom_coeff;
    if (coeff < 1.0e-12) coeff = 1.0e-12;
    double vphi = r * std::cbrt(coeff) * std::fabs(omega);
    return (vphi < 1.0e-12) ? 1.0e-12 : vphi;
}

inline double factorized_flux_c(
    double r,
    double omega,
    double H,
    double L_mag,
    const double L_vec[3],
    const double S1_weighted[3],
    const double S2_weighted[3],
    double mass1,
    double mass2,
    double eta,
    double M,
    double vphi_nk,
    const double prefix_table[9][9]
) {
    double v = std::cbrt(std::max(std::fabs(omega), 1.0e-12));
    double mass1_norm = mass1 / M;
    double mass2_norm = mass2 / M;
    double L_clamp = (L_mag < 1.0e-15) ? 1.0e-15 : L_mag;
    double spin_axis[3] = {
        L_vec[0] / L_clamp,
        L_vec[1] / L_clamp,
        L_vec[2] / L_clamp
    };
    double chi1_flux = (S1_weighted[0]*spin_axis[0] + S1_weighted[1]*spin_axis[1] + S1_weighted[2]*spin_axis[2]) / (mass1_norm * mass1_norm);
    double chi2_flux = (S2_weighted[0]*spin_axis[0] + S2_weighted[1]*spin_axis[1] + S2_weighted[2]*spin_axis[2]) / (mass2_norm * mass2_norm);
    double chiS_flux = 0.5 * (chi1_flux + chi2_flux);
    double chiA_flux = 0.5 * (chi1_flux - chi2_flux);
    double dM_flux = (mass1 - mass2) / M;
    double tplspin_flux = (1.0 - 2.0 * eta) * chiS_flux + dM_flux * chiA_flux;

    double flux = 0.0;

    for (int l = 2; l <= 8; ++l) {
        for (int m = 1; m <= l; ++m) {
            int eps = (l + m) % 2;
            double rho_val, aux_val;
            rho_aux_flux_c(l, m, v, eta, chi1_flux, chi2_flux, mass1, mass2, tplspin_flux, &rho_val, &aux_val);
            
            double rholm_pwrl;
            if (std::fabs(eta - 0.25) < 1.0e-12 && (m % 2 != 0)) {
                rholm_pwrl = aux_val;
            } else {
                rholm_pwrl = std::pow(rho_val, l) + aux_val;
            }

            double h_newt = prefix_table[l][m] * std::pow(vphi_nk, l + eps);
            double S_eff = (eps == 0) ? ((H * H - 1.0) / (2.0 * eta) + 1.0) : (v * L_mag);
            
            // Tail factor
            double k = std::fabs((double)m * omega);
            double hathatk = (H * k < 1.0e-12) ? 1.0e-12 : (H * k);
            constexpr double PI_VAL = 3.14159265358979323846;
            double four_pi_k = 4.0 * PI_VAL * hathatk;
            double exp_neg = std::exp(-four_pi_k);
            double denom_exp = (1.0 - exp_neg < 1.0e-16) ? 1.0e-16 : (1.0 - exp_neg);
            double pref_tail = std::sqrt(four_pi_k / denom_exp) / factorial_c(l);
            double prod = 1.0;
            for (int s = 1; s <= l; ++s) {
                prod *= (4.0 * hathatk * hathatk + (double)(s * s));
            }
            double tail = pref_tail * std::sqrt(prod);

            double amp = h_newt * S_eff * rholm_pwrl * tail;
            double flux_mode = ((double)m * omega) * ((double)m * omega) * (amp * amp);
            flux += flux_mode;
        }
    }
    constexpr double PI_VAL = 3.14159265358979323846;
    return flux / (8.0 * PI_VAL * eta);
}

struct EOBDynamicsContext {
    double mass1;
    double mass2;
    double M;
    double eta;
    double mass1_norm;
    double mass2_norm;
    double s1_scale;
    double s2_scale;
    double prefix_table[9][9];
};

inline void init_eob_dynamics_context(double mass1, double mass2, EOBDynamicsContext* ctx) {
    ctx->mass1 = mass1;
    ctx->mass2 = mass2;
    ctx->M = mass1 + mass2;
    ctx->eta = mass1 * mass2 / (ctx->M * ctx->M);
    ctx->mass1_norm = mass1 / ctx->M;
    ctx->mass2_norm = mass2 / ctx->M;
    ctx->s1_scale = ctx->mass1_norm * ctx->mass1_norm;
    ctx->s2_scale = ctx->mass2_norm * ctx->mass2_norm;
    for (int l = 0; l <= 8; ++l) {
        for (int m = 0; m <= 8; ++m) {
            ctx->prefix_table[l][m] = 0.0;
        }
    }
    for (int l = 2; l <= 8; ++l) {
        for (int m = 1; m <= l; ++m) {
            int eps = (l + m) % 2;
            double p = calc_prefix_abs_c(l, m, mass1, mass2, ctx->eta);
            double y = abs_scalar_sph_pi_over2_c(l - eps, -m);
            ctx->prefix_table[l][m] = p * y;
        }
    }
}

// Complete 14D in-engine C++ Cartesian RHS evaluation
inline void eob_rhs_cartesian_c(
    double t,
    const double* y,
    double* dydt,
    const EOBDynamicsContext& ctx
) {
    const double r_vec[3] = {y[0], y[1], y[2]};
    const double p_vec[3] = {y[3], y[4], y[5]};
    const double S1_weighted[3] = {y[6], y[7], y[8]};
    const double S2_weighted[3] = {y[9], y[10], y[11]};

    double r2 = r_vec[0]*r_vec[0] + r_vec[1]*r_vec[1] + r_vec[2]*r_vec[2];
    double r = std::sqrt(r2);
    double r_clamp = (r < 1.0e-15) ? 1.0e-15 : r;
    double n_hat[3] = {r_vec[0] / r_clamp, r_vec[1] / r_clamp, r_vec[2] / r_clamp};

    double L_vec[3] = {
        r_vec[1]*p_vec[2] - r_vec[2]*p_vec[1],
        r_vec[2]*p_vec[0] - r_vec[0]*p_vec[2],
        r_vec[0]*p_vec[1] - r_vec[1]*p_vec[0]
    };
    double L_mag = std::sqrt(L_vec[0]*L_vec[0] + L_vec[1]*L_vec[1] + L_vec[2]*L_vec[2]);
    double L_clamp = (L_mag < 1.0e-15) ? 1.0e-15 : L_mag;

    HCoeffs h_inst;
    instantaneous_hcoeffs_c(ctx.eta, L_vec, S1_weighted, S2_weighted, &h_inst);

    // Tortoise prelude
    double u = 1.0 / r_clamp;
    double u2 = u * u;
    double u3 = u2 * u;
    double u4 = u2 * u2;
    double u5 = u4 * u;
    double logu = std::log(u);
    double sigma_vec[3] = {S1_weighted[0] + S2_weighted[0], S1_weighted[1] + S2_weighted[1], S1_weighted[2] + S2_weighted[2]};
    double a = std::sqrt(sigma_vec[0]*sigma_vec[0] + sigma_vec[1]*sigma_vec[1] + sigma_vec[2]*sigma_vec[2]);
    double a2 = a * a;
    double w2 = r2 + a2;

    double D_term = 1.0 + 6.0 * ctx.eta * u2 + 2.0 * (26.0 - 3.0 * ctx.eta) * ctx.eta * u3;
    double D = 1.0 + std::log(D_term);
    double eobD_r = (u2 / (D * D)) * (12.0 * ctx.eta * u + 6.0 * (26.0 - 3.0 * ctx.eta) * ctx.eta * u2) / D_term;

    double m1PlusetaKK = -1.0 + ctx.eta * h_inst.KK;
    double bulk = 1.0 / (m1PlusetaKK * m1PlusetaKK) + (2.0 * u) / m1PlusetaKK + a2 * u2;
    double log_arg = 1.0 + h_inst.k1 * u + h_inst.k2 * u2 + h_inst.k3 * u3 + h_inst.k4 * u4 + h_inst.k5 * u5 + h_inst.k5l * u5 * logu;
    double logTerms = 1.0 + ctx.eta * h_inst.k0 + ctx.eta * std::log(std::fabs(log_arg));
    double deltaU = bulk * logTerms;
    double deltaT = r2 * deltaU;
    double dlogarg_du = h_inst.k1 + u * (2.0 * h_inst.k2 + u * (3.0 * h_inst.k3 + u * (4.0 * h_inst.k4 + 5.0 * (h_inst.k5 + h_inst.k5l * logu) * u)));
    double deltaU_u = 2.0 * (1.0 / m1PlusetaKK + a2 * u) * logTerms + bulk * (ctx.eta * dlogarg_du) / log_arg;
    double deltaU_r = -u2 * deltaU_u;
    double deltaR = deltaT * D;
    double csi = std::sqrt(std::max(deltaT * deltaR, 0.0)) / w2;
    double csi_fac = (csi < 1.0e-15) ? 1.0e-15 : csi;
    double dcsi = csi * (2.0 / r_clamp + deltaU_r / deltaU)
                 + (csi * csi * csi) / (2.0 * r2 * r2 * deltaU * deltaU) * (r_clamp * (-4.0 * w2) / D - eobD_r * (w2 * w2));

    // Tortoise matrices
    double Tmat[3][3], invTmat[3][3], dTijdXk[3][3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            double tij = (r_vec[i] * r_vec[j] / r2) * (csi_fac - 1.0);
            double inv_tij = -((csi_fac - 1.0) / csi_fac) * (r_vec[i] * r_vec[j] / r2);
            if (i == j) {
                tij += 1.0;
                inv_tij += 1.0;
            }
            Tmat[i][j] = tij;
            Tmat[j][i] = tij;
            invTmat[i][j] = inv_tij;
            invTmat[j][i] = inv_tij;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                double delta_jk = (j == k) ? 1.0 : 0.0;
                double delta_ik = (i == k) ? 1.0 : 0.0;
                dTijdXk[i][j][k] = (r_vec[i] * delta_jk + delta_ik * r_vec[j]) * (csi_fac - 1.0) / r2
                                 + (r_vec[i] * r_vec[j] * r_vec[k] / (r2 * r_clamp)) * (-2.0 / r_clamp * (csi_fac - 1.0) + dcsi);
            }
        }
    }

    // dH/dpvec
    double dH_dpvec[3];
    double p_tmp[3] = {p_vec[0], p_vec[1], p_vec[2]};
    constexpr double STEP_DERIV = 2.0e-3;
    for (int axis = 0; axis < 3; ++axis) {
        auto f = [&](double p_val) -> double {
            double orig = p_tmp[axis];
            p_tmp[axis] = p_val;
            double h_val = eob_hamiltonian_c(
                r_vec, p_tmp, S1_weighted, S2_weighted,
                ctx.mass1, ctx.mass2, ctx.eta, ctx.M,
                h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
                h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
                h_inst.b3, h_inst.bb3,
                1, true
            ) / ctx.eta;
            p_tmp[axis] = orig;
            return h_val;
        };
        dH_dpvec[axis] = gsl_deriv_central_c(f, p_vec[axis], STEP_DERIV);
    }

    // dxdt = Tmat * dH_dpvec
    double dxdt[3] = {
        Tmat[0][0]*dH_dpvec[0] + Tmat[0][1]*dH_dpvec[1] + Tmat[0][2]*dH_dpvec[2],
        Tmat[1][0]*dH_dpvec[0] + Tmat[1][1]*dH_dpvec[1] + Tmat[1][2]*dH_dpvec[2],
        Tmat[2][0]*dH_dpvec[0] + Tmat[2][1]*dH_dpvec[1] + Tmat[2][2]*dH_dpvec[2]
    };

    // Precessing orbital frequency omega
    double rCrossV_x = r_vec[1] * dxdt[2] - r_vec[2] * dxdt[1];
    double rCrossV_y = r_vec[2] * dxdt[0] - r_vec[0] * dxdt[2];
    double rCrossV_z = r_vec[0] * dxdt[1] - r_vec[1] * dxdt[0];
    double omega = std::sqrt(rCrossV_x*rCrossV_x + rCrossV_y*rCrossV_y + rCrossV_z*rCrossV_z) / r2;
    if (omega < 1.0e-12) omega = 1.0e-12;

    // Non-Keplerian vphi_nk
    double vphi_nk = non_keplerian_vphi_c(
        r_vec, p_vec, dxdt, S1_weighted, S2_weighted,
        ctx.mass1, ctx.mass2, ctx.eta, ctx.M, omega, r_clamp
    );

    // Hamiltonian H_val for radiation reaction
    double H_val = eob_hamiltonian_c(
        r_vec, p_vec, S1_weighted, S2_weighted,
        ctx.mass1, ctx.mass2, ctx.eta, ctx.M,
        h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
        h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
        h_inst.b3, h_inst.bb3,
        1, true
    );

    // Factorized flux
    double flux = factorized_flux_c(
        r_clamp, omega, H_val, L_mag, L_vec,
        S1_weighted, S2_weighted, ctx.mass1, ctx.mass2, ctx.eta, ctx.M,
        vphi_nk, ctx.prefix_table
    );

    // dH/dx at fixed non-tortoise momentum
    double pr_star = p_vec[0]*n_hat[0] + p_vec[1]*n_hat[1] + p_vec[2]*n_hat[2];
    double p_non_tortoise[3] = {
        p_vec[0] - n_hat[0] * pr_star * (csi_fac - 1.0) / csi_fac,
        p_vec[1] - n_hat[1] * pr_star * (csi_fac - 1.0) / csi_fac,
        p_vec[2] - n_hat[2] * pr_star * (csi_fac - 1.0) / csi_fac
    };
    double dH_dx[3];
    double r_tmp[3] = {r_vec[0], r_vec[1], r_vec[2]};
    for (int axis = 0; axis < 3; ++axis) {
        auto f = [&](double r_val) -> double {
            double orig = r_tmp[axis];
            r_tmp[axis] = r_val;
            double h_val = eob_hamiltonian_c(
                r_tmp, p_non_tortoise, S1_weighted, S2_weighted,
                ctx.mass1, ctx.mass2, ctx.eta, ctx.M,
                h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
                h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
                h_inst.b3, h_inst.bb3,
                2, false
            ) / ctx.eta;

            r_tmp[axis] = orig;
            return h_val;
        };
        dH_dx[axis] = gsl_deriv_central_c(f, r_vec[axis], STEP_DERIV);
    }

    // dpdt
    double pdot_t1[3] = {
        -(dH_dx[0]*Tmat[0][0] + dH_dx[1]*Tmat[0][1] + dH_dx[2]*Tmat[0][2]),
        -(dH_dx[0]*Tmat[1][0] + dH_dx[1]*Tmat[1][1] + dH_dx[2]*Tmat[1][2]),
        -(dH_dx[0]*Tmat[2][0] + dH_dx[1]*Tmat[2][1] + dH_dx[2]*Tmat[2][2])
    };
    double flux_denom = omega * L_mag;
    if (flux_denom < 1.0e-12) flux_denom = 1.0e-12;
    double flux_fac = -flux / flux_denom;
    double pdot_rr[3] = {
        flux_fac * p_vec[0],
        flux_fac * p_vec[1],
        flux_fac * p_vec[2]
    };
    double pdot_t3[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        double acc_i = 0.0;
        for (int j = 0; j < 3; ++j) {
            double acc_ij = 0.0;
            for (int l = 0; l < 3; ++l) {
                double acc_ijl = 0.0;
                for (int k = 0; k < 3; ++k) {
                    acc_ijl += dTijdXk[i][k][j] * invTmat[k][l];
                }
                acc_ij += acc_ijl * p_vec[l];
            }
            acc_i += acc_ij * dxdt[j];
        }
        pdot_t3[i] = acc_i;
    }
    double dpdt[3] = {
        pdot_t1[0] + pdot_rr[0] + pdot_t3[0],
        pdot_t1[1] + pdot_rr[1] + pdot_t3[1],
        pdot_t1[2] + pdot_rr[2] + pdot_t3[2]
    };

    // Spin derivatives dH/dS1, dH/dS2
    double dH_dS1[3], dH_dS2[3];
    double s1_tmp[3] = {S1_weighted[0], S1_weighted[1], S1_weighted[2]};
    double s2_tmp[3] = {S2_weighted[0], S2_weighted[1], S2_weighted[2]};
    double step1 = STEP_DERIV * ctx.s1_scale;
    double step2 = STEP_DERIV * ctx.s2_scale;
    for (int axis = 0; axis < 3; ++axis) {
        auto f1 = [&](double s_val) -> double {
            double orig = s1_tmp[axis];
            s1_tmp[axis] = s_val;
            double h_val = eob_hamiltonian_c(
                r_vec, p_vec, s1_tmp, s2_tmp,
                ctx.mass1, ctx.mass2, ctx.eta, ctx.M,
                h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
                h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
                h_inst.b3, h_inst.bb3,
                1, true
            ) / ctx.eta;
            s1_tmp[axis] = orig;
            return h_val;
        };
        dH_dS1[axis] = gsl_deriv_central_c(f1, S1_weighted[axis], step1);

        auto f2 = [&](double s_val) -> double {
            double orig = s2_tmp[axis];
            s2_tmp[axis] = s_val;
            double h_val = eob_hamiltonian_c(
                r_vec, p_vec, s1_tmp, s2_tmp,
                ctx.mass1, ctx.mass2, ctx.eta, ctx.M,
                h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
                h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
                h_inst.b3, h_inst.bb3,
                1, true
            ) / ctx.eta;
            s2_tmp[axis] = orig;
            return h_val;
        };
        dH_dS2[axis] = gsl_deriv_central_c(f2, S2_weighted[axis], step2);
    }


    double dS1_weighted[3] = {
        ctx.eta * (dH_dS1[1] * S1_weighted[2] - dH_dS1[2] * S1_weighted[1]),
        ctx.eta * (dH_dS1[2] * S1_weighted[0] - dH_dS1[0] * S1_weighted[2]),
        ctx.eta * (dH_dS1[0] * S1_weighted[1] - dH_dS1[1] * S1_weighted[0])
    };
    double dS2_weighted[3] = {
        ctx.eta * (dH_dS2[1] * S2_weighted[2] - dH_dS2[2] * S2_weighted[1]),
        ctx.eta * (dH_dS2[2] * S2_weighted[0] - dH_dS2[0] * S2_weighted[2]),
        ctx.eta * (dH_dS2[0] * S2_weighted[1] - dH_dS2[1] * S2_weighted[0])
    };

    // Precession phase rates (phase_dot, zeta_dot)
    double Lx = L_vec[0], Ly = L_vec[1], Lz = L_vec[2];
    double Lhatx = Lx / L_clamp;
    double Lhaty = Ly / L_clamp;
    double Lhatz = Lz / L_clamp;

    double dLx = dxdt[1] * p_vec[2] - dxdt[2] * p_vec[1] + r_vec[1] * dpdt[2] - r_vec[2] * dpdt[1];
    double dLy = dxdt[2] * p_vec[0] - dxdt[0] * p_vec[2] + r_vec[2] * dpdt[0] - r_vec[0] * dpdt[2];
    double dLz = dxdt[0] * p_vec[1] - dxdt[1] * p_vec[0] + r_vec[0] * dpdt[1] - r_vec[1] * dpdt[0];

    double dMagL = (Lx * dLx + Ly * dLy + Lz * dLz) / L_clamp;
    double dLhatx = (dLx * L_clamp - Lx * dMagL) / (L_clamp * L_clamp);
    double dLhaty = (dLy * L_clamp - Ly * dMagL) / (L_clamp * L_clamp);

    double alphadotcosi = 0.0;
    if (Lhatx != 0.0 || Lhaty != 0.0) {
        alphadotcosi = Lhatz * (Lhatx * dLhaty - Lhaty * dLhatx) / (Lhatx * Lhatx + Lhaty * Lhaty);
    }
    double phase_dot = omega - alphadotcosi;
    double zeta_dot = alphadotcosi;

    // Store into dydt
    dydt[0] = dxdt[0]; dydt[1] = dxdt[1]; dydt[2] = dxdt[2];
    dydt[3] = dpdt[0]; dydt[4] = dpdt[1]; dydt[5] = dpdt[2];
    dydt[6] = dS1_weighted[0]; dydt[7] = dS1_weighted[1]; dydt[8] = dS1_weighted[2];
    dydt[9] = dS2_weighted[0]; dydt[10] = dS2_weighted[1]; dydt[11] = dS2_weighted[2];
    dydt[12] = phase_dot;
    dydt[13] = zeta_dot;
}

torch::Tensor eob_rhs_cartesian_native(
    const torch::Tensor& y_tensor,
    double mass1,
    double mass2,
    double t
) {
    validate_tensor(y_tensor, "y");
    TORCH_CHECK(y_tensor.numel() == 14, "y must be 14-dimensional");
    auto options = y_tensor.options();
    auto dydt_t = torch::empty({14}, options);

    EOBDynamicsContext ctx;
    init_eob_dynamics_context(mass1, mass2, &ctx);

    eob_rhs_cartesian_c(
        t,
        y_tensor.data_ptr<double>(),
        dydt_t.data_ptr<double>(),
        ctx
    );
    return dydt_t;
}

// Initial conditions spherical derivatives and root finding
inline void ic_spherical_derivatives_c(
    double mass1,
    double mass2,
    double eta,
    double M,
    double r,
    double ptheta,
    double pphi,
    const double S1[3],
    const double S2[3],
    double* out_dHdr,
    double* out_dHdptheta,
    double* out_dHdpphi,
    HCoeffs* inout_hcoeffs,
    double py_cart = std::numeric_limits<double>::quiet_NaN(),
    double pz_cart = std::numeric_limits<double>::quiet_NaN(),
    const double* S1_weighted_override = nullptr,
    const double* S2_weighted_override = nullptr
) {
    double py = std::isnan(py_cart) ? (pphi / r) : py_cart;
    double pz = std::isnan(pz_cart) ? (-ptheta / r) : pz_cart;
    double r_vec[3] = {r, 0.0, 0.0};
    double p_vec[3] = {0.0, py, pz};
    double L_vec[3] = {
        0.0,
        -r * pz,
        r * py
    };
    double s1_m2[3], s2_m2[3];
    if (S1_weighted_override != nullptr) {
        s1_m2[0] = S1_weighted_override[0];
        s1_m2[1] = S1_weighted_override[1];
        s1_m2[2] = S1_weighted_override[2];
    } else {
        double s1_scale = (mass1 * mass1) / (M * M);
        s1_m2[0] = S1[0] * s1_scale;
        s1_m2[1] = S1[1] * s1_scale;
        s1_m2[2] = S1[2] * s1_scale;
    }
    if (S2_weighted_override != nullptr) {
        s2_m2[0] = S2_weighted_override[0];
        s2_m2[1] = S2_weighted_override[1];
        s2_m2[2] = S2_weighted_override[2];
    } else {
        double s2_scale = (mass2 * mass2) / (M * M);
        s2_m2[0] = S2[0] * s2_scale;
        s2_m2[1] = S2[1] * s2_scale;
        s2_m2[2] = S2[2] * s2_scale;
    }

    HCoeffs entry_hcoeffs = *inout_hcoeffs;
    HCoeffs h_inst;
    instantaneous_hcoeffs_c(eta, L_vec, s1_m2, s2_m2, &h_inst);
    *inout_hcoeffs = h_inst;

    double r_clamp = (r < 1.0e-15) ? 1.0e-15 : r;
    double r2 = r_clamp * r_clamp;
    double u = 1.0 / r_clamp;
    double u2 = u * u;
    double u3 = u2 * u;
    double u4 = u2 * u2;
    double u5 = u4 * u;
    double logu = std::log(u);
    double sigma_vec[3] = {s1_m2[0] + s2_m2[0], s1_m2[1] + s2_m2[1], s1_m2[2] + s2_m2[2]};
    double a2 = sigma_vec[0]*sigma_vec[0] + sigma_vec[1]*sigma_vec[1] + sigma_vec[2]*sigma_vec[2];
    double a = std::sqrt(a2);
    double w2 = r2 + a2;

    double D_term = 1.0 + 6.0 * eta * u2 + 2.0 * (26.0 - 3.0 * eta) * eta * u3;
    double D = 1.0 + std::log(D_term);
    double eobD_r = (u2 / (D * D)) * (12.0 * eta * u + 6.0 * (26.0 - 3.0 * eta) * eta * u2) / D_term;

    double m1PlusetaKK = -1.0 + eta * entry_hcoeffs.KK;
    double bulk = 1.0 / (m1PlusetaKK * m1PlusetaKK) + (2.0 * u) / m1PlusetaKK + a2 * u2;
    double log_arg = 1.0 + entry_hcoeffs.k1 * u + entry_hcoeffs.k2 * u2 + entry_hcoeffs.k3 * u3 + entry_hcoeffs.k4 * u4 + entry_hcoeffs.k5 * u5 + entry_hcoeffs.k5l * u5 * logu;
    double logTerms = 1.0 + eta * entry_hcoeffs.k0 + eta * std::log(std::fabs(log_arg));
    double deltaU = bulk * logTerms;
    double deltaT = r2 * deltaU;
    double dlogarg_du = entry_hcoeffs.k1 + u * (2.0 * entry_hcoeffs.k2 + u * (3.0 * entry_hcoeffs.k3 + u * (4.0 * entry_hcoeffs.k4 + 5.0 * (entry_hcoeffs.k5 + entry_hcoeffs.k5l * logu) * u)));
    double deltaU_u = 2.0 * (1.0 / m1PlusetaKK + a2 * u) * logTerms + bulk * (eta * dlogarg_du) / log_arg;
    double deltaU_r = -u2 * deltaU_u;
    double deltaR = deltaT * D;
    double csi = std::sqrt(std::max(deltaT * deltaR, 0.0)) / w2;
    double csi_fac = (csi < 1.0e-15) ? 1.0e-15 : csi;
    double dcsi = csi * (2.0 / r_clamp + deltaU_r / deltaU)
                 + (csi * csi * csi) / (2.0 * r2 * r2 * deltaU * deltaU) * (r_clamp * (-4.0 * w2) / D - eobD_r * (w2 * w2));

    double Tmat[3][3], invTmat[3][3], dTijdXk[3][3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            double tij = (r_vec[i] * r_vec[j] / r2) * (csi_fac - 1.0);
            double inv_tij = -((csi_fac - 1.0) / csi_fac) * (r_vec[i] * r_vec[j] / r2);
            if (i == j) {
                tij += 1.0;
                inv_tij += 1.0;
            }
            Tmat[i][j] = tij;
            Tmat[j][i] = tij;
            invTmat[i][j] = inv_tij;
            invTmat[j][i] = inv_tij;
        }
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                double delta_jk = (j == k) ? 1.0 : 0.0;
                double delta_ik = (i == k) ? 1.0 : 0.0;
                dTijdXk[i][j][k] = (r_vec[i] * delta_jk + delta_ik * r_vec[j]) * (csi_fac - 1.0) / r2
                                 + (r_vec[i] * r_vec[j] * r_vec[k] / (r2 * r_clamp)) * (-2.0 / r_clamp * (csi_fac - 1.0) + dcsi);
            }
        }
    }

    double p_non_tortoise[3] = {p_vec[0], p_vec[1], p_vec[2]};

    constexpr double STEP_DERIV = 2.0e-3;
    double dH_dx[3];
    double r_tmp[3] = {r_vec[0], r_vec[1], r_vec[2]};
    for (int axis = 0; axis < 3; ++axis) {
        auto f = [&](double r_val) -> double {
            double orig = r_tmp[axis];
            r_tmp[axis] = r_val;
            double h_val = eob_hamiltonian_c(
                r_tmp, p_non_tortoise, s1_m2, s2_m2,
                mass1, mass2, eta, M,
                h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
                h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
                h_inst.b3, h_inst.bb3,
                2, false
            ) / eta;
            r_tmp[axis] = orig;
            return h_val;
        };
        dH_dx[axis] = gsl_deriv_central_c(f, r_vec[axis], STEP_DERIV);
    }

    double dH_dpvec[3];
    double p_tmp[3] = {p_vec[0], p_vec[1], p_vec[2]};
    for (int axis = 0; axis < 3; ++axis) {
        auto f = [&](double p_val) -> double {
            double orig = p_tmp[axis];
            p_tmp[axis] = p_val;
            double h_val = eob_hamiltonian_c(
                r_vec, p_tmp, s1_m2, s2_m2,
                mass1, mass2, eta, M,
                h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
                h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
                h_inst.b3, h_inst.bb3,
                1, true
            ) / eta;
            p_tmp[axis] = orig;
            return h_val;
        };
        dH_dpvec[axis] = gsl_deriv_central_c(f, p_vec[axis], STEP_DERIV);
    }

    double dxdt[3] = {
        Tmat[0][0]*dH_dpvec[0] + Tmat[0][1]*dH_dpvec[1] + Tmat[0][2]*dH_dpvec[2],
        Tmat[1][0]*dH_dpvec[0] + Tmat[1][1]*dH_dpvec[1] + Tmat[1][2]*dH_dpvec[2],
        Tmat[2][0]*dH_dpvec[0] + Tmat[2][1]*dH_dpvec[1] + Tmat[2][2]*dH_dpvec[2]
    };

    double pdot_t1[3] = {
        -(dH_dx[0]*Tmat[0][0] + dH_dx[1]*Tmat[0][1] + dH_dx[2]*Tmat[0][2]),
        -(dH_dx[0]*Tmat[1][0] + dH_dx[1]*Tmat[1][1] + dH_dx[2]*Tmat[1][2]),
        -(dH_dx[0]*Tmat[2][0] + dH_dx[1]*Tmat[2][1] + dH_dx[2]*Tmat[2][2])
    };

    double pdot_t3[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i) {
        double acc_i = 0.0;
        for (int j = 0; j < 3; ++j) {
            double acc_ij = 0.0;
            for (int l = 0; l < 3; ++l) {
                double acc_ijl = 0.0;
                for (int k = 0; k < 3; ++k) {
                    acc_ijl += dTijdXk[i][k][j] * invTmat[k][l];
                }
                acc_ij += acc_ijl * p_vec[l];
            }
            acc_i += acc_ij * dxdt[j];
        }
        pdot_t3[i] = acc_i;
    }

    double dpdt[3] = {
        pdot_t1[0] + pdot_t3[0],
        pdot_t1[1] + pdot_t3[1],
        pdot_t1[2] + pdot_t3[2]
    };

    double dHdx = -dpdt[0];
    double dHdpy = dxdt[1];
    double dHdpz = dxdt[2];

    double r2_denom = (r2 < 1.0e-15) ? 1.0e-15 : r2;
    double r_denom = (r_clamp < 1.0e-15) ? 1.0e-15 : r_clamp;

    *out_dHdr = dHdx - dHdpy * pphi / r2_denom + dHdpz * ptheta / r2_denom;
    *out_dHdptheta = -dHdpz / r_denom;
    *out_dHdpphi = dHdpy / r_denom;
}

inline bool gsl_multiroot_hybrids_3d_c(
    const std::function<void(const double[3], double[3])>& residual,
    const double guess[3],
    double epsabs,
    int max_iter,
    double sol_x[3],
    double final_f[3]
) {
    constexpr double SQRT_DBL_EPSILON = 1.4901161193847656e-08;

    double x[3] = {guess[0], guess[1], guess[2]};
    double f[3];
    residual(x, f);

    auto enorm = [](const double v[3]) {
        return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    };
    auto scaled_enorm = [](const double diag[3], const double v[3]) {
        double d0 = diag[0]*v[0], d1 = diag[1]*v[1], d2 = diag[2]*v[2];
        return std::sqrt(d0*d0 + d1*d1 + d2*d2);
    };

    auto fdjac = [&](const double cur_x[3], const double cur_f[3], double J[3][3]) {
        double x_tr[3] = {cur_x[0], cur_x[1], cur_x[2]};
        for (int j = 0; j < 3; ++j) {
            double xj = cur_x[j];
            double step = SQRT_DBL_EPSILON * std::fabs(xj);
            if (step == 0.0) step = SQRT_DBL_EPSILON;
            x_tr[j] = xj + step;
            double f_tr[3];
            residual(x_tr, f_tr);
            x_tr[j] = xj;
            for (int i = 0; i < 3; ++i) {
                J[i][j] = (f_tr[i] - cur_f[i]) / step;
            }
        }
    };

    auto update_diag = [](const double J[3][3], double diag[3], bool init) {
        for (int j = 0; j < 3; ++j) {
            double tot = J[0][j]*J[0][j] + J[1][j]*J[1][j] + J[2][j]*J[2][j];
            if (tot == 0.0) tot = 1.0;
            double norm = std::sqrt(tot);
            if (init) {
                diag[j] = norm;
            } else if (norm > diag[j]) {
                diag[j] = norm;
            }
        }
    };

    auto qr_decomp = [](const double J[3][3], double Q[3][3], double R[3][3]) {
        double A[3][3];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) A[i][j] = J[i][j];
        double tau[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i) {
            int len = 3 - i;
            if (len == 1) {
                tau[i] = 0.0;
            } else {
                double xnorm = 0.0;
                for (int k = i + 1; k < 3; ++k) xnorm += A[k][i] * A[k][i];
                xnorm = std::sqrt(xnorm);
                if (xnorm == 0.0) {
                    tau[i] = 0.0;
                } else {
                    double alpha = A[i][i];
                    double sign = (alpha >= 0.0) ? 1.0 : -1.0;
                    double beta = -sign * std::hypot(alpha, xnorm);
                    tau[i] = (beta - alpha) / beta;
                    double scale = 1.0 / (alpha - beta);
                    for (int k = i + 1; k < 3; ++k) A[k][i] *= scale;
                    A[i][i] = beta;
                }
            }
            if (i + 1 < 3 && tau[i] != 0.0) {
                double first = A[i][i];
                A[i][i] = 1.0;
                for (int j = i + 1; j < 3; ++j) {
                    double dot = 0.0;
                    for (int k = i; k < 3; ++k) dot += A[k][i] * A[k][j];
                    for (int k = i; k < 3; ++k) A[k][j] -= tau[i] * dot * A[k][i];
                }
                A[i][i] = first;
            }
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
        for (int i = 2; i >= 0; --i) {
            if (tau[i] != 0.0) {
                for (int j = 0; j < 3; ++j) {
                    double dot = Q[i][j];
                    for (int k = i + 1; k < 3; ++k) dot += A[k][i] * Q[k][j];
                    Q[i][j] -= tau[i] * dot;
                    for (int k = i + 1; k < 3; ++k) Q[k][j] -= tau[i] * dot * A[k][i];
                }
            }
        }
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                R[i][j] = (j >= i) ? A[i][j] : 0.0;
            }
        }
    };

    auto givens = [](double a, double b, double& c, double& s) {
        if (b == 0.0) {
            c = 1.0; s = 0.0;
        } else if (std::fabs(b) > std::fabs(a)) {
            double t = -a / b;
            s = 1.0 / std::sqrt(1.0 + t*t);
            c = s * t;
        } else {
            double t = -b / a;
            c = 1.0 / std::sqrt(1.0 + t*t);
            s = c * t;
        }
    };

    auto apply_givens = [&](double Q[3][3], double R[3][3], int i, int j, double c, double s) {
        for (int k = 0; k < 3; ++k) {
            double qi = Q[k][i];
            double qj = Q[k][j];
            Q[k][i] = c * qi - s * qj;
            Q[k][j] = s * qi + c * qj;
        }
        int start = std::min(i, j);
        for (int k = start; k < 3; ++k) {
            double ri = R[i][k];
            double rj = R[j][k];
            R[i][k] = c * ri - s * rj;
            R[j][k] = s * ri + c * rj;
        }
    };

    auto qr_update = [&](double Q[3][3], double R[3][3], double w[3], const double v[3]) {
        for (int k = 2; k >= 1; --k) {
            double c, s;
            givens(w[k - 1], w[k], c, s);
            double wi = w[k - 1];
            double wj = w[k];
            w[k - 1] = c * wi - s * wj;
            w[k] = s * wi + c * wj;
            apply_givens(Q, R, k - 1, k, c, s);
        }
        for (int j = 0; j < 3; ++j) {
            R[0][j] += w[0] * v[j];
        }
        for (int k = 1; k < 3; ++k) {
            double c, s;
            givens(R[k - 1][k - 1], R[k][k - 1], c, s);
            apply_givens(Q, R, k - 1, k, c, s);
            R[k][k - 1] = 0.0;
        }
    };

    auto dogleg = [&](const double R[3][3], const double qtf[3], const double diag[3], double delta, double dx[3]) {
        double newton[3];
        for (int i = 2; i >= 0; --i) {
            double sum = -qtf[i];
            for (int j = i + 1; j < 3; ++j) sum -= R[i][j] * newton[j];
            newton[i] = sum / R[i][i];
        }
        double qnorm = scaled_enorm(diag, newton);
        if (qnorm <= delta) {
            dx[0] = newton[0]; dx[1] = newton[1]; dx[2] = newton[2];
            return;
        }
        double grad[3];
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int i = 0; i < 3; ++i) sum += R[i][j] * qtf[i];
            grad[j] = -sum / diag[j];
        }
        double gnorm = enorm(grad);
        if (gnorm == 0.0) {
            for (int i = 0; i < 3; ++i) dx[i] = (delta / qnorm) * newton[i];
            return;
        }
        double scaled_grad[3];
        for (int i = 0; i < 3; ++i) scaled_grad[i] = (grad[i] / gnorm) / diag[i];
        double rg[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) rg[i] += R[i][j] * scaled_grad[j];
        }
        double temp = enorm(rg);
        double sgnorm = (gnorm / temp) / temp;
        if (sgnorm > delta) {
            for (int i = 0; i < 3; ++i) dx[i] = delta * scaled_grad[i];
            return;
        }
        double bnorm = enorm(qtf);
        double bg = bnorm / gnorm;
        double bq = bnorm / qnorm;
        double dq = delta / qnorm;
        double dq2 = dq * dq;
        double sd = sgnorm / delta;
        double sd2 = sd * sd;
        double term1 = bg * bq * sd;
        double u = term1 - dq;
        double term2 = term1 - dq * sd2 + std::sqrt(u * u + (1.0 - dq2) * (1.0 - sd2));
        double alpha = dq * (1.0 - sd2) / term2;
        double beta = (1.0 - alpha) * sgnorm;
        for (int i = 0; i < 3; ++i) {
            dx[i] = alpha * newton[i] + beta * scaled_grad[i];
        }
    };

    double J[3][3], Q[3][3], R[3][3], diag[3];
    fdjac(x, f, J);
    update_diag(J, diag, true);
    double scaled_xnorm = scaled_enorm(diag, x);
    double delta = (scaled_xnorm > 0.0) ? (100.0 * scaled_xnorm) : 100.0;
    qr_decomp(J, Q, R);

    int iteration = 1;
    double fnorm = enorm(f);
    int ncfail = 0, ncsuc = 0, nslow1 = 0, nslow2 = 0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double old_fnorm = fnorm;
        double qtf[3] = {0.0, 0.0, 0.0};
        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) qtf[j] += Q[i][j] * f[i];
        }
        double dx[3];
        dogleg(R, qtf, diag, delta, dx);
        double x_tr[3] = {x[0] + dx[0], x[1] + dx[1], x[2] + dx[2]};
        double pnorm = scaled_enorm(diag, dx);
        if (iteration == 1 && pnorm < delta) {
            delta = pnorm;
        }
        double f_tr[3];
        residual(x_tr, f_tr);
        double df[3] = {f_tr[0] - f[0], f_tr[1] - f[1], f_tr[2] - f[2]};
        double fnorm1 = enorm(f_tr);
        double actred = (fnorm1 < old_fnorm) ? (1.0 - (fnorm1 / old_fnorm) * (fnorm1 / old_fnorm)) : -1.0;

        double rdx[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; ++i) {
            for (int j = i; j < 3; ++j) rdx[i] += R[i][j] * dx[j];
        }
        double v_pred[3] = {qtf[0] + rdx[0], qtf[1] + rdx[1], qtf[2] + rdx[2]};
        double fnorm1p = enorm(v_pred);
        double prered = (fnorm1p < old_fnorm) ? (1.0 - (fnorm1p / old_fnorm) * (fnorm1p / old_fnorm)) : 0.0;
        double ratio = (prered > 0.0) ? (actred / prered) : 0.0;

        if (ratio < 0.1) {
            ncsuc = 0;
            ncfail += 1;
            delta *= 0.5;
        } else {
            ncfail = 0;
            ncsuc += 1;
            if (ratio >= 0.5 || ncsuc > 1) {
                delta = std::max(delta, pnorm / 0.5);
            }
            if (std::fabs(ratio - 1.0) <= 0.1) {
                delta = pnorm / 0.5;
            }
        }

        if (ratio >= 0.0001) {
            x[0] = x_tr[0]; x[1] = x_tr[1]; x[2] = x_tr[2];
            f[0] = f_tr[0]; f[1] = f_tr[1]; f[2] = f_tr[2];
            fnorm = fnorm1;
            iteration += 1;
        }

        nslow1 += 1;
        if (actred >= 0.001) nslow1 = 0;
        if (actred >= 0.1) nslow2 = 0;

        if (ncfail == 2) {
            fdjac(x, f, J);
            nslow2 += 1;
            if (iteration == 1) {
                update_diag(J, diag, true);
                scaled_xnorm = scaled_enorm(diag, x);
                delta = (scaled_xnorm > 0.0) ? (100.0 * scaled_xnorm) : 100.0;
            } else {
                update_diag(J, diag, false);
            }
            qr_decomp(J, Q, R);
        } else {
            if (pnorm == 0.0) return false;
            double qtdf[3] = {0.0, 0.0, 0.0};
            for (int j = 0; j < 3; ++j) {
                for (int i = 0; i < 3; ++i) qtdf[j] += Q[i][j] * df[i];
            }
            double w[3] = {
                (qtdf[0] - rdx[0]) / pnorm,
                (qtdf[1] - rdx[1]) / pnorm,
                (qtdf[2] - rdx[2]) / pnorm
            };
            double v[3] = {
                diag[0] * diag[0] * dx[0] / pnorm,
                diag[1] * diag[1] * dx[1] / pnorm,
                diag[2] * diag[2] * dx[2] / pnorm
            };
            qr_update(Q, R, w, v);
            if (nslow2 == 5 || nslow1 == 10) return false;
        }

        if ((std::fabs(f[0]) + std::fabs(f[1]) + std::fabs(f[2])) < epsabs) {
            sol_x[0] = x[0]; sol_x[1] = x[1]; sol_x[2] = x[2];
            final_f[0] = f[0]; final_f[1] = f[1]; final_f[2] = f[2];
            return true;
        }
    }
    return false;
}

inline bool initial_cartesian_conditions_c(
    double mass1,
    double mass2,
    double spin1x,
    double spin1y,
    double spin1z,
    double spin2x,
    double spin2y,
    double spin2z,
    double f_lower,
    double out_state[14]
) {
    constexpr double MTSUN_SI = 4.925490947641266978197229498498379006e-6;
    constexpr double PI_VAL = 3.14159265358979323846;

    double M = mass1 + mass2;
    double eta = (mass1 * mass2) / (M * M);
    double M_sec = M * MTSUN_SI;
    double omega_target = PI_VAL * f_lower * M_sec;
    if (!std::isfinite(omega_target) || omega_target <= 0.0) {
        return false;
    }

    double v0 = std::cbrt(omega_target);
    double v0_sq = v0 * v0;
    double x0_sq = (1.0 / v0_sq) * (1.0 / v0_sq) - 36.0;
    if (!std::isfinite(x0_sq) || x0_sq <= 0.0) {
        return false;
    }
    double x0 = std::sqrt(x0_sq);
    double guess[3] = {x0, 2.0 * v0, 200.0e-3};

    double S1[3] = {spin1x, spin1y, spin1z};
    double S2[3] = {spin2x, spin2y, spin2z};

    HCoeffs h_state;
    double s1_scale = (mass1 * mass1) / (M * M);
    double s2_scale = (mass2 * mass2) / (M * M);
    double s1_init[3] = {S1[0] * s1_scale, S1[1] * s1_scale, S1[2] * s1_scale};
    double s2_init[3] = {S2[0] * s2_scale, S2[1] * s2_scale, S2[2] * s2_scale};
    double L_init[3] = {0.0, 0.0, 1.0};
    instantaneous_hcoeffs_c(eta, L_init, s1_init, s2_init, &h_state);

    auto residual = [&](const double scaled[3], double res[3]) {
        double x_scaled = scaled[0];
        double py_scaled = scaled[1];
        double pz_scaled = scaled[2];
        double r = std::sqrt(x_scaled * x_scaled + 36.0);
        double py = py_scaled / 2.0;
        double pz = pz_scaled / 200.0;
        double ptheta = -r * pz;
        double pphi = r * py;

        double dHdr = 0.0, dHdptheta = 0.0, dHdpphi = 0.0;
        ic_spherical_derivatives_c(
            mass1, mass2, eta, M,
            r, ptheta, pphi, S1, S2,
            &dHdr, &dHdptheta, &dHdpphi,
            &h_state,
            py, pz
        );
        res[0] = dHdr;
        res[1] = dHdptheta;
        res[2] = dHdpphi - omega_target;
        if (!std::isfinite(res[0]) || !std::isfinite(res[1]) || !std::isfinite(res[2])) {
            res[0] = 1.0e3; res[1] = 1.0e3; res[2] = 1.0e3;
        }
    };

    double sol_x[3] = {guess[0], guess[1], guess[2]};
    double final_res[3] = {1.0e3, 1.0e3, 1.0e3};
    bool ok = gsl_multiroot_hybrids_3d_c(residual, guess, 1.0e-9, 10000, sol_x, final_res);
    double res_norm = std::sqrt(final_res[0]*final_res[0] + final_res[1]*final_res[1] + final_res[2]*final_res[2]);
    if (!ok || !std::isfinite(res_norm) || res_norm > 1.0e-7) {
        return false;
    }

    double x_scaled = sol_x[0];
    double py_scaled = sol_x[1];
    double pz_scaled = sol_x[2];
    double r = std::sqrt(x_scaled * x_scaled + 36.0);
    double py = py_scaled / 2.0;
    double pz = pz_scaled / 200.0;

    double p_unrotated_norm = std::sqrt(py * py + pz * pz);
    if (p_unrotated_norm <= 0.0) return false;
    double py_hat = py / p_unrotated_norm;
    double pz_hat = pz / p_unrotated_norm;
    double p_norm = py_hat * py + pz_hat * pz;

    double rotMatrix2[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, py_hat, pz_hat},
        {0.0, -pz_hat, py_hat}
    };

    double S1_ic[3], S2_ic[3], S1_deriv_ic[3], S2_deriv_ic[3], S1_norm_ic[3], S2_norm_ic[3];
    for (int i = 0; i < 3; ++i) {
        S1_ic[i] = rotMatrix2[i][0]*S1[0] + rotMatrix2[i][1]*S1[1] + rotMatrix2[i][2]*S1[2];
        S2_ic[i] = rotMatrix2[i][0]*S2[0] + rotMatrix2[i][1]*S2[1] + rotMatrix2[i][2]*S2[2];
        S1_deriv_ic[i] = S1_ic[i] * s1_scale;
        S2_deriv_ic[i] = S2_ic[i] * s2_scale;
        S1_norm_ic[i] = S1_deriv_ic[i];
        S2_norm_ic[i] = S2_deriv_ic[i];
    }
    double py_ic = p_norm;
    double pz_ic = 0.0;
    double ptheta = 0.0;
    double pphi = r * p_norm;
    double L_vec_ic[3] = {0.0, 0.0, pphi};

    auto deriv2_fn_dHdr = [&](double r_eval) -> double {
        double dHdr_val = 0.0, dHdptheta_val = 0.0, dHdpphi_val = 0.0;
        HCoeffs h_tmp = h_state;
        ic_spherical_derivatives_c(
            mass1, mass2, eta, M,
            r_eval, ptheta, pphi, S1_ic, S2_ic,
            &dHdr_val, &dHdptheta_val, &dHdpphi_val,
            &h_tmp,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            S1_deriv_ic, S2_deriv_ic
        );
        return dHdr_val;
    };

    auto deriv2_fn_dHdpphi = [&](double r_eval) -> double {
        double dHdr_val = 0.0, dHdptheta_val = 0.0, dHdpphi_val = 0.0;
        HCoeffs h_tmp = h_state;
        ic_spherical_derivatives_c(
            mass1, mass2, eta, M,
            r_eval, ptheta, pphi, S1_ic, S2_ic,
            &dHdr_val, &dHdptheta_val, &dHdpphi_val,
            &h_tmp,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            S1_deriv_ic, S2_deriv_ic
        );
        return dHdpphi_val;
    };

    double d2Hdr2 = robust_gsl_derivative_c(deriv2_fn_dHdr, r, 3.0e-3);
    double d2Hdrdpphi = robust_gsl_derivative_c(deriv2_fn_dHdpphi, r, 3.0e-3);

    double r_vec_ic[3] = {r, 0.0, 0.0};
    double p_vec_ic[3] = {0.0, py_ic, 0.0};
    HCoeffs h_base;
    instantaneous_hcoeffs_c(eta, L_vec_ic, S1_deriv_ic, S2_deriv_ic, &h_base);

    double dH_dpvec_base[3];
    double p_tmp[3] = {p_vec_ic[0], p_vec_ic[1], p_vec_ic[2]};
    constexpr double STEP_DERIV = 2.0e-3;
    for (int axis = 0; axis < 3; ++axis) {
        auto f = [&](double p_val) -> double {
            double orig = p_tmp[axis];
            p_tmp[axis] = p_val;
            double h_val = eob_hamiltonian_c(
                r_vec_ic, p_tmp, S1_deriv_ic, S2_deriv_ic,
                mass1, mass2, eta, M,
                h_base.k0, h_base.k1, h_base.k2, h_base.k3, h_base.k4, h_base.k5, h_base.k5l,
                h_base.KK, h_base.d1, h_base.d1v2, h_base.dheffSS, h_base.dheffSSv2,
                h_base.b3, h_base.bb3,
                1, true
            ) / eta;
            p_tmp[axis] = orig;
            return h_val;
        };
        dH_dpvec_base[axis] = gsl_deriv_central_c(f, p_vec_ic[axis], STEP_DERIV);
    }
    double dHdpphi = dH_dpvec_base[1] / r;
    if (std::fabs(d2Hdrdpphi) < 1.0e-14 || std::fabs(d2Hdr2) < 1.0e-14) {
        return false;
    }
    double dEdr = -dHdpphi * d2Hdr2 / d2Hdrdpphi;

    EOBDynamicsContext dyn_ctx;
    init_eob_dynamics_context(mass1, mass2, &dyn_ctx);

    double dxdt_base[3] = {0.0, dH_dpvec_base[1], dH_dpvec_base[2]};
    double vphi_nk = non_keplerian_vphi_c(
        r_vec_ic, p_vec_ic, dxdt_base, S1_norm_ic, S2_norm_ic,
        mass1, mass2, eta, M, omega_target, r
    );

    double H_val = eob_hamiltonian_c(
        r_vec_ic, p_vec_ic, S1_norm_ic, S2_norm_ic,
        mass1, mass2, eta, M,
        h_base.k0, h_base.k1, h_base.k2, h_base.k3, h_base.k4, h_base.k5, h_base.k5l,
        h_base.KK, h_base.d1, h_base.d1v2, h_base.dheffSS, h_base.dheffSSv2,
        h_base.b3, h_base.bb3,
        1, true
    );

    double flux = factorized_flux_c(
        r, omega_target, H_val, pphi, L_vec_ic,
        S1_norm_ic, S2_norm_ic, mass1, mass2, eta, M,
        vphi_nk, dyn_ctx.prefix_table
    );

    double pr_probe = 1.0e-3;
    double p_probe_vec[3] = {pr_probe, py_ic, 0.0};
    HCoeffs h_probe;
    instantaneous_hcoeffs_c(eta, L_vec_ic, S1_deriv_ic, S2_deriv_ic, &h_probe);

    double dH_dpvec_probe0 = 0.0;
    {
        double p_pr_tmp[3] = {p_probe_vec[0], p_probe_vec[1], p_probe_vec[2]};
        auto f = [&](double p_val) -> double {
            double orig = p_pr_tmp[0];
            p_pr_tmp[0] = p_val;
            double h_val = eob_hamiltonian_c(
                r_vec_ic, p_pr_tmp, S1_deriv_ic, S2_deriv_ic,
                mass1, mass2, eta, M,
                h_probe.k0, h_probe.k1, h_probe.k2, h_probe.k3, h_probe.k4, h_probe.k5, h_probe.k5l,
                h_probe.KK, h_probe.d1, h_probe.d1v2, h_probe.dheffSS, h_probe.dheffSSv2,
                h_probe.b3, h_probe.bb3,
                1, true
            ) / eta;
            p_pr_tmp[0] = orig;
            return h_val;
        };
        dH_dpvec_probe0 = gsl_deriv_central_c(f, p_probe_vec[0], STEP_DERIV);
    }

    double r_clamp = (r < 1.0e-15) ? 1.0e-15 : r;
    double r2 = r_clamp * r_clamp;
    double u = 1.0 / r_clamp;
    double u2 = u * u;
    double u3 = u2 * u;
    double u4 = u2 * u2;
    double u5 = u4 * u;
    double logu = std::log(u);
    double sigma_vec[3] = {S1_deriv_ic[0] + S2_deriv_ic[0], S1_deriv_ic[1] + S2_deriv_ic[1], S1_deriv_ic[2] + S2_deriv_ic[2]};
    double a2 = sigma_vec[0]*sigma_vec[0] + sigma_vec[1]*sigma_vec[1] + sigma_vec[2]*sigma_vec[2];
    double w2 = r2 + a2;
    double D_term = 1.0 + 6.0 * eta * u2 + 2.0 * (26.0 - 3.0 * eta) * eta * u3;
    double D = 1.0 + std::log(D_term);
    double m1PlusetaKK = -1.0 + eta * h_probe.KK;
    double bulk = 1.0 / (m1PlusetaKK * m1PlusetaKK) + (2.0 * u) / m1PlusetaKK + a2 * u2;
    double log_arg = 1.0 + h_probe.k1 * u + h_probe.k2 * u2 + h_probe.k3 * u3 + h_probe.k4 * u4 + h_probe.k5 * u5 + h_probe.k5l * u5 * logu;
    double logTerms = 1.0 + eta * h_probe.k0 + eta * std::log(std::fabs(log_arg));
    double deltaU = bulk * logTerms;
    double deltaT = r2 * deltaU;
    double deltaR = deltaT * D;
    double csi_ic = std::sqrt(std::max(deltaT * deltaR, 0.0)) / w2;
    if (csi_ic < 1.0e-15) csi_ic = 1.0e-15;

    double dxdt_probe0 = csi_ic * dH_dpvec_probe0;
    double dHdpr = csi_ic * dxdt_probe0;
    if (std::fabs(dHdpr) < 1.0e-14 || std::fabs(dEdr) < 1.0e-14) {
        return false;
    }
    double r_dot = -flux / dEdr;
    double pr_non_tortoise = r_dot / (dHdpr / pr_probe);
    double pr_star = csi_ic * pr_non_tortoise;
    if (!std::isfinite(pr_star)) {
        return false;
    }

    double S1_norm[3] = {S1[0] * s1_scale, S1[1] * s1_scale, S1[2] * s1_scale};
    double S2_norm[3] = {S2[0] * s2_scale, S2[1] * s2_scale, S2[2] * s2_scale};

    out_state[0] = r;
    out_state[1] = 0.0;
    out_state[2] = 0.0;
    out_state[3] = pr_star;
    out_state[4] = py;
    out_state[5] = pz;
    out_state[6] = S1_norm[0];
    out_state[7] = S1_norm[1];
    out_state[8] = S1_norm[2];
    out_state[9] = S2_norm[0];
    out_state[10] = S2_norm[1];
    out_state[11] = S2_norm[2];
    out_state[12] = 0.0;
    out_state[13] = 0.0;

    return true;
}

torch::Tensor initial_cartesian_conditions_native(
    double mass1,
    double mass2,
    double spin1x,
    double spin1y,
    double spin1z,
    double spin2x,
    double spin2y,
    double spin2z,
    double f_lower
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    double out_state[14];
    bool ok = initial_cartesian_conditions_c(
        mass1, mass2,
        spin1x, spin1y, spin1z,
        spin2x, spin2y, spin2z,
        f_lower,
        out_state
    );
    if (!ok) {
        return torch::empty({0}, options);
    }
    auto tensor = torch::empty({14}, options);
    std::copy(out_state, out_state + 14, tensor.data_ptr<double>());
    return tensor;
}

torch::Tensor ic_spherical_derivatives_native(
    double mass1,
    double mass2,
    double r,
    double ptheta,
    double pphi,
    const torch::Tensor& s1_tensor,
    const torch::Tensor& s2_tensor
) {
    validate_tensor(s1_tensor, "S1");
    validate_tensor(s2_tensor, "S2");
    TORCH_CHECK(s1_tensor.numel() == 3 && s2_tensor.numel() == 3, "S1 and S2 must be 3-vectors");
    double M = mass1 + mass2;
    double eta = mass1 * mass2 / (M * M);
    const double* S1 = s1_tensor.data_ptr<double>();
    const double* S2 = s2_tensor.data_ptr<double>();
    double s1_scale = (mass1 * mass1) / (M * M);
    double s2_scale = (mass2 * mass2) / (M * M);
    double s1_init[3] = {S1[0] * s1_scale, S1[1] * s1_scale, S1[2] * s1_scale};
    double s2_init[3] = {S2[0] * s2_scale, S2[1] * s2_scale, S2[2] * s2_scale};
    double L_init[3] = {0.0, 0.0, 1.0};
    HCoeffs h_state;
    instantaneous_hcoeffs_c(eta, L_init, s1_init, s2_init, &h_state);

    double dHdr = 0.0, dHdptheta = 0.0, dHdpphi = 0.0;
    ic_spherical_derivatives_c(
        mass1, mass2, eta, M,
        r, ptheta, pphi, S1, S2,
        &dHdr, &dHdptheta, &dHdpphi,
        &h_state
    );
    auto options = s1_tensor.options();
    auto out = torch::empty({3}, options);
    double* d = out.data_ptr<double>();
    d[0] = dHdr; d[1] = dHdptheta; d[2] = dHdpphi;
    return out;
}

py::tuple integrate_cartesian_native(
    const torch::Tensor& y0_tensor,
    double mass1,
    double mass2,
    double t0,
    double t1,
    double h0,
    double rtol,
    double atol,
    int64_t max_steps,
    double h_min,
    double h_max,
    int stop_mode,
    double initial_prev_omega,
    double initial_prev_dr,
    int initial_omega_peaked,
    bool return_diagnostics
) {
    validate_tensor(y0_tensor, "y0");
    TORCH_CHECK(y0_tensor.numel() == 14, "y0 must be 14-dimensional");
    int64_t dim = 14;
    auto options = y0_tensor.options();

    EOBDynamicsContext ctx;
    init_eob_dynamics_context(mass1, mass2, &ctx);

    auto rhs_fn = [&](double t_val, const double* y_val, double* dydt_val) {
        eob_rhs_cartesian_c(t_val, y_val, dydt_val, ctx);
    };

    double t = t0;
    double h = h0;
    double prev_omega = initial_prev_omega;
    double prev_dr = initial_prev_dr;
    int omega_peaked = initial_omega_peaked;
    int stop_reason_int = -1;

    const double finfo_tiny = std::numeric_limits<double>::min();
    const int max_retries = 1;
    int retries_left = max_retries;
    int64_t accepted_steps = 0;
    int64_t rejected_steps = 0;
    int64_t attempted_steps = 0;
    bool hit_max_steps = false;
    bool stopped_or_finished = false;

    std::vector<double> t_buf;
    std::vector<double> y_buf;
    t_buf.reserve(std::min<int64_t>(max_steps, 8192));
    y_buf.reserve(std::min<int64_t>(max_steps * dim, 8192 * dim));

    double y_curr[14];
    double d_in[14];
    double d_out[14];
    double y_try[14];
    double e_try[14];

    const double* y0_data = y0_tensor.data_ptr<double>();
    for (int64_t i = 0; i < 14; ++i) {
        y_curr[i] = y0_data[i];
    }

    rhs_fn(t, y_curr, d_in);

    for (int64_t step = 0; step < max_steps; ++step) {
        attempted_steps = step + 1;
        if ((t + h) > t1) {
            h = t1 - t;
        }

        bool trial_failed = false;
        try {
            rkf45_step_kernel(
                rhs_fn,
                t,
                y_curr,
                h,
                14,
                d_in,
                y_try,
                e_try,
                nullptr,
                nullptr,
                d_out,
                true
            );
            for (int64_t i = 0; i < 14; ++i) {
                if (!std::isfinite(y_try[i]) || !std::isfinite(d_out[i]) || !std::isfinite(e_try[i])) {
                    trial_failed = true;
                    break;
                }
            }
        } catch (const std::exception&) {
            trial_failed = true;
        }

        if (trial_failed) {
            rejected_steps++;
            retries_left--;
            h = h / 10.0;
            if (retries_left < 0 || (h_min > 0.0 && h < h_min)) {
                if (accepted_steps > 0) {
                    break;
                }
                throw std::runtime_error("NaN/inf encountered in ODE trial step");
            }
            continue;
        }

        // GSL error ratio & step adaptation
        double worst_err_norm = 0.0;
        for (int64_t i = 0; i < 14; ++i) {
            double scale = atol + rtol * std::fabs(y_try[i]);
            double ratio_i = std::fabs(e_try[i]) / (scale > 0.0 ? scale : finfo_tiny);
            if (!std::isfinite(ratio_i)) ratio_i = std::numeric_limits<double>::infinity();
            if (ratio_i > worst_err_norm) worst_err_norm = ratio_i;
        }

        double h_next = h;
        bool accepted = true;

        if (worst_err_norm > 1.1 && h > finfo_tiny) {
            double ratio = 0.9 * std::pow(std::max(worst_err_norm, finfo_tiny), -0.2);
            if (ratio < 0.2) ratio = 0.2;
            double h_new = h * ratio;
            if (h_min > 0.0 && h_new < h_min) h_new = h_min;
            if (h_new < h) {
                h = h_new;
                accepted = false;
                rejected_steps++;
                continue;
            }
            h_next = h_new;
        } else if (worst_err_norm < 0.5) {
            double ratio = 0.9 * std::pow(std::max(worst_err_norm, finfo_tiny), -1.0 / 6.0);
            if (ratio < 1.0) ratio = 1.0;
            if (ratio > 5.0) ratio = 5.0;
            h_next = h * ratio;
        }

        if (accepted) {
            t += h;
            for (int64_t i = 0; i < 14; ++i) {
                y_curr[i] = y_try[i];
                d_in[i] = d_out[i];
            }

            t_buf.push_back(t);
            for (int64_t i = 0; i < 14; ++i) {
                y_buf.push_back(y_curr[i]);
            }
            accepted_steps++;

            bool should_stop = false;
            if (stop_mode == STOP_MODE_CARTESIAN) {
                if (check_cartesian_stop(y_curr, d_in, prev_omega, prev_dr, omega_peaked, stop_reason_int)) {
                    should_stop = true;
                }
            }

            if (should_stop || t >= t1) {
                stopped_or_finished = true;
                break;
            }
            retries_left = max_retries;
        }

        h = h_next;
        if (h_max > 0.0 && h > h_max) h = h_max;
    }

    if (!stopped_or_finished && max_steps > 0) {
        hit_max_steps = true;
    }

    auto t_out = torch::empty({accepted_steps}, options);
    auto y_out = torch::empty({accepted_steps, dim}, options);
    if (accepted_steps > 0) {
        std::copy(t_buf.begin(), t_buf.end(), t_out.data_ptr<double>());
        std::copy(y_buf.begin(), y_buf.end(), y_out.data_ptr<double>());
    }

    py::tuple traj = py::make_tuple(t_out, y_out);
    if (!return_diagnostics) {
        return traj;
    }

    py::dict diag;
    diag["accepted_steps"] = accepted_steps;
    diag["rejected_steps"] = rejected_steps;
    diag["attempted_steps"] = attempted_steps;
    diag["max_steps"] = max_steps;
    diag["hit_max_steps"] = hit_max_steps;
    diag["t_end"] = t;
    diag["h_final"] = h;
    diag["prev_omega"] = prev_omega;
    diag["prev_dr"] = prev_dr;
    diag["omega_peaked"] = omega_peaked;
    if (stop_reason_int >= 0) {
        diag["stop_reason"] = stop_reason_int;
    } else {
        diag["stop_reason"] = py::none();
    }

    return py::make_tuple(traj, diag);
}

torch::Tensor calcomega_lal_polar_derivative_native(
    const torch::Tensor& r_vec_t,
    const torch::Tensor& p_vec_t,
    const torch::Tensor& s1_t,
    const torch::Tensor& s2_t,
    double mass1,
    double mass2,
    const c10::optional<torch::Tensor>& rdot_vec_opt,
    const c10::optional<torch::Tensor>& s1_weighted_opt,
    const c10::optional<torch::Tensor>& s2_weighted_opt
) {
    validate_tensor(r_vec_t, "r_vec");
    validate_tensor(p_vec_t, "p_vec");
    validate_tensor(s1_t, "s1");
    validate_tensor(s2_t, "s2");

    double eta = (mass1 * mass2) / ((mass1 + mass2) * (mass1 + mass2));
    double M = mass1 + mass2;

    const double* rdot_ptr = nullptr;
    if (rdot_vec_opt.has_value() && rdot_vec_opt.value().defined()) {
        validate_tensor(rdot_vec_opt.value(), "rdot_vec");
        rdot_ptr = rdot_vec_opt.value().data_ptr<double>();
    }
    const double* s1_weighted_ptr = nullptr;
    if (s1_weighted_opt.has_value() && s1_weighted_opt.value().defined()) {
        validate_tensor(s1_weighted_opt.value(), "s1_weighted");
        s1_weighted_ptr = s1_weighted_opt.value().data_ptr<double>();
    }
    const double* s2_weighted_ptr = nullptr;
    if (s2_weighted_opt.has_value() && s2_weighted_opt.value().defined()) {
        validate_tensor(s2_weighted_opt.value(), "s2_weighted");
        s2_weighted_ptr = s2_weighted_opt.value().data_ptr<double>();
    }

    if (r_vec_t.dim() == 1) {
        TORCH_CHECK(r_vec_t.numel() == 3, "r_vec must have length 3");
        TORCH_CHECK(p_vec_t.numel() == 3, "p_vec must have length 3");
        TORCH_CHECK(s1_t.numel() == 3, "s1 must have length 3");
        TORCH_CHECK(s2_t.numel() == 3, "s2 must have length 3");

        double val = calcomega_lal_polar_derivative_point_c(
            r_vec_t.data_ptr<double>(),
            p_vec_t.data_ptr<double>(),
            s1_t.data_ptr<double>(),
            s2_t.data_ptr<double>(),
            mass1, mass2, eta, M,
            rdot_ptr, s1_weighted_ptr, s2_weighted_ptr
        );
        return torch::tensor(val, r_vec_t.options());
    }

    TORCH_CHECK(r_vec_t.dim() == 2 && r_vec_t.size(1) == 3, "r_vec must have shape (N, 3)");
    int64_t N = r_vec_t.size(0);
    TORCH_CHECK(p_vec_t.size(0) == N && p_vec_t.size(1) == 3, "p_vec must have shape (N, 3)");

    bool s1_is_2d = (s1_t.dim() == 2);
    bool s2_is_2d = (s2_t.dim() == 2);

    const double* r_data = r_vec_t.data_ptr<double>();
    const double* p_data = p_vec_t.data_ptr<double>();
    const double* s1_data = s1_t.data_ptr<double>();
    const double* s2_data = s2_t.data_ptr<double>();

    auto out = torch::empty({N}, r_vec_t.options());
    double* out_data = out.data_ptr<double>();

    for (int64_t i = 0; i < N; ++i) {
        const double* r_i = r_data + 3 * i;
        const double* p_i = p_data + 3 * i;
        const double* s1_i = s1_is_2d ? (s1_data + 3 * i) : s1_data;
        const double* s2_i = s2_is_2d ? (s2_data + 3 * i) : s2_data;

        const double* rdot_i = rdot_ptr ? (rdot_vec_opt.value().dim() == 2 ? (rdot_ptr + 3 * i) : rdot_ptr) : nullptr;
        const double* s1_w_i = s1_weighted_ptr ? (s1_weighted_opt.value().dim() == 2 ? (s1_weighted_ptr + 3 * i) : s1_weighted_ptr) : nullptr;
        const double* s2_w_i = s2_weighted_ptr ? (s2_weighted_opt.value().dim() == 2 ? (s2_weighted_ptr + 3 * i) : s2_weighted_ptr) : nullptr;

        out_data[i] = calcomega_lal_polar_derivative_point_c(
            r_i, p_i, s1_i, s2_i, mass1, mass2, eta, M,
            rdot_i, s1_w_i, s2_w_i
        );
    }
    return out;
}

torch::Tensor non_keplerian_vphi_native(
    const torch::Tensor& r_t,
    const torch::Tensor& omega_t,
    const torch::Tensor& phi_t,
    const torch::Tensor& L_vec_t,
    const torch::Tensor& s1_t,
    const torch::Tensor& s2_t,
    double mass1,
    double mass2,
    bool aligned_spins,
    const c10::optional<torch::Tensor>& r_vec_opt,
    const c10::optional<torch::Tensor>& p_vec_opt,
    const c10::optional<torch::Tensor>& rdot_vec_opt,
    const c10::optional<torch::Tensor>& s1_weighted_opt,
    const c10::optional<torch::Tensor>& s2_weighted_opt
) {
    validate_tensor(r_t, "r");
    validate_tensor(omega_t, "omega");
    validate_tensor(phi_t, "phi");
    validate_tensor(L_vec_t, "L_vec");
    validate_tensor(s1_t, "s1");
    validate_tensor(s2_t, "s2");

    double eta = (mass1 * mass2) / ((mass1 + mass2) * (mass1 + mass2));
    double M = mass1 + mass2;

    const double* r_vec_ptr = nullptr;
    if (r_vec_opt.has_value() && r_vec_opt.value().defined()) {
        validate_tensor(r_vec_opt.value(), "r_vec");
        r_vec_ptr = r_vec_opt.value().data_ptr<double>();
    }
    const double* p_vec_ptr = nullptr;
    if (p_vec_opt.has_value() && p_vec_opt.value().defined()) {
        validate_tensor(p_vec_opt.value(), "p_vec");
        p_vec_ptr = p_vec_opt.value().data_ptr<double>();
    }
    const double* rdot_ptr = nullptr;
    if (rdot_vec_opt.has_value() && rdot_vec_opt.value().defined()) {
        validate_tensor(rdot_vec_opt.value(), "rdot_vec");
        rdot_ptr = rdot_vec_opt.value().data_ptr<double>();
    }
    const double* s1_weighted_ptr = nullptr;
    if (s1_weighted_opt.has_value() && s1_weighted_opt.value().defined()) {
        validate_tensor(s1_weighted_opt.value(), "s1_weighted");
        s1_weighted_ptr = s1_weighted_opt.value().data_ptr<double>();
    }
    const double* s2_weighted_ptr = nullptr;
    if (s2_weighted_opt.has_value() && s2_weighted_opt.value().defined()) {
        validate_tensor(s2_weighted_opt.value(), "s2_weighted");
        s2_weighted_ptr = s2_weighted_opt.value().data_ptr<double>();
    }

    if (r_t.dim() == 0) {
        double val = non_keplerian_vphi_point_c(
            r_t.item<double>(),
            omega_t.item<double>(),
            phi_t.item<double>(),
            L_vec_t.data_ptr<double>(),
            s1_t.data_ptr<double>(),
            s2_t.data_ptr<double>(),
            mass1, mass2, eta, M,
            aligned_spins,
            r_vec_ptr, p_vec_ptr, rdot_ptr, s1_weighted_ptr, s2_weighted_ptr
        );
        return torch::tensor(val, r_t.options());
    }

    int64_t N = r_t.numel();
    auto out = torch::empty({N}, r_t.options());
    double* out_data = out.data_ptr<double>();

    const double* r_data = r_t.data_ptr<double>();
    const double* omega_data = omega_t.data_ptr<double>();
    const double* phi_data = phi_t.data_ptr<double>();
    const double* L_data = L_vec_t.data_ptr<double>();
    const double* s1_data = s1_t.data_ptr<double>();
    const double* s2_data = s2_t.data_ptr<double>();

    bool L_is_2d = (L_vec_t.dim() == 2);
    bool s1_is_2d = (s1_t.dim() == 2);
    bool s2_is_2d = (s2_t.dim() == 2);
    bool r_vec_is_2d = r_vec_ptr && (r_vec_opt.value().dim() == 2);
    bool p_vec_is_2d = p_vec_ptr && (p_vec_opt.value().dim() == 2);
    bool rdot_is_2d = rdot_ptr && (rdot_vec_opt.value().dim() == 2);
    bool s1_w_is_2d = s1_weighted_ptr && (s1_weighted_opt.value().dim() == 2);
    bool s2_w_is_2d = s2_weighted_ptr && (s2_weighted_opt.value().dim() == 2);

    for (int64_t i = 0; i < N; ++i) {
        double r_i = r_data[i];
        double om_i = omega_data[i];
        double phi_i = phi_data[i];
        const double* L_i = L_is_2d ? (L_data + 3 * i) : L_data;
        const double* s1_i = s1_is_2d ? (s1_data + 3 * i) : s1_data;
        const double* s2_i = s2_is_2d ? (s2_data + 3 * i) : s2_data;

        const double* r_vec_i = r_vec_ptr ? (r_vec_is_2d ? (r_vec_ptr + 3 * i) : r_vec_ptr) : nullptr;
        const double* p_vec_i = p_vec_ptr ? (p_vec_is_2d ? (p_vec_ptr + 3 * i) : p_vec_ptr) : nullptr;
        const double* rdot_i = rdot_ptr ? (rdot_is_2d ? (rdot_ptr + 3 * i) : rdot_ptr) : nullptr;
        const double* s1_w_i = s1_weighted_ptr ? (s1_w_is_2d ? (s1_weighted_ptr + 3 * i) : s1_weighted_ptr) : nullptr;
        const double* s2_w_i = s2_weighted_ptr ? (s2_w_is_2d ? (s2_weighted_ptr + 3 * i) : s2_weighted_ptr) : nullptr;

        out_data[i] = non_keplerian_vphi_point_c(
            r_i, om_i, phi_i, L_i, s1_i, s2_i, mass1, mass2, eta, M,
            aligned_spins,
            r_vec_i, p_vec_i, rdot_i, s1_w_i, s2_w_i
        );
    }
    return out;
}

torch::Tensor eob_hamiltonian_trajectory_native(
    const torch::Tensor& r_vec_t,
    const torch::Tensor& p_vec_t,
    const torch::Tensor& s1_t,
    const torch::Tensor& s2_t,
    double mass1,
    double mass2
) {
    validate_tensor(r_vec_t, "r_vec");
    validate_tensor(p_vec_t, "p_vec");
    int64_t n = r_vec_t.size(0);
    auto h_out = torch::empty({n}, r_vec_t.options());
    double* h_data = h_out.data_ptr<double>();
    const double* r_data = r_vec_t.data_ptr<double>();
    const double* p_data = p_vec_t.data_ptr<double>();
    const double* s1_data = s1_t.data_ptr<double>();
    const double* s2_data = s2_t.data_ptr<double>();
    bool s1_is_2d = s1_t.dim() == 2;
    bool s2_is_2d = s2_t.dim() == 2;

    EOBDynamicsContext ctx;
    init_eob_dynamics_context(mass1, mass2, &ctx);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        const double* r_row = r_data + i * 3;
        const double* p_row = p_data + i * 3;
        const double* s1_row = s1_is_2d ? (s1_data + i * 3) : s1_data;
        const double* s2_row = s2_is_2d ? (s2_data + i * 3) : s2_data;

        double S1_weighted[3] = {
            s1_row[0] * ctx.s1_scale,
            s1_row[1] * ctx.s1_scale,
            s1_row[2] * ctx.s1_scale
        };
        double S2_weighted[3] = {
            s2_row[0] * ctx.s2_scale,
            s2_row[1] * ctx.s2_scale,
            s2_row[2] * ctx.s2_scale
        };

        // Instantaneous L_vec
        double L_vec[3] = {
            r_row[1] * p_row[2] - r_row[2] * p_row[1],
            r_row[2] * p_row[0] - r_row[0] * p_row[2],
            r_row[0] * p_row[1] - r_row[1] * p_row[0]
        };
        double L_norm = std::sqrt(L_vec[0]*L_vec[0] + L_vec[1]*L_vec[1] + L_vec[2]*L_vec[2]);
        double L_clamp = (L_norm < 1.0e-15) ? 1.0e-15 : L_norm;
        double Lhat[3] = {L_vec[0] / L_clamp, L_vec[1] / L_clamp, L_vec[2] / L_clamp};

        double chi1_dot_L = (s1_row[0] * Lhat[0] + s1_row[1] * Lhat[1] + s1_row[2] * Lhat[2]);
        double chi2_dot_L = (s2_row[0] * Lhat[0] + s2_row[1] * Lhat[1] + s2_row[2] * Lhat[2]);
        double chi_eff = (ctx.mass1 * chi1_dot_L + ctx.mass2 * chi2_dot_L) / ctx.M;
        double a = std::sqrt(
            (S1_weighted[0] + S2_weighted[0]) * (S1_weighted[0] + S2_weighted[0]) +
            (S1_weighted[1] + S2_weighted[1]) * (S1_weighted[1] + S2_weighted[1]) +
            (S1_weighted[2] + S2_weighted[2]) * (S1_weighted[2] + S2_weighted[2])
        );

        HCoeffs h_inst;
        compute_spin_aligned_hcoeffs_c(ctx.eta, a, chi_eff, &h_inst);

        h_data[i] = eob_hamiltonian_c(
            r_row, p_row, S1_weighted, S2_weighted,
            ctx.mass1, ctx.mass2, ctx.eta, ctx.M,
            h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
            h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
            h_inst.b3, h_inst.bb3,
            2, true
        );
    }
    return h_out;
}

torch::Tensor omega_from_hamiltonian_velocity_native(
    const torch::Tensor& r_vec_t,
    const torch::Tensor& p_vec_t,
    const torch::Tensor& s1_t,
    const torch::Tensor& s2_t,
    double mass1,
    double mass2
) {
    validate_tensor(r_vec_t, "r_vec");
    validate_tensor(p_vec_t, "p_vec");
    int64_t n = r_vec_t.size(0);
    auto om_out = torch::empty({n}, r_vec_t.options());
    double* om_data = om_out.data_ptr<double>();
    const double* r_data = r_vec_t.data_ptr<double>();
    const double* p_data = p_vec_t.data_ptr<double>();
    const double* s1_data = s1_t.data_ptr<double>();
    const double* s2_data = s2_t.data_ptr<double>();
    bool s1_is_2d = s1_t.dim() == 2;
    bool s2_is_2d = s2_t.dim() == 2;

    EOBDynamicsContext ctx;
    init_eob_dynamics_context(mass1, mass2, &ctx);

    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        const double* r_row = r_data + i * 3;
        const double* p_row = p_data + i * 3;
        const double* s1_row = s1_is_2d ? (s1_data + i * 3) : s1_data;
        const double* s2_row = s2_is_2d ? (s2_data + i * 3) : s2_data;

        double S1_weighted[3] = {
            s1_row[0] * ctx.s1_scale,
            s1_row[1] * ctx.s1_scale,
            s1_row[2] * ctx.s1_scale
        };
        double S2_weighted[3] = {
            s2_row[0] * ctx.s2_scale,
            s2_row[1] * ctx.s2_scale,
            s2_row[2] * ctx.s2_scale
        };

        // Instantaneous L_vec
        double L_vec[3] = {
            r_row[1] * p_row[2] - r_row[2] * p_row[1],
            r_row[2] * p_row[0] - r_row[0] * p_row[2],
            r_row[0] * p_row[1] - r_row[1] * p_row[0]
        };
        double L_norm = std::sqrt(L_vec[0]*L_vec[0] + L_vec[1]*L_vec[1] + L_vec[2]*L_vec[2]);
        double L_clamp = (L_norm < 1.0e-15) ? 1.0e-15 : L_norm;
        double Lhat[3] = {L_vec[0] / L_clamp, L_vec[1] / L_clamp, L_vec[2] / L_clamp};

        double chi1_dot_L = (s1_row[0] * Lhat[0] + s1_row[1] * Lhat[1] + s1_row[2] * Lhat[2]);
        double chi2_dot_L = (s2_row[0] * Lhat[0] + s2_row[1] * Lhat[1] + s2_row[2] * Lhat[2]);
        double chi_eff = (ctx.mass1 * chi1_dot_L + ctx.mass2 * chi2_dot_L) / ctx.M;
        double a = std::sqrt(
            (S1_weighted[0] + S2_weighted[0]) * (S1_weighted[0] + S2_weighted[0]) +
            (S1_weighted[1] + S2_weighted[1]) * (S1_weighted[1] + S2_weighted[1]) +
            (S1_weighted[2] + S2_weighted[2]) * (S1_weighted[2] + S2_weighted[2])
        );

        HCoeffs h_inst;
        compute_spin_aligned_hcoeffs_c(ctx.eta, a, chi_eff, &h_inst);

        double r2 = r_row[0]*r_row[0] + r_row[1]*r_row[1] + r_row[2]*r_row[2];
        double r_mag = std::sqrt(r2);
        double r_clamp = (r_mag < 1.0e-15) ? 1.0e-15 : r_mag;
        double u = 1.0 / r_clamp;
        double u2 = u * u;
        double u3 = u2 * u;
        double u4 = u2 * u2;
        double u5 = u4 * u;
        double a2 = a * a;
        double w2 = a2 + r2;
        double inv_w2 = 1.0 / w2;

        double m1PlusetaKK = -1.0 + ctx.eta * h_inst.KK;
        double logu = std::log(u);
        double log_arg = 1.0 + ctx.eta * (h_inst.k0 + h_inst.k1 * u + h_inst.k2 * u2 + h_inst.k3 * u3 + h_inst.k4 * u4 + (h_inst.k5 + h_inst.k5l * logu) * u5);
        if (log_arg < 1.0e-15) log_arg = 1.0e-15;
        double logTerms = std::log(log_arg);
        double bulk = 1.0 / (m1PlusetaKK * m1PlusetaKK) + 2.0 * u / m1PlusetaKK + a2 * u2;
        double deltaU = bulk * logTerms + 1.0 - 2.0 * u;
        double D = 1.0 + std::log(1.0 + 6.0 * ctx.eta * u2 + 2.0 * (26.0 - 3.0 * ctx.eta) * ctx.eta * u3);
        double deltaT = r2 * deltaU;
        double deltaR = deltaT * D;
        double csi = std::sqrt(std::max(deltaT * deltaR, 0.0)) * inv_w2;
        double csi_fac = (csi < 1.0e-15) ? 1.0e-15 : csi;

        // Tortoise matrix Tmat
        double Tmat[3][3];
        for (int ax1 = 0; ax1 < 3; ++ax1) {
            for (int ax2 = 0; ax2 <= ax1; ++ax2) {
                double tij = (r_row[ax1] * r_row[ax2] / r2) * (csi_fac - 1.0);
                if (ax1 == ax2) tij += 1.0;
                Tmat[ax1][ax2] = tij;
                Tmat[ax2][ax1] = tij;
            }
        }

        // dH/dp
        double dH_dp[3];
        double p_tmp[3] = {p_row[0], p_row[1], p_row[2]};
        constexpr double STEP_DERIV = 2.0e-3;
        for (int axis = 0; axis < 3; ++axis) {
            auto f = [&](double p_val) -> double {
                double orig = p_tmp[axis];
                p_tmp[axis] = p_val;
                double h_val = eob_hamiltonian_c(
                    r_row, p_tmp, S1_weighted, S2_weighted,
                    ctx.mass1, ctx.mass2, ctx.eta, ctx.M,
                    h_inst.k0, h_inst.k1, h_inst.k2, h_inst.k3, h_inst.k4, h_inst.k5, h_inst.k5l,
                    h_inst.KK, h_inst.d1, h_inst.d1v2, h_inst.dheffSS, h_inst.dheffSSv2,
                    h_inst.b3, h_inst.bb3,
                    1, true
                ) / ctx.eta;
                p_tmp[axis] = orig;
                return h_val;
            };
            dH_dp[axis] = gsl_deriv_central_c(f, p_row[axis], STEP_DERIV);
        }

        // dxdt = Tmat * dH_dp
        double dxdt[3] = {
            Tmat[0][0]*dH_dp[0] + Tmat[0][1]*dH_dp[1] + Tmat[0][2]*dH_dp[2],
            Tmat[1][0]*dH_dp[0] + Tmat[1][1]*dH_dp[1] + Tmat[1][2]*dH_dp[2],
            Tmat[2][0]*dH_dp[0] + Tmat[2][1]*dH_dp[1] + Tmat[2][2]*dH_dp[2]
        };

        // r x dxdt
        double rx = r_row[1] * dxdt[2] - r_row[2] * dxdt[1];
        double ry = r_row[2] * dxdt[0] - r_row[0] * dxdt[2];
        double rz = r_row[0] * dxdt[1] - r_row[1] * dxdt[0];
        double rCrossV_mag = std::sqrt(rx*rx + ry*ry + rz*rz);
        om_data[i] = rCrossV_mag / ((r2 < 1.0e-24) ? 1.0e-24 : r2);
    }
    return om_out;
}

static const double fact_table_wigner[] = {
    1.0, 1.0, 2.0, 6.0, 24.0, 120.0, 720.0, 5040.0, 40320.0, 362880.0,
    3628800.0, 39916800.0, 479001600.0, 6227020800.0, 87178291200.0,
    1307674368000.0, 20922789888000.0, 355687428096000.0, 6402373705728000.0
};

inline double wigner_d_element_c(int l, int mp, int m, double cosb2, double sinb2) {
    int k_min = std::max(0, m - mp);
    int k_max = std::min(l + m, l - mp);
    double pref = std::sqrt(
        fact_table_wigner[l + m] * fact_table_wigner[l - m] * fact_table_wigner[l + mp] * fact_table_wigner[l - mp]
    );
    double sum = 0.0;
    for (int k = k_min; k <= k_max; ++k) {
        double denom = fact_table_wigner[l + m - k] * fact_table_wigner[k] * fact_table_wigner[mp - m + k] * fact_table_wigner[l - mp - k];
        double coef = pref / denom;
        double sign = ((k - mp + m) % 2 != 0) ? -1.0 : 1.0;
        int cos_pow = 2 * l + m - mp - 2 * k;
        int sin_pow = mp - m + 2 * k;
        double term = sign * coef * std::pow(cosb2, cos_pow) * std::pow(sinb2, sin_pow);
        sum += term;
    }
    if ((m - mp) % 2 != 0) sum = -sum;
    return sum;
}

py::dict rotate_modes_jframe_native(
    const py::dict& modes_in,
    const torch::Tensor& alpha_u_t,
    const torch::Tensor& beta_u_t,
    const torch::Tensor& gamma_u_t,
    const std::vector<int>& l_values
) {
    validate_tensor(alpha_u_t, "alpha_u");
    validate_tensor(beta_u_t, "beta_u");
    validate_tensor(gamma_u_t, "gamma_u");
    int64_t n = alpha_u_t.size(0);
    const double* alpha_data = alpha_u_t.data_ptr<double>();
    const double* beta_data = beta_u_t.data_ptr<double>();
    const double* gamma_data = gamma_u_t.data_ptr<double>();

    py::dict out;
    for (int l : l_values) {
        int dim = 2 * l + 1;
        std::vector<const std::complex<double>*> h_pos(l + 1, nullptr);
        for (int mp = 1; mp <= l; ++mp) {
            py::tuple key = py::make_tuple(l, mp);
            if (modes_in.contains(key)) {
                torch::Tensor t = modes_in[key].cast<torch::Tensor>();
                if (t.numel() == n && t.is_complex()) {
                    h_pos[mp] = reinterpret_cast<const std::complex<double>*>(t.data_ptr<c10::complex<double>>());
                }
            }
        }

        std::vector<torch::Tensor> out_tensors;
        std::vector<std::complex<double>*> out_ptrs;
        out_tensors.reserve(dim);
        out_ptrs.reserve(dim);
        for (int m = -l; m <= l; ++m) {
            auto t = torch::zeros({n}, torch::dtype(torch::kComplexDouble).device(alpha_u_t.device()));
            out_ptrs.push_back(reinterpret_cast<std::complex<double>*>(t.data_ptr<c10::complex<double>>()));
            out_tensors.push_back(t);
        }

        double parity_l = ((l % 2) != 0) ? -1.0 : 1.0;

        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            double a_val = alpha_data[i];
            double b_val = beta_data[i];
            double g_val = gamma_data[i];
            double cosb2 = std::cos(b_val * 0.5);
            double sinb2 = std::sin(b_val * 0.5);

            std::complex<double> exp_a[2 * 8 + 1];
            std::complex<double> exp_g[2 * 8 + 1];
            for (int m = -l; m <= l; ++m) {
                exp_a[m + l] = std::polar(1.0, -m * a_val);
                exp_g[m + l] = std::polar(1.0, -m * g_val);
            }

            for (int m_idx = 0; m_idx < dim; ++m_idx) {
                int m = m_idx - l;
                std::complex<double> sum_val(0.0, 0.0);

                for (int mp_idx = 0; mp_idx < dim; ++mp_idx) {
                    int mp = mp_idx - l;
                    std::complex<double> h_val(0.0, 0.0);
                    if (mp > 0 && h_pos[mp] != nullptr) {
                        h_val = h_pos[mp][i];
                    } else if (mp < 0 && h_pos[-mp] != nullptr) {
                        h_val = parity_l * std::conj(h_pos[-mp][i]);
                    } else {
                        continue;
                    }

                    double d_elem = wigner_d_element_c(l, mp, m, cosb2, sinb2);
                    std::complex<double> D_val = exp_a[m + l] * d_elem * exp_g[mp + l];
                    sum_val += D_val * h_val;
                }
                out_ptrs[m_idx][i] = sum_val;
            }
        }

        for (int m_idx = 0; m_idx < dim; ++m_idx) {
            int m = m_idx - l;
            out[py::make_tuple(l, m)] = out_tensors[m_idx];
        }
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "Native C++ RKF45 ODE integrator for SEOBNR dynamics";
    module.def("rkf45_step_native", &rkf45_step_native, "Single Fehlberg 4(5) step");
    module.def("integrate_native", &integrate_native, "Adaptive RKF45 trajectory integrator");
    module.def("integrate_cpp_benchmark", &integrate_cpp_benchmark, "Pure C++ RHS RKF45 benchmark integrator");
    module.def("natural_spline_coeffs_native", &natural_spline_coeffs_native, "Fast natural cubic spline solver");
    module.def("natural_spline_eval_derivs_native", &natural_spline_eval_derivs_native, "Fast natural cubic spline evaluate (val, d1, d2)");
    module.def("natural_spline_interpolate_native", &natural_spline_interpolate_native, "Fast natural cubic spline interpolate vector");
    module.def("calcomega_polar_derivative_core_native", &calcomega_polar_derivative_core_native, "Fast C++ CalcOmega polar derivative");
    module.def("calcomega_lal_polar_derivative_native", &calcomega_lal_polar_derivative_native, "Fast C++ CalcOmega LAL polar derivative");
    module.def("non_keplerian_vphi_native", &non_keplerian_vphi_native, "Fast C++ Non-Keplerian vPhi evaluator");

    module.def("eob_hamiltonian_native", &eob_hamiltonian_native, "Fast C++ EOB Cartesian Hamiltonian");
    module.def("eob_hamiltonian_trajectory_native", &eob_hamiltonian_trajectory_native, "Fast C++ EOB Cartesian Hamiltonian across trajectory");
    module.def("omega_from_hamiltonian_velocity_native", &omega_from_hamiltonian_velocity_native, "Fast C++ orbital frequency from Cartesian trajectory");
    module.def("rotate_modes_jframe_native", &rotate_modes_jframe_native, "Fast C++ Wigner-D J-frame mode rotation");
    module.def("dH_dx_cartesian_native", &dH_dx_cartesian_native, "Fast C++ Cartesian d(H/eta)/dx");
    module.def("dH_dp_cartesian_native", &dH_dp_cartesian_native, "Fast C++ Cartesian d(H/eta)/dp");
    module.def("dH_dspin_cartesian_native", &dH_dspin_cartesian_native, "Fast C++ Cartesian d(H/eta)/dS1, d(H/eta)/dS2");
    module.def("eob_rhs_cartesian_native", &eob_rhs_cartesian_native, "Fast C++ 14D Cartesian RHS");
    module.def("integrate_cartesian_native", &integrate_cartesian_native, "Fast C++ 14D adaptive Cartesian trajectory integrator");
    module.def("ic_spherical_derivatives_native", &ic_spherical_derivatives_native, "Fast C++ initial-condition spherical Hamiltonian derivatives");
    module.def("initial_cartesian_conditions_native", &initial_cartesian_conditions_native, "Fast C++ 14D Cartesian initial conditions");
}


