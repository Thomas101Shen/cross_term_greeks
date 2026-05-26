#include "mbs/rates/LmmSofr.hpp"

#include "mbs/numerics/CompensatedSum.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace mbs::rates {

namespace {

void validate_lmm_input(const LmmSofrCalibrationInput &input) {
  const std::size_t n = input.forward_start_times.size();
  if (n == 0 || input.accruals.size() != n || input.initial_vols.size() != n ||
      input.correlation.size() != n || input.instruments.empty()) {
    throw std::invalid_argument("calibrate_lmm_sofr_proxy: inconsistent input sizes");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (input.forward_start_times[i] < 0.0 || !(input.accruals[i] > 0.0) ||
        input.initial_vols[i] < 0.0 || input.correlation[i].size() != n) {
      throw std::invalid_argument("calibrate_lmm_sofr_proxy: invalid forward bucket");
    }
    if (i > 0 && !(input.forward_start_times[i] > input.forward_start_times[i - 1])) {
      throw std::invalid_argument(
          "calibrate_lmm_sofr_proxy: start times must increase");
    }
  }
  for (const auto &instrument : input.instruments) {
    if (instrument.expiry < 0.0 || !(instrument.tenor > 0.0) ||
        instrument.market_vol < 0.0 || !(instrument.weight > 0.0)) {
      throw std::invalid_argument("calibrate_lmm_sofr_proxy: invalid instrument");
    }
  }
}

std::vector<std::size_t> active_indices(const LmmSofrCalibrationInput &input,
                                        const LmmSofrInstrument &instrument) {
  std::vector<std::size_t> indices;
  const double end_time = instrument.expiry + instrument.tenor;
  for (std::size_t i = 0; i < input.forward_start_times.size(); ++i) {
    const double start = input.forward_start_times[i];
    const double end = start + input.accruals[i];
    if (start >= instrument.expiry - 1.0e-12 && end <= end_time + 1.0e-12) {
      indices.push_back(i);
    }
  }
  if (indices.empty()) {
    throw std::invalid_argument("calibrate_lmm_sofr_proxy: instrument has no buckets");
  }
  return indices;
}

double model_instrument_vol(const DiscountCurve &curve,
                            const LmmSofrCalibrationInput &input,
                            const std::vector<double> &vols,
                            const LmmSofrInstrument &instrument) {
  const auto indices = active_indices(input, instrument);
  SwaptionProxyInput proxy;
  proxy.expiry = instrument.expiry;

  for (const std::size_t index : indices) {
    proxy.payment_times.push_back(input.forward_start_times[index] +
                                  input.accruals[index]);
    proxy.accruals.push_back(input.accruals[index]);
    proxy.forward_rate_vols.push_back(vols[index]);
  }

  proxy.correlation.resize(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    proxy.correlation[i].reserve(indices.size());
    for (const std::size_t j_index : indices) {
      proxy.correlation[i].push_back(input.correlation[indices[i]][j_index]);
    }
  }

  return derive_swaption_proxy_vol(curve, proxy).black_vol;
}

std::vector<double> model_vols(const DiscountCurve &curve,
                               const LmmSofrCalibrationInput &input,
                               const std::vector<double> &vols) {
  std::vector<double> values;
  values.reserve(input.instruments.size());
  for (const auto &instrument : input.instruments) {
    values.push_back(model_instrument_vol(curve, input, vols, instrument));
  }
  return values;
}

double objective(const DiscountCurve &curve, const LmmSofrCalibrationInput &input,
                 const std::vector<double> &vols) {
  const auto fitted = model_vols(curve, input, vols);
  mbs::numerics::CompensatedSum value;
  for (std::size_t i = 0; i < fitted.size(); ++i) {
    const double error = fitted[i] - input.instruments[i].market_vol;
    value.add(input.instruments[i].weight * error * error);
  }
  return value.value();
}

} // namespace

LmmSofrCalibrationResult calibrate_lmm_sofr_proxy(const DiscountCurve &curve,
                                                  const LmmSofrCalibrationInput &input,
                                                  std::size_t max_iterations,
                                                  double tolerance) {
  validate_lmm_input(input);

  std::vector<double> vols = input.initial_vols;
  double previous_objective = objective(curve, input, vols);

  for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
    for (std::size_t factor = 0; factor < vols.size(); ++factor) {
      const double base = vols[factor];
      const double step = std::max(1.0e-5, 1.0e-4 * std::max(1.0, base));
      vols[factor] = base + step;
      const double up = objective(curve, input, vols);
      vols[factor] = std::max(0.0, base - step);
      const double down = objective(curve, input, vols);
      vols[factor] = base;

      const double gradient = (up - down) / (2.0 * step);
      const double learning_rate = 0.25 / static_cast<double>(iteration + 1);
      vols[factor] = std::max(0.0, base - learning_rate * gradient);
    }

    const double current_objective = objective(curve, input, vols);
    if (std::abs(previous_objective - current_objective) <= tolerance) {
      break;
    }
    previous_objective = current_objective;
  }

  auto fitted = model_vols(curve, input, vols);
  mbs::numerics::CompensatedSum weighted_error;
  mbs::numerics::CompensatedSum weight_sum;
  for (std::size_t i = 0; i < fitted.size(); ++i) {
    const double error = fitted[i] - input.instruments[i].market_vol;
    weighted_error.add(input.instruments[i].weight * error * error);
    weight_sum.add(input.instruments[i].weight);
  }

  return LmmSofrCalibrationResult{
      .forward_vols = std::move(vols),
      .model_vols = std::move(fitted),
      .weighted_rmse = std::sqrt(weighted_error.value() / weight_sum.value())};
}

} // namespace mbs::rates
