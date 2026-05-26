#pragma once

#include "mbs/rates/DiscountCurve.hpp"
#include "mbs/rates/SwaptionVolProxy.hpp"

#include <vector>

namespace mbs::rates {

struct LmmSofrInstrument {
  double expiry = 0.0;
  double tenor = 0.0;
  double market_vol = 0.0;
  double weight = 1.0;
};

struct LmmSofrCalibrationInput {
  std::vector<double> forward_start_times;
  std::vector<double> accruals;
  std::vector<double> initial_vols;
  std::vector<std::vector<double>> correlation;
  std::vector<LmmSofrInstrument> instruments;
};

struct LmmSofrCalibrationResult {
  std::vector<double> forward_vols;
  std::vector<double> model_vols;
  double weighted_rmse = 0.0;
};

LmmSofrCalibrationResult calibrate_lmm_sofr_proxy(const DiscountCurve &curve,
                                                  const LmmSofrCalibrationInput &input,
                                                  std::size_t max_iterations = 32,
                                                  double tolerance = 1.0e-10);

} // namespace mbs::rates
