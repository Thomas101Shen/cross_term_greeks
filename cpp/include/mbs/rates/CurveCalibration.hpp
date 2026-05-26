#pragma once

#include "mbs/rates/Bootstrap.hpp"

#include <vector>

namespace mbs::rates {

struct CalibrationError {
  double maturity = 0.0;
  double market_quote = 0.0;
  double model_quote = 0.0;
  double error = 0.0;
};

struct CurveCalibrationResult {
  DiscountCurve curve;
  std::vector<CalibrationError> errors;
  double rmse = 0.0;
};

CurveCalibrationResult calibrate_bootstrap_curve(
    const std::vector<RateInstrument> &instruments,
    numerics::InterpolationMethod method = numerics::InterpolationMethod::LogLinear);

} // namespace mbs::rates
