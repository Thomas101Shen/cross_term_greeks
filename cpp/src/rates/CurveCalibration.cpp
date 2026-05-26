#include "mbs/rates/CurveCalibration.hpp"

#include "mbs/numerics/CompensatedSum.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace mbs::rates {

namespace {

double model_quote(const DiscountCurve &curve, const RateInstrument &instrument) {
  switch (instrument.type) {
  case RateInstrumentType::Deposit:
    return (1.0 / curve.discount(instrument.maturity) - 1.0) / instrument.maturity;
  case RateInstrumentType::ZeroCouponBond:
    return curve.discount(instrument.maturity);
  case RateInstrumentType::ParSwap:
    return par_swap_rate(curve, instrument.maturity, instrument.fixed_frequency);
  }
  throw std::invalid_argument("calibrate_bootstrap_curve: unsupported instrument type");
}

} // namespace

CurveCalibrationResult
calibrate_bootstrap_curve(const std::vector<RateInstrument> &instruments,
                          numerics::InterpolationMethod method) {
  DiscountCurve curve = bootstrap_discount_curve(instruments, method);

  std::vector<CalibrationError> errors;
  errors.reserve(instruments.size());
  numerics::CompensatedSum sum_squared_error;
  for (const auto &instrument : instruments) {
    const double model = model_quote(curve, instrument);
    const double error = model - instrument.quote;
    sum_squared_error.add(error * error);
    errors.push_back(CalibrationError{.maturity = instrument.maturity,
                                      .market_quote = instrument.quote,
                                      .model_quote = model,
                                      .error = error});
  }

  CurveCalibrationResult result{.curve = std::move(curve), .errors = std::move(errors)};
  result.rmse = std::sqrt(sum_squared_error.value() / instruments.size());
  return result;
}

} // namespace mbs::rates
