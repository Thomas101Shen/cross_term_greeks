#include "mbs/rates/Bootstrap.hpp"
#include "mbs/rates/CurveCalibration.hpp"
#include "mbs/rates/LmmSofr.hpp"
#include "mbs/rates/SwaptionVolProxy.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace {

bool nearly_equal(double lhs, double rhs, double tolerance = 1.0e-10) {
  return std::abs(lhs - rhs) < tolerance;
}

std::vector<std::vector<double>> identity(std::size_t n) {
  std::vector<std::vector<double>> matrix(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) {
    matrix[i][i] = 1.0;
  }
  return matrix;
}

} // namespace

int main() {
  using mbs::numerics::InterpolationMethod;
  using mbs::rates::bootstrap_discount_curve;
  using mbs::rates::calibrate_bootstrap_curve;
  using mbs::rates::calibrate_lmm_sofr_proxy;
  using mbs::rates::CurveNode;
  using mbs::rates::derive_swaption_proxy_vol;
  using mbs::rates::DiscountCurve;
  using mbs::rates::LmmSofrCalibrationInput;
  using mbs::rates::LmmSofrInstrument;
  using mbs::rates::par_swap_rate;
  using mbs::rates::RateInstrument;
  using mbs::rates::RateInstrumentType;
  using mbs::rates::SwaptionProxyInput;

  const DiscountCurve curve({CurveNode{.time = 1.0, .discount = 0.98},
                             CurveNode{.time = 2.0, .discount = 0.94},
                             CurveNode{.time = 3.0, .discount = 0.89}},
                            InterpolationMethod::LogLinear);
  assert(nearly_equal(curve.discount(2.0), 0.94));
  assert(curve.zero_rate(2.0) > 0.0);
  assert(curve.forward_rate(1.0, 2.0) > 0.0);

  const std::vector<RateInstrument> instruments{
      RateInstrument{.type = RateInstrumentType::Deposit,
                     .maturity = 0.5,
                     .quote = 0.04,
                     .fixed_frequency = 2.0},
      RateInstrument{.type = RateInstrumentType::ZeroCouponBond,
                     .maturity = 1.0,
                     .quote = 0.96,
                     .fixed_frequency = 1.0},
      RateInstrument{.type = RateInstrumentType::ParSwap,
                     .maturity = 2.0,
                     .quote = 0.045,
                     .fixed_frequency = 1.0},
  };

  const auto bootstrapped = bootstrap_discount_curve(instruments);
  assert(nearly_equal(bootstrapped.discount(1.0), 0.96));
  assert(nearly_equal(par_swap_rate(bootstrapped, 2.0), 0.045));

  const auto calibrated = calibrate_bootstrap_curve(instruments);
  assert(calibrated.rmse < 1.0e-12);
  assert(calibrated.errors.size() == instruments.size());

  const SwaptionProxyInput proxy{
      .expiry = 1.0,
      .payment_times = {2.0, 3.0},
      .accruals = {1.0, 1.0},
      .forward_rate_vols = {0.20, 0.30},
      .correlation = {{1.0, 0.25}, {0.25, 1.0}},
  };
  const auto proxy_result = derive_swaption_proxy_vol(curve, proxy);
  assert(proxy_result.forward_swap_rate > 0.0);
  assert(proxy_result.annuity > 0.0);
  assert(proxy_result.black_vol > 0.0);
  assert(proxy_result.weights.size() == 2);

  const LmmSofrCalibrationInput lmm_input{
      .forward_start_times = {1.0, 2.0},
      .accruals = {1.0, 1.0},
      .initial_vols = {0.18, 0.28},
      .correlation = identity(2),
      .instruments = {LmmSofrInstrument{.expiry = 1.0,
                                        .tenor = 2.0,
                                        .market_vol = proxy_result.black_vol,
                                        .weight = 1.0}},
  };
  const auto lmm = calibrate_lmm_sofr_proxy(curve, lmm_input, 8);
  assert(lmm.forward_vols.size() == 2);
  assert(lmm.model_vols.size() == 1);
  assert(lmm.weighted_rmse >= 0.0);

  return 0;
}
