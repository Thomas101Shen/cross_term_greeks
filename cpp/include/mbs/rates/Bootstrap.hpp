#pragma once

#include "mbs/rates/DiscountCurve.hpp"

#include <vector>

namespace mbs::rates {

enum class RateInstrumentType {
  Deposit,
  ZeroCouponBond,
  ParSwap,
};

struct RateInstrument {
  RateInstrumentType type = RateInstrumentType::ZeroCouponBond;
  double maturity = 0.0;
  double quote = 0.0;
  double fixed_frequency = 1.0;
};

DiscountCurve bootstrap_discount_curve(
    const std::vector<RateInstrument> &instruments,
    numerics::InterpolationMethod method = numerics::InterpolationMethod::LogLinear);

double par_swap_rate(const DiscountCurve &curve, double maturity,
                     double fixed_frequency = 1.0);

} // namespace mbs::rates
