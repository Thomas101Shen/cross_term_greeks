#pragma once

#include "mbs/rates/DiscountCurve.hpp"

#include <vector>

namespace mbs::rates {

struct SwaptionProxyInput {
  double expiry = 0.0;
  std::vector<double> payment_times;
  std::vector<double> accruals;
  std::vector<double> forward_rate_vols;
  std::vector<std::vector<double>> correlation;
};

struct SwaptionProxyResult {
  double forward_swap_rate = 0.0;
  double annuity = 0.0;
  double black_vol = 0.0;
  std::vector<double> weights;
};

SwaptionProxyResult derive_swaption_proxy_vol(const DiscountCurve &curve,
                                              const SwaptionProxyInput &input);

} // namespace mbs::rates
