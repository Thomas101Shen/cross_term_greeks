#include "mbs/rates/SwaptionVolProxy.hpp"

#include "mbs/numerics/CompensatedSum.hpp"

#include <cmath>
#include <stdexcept>

namespace mbs::rates {

namespace {

void validate_proxy_input(const SwaptionProxyInput &input) {
  const std::size_t n = input.payment_times.size();
  if (!(input.expiry >= 0.0) || n == 0 || input.accruals.size() != n ||
      input.forward_rate_vols.size() != n || input.correlation.size() != n) {
    throw std::invalid_argument("derive_swaption_proxy_vol: inconsistent input sizes");
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!(input.payment_times[i] > input.expiry) || !(input.accruals[i] > 0.0) ||
        input.forward_rate_vols[i] < 0.0 || input.correlation[i].size() != n) {
      throw std::invalid_argument("derive_swaption_proxy_vol: invalid input");
    }
  }
}

} // namespace

SwaptionProxyResult derive_swaption_proxy_vol(const DiscountCurve &curve,
                                              const SwaptionProxyInput &input) {
  validate_proxy_input(input);

  const std::size_t n = input.payment_times.size();
  std::vector<double> discounted_accruals;
  discounted_accruals.reserve(n);

  mbs::numerics::CompensatedSum annuity_sum;
  for (std::size_t i = 0; i < n; ++i) {
    const double discounted_accrual =
        input.accruals[i] * curve.discount(input.payment_times[i]);
    discounted_accruals.push_back(discounted_accrual);
    annuity_sum.add(discounted_accrual);
  }
  const double annuity = annuity_sum.value();
  if (!(annuity > 0.0)) {
    throw std::invalid_argument("derive_swaption_proxy_vol: annuity must be positive");
  }

  const double start_discount = curve.discount(input.expiry);
  const double end_discount = curve.discount(input.payment_times.back());
  const double forward_swap_rate = (start_discount - end_discount) / annuity;
  if (!(forward_swap_rate > 0.0)) {
    throw std::invalid_argument(
        "derive_swaption_proxy_vol: forward swap rate must be positive");
  }

  std::vector<double> weights;
  weights.reserve(n);
  for (const double discounted_accrual : discounted_accruals) {
    weights.push_back(discounted_accrual / annuity);
  }

  mbs::numerics::CompensatedSum variance_sum;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      variance_sum.add(weights[i] * weights[j] * input.forward_rate_vols[i] *
                       input.forward_rate_vols[j] * input.correlation[i][j]);
    }
  }
  double variance = variance_sum.value();
  if (variance < 0.0 && variance > -1.0e-14) {
    variance = 0.0;
  }
  if (variance < 0.0) {
    throw std::invalid_argument("derive_swaption_proxy_vol: negative variance");
  }

  return SwaptionProxyResult{.forward_swap_rate = forward_swap_rate,
                             .annuity = annuity,
                             .black_vol = std::sqrt(variance),
                             .weights = std::move(weights)};
}

} // namespace mbs::rates
