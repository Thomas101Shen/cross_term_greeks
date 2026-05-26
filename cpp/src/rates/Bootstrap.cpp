#include "mbs/rates/Bootstrap.hpp"

#include "mbs/numerics/CompensatedSum.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace mbs::rates {

namespace {

constexpr double kTimeTolerance = 1.0e-12;

void validate_instrument(const RateInstrument &instrument) {
  if (!(instrument.maturity > 0.0)) {
    throw std::invalid_argument("bootstrap_discount_curve: maturity must be positive");
  }
  if (instrument.type != RateInstrumentType::ZeroCouponBond &&
      instrument.fixed_frequency <= 0.0) {
    throw std::invalid_argument(
        "bootstrap_discount_curve: fixed frequency must be positive");
  }
}

std::vector<RateInstrument>
sorted_instruments(const std::vector<RateInstrument> &instruments) {
  if (instruments.empty()) {
    throw std::invalid_argument("bootstrap_discount_curve: instruments are empty");
  }

  auto sorted = instruments;
  std::sort(sorted.begin(), sorted.end(),
            [](const RateInstrument &lhs, const RateInstrument &rhs) {
              return lhs.maturity < rhs.maturity;
            });
  for (const auto &instrument : sorted) {
    validate_instrument(instrument);
  }
  for (std::size_t i = 1; i < sorted.size(); ++i) {
    if (std::abs(sorted[i].maturity - sorted[i - 1].maturity) <= kTimeTolerance) {
      throw std::invalid_argument("bootstrap_discount_curve: duplicate maturities");
    }
  }
  return sorted;
}

double known_discount(const std::vector<CurveNode> &nodes, double time,
                      numerics::InterpolationMethod method) {
  if (time == 0.0) {
    return 1.0;
  }
  for (const auto &node : nodes) {
    if (std::abs(node.time - time) <= kTimeTolerance) {
      return node.discount;
    }
  }
  if (nodes.size() < 2 || time > nodes.back().time) {
    throw std::invalid_argument(
        "bootstrap_discount_curve: cannot infer intermediate coupon discount");
  }
  return DiscountCurve(nodes, method, true).discount(time);
}

double bootstrap_swap_discount(const std::vector<CurveNode> &nodes,
                               const RateInstrument &swap,
                               numerics::InterpolationMethod method) {
  const double accrual = 1.0 / swap.fixed_frequency;
  numerics::CompensatedSum annuity_before_final;
  for (double pay_time = accrual; pay_time < swap.maturity - kTimeTolerance;
       pay_time += accrual) {
    annuity_before_final.add(accrual * known_discount(nodes, pay_time, method));
  }

  const double final_accrual =
      swap.maturity -
      std::floor((swap.maturity - kTimeTolerance) * swap.fixed_frequency) /
          swap.fixed_frequency;
  const double last_accrual = final_accrual > kTimeTolerance ? final_accrual : accrual;
  const double numerator = 1.0 - swap.quote * annuity_before_final.value();
  const double denominator = 1.0 + swap.quote * last_accrual;
  if (!(denominator > 0.0) || !(numerator > 0.0)) {
    throw std::invalid_argument("bootstrap_discount_curve: invalid swap quote");
  }
  return numerator / denominator;
}

} // namespace

DiscountCurve bootstrap_discount_curve(const std::vector<RateInstrument> &instruments,
                                       numerics::InterpolationMethod method) {
  const auto sorted = sorted_instruments(instruments);
  std::vector<CurveNode> nodes;
  nodes.reserve(sorted.size());

  for (const auto &instrument : sorted) {
    double discount = 0.0;
    switch (instrument.type) {
    case RateInstrumentType::Deposit:
      discount = 1.0 / (1.0 + instrument.quote * instrument.maturity);
      break;
    case RateInstrumentType::ZeroCouponBond:
      discount = instrument.quote;
      break;
    case RateInstrumentType::ParSwap:
      discount = bootstrap_swap_discount(nodes, instrument, method);
      break;
    }

    if (!(discount > 0.0)) {
      throw std::invalid_argument(
          "bootstrap_discount_curve: discount must be positive");
    }
    nodes.push_back(CurveNode{.time = instrument.maturity, .discount = discount});
  }

  if (nodes.size() == 1) {
    nodes.insert(nodes.begin(), CurveNode{.time = 0.0, .discount = 1.0});
  }
  return DiscountCurve(std::move(nodes), method);
}

double par_swap_rate(const DiscountCurve &curve, double maturity,
                     double fixed_frequency) {
  if (!(maturity > 0.0) || !(fixed_frequency > 0.0)) {
    throw std::invalid_argument("par_swap_rate: invalid swap schedule");
  }

  const double accrual = 1.0 / fixed_frequency;
  numerics::CompensatedSum annuity;
  for (double pay_time = accrual; pay_time < maturity - kTimeTolerance;
       pay_time += accrual) {
    annuity.add(accrual * curve.discount(pay_time));
  }
  const double final_accrual =
      maturity -
      std::floor((maturity - kTimeTolerance) * fixed_frequency) / fixed_frequency;
  const double last_accrual = final_accrual > kTimeTolerance ? final_accrual : accrual;
  annuity.add(last_accrual * curve.discount(maturity));
  return (1.0 - curve.discount(maturity)) / annuity.value();
}

} // namespace mbs::rates
