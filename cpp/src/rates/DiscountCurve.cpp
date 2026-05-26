#include "mbs/rates/DiscountCurve.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace mbs::rates {

namespace {

void validate_nodes(const std::vector<CurveNode> &nodes) {
  if (nodes.size() < 2) {
    throw std::invalid_argument("DiscountCurve: at least two nodes are required");
  }
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].time < 0.0 || !(nodes[i].discount > 0.0)) {
      throw std::invalid_argument("DiscountCurve: invalid node");
    }
    if (i > 0 && !(nodes[i].time > nodes[i - 1].time)) {
      throw std::invalid_argument("DiscountCurve: node times must increase");
    }
  }
}

std::vector<double> times_from_nodes(const std::vector<CurveNode> &nodes) {
  std::vector<double> times;
  times.reserve(nodes.size());
  for (const auto &node : nodes) {
    times.push_back(node.time);
  }
  return times;
}

std::vector<double> discounts_from_nodes(const std::vector<CurveNode> &nodes) {
  std::vector<double> discounts;
  discounts.reserve(nodes.size());
  for (const auto &node : nodes) {
    discounts.push_back(node.discount);
  }
  return discounts;
}

} // namespace

DiscountCurve::DiscountCurve(std::vector<CurveNode> nodes,
                             numerics::InterpolationMethod method,
                             bool allow_extrapolation)
    : nodes_(std::move(nodes)),
      interpolator_(times_from_nodes(nodes_), discounts_from_nodes(nodes_), method,
                    allow_extrapolation) {
  validate_nodes(nodes_);
}

DiscountCurve DiscountCurve::from_zero_rates(const std::vector<double> &times,
                                             const std::vector<double> &zero_rates,
                                             numerics::InterpolationMethod method) {
  if (times.size() != zero_rates.size()) {
    throw std::invalid_argument("DiscountCurve: time and zero-rate sizes must match");
  }

  std::vector<CurveNode> nodes;
  nodes.reserve(times.size());
  for (std::size_t i = 0; i < times.size(); ++i) {
    nodes.push_back(
        CurveNode{.time = times[i], .discount = std::exp(-zero_rates[i] * times[i])});
  }
  return DiscountCurve(std::move(nodes), method);
}

double DiscountCurve::discount(double time) const {
  if (time == 0.0) {
    return 1.0;
  }
  if (time < 0.0) {
    throw std::invalid_argument("DiscountCurve: time cannot be negative");
  }
  return interpolator_.value(time);
}

double DiscountCurve::zero_rate(double time) const {
  if (!(time > 0.0)) {
    throw std::invalid_argument("DiscountCurve: zero-rate time must be positive");
  }
  return -std::log(discount(time)) / time;
}

double DiscountCurve::forward_rate(double start_time, double end_time) const {
  if (!(end_time > start_time) || start_time < 0.0) {
    throw std::invalid_argument("DiscountCurve: invalid forward period");
  }
  return std::log(discount(start_time) / discount(end_time)) / (end_time - start_time);
}

} // namespace mbs::rates
