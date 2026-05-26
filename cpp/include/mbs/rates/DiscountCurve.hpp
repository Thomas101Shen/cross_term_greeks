#pragma once

#include "mbs/numerics/Interpolation.hpp"

#include <vector>

namespace mbs::rates {

struct CurveNode {
  double time = 0.0;
  double discount = 1.0;
};

class DiscountCurve {
public:
  DiscountCurve(
      std::vector<CurveNode> nodes,
      numerics::InterpolationMethod method = numerics::InterpolationMethod::LogLinear,
      bool allow_extrapolation = false);

  static DiscountCurve from_zero_rates(
      const std::vector<double> &times, const std::vector<double> &zero_rates,
      numerics::InterpolationMethod method = numerics::InterpolationMethod::LogLinear);

  double discount(double time) const;
  double zero_rate(double time) const;
  double forward_rate(double start_time, double end_time) const;

  const std::vector<CurveNode> &nodes() const { return nodes_; }

private:
  std::vector<CurveNode> nodes_;
  numerics::Interpolator interpolator_;
};

} // namespace mbs::rates
