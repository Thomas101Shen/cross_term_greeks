#pragma once

#include <vector>

namespace mbs::numerics {

enum class InterpolationMethod {
  Linear,
  LogLinear,
  FlatForward,
  MonotoneCubic,
};

class Interpolator {
public:
  Interpolator(std::vector<double> x, std::vector<double> y,
               InterpolationMethod method = InterpolationMethod::Linear,
               bool allow_extrapolation = false);

  double value(double x_value) const;

  const std::vector<double> &x() const { return x_; }
  const std::vector<double> &y() const { return y_; }
  InterpolationMethod method() const { return method_; }

private:
  std::vector<double> x_;
  std::vector<double> y_;
  std::vector<double> slopes_;
  InterpolationMethod method_;
  bool allow_extrapolation_;
};

} // namespace mbs::numerics
