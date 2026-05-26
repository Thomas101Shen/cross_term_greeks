#include "mbs/numerics/Interpolation.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

bool nearly_equal(double lhs, double rhs, double tolerance = 1.0e-12) {
  return std::abs(lhs - rhs) < tolerance;
}

} // namespace

int main() {
  using mbs::numerics::InterpolationMethod;
  using mbs::numerics::Interpolator;

  const Interpolator linear({1.0, 2.0, 3.0}, {10.0, 20.0, 50.0});
  assert(nearly_equal(linear.value(1.5), 15.0));

  const Interpolator log_linear({1.0, 2.0}, {0.99, 0.9801},
                                InterpolationMethod::LogLinear);
  assert(nearly_equal(log_linear.value(1.5), 0.9850375627355539));

  const Interpolator monotone({0.0, 1.0, 2.0, 3.0}, {0.0, 1.0, 1.5, 1.75},
                              InterpolationMethod::MonotoneCubic);
  const double mid = monotone.value(1.5);
  assert(mid >= 1.0);
  assert(mid <= 1.5);

  const Interpolator extrapolating({1.0, 2.0}, {10.0, 20.0},
                                   InterpolationMethod::Linear, true);
  assert(nearly_equal(extrapolating.value(0.5), 5.0));

  bool rejected_unsorted = false;
  try {
    (void)Interpolator({1.0, 1.0}, {1.0, 2.0});
  } catch (const std::invalid_argument &) {
    rejected_unsorted = true;
  }
  assert(rejected_unsorted);

  return 0;
}
