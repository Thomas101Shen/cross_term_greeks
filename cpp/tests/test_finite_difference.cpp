#include "mbs/numerics/FiniteDifference.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

bool nearly_equal(double lhs, double rhs) { return std::abs(lhs - rhs) < 1.0e-12; }

} // namespace

int main() {
  using mbs::numerics::central_cross;
  using mbs::numerics::central_second;
  using mbs::numerics::cross_pnl;
  using mbs::numerics::second_order_pnl;

  assert(nearly_equal(central_second(10.0, 13.0, 11.0, 2.0), 1.0));
  assert(nearly_equal(second_order_pnl(0.5, -2.5), 1.5625));

  assert(nearly_equal(central_cross(103.0, 99.0, 98.0, 100.0, 1.0, 2.0), 0.75));
  assert(nearly_equal(cross_pnl(0.75, 5.0, -2.5), -9.375));

  bool rejected_zero_scalar_bump = false;
  try {
    (void)central_second(10.0, 11.0, 9.0, 0.0);
  } catch (const std::invalid_argument &) {
    rejected_zero_scalar_bump = true;
  }
  assert(rejected_zero_scalar_bump);

  bool rejected_zero_cross_bump = false;
  try {
    (void)central_cross(1.0, 1.0, 1.0, 1.0, 1.0, 0.0);
  } catch (const std::invalid_argument &) {
    rejected_zero_cross_bump = true;
  }
  assert(rejected_zero_cross_bump);

  return 0;
}
