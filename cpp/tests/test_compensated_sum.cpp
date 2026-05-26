#include "mbs/numerics/CompensatedSum.hpp"

#include <cassert>
#include <cmath>

namespace {

bool nearly_equal(double lhs, double rhs) { return std::abs(lhs - rhs) < 1.0e-12; }

} // namespace

int main() {
  mbs::numerics::CompensatedSum sum;
  sum.add(1.0e16);
  sum.add(1.0);
  sum.add(-1.0e16);
  assert(nearly_equal(sum.value(), 1.0));

  mbs::numerics::CompensatedSum small_terms;
  for (int i = 0; i < 1000; ++i) {
    small_terms.add(0.001);
  }
  assert(nearly_equal(small_terms.value(), 1.0));

  return 0;
}
