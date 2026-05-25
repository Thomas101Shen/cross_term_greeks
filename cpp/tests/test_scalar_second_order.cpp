#include "mbs/attribution/ScalarSecondOrderCalculator.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

constexpr double kTolerance = 1.0e-12;

bool nearly_equal(double lhs, double rhs) { return std::abs(lhs - rhs) < kTolerance; }

mbs::scenarios::ScenarioPrice base_price() {
  return {
      .run_id = "RUN_001",
      .security_id = "FNMA_30Y_6",
      .asof_date = "2026-05-25",
      .scenario_id = "BASE",
      .factor_type_1 = "BASE",
      .factor_name_1 = "BASE",
      .shock_1 = 0.0,
      .price = 100.0,
  };
}

mbs::scenarios::ScenarioPrice scalar_price(const mbs::factors::FactorKey &factor,
                                           double shock, double price) {
  return {
      .run_id = "RUN_001",
      .security_id = "FNMA_30Y_6",
      .asof_date = "2026-05-25",
      .scenario_id = factor.name,
      .factor_type_1 = factor.type,
      .factor_name_1 = factor.name,
      .shock_1 = shock,
      .price = price,
  };
}

} // namespace

int main() {
  using mbs::attribution::AttributionRequest;
  using mbs::attribution::ScalarSecondOrderCalculator;
  using mbs::factors::FactorKey;
  using mbs::factors::FactorMove;
  using mbs::scenarios::ScenarioLattice;

  const FactorKey vol_short{"VOL", "VOL_SHORT"};
  const ScenarioLattice volga_lattice({
      base_price(),
      scalar_price(vol_short, 1.0, 99.0),
      scalar_price(vol_short, -1.0, 101.5),
  });

  const ScalarSecondOrderCalculator calculator;
  const auto volga = calculator.compute(
      volga_lattice,
      AttributionRequest{
          .run_id = "RUN_001",
          .security_id = "FNMA_30Y_6",
          .component = "VOLGA",
          .method = "scalar_fd",
          .move_1 = FactorMove{.key = vol_short, .bump = 1.0, .realized_move = -2.5},
      });

  assert(volga.component == "VOLGA");
  assert(volga.method == "scalar_fd");
  assert(volga.label == "VOLGA:VOL_SHORT");
  assert(volga.factor_1 == vol_short);
  assert(!volga.factor_2.has_value());
  assert(nearly_equal(volga.greek, 0.5));
  assert(nearly_equal(volga.pnl_price, 1.5625));

  const FactorKey cc_oas{"CC", "CCOAS"};
  const ScenarioLattice cc_lattice({
      base_price(),
      scalar_price(cc_oas, 1.0, 100.25),
      scalar_price(cc_oas, -1.0, 100.75),
  });

  const auto cc_convexity = calculator.compute(
      cc_lattice,
      AttributionRequest{
          .run_id = "RUN_001",
          .security_id = "FNMA_30Y_6",
          .component = "CC_CONVEXITY",
          .method = "scalar_fd",
          .move_1 = FactorMove{.key = cc_oas, .bump = 1.0, .realized_move = 3.0},
      });

  assert(cc_convexity.label == "CC_CONVEXITY:CCOAS");
  assert(nearly_equal(cc_convexity.greek, 1.0));
  assert(nearly_equal(cc_convexity.pnl_price, 4.5));

  bool rejected_cross_request = false;
  try {
    calculator.compute(volga_lattice,
                       AttributionRequest{
                           .run_id = "RUN_001",
                           .security_id = "FNMA_30Y_6",
                           .component = "BAD",
                           .method = "scalar_fd",
                           .move_1 = FactorMove{.key = vol_short, .bump = 1.0},
                           .move_2 = FactorMove{.key = cc_oas, .bump = 1.0},
                       });
  } catch (const std::invalid_argument &) {
    rejected_cross_request = true;
  }
  assert(rejected_cross_request);

  return 0;
}
