#include "mbs/attribution/CrossSecondOrderCalculator.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

namespace {

constexpr double kTolerance = 1.0e-12;

bool nearly_equal(double lhs, double rhs) { return std::abs(lhs - rhs) < kTolerance; }

mbs::scenarios::ScenarioPrice cross_price(const mbs::factors::FactorKey &factor_1,
                                          double shock_1,
                                          const mbs::factors::FactorKey &factor_2,
                                          double shock_2, double price) {
  return {
      .run_id = "RUN_001",
      .security_id = "FNMA_30Y_6",
      .asof_date = "2026-05-25",
      .scenario_id = factor_1.name + "_" + factor_2.name,
      .factor_type_1 = factor_1.type,
      .factor_name_1 = factor_1.name,
      .shock_1 = shock_1,
      .factor_type_2 = factor_2.type,
      .factor_name_2 = factor_2.name,
      .shock_2 = shock_2,
      .price = price,
  };
}

} // namespace

int main() {
  using mbs::attribution::AttributionRequest;
  using mbs::attribution::CrossSecondOrderCalculator;
  using mbs::factors::FactorKey;
  using mbs::factors::FactorMove;
  using mbs::scenarios::ScenarioLattice;

  const FactorKey rate{"RATE", "RATE_LEVEL"};
  const FactorKey vol{"VOL", "VOL_SHORT"};

  const ScenarioLattice lattice({
      cross_price(rate, 1.0, vol, 2.0, 103.0),
      cross_price(rate, 1.0, vol, -2.0, 99.0),
      cross_price(rate, -1.0, vol, 2.0, 98.0),
      cross_price(rate, -1.0, vol, -2.0, 100.0),
  });

  const CrossSecondOrderCalculator calculator;
  const auto result = calculator.compute(
      lattice, AttributionRequest{
                   .run_id = "RUN_001",
                   .security_id = "FNMA_30Y_6",
                   .component = "RATE_VOL",
                   .method = "cross_fd",
                   .move_1 = FactorMove{.key = rate, .bump = 1.0, .realized_move = 5.0},
                   .move_2 = FactorMove{.key = vol, .bump = 2.0, .realized_move = -2.5},
               });

  assert(result.component == "RATE_VOL");
  assert(result.label == "RATE_VOL:RATE_LEVEL x VOL_SHORT");
  assert(result.factor_1 == rate);
  assert(result.factor_2 == vol);
  assert(nearly_equal(result.greek, 0.75));
  assert(nearly_equal(result.pnl_price, -9.375));

  const auto reversed_lookup = calculator.compute(
      lattice, AttributionRequest{
                   .run_id = "RUN_001",
                   .security_id = "FNMA_30Y_6",
                   .component = "VOL_RATE",
                   .method = "cross_fd",
                   .move_1 = FactorMove{.key = vol, .bump = 2.0, .realized_move = -2.5},
                   .move_2 = FactorMove{.key = rate, .bump = 1.0, .realized_move = 5.0},
               });

  assert(nearly_equal(reversed_lookup.greek, 0.75));

  bool rejected_scalar_request = false;
  try {
    calculator.compute(lattice, AttributionRequest{
                                    .run_id = "RUN_001",
                                    .security_id = "FNMA_30Y_6",
                                    .component = "BAD",
                                    .method = "cross_fd",
                                    .move_1 = FactorMove{.key = rate, .bump = 1.0},
                                });
  } catch (const std::invalid_argument &) {
    rejected_scalar_request = true;
  }
  assert(rejected_scalar_request);

  return 0;
}
