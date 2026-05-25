#include "mbs/attribution/MatrixHessianCalculator.hpp"

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
  using mbs::attribution::MatrixHessianCalculator;
  using mbs::factors::FactorKey;
  using mbs::factors::FactorMove;
  using mbs::scenarios::ScenarioLattice;

  const FactorKey rate{"RATE", "RATE_LEVEL"};
  const FactorKey vol{"VOL", "VOL_SHORT"};

  const ScenarioLattice lattice({
      base_price(),
      scalar_price(rate, 1.0, 101.0),
      scalar_price(rate, -1.0, 100.0),
      scalar_price(vol, 2.0, 102.0),
      scalar_price(vol, -2.0, 106.0),
      cross_price(rate, 1.0, vol, 2.0, 103.0),
      cross_price(rate, 1.0, vol, -2.0, 99.0),
      cross_price(rate, -1.0, vol, 2.0, 98.0),
      cross_price(rate, -1.0, vol, -2.0, 100.0),
  });

  const MatrixHessianCalculator calculator;
  const AttributionRequest request{
      .run_id = "RUN_001",
      .security_id = "FNMA_30Y_6",
      .component = "SECOND_ORDER",
      .method = "matrix_fd",
      .moves =
          {
              FactorMove{.key = rate, .bump = 1.0, .realized_move = 3.0},
              FactorMove{.key = vol, .bump = 2.0, .realized_move = -1.5},
          },
  };

  const auto components = calculator.compute_components(lattice, request);
  assert(components.size() == 3);
  assert(components[0].method == "matrix_scalar_fd");
  assert(components[1].method == "matrix_scalar_fd");
  assert(components[2].method == "matrix_cross_fd");
  assert(nearly_equal(components[0].greek, 1.0));
  assert(nearly_equal(components[0].pnl_price, 4.5));
  assert(nearly_equal(components[1].greek, 2.0));
  assert(nearly_equal(components[1].pnl_price, 2.25));
  assert(nearly_equal(components[2].greek, 0.75));
  assert(nearly_equal(components[2].pnl_price, -3.375));

  const auto total = calculator.compute(lattice, request);
  assert(total.label == "SECOND_ORDER:matrix");
  assert(nearly_equal(total.pnl_price, 3.375));

  bool rejected_scalar_matrix = false;
  try {
    calculator.compute(lattice, AttributionRequest{
                                    .run_id = "RUN_001",
                                    .security_id = "FNMA_30Y_6",
                                    .component = "BAD",
                                    .moves =
                                        {
                                            FactorMove{.key = rate, .bump = 1.0},
                                        },
                                });
  } catch (const std::invalid_argument &) {
    rejected_scalar_matrix = true;
  }
  assert(rejected_scalar_matrix);

  return 0;
}
