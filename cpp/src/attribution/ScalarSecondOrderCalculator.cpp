#include "mbs/attribution/ScalarSecondOrderCalculator.hpp"

#include "mbs/numerics/FiniteDifference.hpp"

#include <stdexcept>

namespace mbs::attribution {

AttributionResult
ScalarSecondOrderCalculator::compute(const scenarios::ScenarioLattice &lattice,
                                     const AttributionRequest &request) const {
  if (request.move_2.has_value()) {
    throw std::invalid_argument(
        "ScalarSecondOrderCalculator expects exactly one factor move");
  }

  const auto &move = request.move_1;
  const double p_base = lattice.base_price(request.run_id, request.security_id);
  const double p_up =
      lattice.shocked_price(request.run_id, request.security_id, move.key, move.bump);
  const double p_down =
      lattice.shocked_price(request.run_id, request.security_id, move.key, -move.bump);

  const double greek = numerics::central_second(p_base, p_up, p_down, move.bump);

  AttributionResult result;
  result.run_id = request.run_id;
  result.security_id = request.security_id;
  result.component = request.component;
  result.method = request.method.empty() ? "scalar_fd" : request.method;
  result.label = result.component + ":" + move.key.name;
  result.factor_1 = move.key;
  result.greek = greek;
  result.bump_1 = move.bump;
  result.realized_move_1 = move.realized_move;
  result.pnl_price = numerics::second_order_pnl(greek, move.realized_move);

  return result;
}

} // namespace mbs::attribution
