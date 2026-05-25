#include "mbs/attribution/CrossSecondOrderCalculator.hpp"

#include "mbs/numerics/FiniteDifference.hpp"

#include <stdexcept>

namespace mbs::attribution {

AttributionResult
CrossSecondOrderCalculator::compute(const scenarios::ScenarioLattice &lattice,
                                    const AttributionRequest &request) const {
  if (!request.move_2.has_value()) {
    throw std::invalid_argument("CrossSecondOrderCalculator expects two factor moves");
  }

  const auto &move_1 = request.move_1;
  const auto &move_2 = *request.move_2;

  const double p_up_up =
      lattice.cross_price(request.run_id, request.security_id, move_1.key, move_1.bump,
                          move_2.key, move_2.bump);
  const double p_up_down =
      lattice.cross_price(request.run_id, request.security_id, move_1.key, move_1.bump,
                          move_2.key, -move_2.bump);
  const double p_down_up =
      lattice.cross_price(request.run_id, request.security_id, move_1.key, -move_1.bump,
                          move_2.key, move_2.bump);
  const double p_down_down =
      lattice.cross_price(request.run_id, request.security_id, move_1.key, -move_1.bump,
                          move_2.key, -move_2.bump);

  const double greek = numerics::central_cross(p_up_up, p_up_down, p_down_up,
                                               p_down_down, move_1.bump, move_2.bump);

  AttributionResult result;
  result.run_id = request.run_id;
  result.security_id = request.security_id;
  result.component = request.component;
  result.method = request.method.empty() ? "cross_fd" : request.method;
  result.label = result.component + ":" + move_1.key.name + " x " + move_2.key.name;
  result.factor_1 = move_1.key;
  result.factor_2 = move_2.key;
  result.greek = greek;
  result.bump_1 = move_1.bump;
  result.bump_2 = move_2.bump;
  result.realized_move_1 = move_1.realized_move;
  result.realized_move_2 = move_2.realized_move;
  result.pnl_price =
      numerics::cross_pnl(greek, move_1.realized_move, move_2.realized_move);

  return result;
}

} // namespace mbs::attribution
