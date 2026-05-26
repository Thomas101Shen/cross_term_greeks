#include "mbs/attribution/MatrixHessianCalculator.hpp"

#include "mbs/attribution/CrossSecondOrderCalculator.hpp"
#include "mbs/attribution/ScalarSecondOrderCalculator.hpp"
#include "mbs/numerics/CompensatedSum.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace mbs::attribution {

namespace {

std::vector<factors::FactorMove> requested_moves(const AttributionRequest &request) {
  if (!request.moves.empty()) {
    return request.moves;
  }

  std::vector<factors::FactorMove> moves;
  moves.push_back(request.move_1);
  if (request.move_2.has_value()) {
    moves.push_back(*request.move_2);
  }
  return moves;
}

AttributionRequest component_request(const AttributionRequest &request,
                                     const factors::FactorMove &move) {
  AttributionRequest component = request;
  component.method = "matrix_scalar_fd";
  component.move_1 = move;
  component.move_2.reset();
  component.moves.clear();
  return component;
}

AttributionRequest component_request(const AttributionRequest &request,
                                     const factors::FactorMove &move_1,
                                     const factors::FactorMove &move_2) {
  AttributionRequest component = request;
  component.method = "matrix_cross_fd";
  component.move_1 = move_1;
  component.move_2 = move_2;
  component.moves.clear();
  return component;
}

} // namespace

AttributionResult
MatrixHessianCalculator::compute(const scenarios::ScenarioLattice &lattice,
                                 const AttributionRequest &request) const {
  const auto components = compute_components(lattice, request);

  AttributionResult total;
  total.run_id = request.run_id;
  total.security_id = request.security_id;
  total.component = request.component;
  total.method = request.method.empty() ? "matrix_fd" : request.method;
  total.label = total.component + ":matrix";

  numerics::CompensatedSum pnl_sum;
  for (const auto &component : components) {
    pnl_sum.add(component.pnl_price);
  }
  total.pnl_price = pnl_sum.value();

  return total;
}

std::vector<AttributionResult>
MatrixHessianCalculator::compute_components(const scenarios::ScenarioLattice &lattice,
                                            const AttributionRequest &request) const {
  const auto moves = requested_moves(request);
  if (moves.size() < 2) {
    throw std::invalid_argument(
        "MatrixHessianCalculator expects at least two factor moves");
  }

  std::vector<AttributionResult> components;
  const ScalarSecondOrderCalculator scalar_calculator;
  const CrossSecondOrderCalculator cross_calculator;

  for (const auto &move : moves) {
    components.push_back(
        scalar_calculator.compute(lattice, component_request(request, move)));
  }

  for (std::size_t i = 0; i < moves.size(); ++i) {
    for (std::size_t j = i + 1; j < moves.size(); ++j) {
      components.push_back(cross_calculator.compute(
          lattice, component_request(request, moves[i], moves[j])));
    }
  }

  return components;
}

} // namespace mbs::attribution
