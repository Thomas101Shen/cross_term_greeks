#pragma once

#include "mbs/attribution/IAttributionCalculator.hpp"

#include <vector>

namespace mbs::attribution {

class MatrixHessianCalculator final : public IAttributionCalculator {
public:
  AttributionResult compute(const scenarios::ScenarioLattice &lattice,
                            const AttributionRequest &request) const override;

  std::vector<AttributionResult>
  compute_components(const scenarios::ScenarioLattice &lattice,
                     const AttributionRequest &request) const;
};

} // namespace mbs::attribution
