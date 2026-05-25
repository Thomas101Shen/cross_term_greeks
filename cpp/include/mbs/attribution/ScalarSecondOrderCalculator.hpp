#pragma once

#include "mbs/attribution/IAttributionCalculator.hpp"

namespace mbs::attribution {

class ScalarSecondOrderCalculator final : public IAttributionCalculator {
public:
  AttributionResult compute(const scenarios::ScenarioLattice &lattice,
                            const AttributionRequest &request) const override;
};

} // namespace mbs::attribution
