#pragma once

#include "mbs/attribution/AttributionRequest.hpp"
#include "mbs/attribution/AttributionResult.hpp"
#include "mbs/scenarios/ScenarioLattice.hpp"

namespace mbs::attribution {

class IAttributionCalculator {
public:
  virtual ~IAttributionCalculator() = default;

  virtual AttributionResult compute(const scenarios::ScenarioLattice &lattice,
                                    const AttributionRequest &request) const = 0;
};

} // namespace mbs::attribution
