#pragma once

#include "mbs/factors/FactorKey.hpp"
#include "mbs/scenarios/ScenarioPrice.hpp"

#include <string>
#include <vector>

namespace mbs::scenarios {

class ScenarioLattice {
public:
  ScenarioLattice() = default;
  explicit ScenarioLattice(std::vector<ScenarioPrice> prices);

  void add(ScenarioPrice price);

  double base_price(const std::string &run_id, const std::string &security_id) const;

  double shocked_price(const std::string &run_id, const std::string &security_id,
                       const factors::FactorKey &factor, double shock) const;

  double cross_price(const std::string &run_id, const std::string &security_id,
                     const factors::FactorKey &factor_1, double shock_1,
                     const factors::FactorKey &factor_2, double shock_2) const;

private:
  std::vector<ScenarioPrice> prices_;
};

} // namespace mbs::scenarios
