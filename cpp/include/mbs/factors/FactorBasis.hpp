#pragma once

#include "mbs/factors/FactorKey.hpp"

#include <string>
#include <vector>

namespace mbs::factors {

struct FactorBasis {
  std::string name;
  std::vector<FactorKey> input_factors;
  std::vector<FactorKey> factors;
  std::vector<std::vector<double>> loadings;
};

} // namespace mbs::factors
