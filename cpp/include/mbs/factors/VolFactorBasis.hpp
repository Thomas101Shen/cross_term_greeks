#pragma once

#include "mbs/factors/FactorBasis.hpp"

#include <string>
#include <utility>
#include <vector>

namespace mbs::factors {

inline FactorBasis make_vol_factor_basis(std::string name,
                                         std::vector<FactorKey> raw_vol_buckets,
                                         std::vector<FactorKey> factors,
                                         std::vector<std::vector<double>> loadings) {
  return FactorBasis{
      .name = std::move(name),
      .input_factors = std::move(raw_vol_buckets),
      .factors = std::move(factors),
      .loadings = std::move(loadings),
  };
}

} // namespace mbs::factors
