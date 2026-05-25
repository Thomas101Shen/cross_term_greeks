#pragma once

#include "mbs/factors/FactorKey.hpp"

#include <string>

namespace mbs::factors {

struct FactorMove {
  FactorKey key;
  double bump = 0.0;
  double realized_move = 0.0;
};

} // namespace mbs::factors
