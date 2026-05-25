#pragma once

#include "mbs/factors/FactorKey.hpp"

#include <optional>
#include <string>

namespace mbs::attribution {

struct AttributionResult {
  std::string run_id;
  std::string security_id;

  std::string component;
  std::string method;
  std::string label;

  std::optional<factors::FactorKey> factor_1;
  std::optional<factors::FactorKey> factor_2;

  double greek = 0.0;
  double bump_1 = 0.0;
  double bump_2 = 0.0;
  double realized_move_1 = 0.0;
  double realized_move_2 = 0.0;
  double pnl_price = 0.0;
};

} // namespace mbs::attribution
