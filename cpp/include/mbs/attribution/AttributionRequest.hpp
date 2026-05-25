#pragma once

#include "mbs/factors/FactorMove.hpp"

#include <optional>
#include <string>
#include <vector>

namespace mbs::attribution {

struct AttributionRequest {
  std::string run_id;
  std::string security_id;
  std::string component;
  std::string method;

  factors::FactorMove move_1;
  std::optional<factors::FactorMove> move_2;
  std::vector<factors::FactorMove> moves;
};

} // namespace mbs::attribution
