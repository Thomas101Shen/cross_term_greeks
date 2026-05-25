#pragma once

#include <optional>
#include <string>

namespace mbs::scenarios {

struct ScenarioPrice {
  std::string run_id;
  std::string security_id;
  std::string asof_date;

  std::string scenario_id;

  std::string factor_type_1; // "VOL", "RATE", "OAS", "CC", "BASE"
  std::string factor_name_1; // "VOL_SHORT", "VOL_MID", etc.
  double shock_1 = 0.0;

  std::optional<std::string> factor_type_2;
  std::optional<std::string> factor_name_2;
  std::optional<double> shock_2;

  double price = 0.0;
};

} // namespace mbs::scenarios
