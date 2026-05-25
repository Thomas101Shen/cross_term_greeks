#pragma once

#include <string>

namespace mbs::factors {

struct FactorKey {
  std::string type;
  std::string name;

  friend bool operator==(const FactorKey &, const FactorKey &) = default;
};

} // namespace mbs::factors
