#include "mbs/scenarios/ScenarioLattice.hpp"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace mbs::scenarios {

namespace {

constexpr double kShockTolerance = 1.0e-12;

bool same_shock(double lhs, double rhs) {
  return std::abs(lhs - rhs) <= kShockTolerance;
}

bool same_factor(const std::string &type, const std::string &name,
                 const factors::FactorKey &factor) {
  return type == factor.type && name == factor.name;
}

std::string describe_missing(const std::string &run_id, const std::string &security_id,
                             const std::string &scenario) {
  return "ScenarioLattice: missing " + scenario + " price for run_id=" + run_id +
         ", security_id=" + security_id;
}

} // namespace

ScenarioLattice::ScenarioLattice(std::vector<ScenarioPrice> prices)
    : prices_(std::move(prices)) {}

void ScenarioLattice::add(ScenarioPrice price) { prices_.push_back(std::move(price)); }

double ScenarioLattice::base_price(const std::string &run_id,
                                   const std::string &security_id) const {
  for (const auto &price : prices_) {
    if (price.run_id == run_id && price.security_id == security_id &&
        price.factor_type_1 == "BASE" && !price.factor_type_2.has_value()) {
      return price.price;
    }
  }

  throw std::out_of_range(describe_missing(run_id, security_id, "base"));
}

double ScenarioLattice::shocked_price(const std::string &run_id,
                                      const std::string &security_id,
                                      const factors::FactorKey &factor,
                                      double shock) const {
  for (const auto &price : prices_) {
    if (price.run_id == run_id && price.security_id == security_id &&
        same_factor(price.factor_type_1, price.factor_name_1, factor) &&
        same_shock(price.shock_1, shock) && !price.factor_type_2.has_value()) {
      return price.price;
    }
  }

  throw std::out_of_range(describe_missing(run_id, security_id, "shocked"));
}

double ScenarioLattice::cross_price(const std::string &run_id,
                                    const std::string &security_id,
                                    const factors::FactorKey &factor_1, double shock_1,
                                    const factors::FactorKey &factor_2,
                                    double shock_2) const {
  for (const auto &price : prices_) {
    if (price.run_id != run_id || price.security_id != security_id) {
      continue;
    }

    if (!price.factor_type_2.has_value() || !price.factor_name_2.has_value() ||
        !price.shock_2.has_value()) {
      continue;
    }

    const bool direct =
        same_factor(price.factor_type_1, price.factor_name_1, factor_1) &&
        same_shock(price.shock_1, shock_1) &&
        same_factor(*price.factor_type_2, *price.factor_name_2, factor_2) &&
        same_shock(*price.shock_2, shock_2);

    const bool reversed =
        same_factor(price.factor_type_1, price.factor_name_1, factor_2) &&
        same_shock(price.shock_1, shock_2) &&
        same_factor(*price.factor_type_2, *price.factor_name_2, factor_1) &&
        same_shock(*price.shock_2, shock_1);

    if (direct || reversed) {
      return price.price;
    }
  }

  throw std::out_of_range(describe_missing(run_id, security_id, "cross"));
}

} // namespace mbs::scenarios
