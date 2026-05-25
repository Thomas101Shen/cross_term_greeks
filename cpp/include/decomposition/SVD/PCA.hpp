#pragma once

#include "decomposition/SVD/SVDInput.hpp"

#include "mbs/factors/FactorBasis.hpp"
#include "mbs/factors/FactorKey.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace decomposition::SVD {

struct PCAOptions {
  bool center_columns = true;
  std::size_t components = 0;
  std::string factor_type = "PCA";
  std::string factor_name_prefix = "PC";
};

struct PCAResult {
  DenseMatrix loadings;
  DenseMatrix scores;
  std::vector<double> singular_values;
  std::vector<double> explained_variance;
  std::vector<double> explained_variance_ratio;
  std::vector<double> column_means;
  std::vector<std::string> input_labels;
  std::vector<std::string> component_labels;
  std::string component_type;

  [[nodiscard]] std::vector<double> project(const std::vector<double> &raw_move) const;
  [[nodiscard]] mbs::factors::FactorBasis
  factor_basis(const std::string &basis_name,
               const std::vector<mbs::factors::FactorKey> &input_factors) const;
};

class PCATransformer {
public:
  explicit PCATransformer(PCAOptions options = {});

  [[nodiscard]] PCAResult fit_transform(const SVDInput &input) const;

private:
  PCAOptions options_;
};

} // namespace decomposition::SVD
