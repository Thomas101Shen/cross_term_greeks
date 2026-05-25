#include "decomposition/SVD/PCA.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace decomposition::SVD {

namespace {

std::vector<double> column_means(const DenseMatrix &matrix) {
  std::vector<double> means(matrix.cols(), 0.0);
  for (std::size_t col = 0; col < matrix.cols(); ++col) {
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      means[col] += matrix(row, col);
    }
    means[col] /= static_cast<double>(matrix.rows());
  }
  return means;
}

DenseMatrix centered_matrix(const DenseMatrix &matrix,
                            const std::vector<double> &means) {
  DenseMatrix centered = DenseMatrix::zeros(matrix.rows(), matrix.cols());
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
      centered(row, col) = matrix(row, col) - means[col];
    }
  }
  return centered;
}

std::vector<std::string> component_labels(const PCAOptions &options,
                                          std::size_t component_count) {
  std::vector<std::string> labels;
  labels.reserve(component_count);
  for (std::size_t index = 0; index < component_count; ++index) {
    labels.push_back(options.factor_name_prefix + std::to_string(index + 1));
  }
  return labels;
}

} // namespace

std::vector<double> PCAResult::project(const std::vector<double> &raw_move) const {
  if (raw_move.size() != loadings.rows()) {
    throw std::invalid_argument("PCAResult::project: raw move size mismatch");
  }

  std::vector<double> projected(loadings.cols(), 0.0);
  for (std::size_t component = 0; component < loadings.cols(); ++component) {
    for (std::size_t input = 0; input < loadings.rows(); ++input) {
      projected[component] += raw_move[input] * loadings(input, component);
    }
  }
  return projected;
}

mbs::factors::FactorBasis PCAResult::factor_basis(
    const std::string &basis_name,
    const std::vector<mbs::factors::FactorKey> &input_factors) const {
  if (input_factors.size() != loadings.rows()) {
    throw std::invalid_argument("PCAResult::factor_basis: input factor count mismatch");
  }

  std::vector<mbs::factors::FactorKey> factors;
  factors.reserve(component_labels.size());
  for (const auto &label : component_labels) {
    factors.push_back(mbs::factors::FactorKey{.type = component_type, .name = label});
  }

  std::vector<std::vector<double>> loading_rows(loadings.rows());
  for (std::size_t row = 0; row < loadings.rows(); ++row) {
    loading_rows[row].reserve(loadings.cols());
    for (std::size_t col = 0; col < loadings.cols(); ++col) {
      loading_rows[row].push_back(loadings(row, col));
    }
  }

  return mbs::factors::FactorBasis{
      .name = basis_name,
      .input_factors = input_factors,
      .factors = std::move(factors),
      .loadings = std::move(loading_rows),
  };
}

PCATransformer::PCATransformer(PCAOptions options) : options_(std::move(options)) {}

PCAResult PCATransformer::fit_transform(const SVDInput &input) const {
  if (input.matrix.rows() < 2) {
    throw std::invalid_argument("PCATransformer: at least two observations required");
  }

  const auto means = options_.center_columns
                         ? column_means(input.matrix)
                         : std::vector<double>(input.matrix.cols(), 0.0);
  const DenseMatrix prepared = centered_matrix(input.matrix, means);
  const SVDResult svd = SVDDecomposer().decompose(SVDInput{
      prepared,
      input.row_labels,
      input.column_labels,
  });

  const std::size_t requested_components =
      options_.components == 0 ? input.matrix.cols() : options_.components;
  if (requested_components > input.matrix.cols()) {
    throw std::invalid_argument("PCATransformer: component count too large");
  }

  DenseMatrix loadings = DenseMatrix::zeros(input.matrix.cols(), requested_components);
  DenseMatrix scores = DenseMatrix::zeros(input.matrix.rows(), requested_components);
  std::vector<double> singular_values(requested_components);
  std::vector<double> explained_variance(requested_components);
  std::vector<double> explained_variance_ratio(requested_components);

  double total_variance = 0.0;
  for (double singular_value : svd.singular_values) {
    total_variance += singular_value * singular_value;
  }

  for (std::size_t component = 0; component < requested_components; ++component) {
    singular_values[component] = svd.singular_values[component];
    explained_variance[component] = singular_values[component] *
                                    singular_values[component] /
                                    static_cast<double>(input.matrix.rows() - 1);
    explained_variance_ratio[component] =
        total_variance == 0.0
            ? 0.0
            : (singular_values[component] * singular_values[component]) /
                  total_variance;

    for (std::size_t input_col = 0; input_col < input.matrix.cols(); ++input_col) {
      loadings(input_col, component) = svd.right_singular_vectors(input_col, component);
    }
    for (std::size_t observation = 0; observation < input.matrix.rows();
         ++observation) {
      scores(observation, component) =
          svd.left_singular_vectors(observation, component) *
          singular_values[component];
    }
  }

  return PCAResult{
      .loadings = std::move(loadings),
      .scores = std::move(scores),
      .singular_values = std::move(singular_values),
      .explained_variance = std::move(explained_variance),
      .explained_variance_ratio = std::move(explained_variance_ratio),
      .column_means = means,
      .input_labels = input.column_labels,
      .component_labels = component_labels(options_, requested_components),
      .component_type = options_.factor_type,
  };
}

} // namespace decomposition::SVD
