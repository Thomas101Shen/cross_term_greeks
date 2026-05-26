#include "decomposition/SVD/PCA.hpp"

#include "mbs/numerics/CompensatedSum.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace decomposition::SVD {

namespace {

std::vector<double> column_means(const Matrix &matrix) {
  std::vector<double> means(matrix.cols(), 0.0);
  for (std::size_t col = 0; col < matrix.cols(); ++col) {
    mbs::numerics::CompensatedSum sum;
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
      sum.add(matrix(row, col));
    }
    means[col] = sum.value() / static_cast<double>(matrix.rows());
  }
  return means;
}

Matrix centered_matrix(const Matrix &matrix, const std::vector<double> &means) {
  Matrix centered = Matrix::zeros(matrix.rows(), matrix.cols());
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

Matrix covariance_matrix(const Matrix &matrix) {
  if (matrix.rows() < 2) {
    throw std::invalid_argument(
        "covariance_matrix: at least two observations required");
  }

  Matrix covariance = Matrix::zeros(matrix.cols(), matrix.cols());
  const double denominator = static_cast<double>(matrix.rows() - 1);
  for (std::size_t lhs = 0; lhs < matrix.cols(); ++lhs) {
    for (std::size_t rhs = lhs; rhs < matrix.cols(); ++rhs) {
      mbs::numerics::CompensatedSum sum;
      for (std::size_t row = 0; row < matrix.rows(); ++row) {
        sum.add(matrix(row, lhs) * matrix(row, rhs));
      }
      covariance(lhs, rhs) = sum.value() / denominator;
    }
  }
  for (std::size_t lhs = 0; lhs < matrix.cols(); ++lhs) {
    for (std::size_t rhs = lhs + 1; rhs < matrix.cols(); ++rhs) {
      covariance(rhs, lhs) = covariance(lhs, rhs);
    }
  }
  return covariance;
}

void require_demeaned_columns(const Matrix &matrix) {
  constexpr double kDemeanTolerance = 1.0e-10;
  const auto means = column_means(matrix);
  for (const double mean : means) {
    if (std::abs(mean) > kDemeanTolerance) {
      throw std::invalid_argument(
          "PCATransformer: covariance solver requires demeaned columns; "
          "set center_columns=true or pass pre-demeaned observations");
    }
  }
}

Matrix scores_from_loadings(const Matrix &matrix, const Matrix &loadings) {
  Matrix scores = Matrix::zeros(matrix.rows(), loadings.cols());
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t component = 0; component < loadings.cols(); ++component) {
      mbs::numerics::CompensatedSum sum;
      for (std::size_t col = 0; col < matrix.cols(); ++col) {
        sum.add(matrix(row, col) * loadings(col, component));
      }
      scores(row, component) = sum.value();
    }
  }
  return scores;
}

struct PCASolution {
  Matrix loadings;
  Matrix scores;
  std::vector<double> singular_values;
  std::vector<double> explained_variance;
  std::vector<double> explained_variance_ratio;
};

PCASolution solve_with_svd(const Matrix &prepared, std::size_t component_count,
                           const SVDInput &input) {
  const SVDResult svd = SVDDecomposer().decompose(SVDInput{
      prepared,
      input.row_labels,
      input.column_labels,
  });

  Matrix loadings = Matrix::zeros(input.matrix.cols(), component_count);
  Matrix scores = Matrix::zeros(input.matrix.rows(), component_count);
  std::vector<double> singular_values(component_count);
  std::vector<double> explained_variance(component_count);
  std::vector<double> explained_variance_ratio(component_count);

  mbs::numerics::CompensatedSum total_variance_sum;
  for (double singular_value : svd.singular_values) {
    total_variance_sum.add(singular_value * singular_value);
  }
  const double total_variance = total_variance_sum.value();

  for (std::size_t component = 0; component < component_count; ++component) {
    singular_values[component] = svd.singular_values[component];
    explained_variance[component] = singular_values[component] *
                                    singular_values[component] /
                                    static_cast<double>(input.matrix.rows() - 1);
    explained_variance_ratio[component] =
        total_variance == 0.0
            ? 0.0
            : singular_values[component] * singular_values[component] / total_variance;

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

  return PCASolution{.loadings = std::move(loadings),
                     .scores = std::move(scores),
                     .singular_values = std::move(singular_values),
                     .explained_variance = std::move(explained_variance),
                     .explained_variance_ratio = std::move(explained_variance_ratio)};
}

PCASolution solve_with_covariance(const Matrix &prepared, std::size_t component_count) {
  require_demeaned_columns(prepared);
  const Matrix covariance = covariance_matrix(prepared);
  const auto eigen = SymmetricEigenDecomposer().decompose(covariance);

  Matrix loadings = Matrix::zeros(prepared.cols(), component_count);
  std::vector<double> singular_values(component_count);
  std::vector<double> explained_variance(component_count);
  std::vector<double> explained_variance_ratio(component_count);

  mbs::numerics::CompensatedSum total_variance_sum;
  for (const double value : eigen.values) {
    total_variance_sum.add(std::max(value, 0.0));
  }
  const double total_variance = total_variance_sum.value();

  for (std::size_t component = 0; component < component_count; ++component) {
    explained_variance[component] = std::max(eigen.values[component], 0.0);
    singular_values[component] = std::sqrt(explained_variance[component] *
                                           static_cast<double>(prepared.rows() - 1));
    explained_variance_ratio[component] =
        total_variance == 0.0 ? 0.0 : explained_variance[component] / total_variance;

    for (std::size_t row = 0; row < prepared.cols(); ++row) {
      loadings(row, component) = eigen.vectors(row, component);
    }
  }

  Matrix scores = scores_from_loadings(prepared, loadings);
  return PCASolution{.loadings = std::move(loadings),
                     .scores = std::move(scores),
                     .singular_values = std::move(singular_values),
                     .explained_variance = std::move(explained_variance),
                     .explained_variance_ratio = std::move(explained_variance_ratio)};
}

PCASolution solve_pca(const Matrix &prepared, const SVDInput &input,
                      const PCAOptions &options, std::size_t component_count) {
  switch (options.solver) {
  case PCASolver::SVD:
    return solve_with_svd(prepared, component_count, input);
  case PCASolver::Covariance:
    return solve_with_covariance(prepared, component_count);
  }
  throw std::invalid_argument("PCATransformer: unsupported solver");
}

} // namespace

std::vector<double> PCAResult::project(const std::vector<double> &raw_move) const {
  if (raw_move.size() != loadings.rows()) {
    throw std::invalid_argument("PCAResult::project: raw move size mismatch");
  }

  std::vector<double> projected(loadings.cols(), 0.0);
  for (std::size_t component = 0; component < loadings.cols(); ++component) {
    mbs::numerics::CompensatedSum sum;
    for (std::size_t input = 0; input < loadings.rows(); ++input) {
      sum.add(raw_move[input] * loadings(input, component));
    }
    projected[component] = sum.value();
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
  const Matrix prepared = centered_matrix(input.matrix, means);
  const std::size_t requested_components =
      options_.components == 0 ? input.matrix.cols() : options_.components;
  if (requested_components > input.matrix.cols()) {
    throw std::invalid_argument("PCATransformer: component count too large");
  }

  auto solution = solve_pca(prepared, input, options_, requested_components);

  return PCAResult{
      .loadings = std::move(solution.loadings),
      .scores = std::move(solution.scores),
      .singular_values = std::move(solution.singular_values),
      .explained_variance = std::move(solution.explained_variance),
      .explained_variance_ratio = std::move(solution.explained_variance_ratio),
      .column_means = means,
      .input_labels = input.column_labels,
      .component_labels = component_labels(options_, requested_components),
      .component_type = options_.factor_type,
  };
}

} // namespace decomposition::SVD
