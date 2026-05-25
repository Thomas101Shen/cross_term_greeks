#include "decomposition/SVD/SVDInput.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace decomposition::SVD {

namespace {

DenseMatrix transpose(const DenseMatrix &matrix) {
  DenseMatrix result = DenseMatrix::zeros(matrix.cols(), matrix.rows());
  for (std::size_t row = 0; row < matrix.rows(); ++row) {
    for (std::size_t col = 0; col < matrix.cols(); ++col) {
      result(col, row) = matrix(row, col);
    }
  }
  return result;
}

DenseMatrix multiply(const DenseMatrix &lhs, const DenseMatrix &rhs) {
  if (lhs.cols() != rhs.rows()) {
    throw std::invalid_argument("multiply: incompatible matrix shapes");
  }

  DenseMatrix result = DenseMatrix::zeros(lhs.rows(), rhs.cols());
  for (std::size_t row = 0; row < lhs.rows(); ++row) {
    for (std::size_t col = 0; col < rhs.cols(); ++col) {
      double value = 0.0;
      for (std::size_t inner = 0; inner < lhs.cols(); ++inner) {
        value += lhs(row, inner) * rhs(inner, col);
      }
      result(row, col) = value;
    }
  }
  return result;
}

struct EigenResult {
  std::vector<double> values;
  DenseMatrix vectors;
};

EigenResult jacobi_eigen_symmetric(const DenseMatrix &input, double tolerance,
                                   std::size_t max_sweeps) {
  if (input.rows() != input.cols()) {
    throw std::invalid_argument("jacobi_eigen_symmetric: matrix must be square");
  }

  const std::size_t size = input.rows();
  DenseMatrix matrix = input;
  DenseMatrix vectors = DenseMatrix::identity(size);

  for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
    std::size_t pivot_row = 0;
    std::size_t pivot_col = 1;
    double max_off_diagonal = 0.0;

    for (std::size_t row = 0; row < size; ++row) {
      for (std::size_t col = row + 1; col < size; ++col) {
        const double candidate = std::abs(matrix(row, col));
        if (candidate > max_off_diagonal) {
          max_off_diagonal = candidate;
          pivot_row = row;
          pivot_col = col;
        }
      }
    }

    if (max_off_diagonal <= tolerance || size == 1) {
      break;
    }

    const double app = matrix(pivot_row, pivot_row);
    const double aqq = matrix(pivot_col, pivot_col);
    const double apq = matrix(pivot_row, pivot_col);
    const double angle = 0.5 * std::atan2(2.0 * apq, aqq - app);
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);

    for (std::size_t index = 0; index < size; ++index) {
      if (index != pivot_row && index != pivot_col) {
        const double aip = matrix(index, pivot_row);
        const double aiq = matrix(index, pivot_col);
        matrix(index, pivot_row) = cosine * aip - sine * aiq;
        matrix(pivot_row, index) = matrix(index, pivot_row);
        matrix(index, pivot_col) = sine * aip + cosine * aiq;
        matrix(pivot_col, index) = matrix(index, pivot_col);
      }
    }

    matrix(pivot_row, pivot_row) =
        cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
    matrix(pivot_col, pivot_col) =
        sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
    matrix(pivot_row, pivot_col) = 0.0;
    matrix(pivot_col, pivot_row) = 0.0;

    for (std::size_t row = 0; row < size; ++row) {
      const double vip = vectors(row, pivot_row);
      const double viq = vectors(row, pivot_col);
      vectors(row, pivot_row) = cosine * vip - sine * viq;
      vectors(row, pivot_col) = sine * vip + cosine * viq;
    }
  }

  std::vector<double> values(size);
  for (std::size_t index = 0; index < size; ++index) {
    values[index] = matrix(index, index);
  }

  return EigenResult{.values = std::move(values), .vectors = std::move(vectors)};
}

std::vector<std::size_t> descending_order(const std::vector<double> &values) {
  std::vector<std::size_t> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&values](std::size_t lhs, std::size_t rhs) {
    return values[lhs] > values[rhs];
  });
  return order;
}

} // namespace

DenseMatrix DenseMatrix::zeros(std::size_t rows, std::size_t cols) {
  return DenseMatrix(rows, cols, std::vector<double>(rows * cols, 0.0));
}

DenseMatrix DenseMatrix::identity(std::size_t size) {
  DenseMatrix result = zeros(size, size);
  for (std::size_t index = 0; index < size; ++index) {
    result(index, index) = 1.0;
  }
  return result;
}

std::vector<double> DenseMatrix::column(std::size_t col) const {
  if (col >= cols_) {
    throw std::out_of_range("DenseMatrix::column: column out of range");
  }

  std::vector<double> result(rows_);
  for (std::size_t row = 0; row < rows_; ++row) {
    result[row] = (*this)(row, col);
  }
  return result;
}

SVDInput::SVDInput(DenseMatrix input_matrix, std::vector<std::string> input_row_labels,
                   std::vector<std::string> input_column_labels)
    : matrix(std::move(input_matrix)), row_labels(std::move(input_row_labels)),
      column_labels(std::move(input_column_labels)) {
  if (!row_labels.empty() && row_labels.size() != matrix.rows()) {
    throw std::invalid_argument("SVDInput: row label count does not match rows");
  }
  if (!column_labels.empty() && column_labels.size() != matrix.cols()) {
    throw std::invalid_argument("SVDInput: column label count does not match columns");
  }
}

SVDDecomposer::SVDDecomposer(double tolerance, std::size_t max_sweeps)
    : tolerance_(tolerance), max_sweeps_(max_sweeps) {
  if (tolerance_ <= 0.0) {
    throw std::invalid_argument("SVDDecomposer: tolerance must be positive");
  }
  if (max_sweeps_ == 0) {
    throw std::invalid_argument("SVDDecomposer: max_sweeps must be positive");
  }
}

SVDResult SVDDecomposer::decompose(const SVDInput &input) const {
  const DenseMatrix normal = multiply(transpose(input.matrix), input.matrix);
  auto eigen = jacobi_eigen_symmetric(normal, tolerance_, max_sweeps_);
  const auto order = descending_order(eigen.values);

  const std::size_t component_count = input.matrix.cols();
  std::vector<double> singular_values(component_count);
  DenseMatrix right_vectors = DenseMatrix::zeros(input.matrix.cols(), component_count);

  for (std::size_t component = 0; component < component_count; ++component) {
    const std::size_t source = order[component];
    singular_values[component] = std::sqrt(std::max(eigen.values[source], 0.0));
    for (std::size_t row = 0; row < input.matrix.cols(); ++row) {
      right_vectors(row, component) = eigen.vectors(row, source);
    }
  }

  DenseMatrix left_vectors = DenseMatrix::zeros(input.matrix.rows(), component_count);
  for (std::size_t component = 0; component < component_count; ++component) {
    const double singular_value = singular_values[component];
    if (singular_value <= tolerance_) {
      continue;
    }

    for (std::size_t row = 0; row < input.matrix.rows(); ++row) {
      double projection = 0.0;
      for (std::size_t col = 0; col < input.matrix.cols(); ++col) {
        projection += input.matrix(row, col) * right_vectors(col, component);
      }
      left_vectors(row, component) = projection / singular_value;
    }
  }

  return SVDResult{
      .left_singular_vectors = std::move(left_vectors),
      .singular_values = std::move(singular_values),
      .right_singular_vectors = std::move(right_vectors),
      .row_labels = input.row_labels,
      .column_labels = input.column_labels,
  };
}

} // namespace decomposition::SVD
