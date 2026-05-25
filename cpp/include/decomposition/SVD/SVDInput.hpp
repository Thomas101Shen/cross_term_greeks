#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace decomposition::SVD {

class DenseMatrix {
public:
  DenseMatrix() = default;

  DenseMatrix(std::size_t rows, std::size_t cols, std::vector<double> values)
      : rows_(rows), cols_(cols), values_(std::move(values)) {
    if (rows_ == 0 || cols_ == 0) {
      throw std::invalid_argument("DenseMatrix: dimensions must be non-zero");
    }
    if (values_.size() != rows_ * cols_) {
      throw std::invalid_argument("DenseMatrix: value count does not match shape");
    }
  }

  DenseMatrix(const DenseMatrix &) = default;
  DenseMatrix(DenseMatrix &&) noexcept = default;
  DenseMatrix &operator=(const DenseMatrix &) = default;
  DenseMatrix &operator=(DenseMatrix &&) noexcept = default;
  ~DenseMatrix() = default;

  static DenseMatrix zeros(std::size_t rows, std::size_t cols);
  static DenseMatrix identity(std::size_t size);

  [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
  [[nodiscard]] std::size_t cols() const noexcept { return cols_; }
  [[nodiscard]] const std::vector<double> &values() const noexcept { return values_; }

  [[nodiscard]] double operator()(std::size_t row, std::size_t col) const {
    return values_.at(index(row, col));
  }

  double &operator()(std::size_t row, std::size_t col) {
    return values_.at(index(row, col));
  }

  [[nodiscard]] std::vector<double> column(std::size_t col) const;

private:
  [[nodiscard]] std::size_t index(std::size_t row, std::size_t col) const {
    return row * cols_ + col;
  }

  std::size_t rows_ = 0;
  std::size_t cols_ = 0;
  std::vector<double> values_;
};

struct SVDInput {
  DenseMatrix matrix;
  std::vector<std::string> row_labels;
  std::vector<std::string> column_labels;

  SVDInput() = default;
  SVDInput(DenseMatrix input_matrix, std::vector<std::string> input_row_labels = {},
           std::vector<std::string> input_column_labels = {});

  SVDInput(const SVDInput &) = default;
  SVDInput(SVDInput &&) noexcept = default;
  SVDInput &operator=(const SVDInput &) = default;
  SVDInput &operator=(SVDInput &&) noexcept = default;
  ~SVDInput() = default;
};

struct SVDResult {
  DenseMatrix left_singular_vectors;
  std::vector<double> singular_values;
  DenseMatrix right_singular_vectors;
  std::vector<std::string> row_labels;
  std::vector<std::string> column_labels;
};

class SVDDecomposer {
public:
  explicit SVDDecomposer(double tolerance = 1.0e-12, std::size_t max_sweeps = 100);

  [[nodiscard]] SVDResult decompose(const SVDInput &input) const;

private:
  double tolerance_;
  std::size_t max_sweeps_;
};

} // namespace decomposition::SVD
