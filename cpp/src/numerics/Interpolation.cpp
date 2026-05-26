#include "mbs/numerics/Interpolation.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace mbs::numerics {

namespace {

void validate_grid(const std::vector<double> &x, const std::vector<double> &y,
                   InterpolationMethod method) {
  if (x.size() != y.size()) {
    throw std::invalid_argument("Interpolator: x and y sizes must match");
  }
  if (x.size() < 2) {
    throw std::invalid_argument("Interpolator: at least two points are required");
  }
  for (std::size_t i = 1; i < x.size(); ++i) {
    if (!(x[i] > x[i - 1])) {
      throw std::invalid_argument("Interpolator: x grid must be strictly increasing");
    }
  }
  if (method == InterpolationMethod::LogLinear ||
      method == InterpolationMethod::FlatForward) {
    for (const double value : y) {
      if (!(value > 0.0)) {
        throw std::invalid_argument(
            "Interpolator: log interpolation requires positive y");
      }
    }
  }
}

std::vector<double> monotone_slopes(const std::vector<double> &x,
                                    const std::vector<double> &y) {
  const std::size_t n = x.size();
  std::vector<double> delta(n - 1);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    delta[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
  }

  std::vector<double> slopes(n);
  slopes.front() = delta.front();
  slopes.back() = delta.back();
  for (std::size_t i = 1; i + 1 < n; ++i) {
    if (delta[i - 1] * delta[i] <= 0.0) {
      slopes[i] = 0.0;
      continue;
    }
    const double h_prev = x[i] - x[i - 1];
    const double h_next = x[i + 1] - x[i];
    const double w_1 = 2.0 * h_next + h_prev;
    const double w_2 = h_next + 2.0 * h_prev;
    slopes[i] = (w_1 + w_2) / (w_1 / delta[i - 1] + w_2 / delta[i]);
  }
  return slopes;
}

} // namespace

Interpolator::Interpolator(std::vector<double> x, std::vector<double> y,
                           InterpolationMethod method, bool allow_extrapolation)
    : x_(std::move(x)), y_(std::move(y)), method_(method),
      allow_extrapolation_(allow_extrapolation) {
  validate_grid(x_, y_, method_);
  if (method_ == InterpolationMethod::MonotoneCubic) {
    slopes_ = monotone_slopes(x_, y_);
  }
}

double Interpolator::value(double x_value) const {
  if (!allow_extrapolation_ && (x_value < x_.front() || x_value > x_.back())) {
    throw std::out_of_range("Interpolator: x is outside interpolation range");
  }

  std::size_t i = 0;
  if (x_value <= x_.front()) {
    i = 0;
  } else if (x_value >= x_.back()) {
    i = x_.size() - 2;
  } else {
    const auto upper = std::upper_bound(x_.begin(), x_.end(), x_value);
    const std::size_t right =
        static_cast<std::size_t>(std::distance(x_.begin(), upper));
    i = std::min(right - 1, x_.size() - 2);
  }

  const double h = x_[i + 1] - x_[i];
  const double t = (x_value - x_[i]) / h;

  if (method_ == InterpolationMethod::LogLinear ||
      method_ == InterpolationMethod::FlatForward) {
    return std::exp(std::log(y_[i]) + t * (std::log(y_[i + 1]) - std::log(y_[i])));
  }

  if (method_ == InterpolationMethod::MonotoneCubic) {
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    const double h10 = t3 - 2.0 * t2 + t;
    const double h01 = -2.0 * t3 + 3.0 * t2;
    const double h11 = t3 - t2;
    return h00 * y_[i] + h10 * h * slopes_[i] + h01 * y_[i + 1] +
           h11 * h * slopes_[i + 1];
  }

  return y_[i] + t * (y_[i + 1] - y_[i]);
}

} // namespace mbs::numerics
