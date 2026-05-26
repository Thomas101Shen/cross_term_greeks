#pragma once

namespace mbs::numerics {

class CompensatedSum {
public:
  void add(double value) noexcept {
    const long double term = static_cast<long double>(value);
    const long double next = sum_ + term;
    if (abs_ld(sum_) >= abs_ld(term)) {
      compensation_ += (sum_ - next) + term;
    } else {
      compensation_ += (term - next) + sum_;
    }
    sum_ = next;
  }

  [[nodiscard]] double value() const noexcept {
    return static_cast<double>(precise_value());
  }
  [[nodiscard]] long double precise_value() const noexcept {
    return sum_ + compensation_;
  }

private:
  static constexpr long double abs_ld(long double value) noexcept {
    return value < 0.0L ? -value : value;
  }

  long double sum_ = 0.0L;
  long double compensation_ = 0.0L;
};

} // namespace mbs::numerics
