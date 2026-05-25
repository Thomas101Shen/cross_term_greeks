#pragma once

#include <stdexcept>

namespace mbs::numerics {

inline double central_second(double p_base, double p_up, double p_down, double p_bump) {
  if (p_bump == 0.0) {
    throw std::invalid_argument("central_second: bump cannot be zero");
  }
  return (p_up - 2.0 * p_base + p_down) / (p_bump * p_bump);
}

inline double central_cross(double p_up_up, double p_up_down, double p_down_up,
                            double p_down_down, double bump_1, double bump_2) {
  if (bump_1 == 0.0 || bump_2 == 0.0) {
    throw std::invalid_argument("central_cross: bumps cannot be zero");
  }
  return (p_up_up - p_up_down - p_down_up + p_down_down) / (4.0 * bump_1 * bump_2);
}

inline double second_order_pnl(double gamma, double realized_move) {
  return 0.5 * gamma * realized_move * realized_move;
}

inline double cross_pnl(double cross_greek, double realized_move_1,
                        double realized_move_2) {
  return cross_greek * realized_move_1 * realized_move_2;
}

} // namespace mbs::numerics
