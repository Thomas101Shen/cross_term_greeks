#include "decomposition/SVD/PCA.hpp"
#include "decomposition/SVD/SVDInput.hpp"

#include "mbs/factors/FactorKey.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace {

constexpr double kTolerance = 1.0e-10;

bool nearly_equal(double lhs, double rhs) { return std::abs(lhs - rhs) < kTolerance; }

} // namespace

int main() {
  using decomposition::SVD::DenseMatrix;
  using decomposition::SVD::PCAOptions;
  using decomposition::SVD::PCATransformer;
  using decomposition::SVD::SVDDecomposer;
  using decomposition::SVD::SVDInput;
  using mbs::factors::FactorKey;

  static_assert(std::is_nothrow_move_constructible_v<DenseMatrix>);
  static_assert(std::is_nothrow_move_assignable_v<DenseMatrix>);
  static_assert(std::is_nothrow_move_constructible_v<SVDInput>);
  static_assert(std::is_nothrow_move_assignable_v<SVDInput>);

  const SVDInput diagonal_input(DenseMatrix(2, 2, {2.0, 0.0, 0.0, 1.0}),
                                {"OBS_1", "OBS_2"}, {"X", "Y"});
  const auto svd = SVDDecomposer().decompose(diagonal_input);
  assert(svd.singular_values.size() == 2);
  assert(nearly_equal(svd.singular_values[0], 2.0));
  assert(nearly_equal(svd.singular_values[1], 1.0));
  assert(nearly_equal(std::abs(svd.right_singular_vectors(0, 0)), 1.0));
  assert(nearly_equal(std::abs(svd.right_singular_vectors(1, 1)), 1.0));

  const SVDInput pca_input(DenseMatrix(3, 2, {1.0, 0.0, 0.0, 0.0, -1.0, 0.0}),
                           {"T1", "T2", "T3"}, {"VOL_SHORT", "VOL_LONG"});

  const auto pca = PCATransformer(PCAOptions{
                                      .center_columns = true,
                                      .components = 2,
                                      .factor_type = "VOL_PCA",
                                      .factor_name_prefix = "VOL_PC",
                                  })
                       .fit_transform(pca_input);

  assert(pca.loadings.rows() == 2);
  assert(pca.loadings.cols() == 2);
  assert(nearly_equal(pca.column_means[0], 0.0));
  assert(nearly_equal(pca.column_means[1], 0.0));
  assert(nearly_equal(pca.singular_values[0], std::sqrt(2.0)));
  assert(nearly_equal(std::abs(pca.loadings(0, 0)), 1.0));
  assert(nearly_equal(std::abs(pca.loadings(1, 1)), 1.0));
  assert(nearly_equal(pca.explained_variance_ratio[0], 1.0));

  const auto projected = pca.project({2.0, 0.0});
  assert(projected.size() == 2);
  assert(nearly_equal(std::abs(projected[0]), 2.0));
  assert(nearly_equal(projected[1], 0.0));

  const auto basis = pca.factor_basis("vol_pca_basis",
                                      {
                                          FactorKey{.type = "VOL", .name = "VOL_SHORT"},
                                          FactorKey{.type = "VOL", .name = "VOL_LONG"},
                                      });
  assert(basis.name == "vol_pca_basis");
  assert(basis.input_factors.size() == 2);
  assert(basis.factors.size() == 2);
  assert(basis.factors[0].type == "VOL_PCA");
  assert(basis.factors[0].name == "VOL_PC1");
  assert(basis.loadings.size() == 2);
  assert(basis.loadings[0].size() == 2);

  bool rejected_bad_shape = false;
  try {
    (void)DenseMatrix(2, 2, {1.0, 2.0, 3.0});
  } catch (const std::invalid_argument &) {
    rejected_bad_shape = true;
  }
  assert(rejected_bad_shape);

  bool rejected_bad_projection = false;
  try {
    (void)pca.project({1.0});
  } catch (const std::invalid_argument &) {
    rejected_bad_projection = true;
  }
  assert(rejected_bad_projection);

  return 0;
}
