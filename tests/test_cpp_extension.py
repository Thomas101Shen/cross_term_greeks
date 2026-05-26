import unittest

import numpy as np

from cross_term_greeks import fit_pca


class CppExtensionTest(unittest.TestCase):
    def test_cpp_pca_backend_loads(self):
        pca = fit_pca(
            [[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]],
            n_components=2,
            solver="covariance",
            backend="cpp",
        )

        self.assertEqual(pca.loadings.shape, (2, 2))
        self.assertLess(np.linalg.norm(pca.loadings.T @ pca.loadings - np.eye(2)), 1e-12)

    def test_cpp_attribution_calculators_load(self):
        from cross_term_greeks import _cpp

        factor = _cpp.FactorKey("VOL", "VOL_SHORT")
        move = _cpp.FactorMove()
        move.key = factor
        move.bump = 1.0
        move.realized_move = 2.0

        prices = []
        base = _cpp.ScenarioPrice()
        base.run_id = "RUN"
        base.security_id = "SEC"
        base.factor_type_1 = "BASE"
        base.price = 100.0
        prices.append(base)

        up = _cpp.ScenarioPrice()
        up.run_id = "RUN"
        up.security_id = "SEC"
        up.factor_type_1 = "VOL"
        up.factor_name_1 = "VOL_SHORT"
        up.shock_1 = 1.0
        up.price = 103.0
        prices.append(up)

        down = _cpp.ScenarioPrice()
        down.run_id = "RUN"
        down.security_id = "SEC"
        down.factor_type_1 = "VOL"
        down.factor_name_1 = "VOL_SHORT"
        down.shock_1 = -1.0
        down.price = 99.0
        prices.append(down)

        request = _cpp.AttributionRequest()
        request.run_id = "RUN"
        request.security_id = "SEC"
        request.component = "VOLGA"
        request.move_1 = move

        result = _cpp.ScalarSecondOrderCalculator().compute(
            _cpp.ScenarioLattice(prices), request
        )

        self.assertAlmostEqual(result.greek, 2.0)
        self.assertAlmostEqual(result.pnl_price, 4.0)

    def test_cpp_rates_calculators_load(self):
        from cross_term_greeks import _cpp

        curve = _cpp.DiscountCurve.from_zero_rates([1.0, 2.0], [0.04, 0.05])
        self.assertGreater(curve.discount(1.5), 0.0)

        instruments = [
            _cpp.RateInstrument("deposit", 0.5, 0.04),
            _cpp.RateInstrument("zero_coupon_bond", 1.0, 0.95),
        ]
        calibrated = _cpp.calibrate_bootstrap_curve(instruments)
        self.assertGreaterEqual(calibrated.rmse, 0.0)


if __name__ == "__main__":
    unittest.main()
