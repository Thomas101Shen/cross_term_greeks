import unittest

from cross_term_greeks import CrossScenarioPrices, fit_pca, rate_vol_pca_interaction


class PythonWorkflowTest(unittest.TestCase):
    def test_fit_pca_project_move(self):
        pca = fit_pca(
            [[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]],
            n_components=2,
            input_labels=["A", "B"],
            component_type="TEST_PCA",
            component_prefix="TEST_PC",
        )

        projected = pca.project_move([2.0, 0.0])
        self.assertEqual(projected.shape, (2,))
        self.assertLess(abs(abs(projected[0]) - 2.0), 1.0e-12)
        self.assertEqual(
            pca.factor_keys()[0],
            {"type": "TEST_PCA", "name": "TEST_PC1"},
        )

    def test_rate_vol_pca_interaction(self):
        cross_prices = {}
        for rate_index in range(1, 3):
            for vol_index in range(1, 3):
                cross_prices[(f"RATE_PC{rate_index}", f"VOL_PC{vol_index}")] = (
                    CrossScenarioPrices(
                        up_up=103.0,
                        up_down=99.0,
                        down_up=98.0,
                        down_down=100.0,
                    )
                )

        result = rate_vol_pca_interaction(
            rate_history=[[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0]],
            vol_history=[[0.0, 2.0], [0.0, 0.0], [0.0, -2.0]],
            rate_move=[3.0, 0.0],
            vol_move=[0.0, -2.0],
            cross_prices=cross_prices,
            n_rate_components=2,
            n_vol_components=2,
            rate_bump=1.0,
            vol_bump=2.0,
        )

        self.assertEqual(len(result["components"]), 4)
        self.assertLess(abs(result["components"][0]["greek"] - 0.75), 1.0e-12)
        self.assertIn("interaction_pnl_price", result)


if __name__ == "__main__":
    unittest.main()
