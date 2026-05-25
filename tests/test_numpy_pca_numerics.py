import unittest

import numpy as np

from cross_term_greeks import fit_pca


def canonicalize_columns(matrix):
    result = matrix.copy()
    for col in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, col])))
        if result[pivot, col] < 0.0:
            result[:, col] *= -1.0
    return result


class NumpyPCANumericsTest(unittest.TestCase):
    def test_numpy_pca_reconstructs_centered_matrix(self):
        observations = np.array(
            [
                [1.0, 2.0, 3.0],
                [2.0, 1.0, 4.0],
                [3.0, 0.0, 5.0],
                [4.0, -1.0, 6.0],
            ],
            dtype=float,
        )
        pca = fit_pca(observations, n_components=3)

        centered = observations - observations.mean(axis=0)
        reconstructed = pca.scores @ pca.loadings.T

        self.assertLess(np.linalg.norm(centered - reconstructed), 1.0e-12)

    def test_numpy_pca_matches_eigh_covariance_basis_up_to_sign(self):
        observations = np.array(
            [
                [1.0, 0.5, 0.2],
                [0.0, 0.1, 0.0],
                [-1.0, -0.4, -0.1],
                [2.0, 1.0, 0.4],
                [-2.0, -0.9, -0.3],
            ],
            dtype=float,
        )
        pca = fit_pca(observations, n_components=3)

        centered = observations - observations.mean(axis=0)
        covariance = centered.T @ centered / (observations.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        self.assertLess(
            np.linalg.norm(
                canonicalize_columns(pca.loadings)
                - canonicalize_columns(eigenvectors)
            ),
            1.0e-12,
        )
        self.assertLess(
            np.linalg.norm(pca.explained_variance - eigenvalues),
            1.0e-12,
        )

    def test_nearly_collinear_inputs_still_give_orthonormal_loadings(self):
        x = np.linspace(-1.0, 1.0, 12)
        observations = np.column_stack(
            [
                x,
                2.0 * x + 1.0e-10 * np.sin(np.arange(x.size)),
                -0.5 * x + 1.0e-10 * np.cos(np.arange(x.size)),
            ]
        )
        pca = fit_pca(observations, n_components=3)

        gram = pca.loadings.T @ pca.loadings
        self.assertLess(np.linalg.norm(gram - np.eye(3)), 1.0e-12)
        self.assertGreater(pca.explained_variance_ratio[0], 0.999999)


if __name__ == "__main__":
    unittest.main()
