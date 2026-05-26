from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable, Sequence

import numpy as np


def _as_2d_array(values: Iterable[Iterable[float]], name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must have at least 2 rows and 1 column")
    return matrix


def _labels(labels: Sequence[str] | None, count: int, prefix: str) -> list[str]:
    if labels is None:
        return [f"{prefix}{index + 1}" for index in range(count)]
    if len(labels) != count:
        raise ValueError(f"expected {count} labels, got {len(labels)}")
    return list(labels)


def _require_demeaned_columns(matrix: np.ndarray) -> None:
    if np.max(np.abs(matrix.mean(axis=0))) > 1.0e-10:
        raise ValueError(
            "covariance solver requires demeaned columns; "
            "set center=True or pass pre-demeaned observations"
        )


@dataclass(frozen=True)
class PCAResult:
    loadings: np.ndarray
    scores: np.ndarray
    singular_values: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    column_means: np.ndarray
    input_labels: list[str]
    component_labels: list[str]
    component_type: str

    def project_move(self, raw_move: Sequence[float]) -> np.ndarray:
        move = np.asarray(raw_move, dtype=float)
        if move.ndim != 1 or move.shape[0] != self.loadings.shape[0]:
            raise ValueError(
                "raw_move must be a 1D vector with one value per input bucket"
            )
        return move @ self.loadings

    def transform_observations(self, observations: Iterable[Iterable[float]]) -> np.ndarray:
        matrix = np.asarray(observations, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.loadings.shape[0]:
            raise ValueError(
                "observations must be a 2D matrix with the fitted input width"
            )
        return (matrix - self.column_means) @ self.loadings

    def factor_keys(self) -> list[dict[str, str]]:
        return [
            {"type": self.component_type, "name": label}
            for label in self.component_labels
        ]


def fit_pca(
    observations: Iterable[Iterable[float]],
    *,
    n_components: int | None = None,
    input_labels: Sequence[str] | None = None,
    component_type: str = "PCA",
    component_prefix: str = "PC",
    center: bool = True,
    solver: str = "svd",
    backend: str | None = None,
) -> PCAResult:
    matrix = _as_2d_array(observations, "observations")
    normalized_solver = solver.lower().replace("-", "_")
    if normalized_solver not in {"svd", "covariance", "cov"}:
        raise ValueError("solver must be 'svd' or 'covariance'")

    max_components = min(matrix.shape) if normalized_solver == "svd" else matrix.shape[1]
    count = max_components if n_components is None else n_components
    if count < 1 or count > max_components:
        raise ValueError(f"n_components must be between 1 and {max_components}")

    normalized_backend = (
        backend or os.environ.get("CROSS_TERM_GREEKS_BACKEND", "numpy")
    ).lower()
    if normalized_backend not in {"numpy", "cpp"}:
        raise ValueError("backend must be 'numpy' or 'cpp'")
    if normalized_backend == "cpp":
        return _fit_pca_cpp(
            matrix,
            n_components=count,
            input_labels=input_labels,
            component_type=component_type,
            component_prefix=component_prefix,
            center=center,
            solver=normalized_solver,
        )

    means = matrix.mean(axis=0) if center else np.zeros(matrix.shape[1])
    centered = matrix - means

    if normalized_solver == "svd":
        u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        loadings = vt[:count].T
        kept_singular_values = singular_values[:count]
        scores = u[:, :count] * kept_singular_values

        all_variance = singular_values * singular_values / (matrix.shape[0] - 1)
        explained_variance = kept_singular_values * kept_singular_values / (
            matrix.shape[0] - 1
        )
    else:
        _require_demeaned_columns(centered)
        covariance = centered.T @ centered / (matrix.shape[0] - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = eigenvectors[:, order]

        loadings = eigenvectors[:, :count]
        scores = centered @ loadings
        explained_variance = eigenvalues[:count]
        kept_singular_values = np.sqrt(explained_variance * (matrix.shape[0] - 1))
        all_variance = eigenvalues

    total_variance = float(all_variance.sum())
    explained_variance_ratio = (
        np.zeros_like(explained_variance)
        if total_variance == 0.0
        else explained_variance / total_variance
    )

    return PCAResult(
        loadings=loadings,
        scores=scores,
        singular_values=kept_singular_values,
        explained_variance=explained_variance,
        explained_variance_ratio=explained_variance_ratio,
        column_means=means,
        input_labels=_labels(input_labels, matrix.shape[1], "X"),
        component_labels=[f"{component_prefix}{index + 1}" for index in range(count)],
        component_type=component_type,
    )


def _fit_pca_cpp(
    matrix: np.ndarray,
    *,
    n_components: int,
    input_labels: Sequence[str] | None,
    component_type: str,
    component_prefix: str,
    center: bool,
    solver: str,
) -> PCAResult:
    try:
        from . import _cpp
    except ImportError as exc:
        raise RuntimeError(
            "C++ backend requested but cross_term_greeks._cpp is not installed; "
            "build and install a native wheel for this platform"
        ) from exc

    result = _cpp.fit_pca(
        matrix.tolist(),
        n_components=n_components,
        input_labels=_labels(input_labels, matrix.shape[1], "X"),
        component_type=component_type,
        component_prefix=component_prefix,
        center=center,
        solver=solver,
    )
    return PCAResult(
        loadings=np.asarray(result["loadings"], dtype=float),
        scores=np.asarray(result["scores"], dtype=float),
        singular_values=np.asarray(result["singular_values"], dtype=float),
        explained_variance=np.asarray(result["explained_variance"], dtype=float),
        explained_variance_ratio=np.asarray(
            result["explained_variance_ratio"], dtype=float
        ),
        column_means=np.asarray(result["column_means"], dtype=float),
        input_labels=list(result["input_labels"]),
        component_labels=list(result["component_labels"]),
        component_type=str(result["component_type"]),
    )
