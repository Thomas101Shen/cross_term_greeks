from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Iterable, Mapping, Sequence

import numpy as np

from .pca import PCAResult, fit_pca


@dataclass(frozen=True)
class FactorMove:
    name: str
    factor_type: str
    bump: float
    realized_move: float


@dataclass(frozen=True)
class ScenarioPrices:
    base: float
    up: float
    down: float


@dataclass(frozen=True)
class CrossScenarioPrices:
    up_up: float
    up_down: float
    down_up: float
    down_down: float


def scalar_second_order(
    prices: ScenarioPrices,
    move: FactorMove,
    *,
    component: str = "SCALAR_SECOND_ORDER",
) -> dict[str, object]:
    if move.bump == 0.0:
        raise ValueError("scalar bump cannot be zero")
    greek = (prices.up - 2.0 * prices.base + prices.down) / (move.bump * move.bump)
    pnl = 0.5 * greek * move.realized_move * move.realized_move
    return {
        "component": component,
        "method": "scalar_fd",
        "label": f"{component}:{move.name}",
        "factor_1": {"type": move.factor_type, "name": move.name},
        "greek": greek,
        "bump_1": move.bump,
        "realized_move_1": move.realized_move,
        "pnl_price": pnl,
    }


def cross_second_order(
    prices: CrossScenarioPrices,
    move_1: FactorMove,
    move_2: FactorMove,
    *,
    component: str = "CROSS_SECOND_ORDER",
) -> dict[str, object]:
    if move_1.bump == 0.0 or move_2.bump == 0.0:
        raise ValueError("cross bumps cannot be zero")
    greek = (
        prices.up_up - prices.up_down - prices.down_up + prices.down_down
    ) / (4.0 * move_1.bump * move_2.bump)
    pnl = greek * move_1.realized_move * move_2.realized_move
    return {
        "component": component,
        "method": "cross_fd",
        "label": f"{component}:{move_1.name} x {move_2.name}",
        "factor_1": {"type": move_1.factor_type, "name": move_1.name},
        "factor_2": {"type": move_2.factor_type, "name": move_2.name},
        "greek": greek,
        "bump_1": move_1.bump,
        "bump_2": move_2.bump,
        "realized_move_1": move_1.realized_move,
        "realized_move_2": move_2.realized_move,
        "pnl_price": pnl,
    }


def _component_moves(
    pca: PCAResult,
    projected_move: np.ndarray,
    *,
    bump: float | Sequence[float],
) -> list[FactorMove]:
    if np.isscalar(bump):
        bumps = [float(bump)] * len(pca.component_labels)
    else:
        bumps = [float(value) for value in bump]
    if len(bumps) != len(pca.component_labels):
        raise ValueError("bump count must match PCA component count")

    return [
        FactorMove(
            name=label,
            factor_type=pca.component_type,
            bump=bumps[index],
            realized_move=float(projected_move[index]),
        )
        for index, label in enumerate(pca.component_labels)
    ]


def _cross_key(name_1: str, name_2: str) -> tuple[str, str]:
    return (name_1, name_2)


def _get_cross_prices(
    cross_prices: Mapping[tuple[str, str], CrossScenarioPrices],
    name_1: str,
    name_2: str,
) -> CrossScenarioPrices:
    direct = _cross_key(name_1, name_2)
    reversed_key = _cross_key(name_2, name_1)
    if direct in cross_prices:
        return cross_prices[direct]
    if reversed_key in cross_prices:
        reversed_prices = cross_prices[reversed_key]
        return CrossScenarioPrices(
            up_up=reversed_prices.up_up,
            up_down=reversed_prices.down_up,
            down_up=reversed_prices.up_down,
            down_down=reversed_prices.down_down,
        )
    raise KeyError(f"missing cross prices for {name_1} x {name_2}")


def rate_vol_pca_interaction(
    *,
    rate_history: Iterable[Iterable[float]],
    vol_history: Iterable[Iterable[float]],
    rate_move: Sequence[float],
    vol_move: Sequence[float],
    cross_prices: Mapping[tuple[str, str], CrossScenarioPrices],
    n_rate_components: int = 3,
    n_vol_components: int = 3,
    rate_bump: float | Sequence[float] = 1.0,
    vol_bump: float | Sequence[float] = 1.0,
    rate_labels: Sequence[str] | None = None,
    vol_labels: Sequence[str] | None = None,
) -> dict[str, object]:
    rate_pca = fit_pca(
        rate_history,
        n_components=n_rate_components,
        input_labels=rate_labels,
        component_type="RATE_PCA",
        component_prefix="RATE_PC",
    )
    vol_pca = fit_pca(
        vol_history,
        n_components=n_vol_components,
        input_labels=vol_labels,
        component_type="VOL_PCA",
        component_prefix="VOL_PC",
    )

    projected_rate_move = rate_pca.project_move(rate_move)
    projected_vol_move = vol_pca.project_move(vol_move)
    rate_moves = _component_moves(rate_pca, projected_rate_move, bump=rate_bump)
    vol_moves = _component_moves(vol_pca, projected_vol_move, bump=vol_bump)

    components = []
    for rate_component in rate_moves:
        for vol_component in vol_moves:
            prices = _get_cross_prices(
                cross_prices,
                rate_component.name,
                vol_component.name,
            )
            components.append(
                cross_second_order(
                    prices,
                    rate_component,
                    vol_component,
                    component="RATE_VOL_PCA_INTERACTION",
                )
            )

    return {
        "method": "rate_vol_pca_cross_fd",
        "rate_pca": rate_pca,
        "vol_pca": vol_pca,
        "projected_rate_move": projected_rate_move,
        "projected_vol_move": projected_vol_move,
        "components": components,
        "interaction_pnl_price": fsum(
            float(component["pnl_price"]) for component in components
        ),
    }
