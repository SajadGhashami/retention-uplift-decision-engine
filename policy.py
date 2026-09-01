from __future__ import annotations

import numpy as np
import pandas as pd


def target_top_fraction(predicted_uplift: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    n = len(predicted_uplift)
    k = max(1, int(np.ceil(n * fraction)))
    order = np.argsort(-predicted_uplift)
    policy = np.zeros(n, dtype=int)
    policy[order[:k]] = 1
    return policy


def estimate_incremental_policy_value(
    data: pd.DataFrame,
    policy: np.ndarray,
    treatment_probability: float = 0.5,
    offer_cost: float = 4.0,
) -> dict:
    """Estimate incremental conversions and profit via IPW.

    Policy=1 means "send the offer"; policy=0 means "do not send".
    Because treatment was randomized, inverse-probability weighting gives
    an unbiased estimate of the policy's incremental effect versus no offer.
    """
    if len(data) != len(policy):
        raise ValueError("data and policy must have the same length.")
    if not 0 < treatment_probability < 1:
        raise ValueError("treatment_probability must be between 0 and 1.")

    t = data["treatment"].to_numpy()
    y = data["returned_60d"].to_numpy()
    margin = data["expected_margin"].to_numpy()
    p = treatment_probability
    selected = policy.astype(float)

    incremental_conversion_contribution = selected * (
        (t * y / p) - ((1 - t) * y / (1 - p))
    )

    incremental_margin_contribution = selected * (
        (t * y * margin / p) - ((1 - t) * y * margin / (1 - p))
    )

    # Cost is incurred whenever the policy chooses treatment.
    net_profit_contribution = incremental_margin_contribution - selected * offer_cost

    return {
        "target_fraction": float(selected.mean()),
        "targeted_customers": int(selected.sum()),
        "incremental_conversion_rate": float(
            incremental_conversion_contribution.mean()
        ),
        "incremental_conversions": float(
            incremental_conversion_contribution.sum()
        ),
        "incremental_margin": float(incremental_margin_contribution.sum()),
        "offer_cost": float(selected.sum() * offer_cost),
        "incremental_profit": float(net_profit_contribution.sum()),
        "profit_per_customer": float(net_profit_contribution.mean()),
    }


def build_policy_curve(
    data: pd.DataFrame,
    predicted_uplift: np.ndarray,
    fractions: list[float] | None = None,
    offer_cost: float = 4.0,
) -> pd.DataFrame:
    if fractions is None:
        fractions = [i / 10 for i in range(1, 11)]

    rows = []
    for fraction in fractions:
        policy = target_top_fraction(predicted_uplift, fraction)
        rows.append(
            estimate_incremental_policy_value(
                data=data,
                policy=policy,
                offer_cost=offer_cost,
            )
        )
    return pd.DataFrame(rows)


def choose_best_fraction(policy_curve: pd.DataFrame) -> float:
    if policy_curve.empty:
        raise ValueError("policy_curve cannot be empty.")
    best_row = policy_curve.loc[policy_curve["incremental_profit"].idxmax()]
    return float(best_row["target_fraction"])
