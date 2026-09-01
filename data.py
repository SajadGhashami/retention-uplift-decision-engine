from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "recency_days",
    "orders_90d",
    "avg_order_value",
    "tenure_months",
    "discount_share",
    "email_engagement",
    "app_sessions_30d",
    "support_tickets_90d",
    "channel",
    "region",
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic_retention_experiment(
    n: int = 50_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a randomized synthetic retention experiment.

    Treatment is randomized 50/50. Treatment response is heterogeneous,
    allowing an uplift model to learn which customers are persuadable.
    """
    if n < 1_000:
        raise ValueError("n should be at least 1,000 for a stable experiment.")

    rng = np.random.default_rng(seed)

    recency_days = np.clip(rng.gamma(shape=2.2, scale=26.0, size=n), 1, 180)
    orders_90d = np.clip(rng.poisson(lam=3.2, size=n), 0, 18)
    avg_order_value = np.clip(rng.lognormal(mean=4.15, sigma=0.38, size=n), 15, 250)
    tenure_months = np.clip(rng.gamma(shape=2.5, scale=9.0, size=n), 1, 96)
    discount_share = rng.beta(2.0, 4.0, size=n)
    email_engagement = rng.beta(2.2, 3.5, size=n)
    app_sessions_30d = np.clip(
        rng.poisson(lam=2.0 + 5.0 * email_engagement, size=n), 0, 35
    )
    support_tickets_90d = np.clip(rng.poisson(lam=0.45, size=n), 0, 6)

    channel = rng.choice(
        ["organic", "paid_search", "referral", "partner"],
        size=n,
        p=[0.38, 0.30, 0.20, 0.12],
    )
    region = rng.choice(
        ["north", "south", "east", "west", "central"],
        size=n,
        p=[0.17, 0.20, 0.20, 0.18, 0.25],
    )

    # Randomized experiment assignment.
    treatment = rng.binomial(1, 0.50, size=n)

    # Baseline probability of returning without an incentive.
    baseline_logit = (
        -1.45
        - 0.014 * recency_days
        + 0.17 * orders_90d
        + 0.004 * (avg_order_value - 60)
        + 0.009 * tenure_months
        + 0.85 * email_engagement
        + 0.035 * app_sessions_30d
        - 0.18 * support_tickets_90d
        + 0.13 * (channel == "referral")
        - 0.10 * (channel == "paid_search")
        + 0.09 * (region == "central")
    )

    # Heterogeneous offer effect.
    # Medium-recency, promotion-sensitive, digitally engaged customers
    # are more persuadable; extremely recent customers need less incentive.
    treatment_logit_effect = (
        -0.60
        + 1.45 * discount_share
        + 0.75 * email_engagement
        + 0.50 * ((recency_days >= 25) & (recency_days <= 95))
        - 0.48 * (recency_days < 12)
        - 0.18 * (support_tickets_90d >= 3)
        - 0.20 * (orders_90d >= 8)
    )

    p_control = sigmoid(baseline_logit)
    p_treated = sigmoid(baseline_logit + treatment_logit_effect)
    observed_probability = np.where(treatment == 1, p_treated, p_control)
    returned_60d = rng.binomial(1, observed_probability)

    expected_margin = np.clip(
        0.34 * avg_order_value + 1.8 * orders_90d,
        8,
        120,
    )

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "recency_days": recency_days.round(1),
            "orders_90d": orders_90d,
            "avg_order_value": avg_order_value.round(2),
            "tenure_months": tenure_months.round(1),
            "discount_share": discount_share.round(4),
            "email_engagement": email_engagement.round(4),
            "app_sessions_30d": app_sessions_30d,
            "support_tickets_90d": support_tickets_90d,
            "channel": channel,
            "region": region,
            "treatment": treatment,
            "returned_60d": returned_60d,
            "expected_margin": expected_margin.round(2),
            # Latent values are useful only for validating the synthetic setup.
            # They are NOT used to train the models.
            "true_p_control": p_control,
            "true_p_treated": p_treated,
            "true_uplift": p_treated - p_control,
        }
    )

    return df
