import numpy as np

from src.data import make_synthetic_retention_experiment
from src.policy import (
    build_policy_curve,
    choose_best_fraction,
    estimate_incremental_policy_value,
    target_top_fraction,
)


def test_top_fraction_selects_expected_number():
    scores = np.array([0.9, 0.1, 0.7, 0.2, 0.8])
    policy = target_top_fraction(scores, 0.4)

    assert policy.sum() == 2
    assert policy[0] == 1
    assert policy[4] == 1


def test_policy_value_output():
    df = make_synthetic_retention_experiment(n=5_000, seed=3)
    policy = np.ones(len(df), dtype=int)

    value = estimate_incremental_policy_value(df, policy, offer_cost=4.0)

    assert value["targeted_customers"] == len(df)
    assert "incremental_profit" in value
    assert "incremental_conversions" in value


def test_choose_best_fraction_returns_curve_fraction():
    df = make_synthetic_retention_experiment(n=5_000, seed=5)
    scores = df["true_uplift"].to_numpy()
    curve = build_policy_curve(df, scores, fractions=[0.2, 0.5, 1.0])

    best = choose_best_fraction(curve)
    assert best in set(curve["target_fraction"])
