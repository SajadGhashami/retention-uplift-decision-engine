import pandas as pd

from src.data import FEATURE_COLUMNS, make_synthetic_retention_experiment


def test_synthetic_data_schema_and_randomization():
    df = make_synthetic_retention_experiment(n=5_000, seed=7)

    required = set(
        FEATURE_COLUMNS
        + [
            "customer_id",
            "treatment",
            "returned_60d",
            "expected_margin",
            "true_uplift",
        ]
    )

    assert required.issubset(df.columns)
    assert len(df) == 5_000
    assert set(df["treatment"].unique()) <= {0, 1}
    assert set(df["returned_60d"].unique()) <= {0, 1}

    treatment_rate = df["treatment"].mean()
    assert 0.45 < treatment_rate < 0.55


def test_true_average_treatment_effect_is_positive():
    df = make_synthetic_retention_experiment(n=5_000, seed=11)
    assert df["true_uplift"].mean() > 0
