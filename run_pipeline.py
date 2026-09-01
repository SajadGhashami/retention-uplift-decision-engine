from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import make_synthetic_retention_experiment
from src.evaluation import experimental_ate, uplift_by_decile
from src.modeling import fit_propensity_benchmark, fit_t_learner, predictive_auc
from src.policy import (
    build_policy_curve,
    choose_best_fraction,
    estimate_incremental_policy_value,
    target_top_fraction,
)


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("Generating synthetic randomized retention experiment...")
    data = make_synthetic_retention_experiment(n=50_000, seed=42)

    # 60/20/20 split.
    train, temp = train_test_split(
        data,
        test_size=0.40,
        random_state=42,
        stratify=data[["treatment", "returned_60d"]],
    )
    validation, test = train_test_split(
        temp,
        test_size=0.50,
        random_state=42,
        stratify=temp[["treatment", "returned_60d"]],
    )

    print("Fitting predictive benchmark...")
    propensity_model = fit_propensity_benchmark(train)
    benchmark_auc = predictive_auc(propensity_model, test)

    print("Fitting T-learner uplift model...")
    uplift_model = fit_t_learner(train)

    val_uplift = uplift_model.predict_uplift(validation)
    test_uplift = uplift_model.predict_uplift(test)

    print("Selecting targeting policy on validation data...")
    validation_curve = build_policy_curve(
        validation,
        val_uplift,
        offer_cost=4.0,
    )
    best_fraction = choose_best_fraction(validation_curve)

    final_policy = target_top_fraction(test_uplift, best_fraction)
    test_policy_value = estimate_incremental_policy_value(
        test,
        final_policy,
        offer_cost=4.0,
    )

    deciles = uplift_by_decile(test, test_uplift)
    top_decile_uplift = float(deciles.loc[deciles["decile"] == 1, "observed_uplift"].iloc[0])

    overall_treated_rate = float(
        test.loc[test["treatment"] == 1, "returned_60d"].mean()
    )
    overall_control_rate = float(
        test.loc[test["treatment"] == 0, "returned_60d"].mean()
    )

    metrics = {
        "rows_total": int(len(data)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "test_treated_return_rate": overall_treated_rate,
        "test_control_return_rate": overall_control_rate,
        "test_experimental_ate": experimental_ate(test),
        "benchmark_predictive_auc": benchmark_auc,
        "top_decile_observed_uplift": top_decile_uplift,
        "selected_target_fraction": best_fraction,
        **{f"holdout_{k}": v for k, v in test_policy_value.items()},
    }

    validation_curve.to_csv(OUTPUT_DIR / "validation_policy_curve.csv", index=False)
    deciles.to_csv(OUTPUT_DIR / "test_uplift_by_decile.csv", index=False)
    test.assign(predicted_uplift=test_uplift).to_csv(
        OUTPUT_DIR / "test_scored_sample.csv",
        index=False,
    )

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(
        validation_curve["target_fraction"] * 100,
        validation_curve["incremental_profit"],
        marker="o",
    )
    plt.xlabel("Customers targeted (%)")
    plt.ylabel("Estimated incremental profit")
    plt.title("Validation policy curve")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_policy_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(deciles["decile"].astype(str), deciles["observed_uplift"])
    plt.xlabel("Predicted uplift decile (1 = highest)")
    plt.ylabel("Observed treatment-control return-rate difference")
    plt.title("Observed uplift by predicted-uplift decile")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "uplift_by_decile.png", dpi=150)
    plt.close()

    print("\n=== Holdout results ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    print("\nArtifacts written to ./outputs")


if __name__ == "__main__":
    main()
