from __future__ import annotations

import numpy as np
import pandas as pd


def experimental_ate(data: pd.DataFrame) -> float:
    treated = data.loc[data["treatment"] == 1, "returned_60d"].mean()
    control = data.loc[data["treatment"] == 0, "returned_60d"].mean()
    return float(treated - control)


def uplift_by_decile(
    data: pd.DataFrame,
    predicted_uplift: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Observed treatment-control difference within uplift-ranked bins."""
    if len(data) != len(predicted_uplift):
        raise ValueError("data and predicted_uplift must have the same length.")

    frame = data[["treatment", "returned_60d"]].copy()
    frame["predicted_uplift"] = predicted_uplift

    # Rank first so qcut remains stable even with tied model predictions.
    ranks = frame["predicted_uplift"].rank(method="first", ascending=False)
    frame["uplift_decile"] = pd.qcut(ranks, q=n_bins, labels=range(1, n_bins + 1))

    rows = []
    for decile, group in frame.groupby("uplift_decile", observed=True):
        treated = group.loc[group["treatment"] == 1, "returned_60d"]
        control = group.loc[group["treatment"] == 0, "returned_60d"]
        rows.append(
            {
                "decile": int(decile),
                "n": int(len(group)),
                "mean_predicted_uplift": float(group["predicted_uplift"].mean()),
                "treated_return_rate": float(treated.mean()),
                "control_return_rate": float(control.mean()),
                "observed_uplift": float(treated.mean() - control.mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("decile").reset_index(drop=True)
