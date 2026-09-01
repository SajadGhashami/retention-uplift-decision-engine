from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Retention Uplift Decision Engine",
    layout="wide",
)

st.title("Customer Retention Uplift & Experimentation Engine")
st.caption(
    "From randomized experiment → heterogeneous treatment effects → "
    "profit-based targeting policy"
)

metrics_path = Path("outputs/metrics.json")
curve_path = Path("outputs/validation_policy_curve.csv")
decile_path = Path("outputs/test_uplift_by_decile.csv")

if not metrics_path.exists():
    st.warning("Run `python run_pipeline.py` first to generate the results.")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
curve = pd.read_csv(curve_path)
deciles = pd.read_csv(decile_path)

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Experimental ATE",
    f"{100 * metrics['test_experimental_ate']:.1f} pp",
)
c2.metric(
    "Top-decile uplift",
    f"{100 * metrics['top_decile_observed_uplift']:.1f} pp",
)
c3.metric(
    "Selected target share",
    f"{100 * metrics['selected_target_fraction']:.0f}%",
)
c4.metric(
    "Holdout incremental profit",
    f"${metrics['holdout_incremental_profit']:,.0f}",
)

st.subheader("Why uplift instead of propensity?")
st.write(
    "Propensity predicts who is likely to return. Uplift estimates who is "
    "more likely to return *because* of the intervention. The latter is the "
    "quantity needed for efficient targeting."
)

st.subheader("Validation policy curve")
chart_curve = curve[["target_fraction", "incremental_profit"]].copy()
chart_curve["target_fraction"] *= 100
chart_curve = chart_curve.set_index("target_fraction")
st.line_chart(chart_curve)

st.subheader("Holdout uplift by decile")
chart_deciles = deciles[["decile", "observed_uplift"]].set_index("decile")
st.bar_chart(chart_deciles)

st.subheader("Holdout policy economics")
st.dataframe(
    pd.DataFrame(
        {
            "Metric": [
                "Targeted customers",
                "Estimated incremental conversions",
                "Incremental margin",
                "Offer cost",
                "Incremental profit",
            ],
            "Value": [
                f"{metrics['holdout_targeted_customers']:,}",
                f"{metrics['holdout_incremental_conversions']:.1f}",
                f"${metrics['holdout_incremental_margin']:,.0f}",
                f"${metrics['holdout_offer_cost']:,.0f}",
                f"${metrics['holdout_incremental_profit']:,.0f}",
            ],
        }
    ),
    hide_index=True,
    use_container_width=True,
)

st.info(
    "All data are synthetic. The holdout results demonstrate the workflow, "
    "not a claim about any real company's campaign performance."
)
