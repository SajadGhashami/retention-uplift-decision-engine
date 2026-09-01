# Customer Retention Uplift & Experimentation Engine

A portfolio project that moves beyond **"Who is likely to return?"** and answers a more useful business question:

> **Which customers should receive a retention offer because the offer is likely to cause an incremental return?**

The project uses a synthetic randomized retention experiment, heterogeneous treatment-effect modeling, uplift evaluation, and profit-based policy optimization.

## Why this project matters

A standard churn or return-propensity model can rank customers by risk, but that does **not** tell us who will change behavior because of an intervention.

Examples:

- A loyal customer may return with or without an offer → discounting them wastes money.
- A disengaged customer may not return even after an offer → contacting them also wastes money.
- A persuadable customer may return *because* of the offer → this is where incremental value exists.

This project estimates that difference and converts it into a decision rule.

## What the project demonstrates

- **Experimentation / A/B testing** with randomized treatment and control groups
- **Causal ML / uplift modeling** using a T-learner
- **Predictive modeling** as a benchmark
- **Heterogeneous treatment effects**
- **Uplift-by-decile analysis**
- **Policy optimization** using expected incremental contribution margin
- **Train / validation / test separation**
- **Business KPI evaluation**, not only model metrics
- **Reproducible synthetic data generation**
- **Unit tests and GitHub Actions**
- A small **Streamlit decision dashboard**

## Business scenario

A consumer business wants to run a retention campaign. Sending an incentive costs money, so targeting everyone is not optimal.

Each customer is randomly assigned to:

- `treatment = 1`: receives an offer
- `treatment = 0`: receives no offer

The outcome is:

- `returned_60d = 1`: customer returns within 60 days
- `returned_60d = 0`: customer does not return

The synthetic dataset includes behavioral features such as recency, purchase frequency, average order value, digital engagement, discount affinity, support interactions, acquisition channel, and customer tenure.

## Approach

```text
Synthetic customer behavior
          ↓
Randomized A/B experiment
          ↓
Train / validation / test split
          ↓
┌───────────────────────────────┐
│ Predictive benchmark          │
│ P(return | customer features) │
└───────────────────────────────┘
          ↓
┌───────────────────────────────┐
│ T-learner causal model        │
│ P(Y|T=1,X) - P(Y|T=0,X)      │
└───────────────────────────────┘
          ↓
Uplift ranking + decile diagnostics
          ↓
Validation-set policy selection
          ↓
Holdout evaluation of incremental profit
```

## Repository structure

```text
retention-uplift-decision-engine/
├── .github/workflows/tests.yml
├── notebooks/
│   └── 01_uplift_walkthrough.ipynb
├── outputs/
│   └── generated after pipeline execution
├── src/
│   ├── data.py
│   ├── evaluation.py
│   ├── modeling.py
│   └── policy.py
├── tests/
│   ├── test_data.py
│   └── test_policy.py
├── app.py
├── run_pipeline.py
└── requirements.txt
```

## Run locally

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full experiment:

```bash
python run_pipeline.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest -q
```

## Main metrics

The pipeline reports:

- overall treatment and control conversion rates
- experimental average treatment effect
- benchmark predictive ROC-AUC
- top-decile observed uplift
- uplift by customer decile
- selected campaign targeting fraction
- estimated incremental conversions
- estimated incremental contribution margin

The exact values are reproducible with the default random seed.


## Reproducible example result

With the default seed, the synthetic experiment produces the following holdout result:

| Metric | Result |
|---|---:|
| Overall experimental lift | **9.6 pp** |
| Top predicted-uplift decile | **21.1 pp** |
| Predictive benchmark ROC-AUC | **0.665** |
| Validation-selected target share | **20%** |
| Holdout customers targeted | **2,000 / 10,000** |
| Estimated incremental conversions | **426** |
| Estimated incremental contribution profit | **$4,963** |

These numbers are generated from **synthetic data** and are included to make the workflow reproducible, not to represent a real business result.

### Policy optimization

![Validation policy curve](outputs/validation_policy_curve.png)

### Uplift diagnostics

![Observed uplift by decile](outputs/uplift_by_decile.png)

## Modeling choice

The uplift model is intentionally implemented as a transparent **T-learner**:

1. Fit one response model on treatment customers.
2. Fit another response model on control customers.
3. Predict both potential outcomes for every customer.
4. Estimate individual uplift as:

```text
uplift(x) = P(return | treatment, x) - P(return | control, x)
```

For a portfolio project, this makes the causal logic easy to inspect. In production, this could be extended with causal forests, X-learners, DR-learners, meta-learners with propensity correction, or specialized uplift libraries.

## Decision policy

Model quality alone is not the final objective.

For candidate targeting fractions, the project estimates the **incremental contribution margin** of treating the selected customers. The targeting percentage is selected on the validation set and then evaluated once on the holdout test set.

This reduces the risk of choosing a campaign policy simply because it looks good on the same data used to evaluate it.

## Important distinction: propensity vs uplift

A propensity model estimates:

```text
P(customer returns)
```

An uplift model estimates:

```text
P(customer returns if treated)
-
P(customer returns if not treated)
```

Those are different questions.

That distinction is the central idea of this repository.

## Possible extensions

- Doubly robust / DR-learner estimation
- Causal forests
- CUPED variance reduction
- Multiple treatment levels
- Long-term customer value instead of 60-day return
- Treatment fatigue / contact constraints
- Budget-constrained optimization
- Bayesian experiment analysis
- Sequential experimentation
- Production model monitoring

## Data and confidentiality

**All data in this repository are synthetic.**

The project is an independent portfolio implementation of common retention, experimentation, and decision-science problems. It does not contain employer, client, or proprietary data, code, model parameters, or business rules.

## Suggested GitHub description

> Causal ML and experimentation project that converts randomized retention-test data into individualized uplift estimates and a profit-optimized targeting policy.

## Suggested portfolio bullet

> Built an end-to-end retention experimentation and causal-ML pipeline using Python and scikit-learn, estimating heterogeneous treatment effects and optimizing campaign targeting based on holdout incremental contribution margin rather than propensity alone.
