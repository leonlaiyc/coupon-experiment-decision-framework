# A Repeatable Decision Framework for Evaluating Coupon-Driven First-Purchase Experiments

> A decision framework for evaluating whether a short-term lift in new-user
> first purchases is likely to reflect durable customer value, before long-term
> outcomes mature.

*This is a generalized framework illustrated with simulated data. It does not
reference any specific company or proprietary information.*

## The Problem

When an e-commerce platform uses coupons to encourage new users to make their
first purchase, the first question is whether the 30-day first-purchase rate
increases. The harder follow-up is what that increase actually means.

A short-term lift can reflect very different realities:

* **Real incremental demand:** users who would not have purchased within the target window without the coupon
* **Borrowed demand:** purchases that would likely have happened later, pulled forward by the incentive
* **Low-quality discount-driven users:** users who convert once because of the coupon but show weak repeat behavior afterward

In the first few weeks, these outcomes can look similar in the headline metric.
The long-term truth only becomes clearer after LTV and repeat behavior mature.
However, in a fast-moving experimentation environment, the decision to scale,
adjust, or stop often has to be made before that data is fully available.

This framework explores one angle: can early behavioral signals create a more
defensible early read on customer quality and borrowed-demand risk, before
long-term outcomes mature?

The goal is not to replace LTV. The goal is to add an earlier layer of evidence,
then use mature long-term outcomes later to calibrate the next experiment.

## The Framework

The framework breaks the problem into five stages. The structure is designed to
stay consistent across experiments, while the method inside each stage can
improve over time. For example, a simple rule-based read can later be upgraded
to a statistical model or machine learning model without changing the overall
decision flow.

### Stage 1: Measurement Design

Before analyzing results, design the experiment so it can answer a stronger
question: not only whether the first-purchase rate increased, but what kind of
value that increase may represent.

This includes defining the primary metric, guardrail metrics, holdout logic,
observation windows, and post-period tracking needed to assess incremental
demand, customer quality, and borrowed-demand risk.

### Stage 2: Early Signal Layer

While long-term outcomes are still immature, collect early behavioral signals
that may indicate customer quality and demand authenticity.

Examples include repeat visits, browsing depth, cart behavior, category intent,
and dependency on the discount itself.

### Stage 3: Early Read Model

Turn early signals into an initial read on customer quality and borrowed-demand
risk.

This stage is intentionally modular. Version 1 can use transparent rules. Later
versions can use statistical models or machine learning while keeping the rest
of the framework unchanged.

### Stage 4: Decision Mapping

Translate the early read into an actionable recommendation: scale, adjust, or
stop.

The output should make the reasoning explicit, including what supports the
decision, what remains uncertain, and what should be monitored next.

The decision rule, in short:

| Situation | Decision |
|---|---|
| Data quality / readiness fails | Do not decide yet |
| No primary lift | Stop or redesign offer |
| Lift + strong quality + low borrowed risk + acceptable dependency | Scale |
| Lift + weak quality + high borrowed risk + high/elevated dependency | Adjust, not broad scale |
| Lift + inconclusive signals | Continue measurement |
| Lift + mixed evidence | Scale selectively / Adjust |

Extreme cases have clear rules; mixed cases are intentionally routed to selective
rollout, adjustment, or further measurement rather than being over-automated. The
framework provides guardrails, not a lookup table that fully automates every
business decision.

### Stage 5: Backtest and Calibration Loop

Once LTV and repeat behavior mature, compare the early read with actual long-term
outcomes.

This closes the loop. Signals that predicted durable value can be weighted more
heavily in future experiments. Signals that failed can be adjusted, replaced, or
removed. Over time, the framework becomes a learning system rather than a
one-off analysis.

## Framework Flow

The structure stays fixed across experiments; the method inside each stage can
improve over time. Stage 4 is where the flow branches into a decision, and
Stage 5 feeds what it learns back into the next cycle.

```
Stage 1  Measurement Design
   |  primary metric, guardrails, holdout, observation windows, post-period tracking
   v
Stage 2  Early Signal Layer
   |  observable behaviour -> transparent derived signals (0-1 percentile scale)
   v
Stage 3  Early Read Model
   |  signals -> high / medium / low reads (quality, borrowed risk, discount dependency)
   v
Stage 4  Decision Tree
   |  Gate 0  data quality ----fail----> do not decide; fix instrumentation
   |  Gate 1  primary lift ----none----> Stop / redesign offer
   |  Gate 2  discount dependency (early risk proxy)
   |  Gate 3  quality mix
   |  Gate 4  borrowed-risk mix
   |  Gate 5  combine --> Scale | Scale selectively / Adjust | Adjust | Continue | Stop
   v
Stage 5  Backtest, Calibration & Monitoring
   |  5A backtest reads vs mature long-term outcomes
   |  5B next-cycle operating update (stage-by-stage changes for the next cycle)
   |  5C post-launch monitoring design (mode the framework would enter if scaled)
   v
Next experiment  (improved signals, gates, and targeting)
```

## Upgradability

Stage 3 (the Early Read model) is intentionally modular. Version 1 is a
transparent, rule-based read: every score is a simple, inspectable combination of
observable signals. Because the rest of the framework depends only on the *read
columns* Stage 3 emits (`quality_read`, `borrowed_risk_read`,
`discount_dependency_read`), that rule-based v1 can later be replaced by a
statistical model or a machine-learning model without changing Stages 1, 2, 4,
or 5. The framework flow stays the same; only the method inside the stage
improves.

## How to Run

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
py src\simulate.py
jupyter notebook notebooks\coupon_experiment_walkthrough.ipynb
```

`src\simulate.py` regenerates the simulated dataset (seeded and deterministic),
and the notebook then walks through the five stages end to end.

## Scope and Limitations

This is a portfolio project that demonstrates a repeatable decision framework, so
its boundaries are stated explicitly by design:

- This is **simulated** data, not production data. The results are illustrative.
- The project demonstrates a repeatable **decision framework**, not a validated
  production model.
- `realized_ltv_180d` is defined as `net_revenue_180d`, a simplified
  realized-value proxy.
- The current simulated dataset does not include post-purchase engagement
  signals.
- Post-launch monitoring requires rolling cohort data with fields such as
  `period` and `strategy_version`; the current single snapshot does not have them.
- Stage 3 is a transparent rule-based v1 and is intentionally replaceable.

Before real deployment, this would require real data, rolling cohorts, and
production metrics for validation.
