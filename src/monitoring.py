"""Stage 5C - Post-launch Monitoring Design (template, not results).

This module defines *how* the framework would monitor a coupon strategy after it
is scaled. It does not contain monitoring results. The current project dataset is
a single experiment snapshot (one cross-section of users): it supports the Stage
5A backtest workflow and the Stage 5B next-cycle update, but it has no time axis,
so it cannot support empirical post-launch decay monitoring.

Monitoring is only meaningful on *rolling cohort data over time*: the same
strategy measured across successive periods, each row tagged with the period and
the strategy version that produced it. Without those longitudinal keys there is
nothing to track a trend against, so every function here first checks the schema
and reports the dataset as "not monitoring-ready" rather than computing a trend
that would not mean anything. No rolling data is fabricated anywhere in this
module.

Expected rolling table (one row per period x strategy_version x group cohort cell)
----------------------------------------------------------------------------------
period                    : ordered period label (e.g. week or month index)
strategy_version          : which shipped strategy produced this cohort
group                     : 'treatment' or 'holdout' (a holdout is kept after scale)
first_purchase_rate       : cohort 30-day first-purchase rate for that period
quality_read              : cohort high-quality read share for that period (0-1)
borrowed_risk_read        : cohort high borrowed-risk read share for that period
discount_dependency_read  : cohort high discount-dependency read share for that period
realized_ltv              : cohort realized LTV for that period
repeat_purchase_rate      : cohort repeat-purchase rate for that period

The read columns reuse the Stage 3 reads (quality / borrowed risk / discount
dependency); monitoring simply tracks them across periods instead of reading them
once. Same framework, extended from a one-time experiment readout to ongoing
post-launch monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# --- Expected rolling schema ---------------------------------------------------
# The two longitudinal keys a single experiment snapshot fundamentally lacks. A
# snapshot is one cross-section: no period and no strategy_version, so there is no
# way to track it over time.
ROLLING_KEYS = ["period", "strategy_version"]

EXPECTED_ROLLING_SCHEMA = [
    "period",
    "strategy_version",
    "group",
    "first_purchase_rate",
    "quality_read",
    "borrowed_risk_read",
    "discount_dependency_read",
    "realized_ltv",
    "repeat_purchase_rate",
]

NOT_MONITORING_READY = (
    "This dataset is a single experiment snapshot and is not monitoring-ready."
)

# Group labels used after scale: a holdout is kept so lift stays measurable.
_TREATMENT = "treatment"
_HOLDOUT = "holdout"


# --- Schema check --------------------------------------------------------------
@dataclass
class MonitoringSchemaCheck:
    """Result of ``validate_monitoring_schema``."""

    monitoring_ready: bool
    missing_fields: list[str]
    missing_rolling_keys: list[str]
    message: str


def validate_monitoring_schema(df: pd.DataFrame) -> MonitoringSchemaCheck:
    """Report whether ``df`` carries the rolling fields needed for monitoring.

    Expected input
    --------------
    A rolling cohort table with the columns in ``EXPECTED_ROLLING_SCHEMA`` - one
    row per (period, strategy_version, group) cell, measured repeatedly over time.

    A single experiment snapshot has no ``period`` and no ``strategy_version``, so
    it is reported as *not monitoring-ready*. The framework treats this as an
    honest data-boundary statement, not an error: monitoring needs rolling cohort
    data over time, which a one-shot snapshot does not provide.
    """
    present = set(df.columns)
    missing_fields = [c for c in EXPECTED_ROLLING_SCHEMA if c not in present]
    missing_rolling_keys = [c for c in ROLLING_KEYS if c not in present]

    if not missing_fields:
        return MonitoringSchemaCheck(
            True, [], [], "monitoring-ready: rolling schema present."
        )

    if missing_rolling_keys:
        headline = (
            "not monitoring-ready: missing rolling fields "
            f"({', '.join(missing_rolling_keys)})"
        )
    else:
        headline = f"not monitoring-ready: missing fields ({', '.join(missing_fields)})"

    return MonitoringSchemaCheck(
        False, missing_fields, missing_rolling_keys, f"{headline}. {NOT_MONITORING_READY}"
    )


def _require_rolling(df: pd.DataFrame) -> str | None:
    """Return the not-ready message if ``df`` lacks the rolling schema, else None."""
    check = validate_monitoring_schema(df)
    return None if check.monitoring_ready else check.message


# --- Monitoring templates (only run on rolling data; never on a snapshot) ------
# Each template first requires the rolling schema. On a single snapshot they all
# return the not-monitoring-ready message instead of computing a trend - the
# framework does not force a time-series read out of cross-sectional data.
def compute_rolling_lift(df: pd.DataFrame) -> "pd.DataFrame | str":
    """A. Effect decay - treatment-vs-holdout first-purchase lift per period.

    Expected input: rolling schema (see ``EXPECTED_ROLLING_SCHEMA``). Returns, per
    period, the treatment first-purchase rate, the holdout rate, and the lift
    between them, so the trend tracks "is the coupon still incremental?".
    """
    not_ready = _require_rolling(df)
    if not_ready:
        return not_ready
    wide = df.pivot_table(index="period", columns="group", values="first_purchase_rate")
    wide["lift"] = wide.get(_TREATMENT) - wide.get(_HOLDOUT)
    return wide


def compute_holdout_gap(df: pd.DataFrame) -> "pd.DataFrame | str":
    """A. Effect decay - treatment-vs-holdout durable-value gap per period.

    Expected input: rolling schema. Returns the per-period gap (treatment minus
    holdout) on realized LTV and repeat-purchase rate, tracking whether the
    incremental value over holdout is holding up or narrowing over time.
    """
    not_ready = _require_rolling(df)
    if not_ready:
        return not_ready
    gaps: dict[str, pd.Series] = {}
    for metric in ("realized_ltv", "repeat_purchase_rate"):
        wide = df.pivot_table(index="period", columns="group", values=metric)
        gaps[f"{metric}_gap"] = wide.get(_TREATMENT) - wide.get(_HOLDOUT)
    return pd.DataFrame(gaps)


def _treatment_trend(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Per-period value for the treatment cohort, split by strategy_version."""
    treat = df[df["group"] == _TREATMENT]
    return treat.pivot_table(index="period", columns="strategy_version", values=value_col)


def compute_quality_mix_over_time(df: pd.DataFrame) -> "pd.DataFrame | str":
    """B. Quality decay - treatment high-quality read share per period.

    Expected input: rolling schema. Tracks the treatment cohort's high-quality
    read share across periods - is acquired customer quality holding or eroding?
    """
    not_ready = _require_rolling(df)
    if not_ready:
        return not_ready
    return _treatment_trend(df, "quality_read")


def compute_discount_dependency_over_time(df: pd.DataFrame) -> "pd.DataFrame | str":
    """C. Dependency / fatigue - treatment discount-dependency read per period.

    Expected input: rolling schema. Tracks the treatment cohort's high
    discount-dependency read share across periods - are users becoming
    discount-trained over time?
    """
    not_ready = _require_rolling(df)
    if not_ready:
        return not_ready
    return _treatment_trend(df, "discount_dependency_read")


def compute_borrowed_risk_over_time(df: pd.DataFrame) -> "pd.DataFrame | str":
    """B/C. Borrowed-risk read share for the treatment cohort per period.

    Expected input: rolling schema. Tracks the treatment cohort's high
    borrowed-risk read share across periods - is pulled-forward demand building up
    as rollout widens?
    """
    not_ready = _require_rolling(df)
    if not_ready:
        return not_ready
    return _treatment_trend(df, "borrowed_risk_read")


# --- Combined monitoring state -------------------------------------------------
@dataclass
class MonitoringState:
    """Outcome of ``evaluate_monitoring_state``."""

    monitoring_ready: bool
    state: str
    action: str
    message: str
    reads: dict[str, object] = field(default_factory=dict)


def _period_series(frame: pd.DataFrame) -> pd.Series:
    """Collapse a (period x strategy_version) frame to one value per period."""
    return frame.mean(axis=1).dropna()


def _is_rising(frame: pd.DataFrame) -> bool:
    s = _period_series(frame)
    return bool(len(s) >= 2 and s.iloc[-1] > s.iloc[0])


def _is_holding(frame: pd.DataFrame) -> bool:
    s = _period_series(frame)
    return bool(len(s) < 2 or s.iloc[-1] >= s.iloc[0])


def evaluate_monitoring_state(df: pd.DataFrame) -> MonitoringState:
    """Walk the monitoring decision tree on rolling data.

    Expected input: rolling schema. The tree reuses the same reads as Stage 4
    (incremental lift, quality, discount dependency, borrowed risk) but reads them
    as *trends across periods* rather than once:

        1. Incremental lift still positive vs holdout?  no  -> pause / reduce rollout
        2. User quality stable?                         no  -> narrow targeting
        3. Discount dependency increasing?              yes -> adjust incentive
        4. Borrowed-demand risk increasing?             yes -> extend window, hold out

    On a single snapshot there is no trend to walk, so this reports
    not monitoring-ready instead of returning a verdict. It never fabricates a
    decay read from cross-sectional data.
    """
    check = validate_monitoring_schema(df)
    if not check.monitoring_ready:
        return MonitoringState(
            False,
            "not monitoring-ready",
            "Collect rolling cohort data over time (period, strategy_version) before monitoring.",
            check.message,
        )

    # --- Rolling-data branch (template): read each trend latest-vs-first -------
    lift = compute_rolling_lift(df)
    quality = compute_quality_mix_over_time(df)
    dependency = compute_discount_dependency_over_time(df)
    risk = compute_borrowed_risk_over_time(df)
    assert isinstance(lift, pd.DataFrame)  # schema is ready here, so these are frames

    latest_lift = float(lift["lift"].dropna().iloc[-1])
    quality_holding = _is_holding(quality)
    dependency_rising = _is_rising(dependency)
    risk_rising = _is_rising(risk)

    if latest_lift <= 0:
        verdict = "lift no longer incremental vs holdout"
        action = "Pause or reduce rollout; re-estimate targeting / incentive."
    elif not quality_holding:
        verdict = "acquired quality eroding"
        action = "Narrow targeting; reduce exposure to low-quality segments."
    elif dependency_rising:
        verdict = "discount dependency increasing"
        action = "Adjust incentive (lower discount / higher threshold / personalized offer)."
    elif risk_rising:
        verdict = "borrowed-demand risk increasing"
        action = "Extend measurement window; maintain holdout; avoid broad scale."
    else:
        verdict = "stable"
        action = "Continue running, with rolling monitoring."

    return MonitoringState(
        True,
        verdict,
        action,
        f"monitoring state: {verdict}",
        reads={
            "latest_lift": latest_lift,
            "quality_holding": quality_holding,
            "dependency_rising": dependency_rising,
            "risk_rising": risk_rising,
        },
    )


def _main() -> None:
    from pathlib import Path

    data_path = (
        Path(__file__).resolve().parent.parent / "data" / "simulated_coupon_experiment.csv"
    )
    snapshot = pd.read_csv(data_path)

    print("=" * 72)
    print("STAGE 5C - POST-LAUNCH MONITORING DESIGN (schema check on current data)")
    print("=" * 72)
    check = validate_monitoring_schema(snapshot)
    print(check.message)
    print(f"missing rolling keys: {check.missing_rolling_keys}")
    print(f"monitoring state    : {evaluate_monitoring_state(snapshot).state}")
    print("-" * 72)
    print("The current dataset is a single experiment snapshot. Monitoring needs")
    print("rolling cohort data over time (period, strategy_version). No rolling data")
    print("is fabricated here; the framework reports the boundary instead.")


if __name__ == "__main__":
    _main()
