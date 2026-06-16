"""Stage 4 - Decision Tree.

A *gated decision flow*, not a flat mapping table. The experiment passes through
ordered gates; each gate either stops the flow or hands control to the next, and
the final gate combines the collected reads into a single recommendation.

    Gate 0  Experiment Readiness and Data Quality Check  - computable data checks
    Gate 1  Primary Metric Check                         - did the 30d target move?
    Gate 2  Guardrail and Dependency Check               - discount dependency as
                                                           an early *risk proxy*
    Gate 3  Early Quality Read                            - treatment quality mix
    Gate 4  Borrowed-Demand Risk Read                     - treatment risk mix
    Gate 5  Decision Output                               - combine into a call

The input is the experiment dataframe with the Stage 2 derived signals and the
Stage 3 early reads already attached (the output of ``add_early_read``). Stage 4
uses only observable fields and the Stage 3 reads - it never touches latent
ground truth or long-term outcome fields. Those are opened for the first time in
Stage 5 (backtest).

The gates read the cohort at the *mix* level (the share of high/medium/low reads
among purchasers in each arm). They never attach a label to an individual user:
"treatment purchasers show a weaker early-quality mix" is a statement about the
cohort, not about any one person.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# --- Column groups this stage is allowed to see --------------------------------
# Observable fields that should be fully populated for every user. days_to_first_
# purchase is intentionally excluded here - it is legitimately null for non-
# purchasers and is checked by the timing-validity rule instead.
ALWAYS_POPULATED_OBSERVABLE = [
    "user_id",
    "group",
    "coupon_received",
    "signup_day",
    "exposure_day",
    "made_first_purchase_30d",
    "repeat_visits_14d",
    "browse_sessions_14d",
    "product_views_14d",
    "cart_adds_14d",
    "discount_page_views_14d",
    "coupon_used",
]
STAGE2_SIGNALS = ["engagement_intensity", "purchase_intent", "discount_dependency", "early_timing"]
READ_COLS = ["quality_read", "borrowed_risk_read", "discount_dependency_read"]
PRIMARY_METRIC = "made_first_purchase_30d"

# --- Thresholds (documented and conservative) ----------------------------------
SAMPLE_RATIO_TOLERANCE = 0.02   # control share must sit within 50% +/- 2pp
CHI2_CRIT_DF1 = 3.841           # chi-square 0.05 critical value, df=1 (no scipy needed)
Z_CRIT_95 = 1.96                # standard normal two-sided 0.05 critical value (no scipy)
MAX_MISSINGNESS = 0.05          # >5% missing in a populated field is a flag
HIGH_SHARE_THRESHOLD = 0.50     # a band "dominates" a cohort when >50% read that way
COHORT_DELTA_PP = 10.0          # a treatment-vs-control mix gap this large is meaningful
BANDS = ["high", "medium", "low"]


# --- Result containers ---------------------------------------------------------
@dataclass
class CheckResult:
    """One computable sub-check inside a gate (used by Gate 0)."""

    name: str
    passed: bool
    detail: str


@dataclass
class GateResult:
    """The outcome of a single gate.

    ``passed`` means "the flow continues past this gate". Gates 2/3/4 always
    continue (they only raise flags), so their ``passed`` is always True; the
    flag they carry lives in ``metrics``.
    """

    gate: str
    name: str
    passed: bool
    summary: str
    checks: list[CheckResult] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)


@dataclass
class DecisionTreeResult:
    gates: list[GateResult]
    decision: str
    rationale: str
    stopped_at: str | None  # gate id where the flow stopped early, else None


# --- Cohort helpers (mix-level, never individual) ------------------------------
def _purchasers(df: pd.DataFrame, arm: str | None = None) -> pd.DataFrame:
    out = df[df[PRIMARY_METRIC] == 1]
    if arm is not None:
        out = out[out["group"] == arm]
    return out


def _band_mix(df: pd.DataFrame, read_col: str, arm: str) -> dict[str, float]:
    """Share of each high/medium/low band among purchasers in ``arm``."""
    s = _purchasers(df, arm)[read_col]
    n = int(s.notna().sum())
    if n == 0:
        return {lvl: float("nan") for lvl in BANDS}
    return {lvl: float((s == lvl).sum()) / n for lvl in BANDS}


def _high_share(df: pd.DataFrame, read_col: str, arm: str) -> float:
    return _band_mix(df, read_col, arm)["high"]


# --- Gate 0: Experiment Readiness and Data Quality Check -----------------------
def _check_sample_ratio(df: pd.DataFrame) -> CheckResult:
    n = len(df)
    n_c = int((df["group"] == "control").sum())
    n_t = int((df["group"] == "treatment").sum())
    control_share = n_c / n if n else float("nan")
    within_tol = abs(control_share - 0.5) <= SAMPLE_RATIO_TOLERANCE
    # Simple chi-square goodness-of-fit against an even 50/50 split (df=1).
    exp = n / 2.0
    chi2 = ((n_c - exp) ** 2 + (n_t - exp) ** 2) / exp if exp else float("nan")
    balanced_chi2 = chi2 < CHI2_CRIT_DF1
    return CheckResult(
        "sample_ratio",
        within_tol,
        f"control share {control_share:.3f} (control={n_c:,}, treatment={n_t:,}); "
        f"within +/-{SAMPLE_RATIO_TOLERANCE:.0%} tol={within_tol}; "
        f"chi2={chi2:.2f} (<{CHI2_CRIT_DF1} -> balanced={balanced_chi2})",
    )


def _check_duplicate_users(df: pd.DataFrame) -> CheckResult:
    n_dupes = int(df["user_id"].duplicated().sum())
    return CheckResult("duplicate_user", n_dupes == 0, f"{n_dupes} duplicate user_id rows")


def _check_missingness(df: pd.DataFrame) -> CheckResult:
    rates = {c: float(df[c].isna().mean()) for c in ALWAYS_POPULATED_OBSERVABLE if c in df.columns}
    worst_col = max(rates, key=rates.get) if rates else None
    worst = rates[worst_col] if worst_col else 0.0
    return CheckResult(
        "missingness",
        worst <= MAX_MISSINGNESS,
        f"max missing rate {worst:.3%} in '{worst_col}' (threshold {MAX_MISSINGNESS:.0%})",
    )


def _check_coupon_logic(df: pd.DataFrame) -> CheckResult:
    control_used = int(df.loc[df["group"] == "control", "coupon_used"].sum())
    return CheckResult(
        "coupon_logic",
        control_used == 0,
        f"{control_used} control-group rows have coupon_used=1 (expected 0)",
    )


def _check_timing_validity(df: pd.DataFrame) -> CheckResult:
    non_purchasers = df[df[PRIMARY_METRIC] == 0]
    purchasers = df[df[PRIMARY_METRIC] == 1]
    bad_nonpur = int(non_purchasers["days_to_first_purchase"].notna().sum())
    bad_pur = int(purchasers["days_to_first_purchase"].isna().sum())
    ok = bad_nonpur == 0 and bad_pur == 0
    return CheckResult(
        "timing_validity",
        ok,
        f"non-purchasers with a purchase day: {bad_nonpur}; purchasers missing a day: {bad_pur}",
    )


def _check_metric_availability(df: pd.DataFrame) -> CheckResult:
    required = ["group", PRIMARY_METRIC, *STAGE2_SIGNALS, *READ_COLS]
    missing = [c for c in required if c not in df.columns]
    return CheckResult(
        "metric_availability",
        not missing,
        "all primary, Stage 2 signal, and Stage 3 read columns present"
        if not missing
        else f"missing columns: {missing}",
    )


def gate0_data_quality(df: pd.DataFrame) -> GateResult:
    checks = [
        _check_sample_ratio(df),
        _check_duplicate_users(df),
        _check_missingness(df),
        _check_coupon_logic(df),
        _check_timing_validity(df),
        _check_metric_availability(df),
    ]
    passed = all(c.passed for c in checks)
    n_fail = sum(not c.passed for c in checks)
    summary = (
        "Data quality acceptable; all readiness checks pass."
        if passed
        else f"{n_fail} readiness check(s) failed - do not decide yet."
    )
    return GateResult("Gate 0", "Experiment Readiness and Data Quality Check", passed, summary, checks=checks)


# --- Gate 1: Primary Metric Check ----------------------------------------------
def gate1_primary_metric(df: pd.DataFrame) -> GateResult:
    # Gate 1 answers only one question: did the short-term target move by more
    # than sampling noise? It does NOT speak to whether the lift is durable or
    # valuable - that is what Gates 2-4 and the Stage 5 backtest are for.
    control = df.loc[df["group"] == "control", PRIMARY_METRIC]
    treatment = df.loc[df["group"] == "treatment", PRIMARY_METRIC]
    n_c, n_t = int(control.shape[0]), int(treatment.shape[0])
    control_rate = float(control.mean())
    treatment_rate = float(treatment.mean())
    lift = treatment_rate - control_rate
    lift_pp = lift * 100.0

    # Two-proportion z-test against H0: equal rates, using the POOLED standard
    # error. Implemented by hand (numpy only, no scipy).
    pooled = float((control.sum() + treatment.sum()) / (n_c + n_t))
    pooled_se = float(np.sqrt(pooled * (1.0 - pooled) * (1.0 / n_c + 1.0 / n_t)))
    z = lift / pooled_se if pooled_se > 0 else 0.0

    # Approximate 95% CI for the lift, using the UNPOOLED standard error.
    unpooled_se = float(np.sqrt(
        control_rate * (1.0 - control_rate) / n_c
        + treatment_rate * (1.0 - treatment_rate) / n_t
    ))
    ci_lo_pp = (lift - Z_CRIT_95 * unpooled_se) * 100.0
    ci_hi_pp = (lift + Z_CRIT_95 * unpooled_se) * 100.0

    # A primary lift "exists" only if it is positive AND clears the z>1.96 bar.
    has_lift = lift > 0 and z > Z_CRIT_95
    summary = (
        f"30d first-purchase rate: control {control_rate:.2%} (n={n_c:,}) vs "
        f"treatment {treatment_rate:.2%} (n={n_t:,}); lift {lift_pp:+.2f}pp "
        f"(95% CI [{ci_lo_pp:+.2f}, {ci_hi_pp:+.2f}]pp, z={z:.2f}). "
        + ("Primary lift exists (positive and z>1.96)." if has_lift
           else "No significant primary lift (lift<=0 or z<=1.96).")
    )
    return GateResult(
        "Gate 1",
        "Primary Metric Check",
        has_lift,
        summary,
        metrics={
            "control_rate": control_rate,
            "treatment_rate": treatment_rate,
            "lift_pp": lift_pp,
            "z": z,
            "ci95_pp": (ci_lo_pp, ci_hi_pp),
            "n_control": n_c,
            "n_treatment": n_t,
        },
    )


# --- Gate 2: Guardrail and Dependency Check ------------------------------------
def gate2_guardrail_dependency(df: pd.DataFrame) -> GateResult:
    # Discount dependency is used here as an EARLY RISK PROXY, not a final cost
    # metric. A treatment cohort leaning heavily on the discount is a warning
    # sign; the true cost (discount spend, margin) is only settled in Stage 5.
    treat_high = _high_share(df, "discount_dependency_read", "treatment")
    control_high = _high_share(df, "discount_dependency_read", "control")
    delta_pp = (treat_high - control_high) * 100.0

    if treat_high >= HIGH_SHARE_THRESHOLD:
        level = "high"
    elif delta_pp >= COHORT_DELTA_PP:
        level = "elevated"
    else:
        level = "acceptable"

    summary = (
        f"Discount-dependency risk proxy: {treat_high:.1%} of treatment purchasers read 'high' "
        f"vs {control_high:.1%} of control ({delta_pp:+.1f}pp) -> dependency {level}. "
        f"Early risk proxy only; true cost is assessed in Stage 5."
    )
    return GateResult(
        "Gate 2",
        "Guardrail and Dependency Check",
        True,  # a risk flag, not a stop
        summary,
        metrics={"dependency_level": level, "treat_high_share": treat_high, "delta_pp": delta_pp},
    )


# --- Gate 3: Early Quality Read ------------------------------------------------
def gate3_quality_read(df: pd.DataFrame) -> GateResult:
    treat_high = _high_share(df, "quality_read", "treatment")
    control_high = _high_share(df, "quality_read", "control")
    delta_pp = (treat_high - control_high) * 100.0

    if delta_pp <= -COHORT_DELTA_PP:
        strength = "weak"
    elif delta_pp >= COHORT_DELTA_PP:
        strength = "strong"
    else:
        strength = "mixed"

    summary = (
        f"Treatment purchasers show a {strength} early-quality mix: {treat_high:.1%} read 'high' "
        f"quality vs {control_high:.1%} of control ({delta_pp:+.1f}pp). Cohort mix, not an "
        f"individual label."
    )
    return GateResult(
        "Gate 3",
        "Early Quality Read",
        True,
        summary,
        metrics={"quality_strength": strength, "treat_high_share": treat_high, "delta_pp": delta_pp},
    )


# --- Gate 4: Borrowed-Demand Risk Read -----------------------------------------
def gate4_borrowed_risk_read(df: pd.DataFrame) -> GateResult:
    # borrowed_risk_read is a COMBINED signal (early purchase AND weak engagement
    # AND high discount dependency), not "bought early" on its own.
    treat_high = _high_share(df, "borrowed_risk_read", "treatment")
    control_high = _high_share(df, "borrowed_risk_read", "control")
    delta_pp = (treat_high - control_high) * 100.0

    if delta_pp >= COHORT_DELTA_PP:
        level = "high"
    elif delta_pp <= -COHORT_DELTA_PP:
        level = "low"
    else:
        level = "mixed"

    summary = (
        f"Borrowed-demand risk read is {level}: {treat_high:.1%} of treatment purchasers read 'high' "
        f"risk vs {control_high:.1%} of control ({delta_pp:+.1f}pp). Combined signal, not early-buy "
        f"alone; cohort mix, not an individual label."
    )
    return GateResult(
        "Gate 4",
        "Borrowed-Demand Risk Read",
        True,
        summary,
        metrics={"risk_level": level, "treat_high_share": treat_high, "delta_pp": delta_pp},
    )


# --- Gate 5: Decision Output ---------------------------------------------------
def _decide(has_lift: bool, quality: str, risk: str, dependency: str) -> tuple[str, str]:
    """Combine the collected reads into a recommendation.

    Decision table (only reached with a primary lift; no lift stops at Gate 1):
      lift + quality strong + risk low + dependency acceptable
          -> Scale (broaden rollout, keep a holdout)
      lift + quality weak   + risk high + dependency high/elevated
          -> Adjust (no broad scale; redesign incentive/targeting)
      lift + signals inconclusive (all mixed, dependency acceptable)
          -> Continue measurement (extend window or add signals)
      lift + any other mix
          -> Scale selectively / Adjust (target stronger segments, monitor)
    """
    if not has_lift:
        return "Stop or redesign offer", "No primary lift: the short-term target did not move."

    favorable = quality == "strong" and risk == "low" and dependency == "acceptable"
    unfavorable = quality == "weak" and risk == "high" and dependency in ("high", "elevated")
    inconclusive = quality == "mixed" and risk == "mixed" and dependency == "acceptable"

    if favorable:
        return "Scale", "Broaden the rollout while keeping a holdout to keep measuring."
    if unfavorable:
        return (
            "Adjust - not broad scale",
            "Do not scale broadly yet. Adjust targeting or incentive design, keep a holdout, "
            "and continue tracking long-term value.",
        )
    if inconclusive:
        return (
            "Continue measurement",
            "Signals are inconclusive; extend the observation window or add signals before deciding.",
        )
    return (
        "Scale selectively / Adjust",
        "Signals are mixed; target the stronger segments, keep a holdout, and monitor.",
    )


def run_decision_tree(df: pd.DataFrame) -> DecisionTreeResult:
    """Run the full gated flow on a dataframe that already carries the Stage 3 reads.

    The flow short-circuits at Gate 0 (data quality) or Gate 1 (no lift); otherwise
    it collects the Gate 2-4 reads and resolves a decision at Gate 5.
    """
    gates: list[GateResult] = []

    g0 = gate0_data_quality(df)
    gates.append(g0)
    if not g0.passed:
        return DecisionTreeResult(
            gates,
            "Do not decide yet",
            "Fix instrumentation, assignment, or data quality first.",
            stopped_at="Gate 0",
        )

    g1 = gate1_primary_metric(df)
    gates.append(g1)
    if not g1.passed:
        decision, rationale = _decide(False, "", "", "")
        return DecisionTreeResult(gates, decision, rationale, stopped_at="Gate 1")

    g2 = gate2_guardrail_dependency(df)
    g3 = gate3_quality_read(df)
    g4 = gate4_borrowed_risk_read(df)
    gates.extend([g2, g3, g4])

    decision, rationale = _decide(
        True,
        str(g3.metrics["quality_strength"]),
        str(g4.metrics["risk_level"]),
        str(g2.metrics["dependency_level"]),
    )
    gates.append(
        GateResult("Gate 5", "Decision Output", True, f"{decision} - {rationale}",
                   metrics={"decision": decision, "rationale": rationale})
    )
    return DecisionTreeResult(gates, decision, rationale, stopped_at=None)


# --- Reporting -----------------------------------------------------------------
def _status_label(gate: GateResult) -> str:
    if gate.gate == "Gate 0":
        return "PASS" if gate.passed else "STOP"
    if gate.gate == "Gate 1":
        return "PASS (lift)" if gate.passed else "STOP (no lift)"
    if gate.gate == "Gate 5":
        return "DECISION"
    return "CONTINUE"


def format_report(result: DecisionTreeResult) -> str:
    lines: list[str] = ["=" * 72, "STAGE 4 - DECISION TREE", "=" * 72]
    for g in result.gates:
        lines.append(f"[{_status_label(g):<14}] {g.gate}: {g.name}")
        lines.append(f"    {g.summary}")
        for c in g.checks:
            mark = "ok  " if c.passed else "FAIL"
            lines.append(f"      - [{mark}] {c.name}: {c.detail}")
    lines.append("-" * 72)
    if result.stopped_at:
        lines.append(f"Flow stopped at {result.stopped_at}.")
    lines.append(f"FINAL DECISION: {result.decision}")
    lines.append(f"RATIONALE:      {result.rationale}")
    lines.append("=" * 72)
    return "\n".join(lines)


def _main() -> None:
    import sys
    from pathlib import Path

    src_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(src_dir))
    from signals import add_derived_signals
    from early_read import add_early_read

    data_path = src_dir.parent / "data" / "simulated_coupon_experiment.csv"
    df = pd.read_csv(data_path)
    observable = [
        "user_id", "group", "coupon_received", "signup_day", "exposure_day",
        "made_first_purchase_30d", "days_to_first_purchase", "repeat_visits_14d",
        "browse_sessions_14d", "product_views_14d", "cart_adds_14d",
        "discount_page_views_14d", "coupon_used",
    ]
    read_df = add_early_read(add_derived_signals(df[observable]))
    print(format_report(run_decision_tree(read_df)))


if __name__ == "__main__":
    _main()
