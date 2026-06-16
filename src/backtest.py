"""Stage 5A + 5B - Backtest and Calibration.

Stage 5 is the calibration loop. It has three parts:

    5A  Backtest    - compare the Stage 3 early reads against mature long-term
                      outcomes (this module).
    5B  Operating   - turn the backtest readout into concrete, stage-by-stage
        update        next-cycle updates (``next_cycle_updates`` here).
    5C  Monitoring  - post-launch monitoring design once a strategy is scaled
                      (``src/monitoring.py``).

This is the first stage allowed to OPEN the long-term outcome fields
(repeat_purchase_180d, orders_180d, net_revenue_180d, realized_ltv_180d, ...).
Stages 1-4 deliberately never touch them.

What the backtest does and does not claim
-----------------------------------------
The backtest asks whether the early reads *align* with later outcomes - i.e.
whether the cohorts we read early as high quality / high risk / high dependency
actually went on to behave that way. It is a directional check on the read
*instrument*, summarised at the cohort (read-band) level. It does NOT claim the
reads predict any individual user's outcome, and it is not a calibrated
prediction model.

Cohort
------
The tables are computed over all PURCHASERS (both arms pooled). Long-term value
is only meaningful for users who actually purchased, and borrowed_risk_read is
defined for purchasers only, so pooling purchasers gives the cleanest and
largest-sample view of the read instrument.
"""

from __future__ import annotations

import pandas as pd

PRIMARY_METRIC = "made_first_purchase_30d"
BANDS = ["high", "medium", "low"]

# Long-term outcome fields - opened here for the first time.
LONG_TERM_FIELDS = [
    "repeat_purchase_180d",
    "orders_180d",
    "gross_revenue_180d",
    "discount_cost_180d",
    "net_revenue_180d",
    "realized_ltv_180d",
]


def derive_post_period_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``post_period_orders_31_180d`` = orders_180d - made_first_purchase_30d.

    orders_180d counts *all* orders in the 180-day window, which includes the
    very first purchase for anyone who converted. To isolate demand that landed
    *after* the 30-day first-purchase window - the part that borrowed-demand risk
    is really about - we subtract the first-purchase flag (1 for purchasers, 0
    otherwise). Borrowed demand shows up as a weak post-period order count even
    when the first purchase looked fine.
    """
    out = df.copy()
    out["post_period_orders_31_180d"] = out["orders_180d"] - out[PRIMARY_METRIC]
    return out


def purchaser_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Return the backtest cohort: all purchasers (both arms pooled)."""
    return df[df[PRIMARY_METRIC] == 1].copy()


def _band_means(cohort: pd.DataFrame, read_col: str, value_cols: list[str]) -> pd.DataFrame:
    """Mean of ``value_cols`` within each high/medium/low band of ``read_col``."""
    return (
        cohort.groupby(read_col)[value_cols]
        .mean()
        .reindex(BANDS)
        .round(2)
    )


def quality_vs_value(cohort: pd.DataFrame) -> pd.DataFrame:
    """Table 1 - quality read vs long-term value."""
    return _band_means(cohort, "quality_read", ["repeat_purchase_180d", "orders_180d", "realized_ltv_180d"])


def borrowed_risk_vs_outcome(cohort: pd.DataFrame) -> pd.DataFrame:
    """Table 2 - borrowed-risk read vs long-term outcome (needs post-period orders)."""
    return _band_means(
        cohort, "borrowed_risk_read", ["post_period_orders_31_180d", "net_revenue_180d", "realized_ltv_180d"]
    )


def discount_dependency_vs_value(cohort: pd.DataFrame) -> pd.DataFrame:
    """Table 3 - discount-dependency read vs long-term value."""
    return _band_means(
        cohort, "discount_dependency_read", ["repeat_purchase_180d", "net_revenue_180d", "realized_ltv_180d"]
    )


def run_backtest(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Derive post-period orders, build the purchaser cohort, and return the 3 tables."""
    cohort = purchaser_cohort(derive_post_period_orders(df))
    return {
        "quality_vs_value": quality_vs_value(cohort),
        "borrowed_risk_vs_outcome": borrowed_risk_vs_outcome(cohort),
        "discount_dependency_vs_value": discount_dependency_vs_value(cohort),
    }


def next_cycle_updates() -> pd.DataFrame:
    """Stage 5B - the next-cycle operating update (process, not weight tuning).

    Each row maps one observation from this cycle to how a specific stage changes
    in the next cycle - the same five columns shown in the notebook's Stage 5B
    table. This is the loop closing: what this cycle learned becomes next cycle's
    improvement, stage by stage. It deliberately does NOT auto-tune weights or
    thresholds.
    """
    rows = [
        ("Treatment lift exists, but treatment purchasers show weaker quality mix",
         "Coupon has short-term lift, but broad scaling is not justified yet",
         "Predefine target segments before next test",
         "Compare early quality mix by segment",
         "Lift alone cannot trigger Scale"),
        ("High discount dependency aligns with weaker long-term value in this simulated case",
         "Discount dependency is an important early risk proxy",
         "Add discount dependency as a stronger guardrail",
         "Keep discount dependency read; consider adding full-price repeat signal",
         "Gate 2 becomes a stronger guardrail before Scale"),
        ("High borrowed-risk read aligns with weaker post-period orders in this walkthrough",
         "Borrowed-risk read should limit broad rollout until longer outcomes mature",
         "Keep holdout and define 31-180d tracking window",
         "Keep timing x weak engagement x discount dependency as combined risk logic",
         "High borrowed risk blocks broad Scale"),
        ("Early timing alone is insufficient",
         "Fast conversion is not automatically bad",
         "Track timing, but do not use it as a standalone decision metric",
         "Timing remains only one input in the combined risk score",
         "Gate 4 cannot be triggered by timing alone"),
        ("Current data lacks post-purchase engagement",
         "The next test needs stronger early quality evidence",
         "Add post-purchase engagement tracking after first order",
         "Add post-purchase sessions / repeat browsing as new signals",
         "Scale condition becomes stricter if post-purchase engagement is weak"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "This cycle observation",
            "Framework interpretation",
            "Next Stage 1 update",
            "Next Stage 2/3 update",
            "Next Stage 4 update",
        ],
    )


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
    # Stage 5 opens the long-term fields and joins them onto the early reads.
    full = read_df.merge(df[["user_id", *LONG_TERM_FIELDS]], on="user_id")

    tables = run_backtest(full)
    titles = {
        "quality_vs_value": "Table 1 - Quality read vs long-term value",
        "borrowed_risk_vs_outcome": "Table 2 - Borrowed-risk read vs long-term outcome",
        "discount_dependency_vs_value": "Table 3 - Discount-dependency read vs long-term value",
    }
    print("=" * 72)
    print("STAGE 5A - BACKTEST (do early reads ALIGN with later outcomes?)")
    print(f"Cohort: all purchasers, both arms pooled (n={int((full[PRIMARY_METRIC] == 1).sum()):,})")
    print("=" * 72)
    for key, title in titles.items():
        print(f"\n{title}")
        print(tables[key].to_string())
    print("\n" + "=" * 72)
    print("STAGE 5B - NEXT-CYCLE OPERATING UPDATE")
    print("=" * 72)
    print(next_cycle_updates().to_string(index=False))


if __name__ == "__main__":
    _main()
