"""Simulate a coupon-driven first-purchase experiment.

Generates ~10,000 users randomly split into control / treatment. Each user is
drawn from one of six hidden personas that act as ground truth. Three kinds of
fields are written to the output CSV:

* Observable fields  - available during the experiment (used by Stages 1-4)
* Long-term fields   - only observed after 180 days (used by Stage 5)
* Latent fields      - the hidden ground truth, revealed only for validation

The simulation is tuned to satisfy fixed guardrails on the primary metric:

* control   30d first-purchase rate in 16-20%
* treatment 30d first-purchase rate in 22-26%
* headline lift in +5pp to +7pp

All observable signals are drawn from probability distributions with
overlapping ranges, so no single signal cleanly separates the personas.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_USERS = 10_000

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "simulated_coupon_experiment.csv"

# Persona ground truth: (response, value, timing, share). Shares sum to 1.0.
PERSONAS = [
    ("persuadable", "high_value", "normal_timing", 0.12),
    ("persuadable", "discount_driven_low_value", "normal_timing", 0.15),
    ("sure_thing", "high_value", "normal_timing", 0.13),
    ("sure_thing", "medium_value", "pulled_forward", 0.12),
    ("persuadable", "discount_driven_low_value", "pulled_forward", 0.18),
    ("lost_cause", "no_value", "no_purchase", 0.30),
]

# 30-day first-purchase probability by response type and arm.
# response_type drives the headline behaviour:
#   persuadable - low in control, much higher in treatment (the real increment)
#   sure_thing  - high in both arms (the coupon adds almost no first-purchase lift)
#   lost_cause  - very low in both arms
FIRST_PURCHASE_RATE = {
    "persuadable": {"control": 0.10, "treatment": 0.216},
    "sure_thing": {"control": 0.516, "treatment": 0.536},
    "lost_cause": {"control": 0.02, "treatment": 0.03},
}


def _by_value(value: np.ndarray, high: float, medium: float, discount: float, none: float) -> np.ndarray:
    """Map a value_type array onto per-user parameters via np.select."""
    return np.select(
        [
            value == "high_value",
            value == "medium_value",
            value == "discount_driven_low_value",
            value == "no_value",
        ],
        [high, medium, discount, none],
        default=medium,
    )


def simulate(seed: int = SEED, n_users: int = N_USERS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Assignment --------------------------------------------------------
    user_id = np.arange(1, n_users + 1)
    group = rng.choice(["control", "treatment"], size=n_users, p=[0.5, 0.5])
    coupon_received = (group == "treatment").astype(int)

    shares = np.array([p[3] for p in PERSONAS])
    persona_idx = rng.choice(len(PERSONAS), size=n_users, p=shares)
    response = np.array([p[0] for p in PERSONAS])[persona_idx]
    value = np.array([p[1] for p in PERSONAS])[persona_idx]
    timing = np.array([p[2] for p in PERSONAS])[persona_idx]

    # --- Timeline ----------------------------------------------------------
    signup_day = rng.integers(1, 91, size=n_users)
    exposure_day = signup_day + rng.integers(0, 8, size=n_users)

    # --- Primary metric: 30d first purchase --------------------------------
    rate = np.array(
        [FIRST_PURCHASE_RATE[r][g] for r, g in zip(response, group)]
    )
    made_first_purchase_30d = rng.binomial(1, rate)

    # Timing of the first purchase (only meaningful for purchasers).
    # pulled_forward + treatment buys earlier; everyone else buys at normal timing.
    mean_days = np.full(n_users, 13.0)
    pulled_forward_treat = (timing == "pulled_forward") & (group == "treatment")
    mean_days[pulled_forward_treat] = 7.0
    days_raw = np.clip(np.round(rng.normal(mean_days, 5.0)), 1, 30).astype(int)
    days_to_first_purchase = pd.array(days_raw, dtype="Int64")
    days_to_first_purchase[made_first_purchase_30d == 0] = pd.NA

    # --- Early behavioural signals (14d, observed for everyone) ------------
    # Driven by value_type; Poisson draws give overlapping, noisy distributions.
    repeat_visits_14d = rng.poisson(_by_value(value, 4.0, 2.5, 1.5, 0.4))
    browse_sessions_14d = rng.poisson(_by_value(value, 6.0, 4.0, 2.5, 0.6))
    product_views_14d = rng.poisson(_by_value(value, 18.0, 11.0, 7.0, 1.5))
    cart_adds_14d = rng.poisson(_by_value(value, 3.5, 2.0, 1.2, 0.2))

    # Discount-page views: high for discount-driven users; nudged up slightly in
    # treatment because the coupon draws attention to discounts.
    discount_lambda = _by_value(value, 1.0, 2.0, 5.0, 0.5) + 0.5 * (group == "treatment")
    discount_page_views_14d = rng.poisson(discount_lambda)

    # A coupon can only be "used" on a purchase, and only in treatment.
    p_use = _by_value(value, 0.5, 0.7, 0.9, 0.3)
    coupon_used = np.where(
        (group == "treatment") & (made_first_purchase_30d == 1),
        rng.binomial(1, p_use),
        0,
    )

    # --- Long-term outcomes (180d) -----------------------------------------
    # Extra orders beyond the first; pulled_forward treatment borrows future
    # demand, so its post-period order rate is discounted at the group level
    # while Poisson noise keeps individual outcomes varied.
    lam_extra = _by_value(value, 3.5, 1.8, 0.5, 0.1)
    borrow = np.where(pulled_forward_treat, 0.45, 1.0)
    extra_orders = rng.poisson(lam_extra * borrow)
    orders_180d = np.where(made_first_purchase_30d == 1, 1 + extra_orders, 0)
    repeat_purchase_180d = (orders_180d >= 2).astype(int)

    aov = _by_value(value, 60.0, 40.0, 30.0, 25.0)
    aov_user = np.maximum(5.0, rng.normal(aov, aov * 0.25))
    gross_revenue_180d = np.where(made_first_purchase_30d == 1, orders_180d * aov_user, 0.0)

    # Discount cost = the coupon face value (treatment users who redeemed) plus
    # ongoing markdown that discount-driven users keep claiming over time.
    coupon_face = np.where(coupon_used == 1, np.maximum(0.0, rng.normal(10.0, 2.0)), 0.0)
    ongoing_rate = _by_value(value, 0.05, 0.12, 0.35, 0.10)
    discount_cost_180d = coupon_face + ongoing_rate * gross_revenue_180d

    net_revenue_180d = gross_revenue_180d - discount_cost_180d
    realized_ltv_180d = net_revenue_180d

    df = pd.DataFrame(
        {
            # Observable fields
            "user_id": user_id,
            "group": group,
            "coupon_received": coupon_received,
            "signup_day": signup_day,
            "exposure_day": exposure_day,
            "made_first_purchase_30d": made_first_purchase_30d,
            "days_to_first_purchase": days_to_first_purchase,
            "repeat_visits_14d": repeat_visits_14d,
            "browse_sessions_14d": browse_sessions_14d,
            "product_views_14d": product_views_14d,
            "cart_adds_14d": cart_adds_14d,
            "discount_page_views_14d": discount_page_views_14d,
            "coupon_used": coupon_used,
            # Long-term fields (observed after 180 days)
            "repeat_purchase_180d": repeat_purchase_180d,
            "orders_180d": orders_180d,
            "gross_revenue_180d": np.round(gross_revenue_180d, 2),
            "discount_cost_180d": np.round(discount_cost_180d, 2),
            "net_revenue_180d": np.round(net_revenue_180d, 2),
            "realized_ltv_180d": np.round(realized_ltv_180d, 2),
            # Latent ground truth (revealed only for validation)
            "latent_response_type": response,
            "latent_value_type": value,
            "latent_timing_type": timing,
        }
    )
    return df


def sanity_check(df: pd.DataFrame) -> None:
    n_c = int((df["group"] == "control").sum())
    n_t = int((df["group"] == "treatment").sum())
    control_rate = df.loc[df["group"] == "control", "made_first_purchase_30d"].mean()
    treatment_rate = df.loc[df["group"] == "treatment", "made_first_purchase_30d"].mean()
    lift = treatment_rate - control_rate

    ok_c = 0.16 <= control_rate <= 0.20
    ok_t = 0.22 <= treatment_rate <= 0.26
    ok_l = 0.05 <= lift <= 0.07

    print("=" * 60)
    print("SANITY CHECK - primary metric guardrails")
    print("=" * 60)
    print(f"Users: {len(df):,}  (control={n_c:,}, treatment={n_t:,})")
    print(f"Control   30d first-purchase rate: {control_rate:7.2%}   target 16-20%")
    print(f"Treatment 30d first-purchase rate: {treatment_rate:7.2%}   target 22-26%")
    print(f"Headline lift:                     {lift * 100:+6.2f}pp   target +5 to +7pp")
    print("-" * 60)
    print(f"Control in range:    {ok_c}")
    print(f"Treatment in range:  {ok_t}")
    print(f"Lift in range:       {ok_l}")
    print(f"ALL GUARDRAILS MET:  {ok_c and ok_t and ok_l}")
    print("=" * 60)


def main() -> None:
    df = simulate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df):,} rows to {OUTPUT_PATH}")
    sanity_check(df)


if __name__ == "__main__":
    main()
