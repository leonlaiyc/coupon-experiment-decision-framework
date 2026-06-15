"""Stage 3 - Early Read Model (v1, transparent rules).

Turns the Stage 2 derived signals into an early read on customer quality and
borrowed-demand risk. The shape is "score, then band": continuous scores are
computed internally as an intermediate step, but the read exposed for decisions
is a coarse high / medium / low band based on within-cohort tertiles. Bands keep
the output honest (no false precision) and need no hand-picked thresholds.

This stage is intentionally modular. v1 is rule-based and fully inspectable; a
later version could swap in a statistical or ML model without changing the rest
of the framework, as long as it produces the same read columns.

Only the Stage 2 derived signals (themselves built from observable fields) are
read here. No latent or long-term field is touched.

Internal scores (intermediate, 0-1)
-----------------------------------
quality_score             high engagement + high intent + low discount dependency
borrowed_risk_score       early purchase AND weak engagement AND high discount
                          dependency, combined as a product so all three must be
                          present. Early timing alone never raises it. Purchasers
                          only (NaN otherwise - no purchase, no pulled-forward
                          demand to speak of).
discount_dependency_score passthrough of the discount_dependency signal

External reads (high / medium / low)
------------------------------------
quality_read, borrowed_risk_read, discount_dependency_read
    top tertile -> high, middle -> medium, bottom -> low, by percentile.
"""

from __future__ import annotations

import pandas as pd

# Tertile cut points for the high / medium / low banding.
_LOW_CUT = 1 / 3
_HIGH_CUT = 2 / 3


def _tertile_read(score: pd.Series) -> pd.Series:
    """Band a continuous score into low / medium / high by tertile.

    Uses percentile rank so the split adapts to the cohort instead of relying on
    fixed thresholds. Ties share a percentile (equal scores get the same read);
    NaN scores stay NaN (e.g. borrowed risk for non-purchasers).
    """
    pct = score.rank(pct=True)
    read = pd.Series(pd.NA, index=score.index, dtype="object")
    read[pct <= _LOW_CUT] = "low"
    read[(pct > _LOW_CUT) & (pct <= _HIGH_CUT)] = "medium"
    read[pct > _HIGH_CUT] = "high"
    return read


def add_early_read(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with internal scores and external reads added.

    Expects the Stage 2 derived-signal columns (``engagement_intensity``,
    ``purchase_intent``, ``discount_dependency``, ``early_timing``) to be present.
    """
    out = df.copy()

    # --- Internal continuous scores (intermediate only) -------------------

    # quality_score: equal-weight blend. Discount dependency is inverted so that
    # *low* dependency lifts quality.
    out["quality_score"] = (
        out["engagement_intensity"]
        + out["purchase_intent"]
        + (1.0 - out["discount_dependency"])
    ) / 3.0

    # borrowed_risk_score: a deliberate AND of three conditions, written as a
    # product so every factor must be elevated for the score to rise:
    #   - early_timing high          (bought early)
    #   - engagement weak            (low engagement_intensity -> high 1 - it)
    #   - discount_dependency high
    # Buying early on its own does NOT raise the score: if engagement is strong
    # or discount dependency is low, that factor is near zero and the product
    # collapses. Non-purchasers have NaN early_timing, so their risk is NaN.
    out["borrowed_risk_score"] = (
        out["early_timing"]
        * (1.0 - out["engagement_intensity"])
        * out["discount_dependency"]
    )

    # discount_dependency_score: direct passthrough of the Stage 2 signal.
    out["discount_dependency_score"] = out["discount_dependency"]

    # --- External reads (high / medium / low by tertile) ------------------
    out["quality_read"] = _tertile_read(out["quality_score"])
    out["borrowed_risk_read"] = _tertile_read(out["borrowed_risk_score"])
    out["discount_dependency_read"] = _tertile_read(out["discount_dependency_score"])

    return out
