"""Stage 2 - Early Signal Layer.

Turns raw observable behaviour into a small set of transparent, rule-based
derived signals. This is a signal layer, not a model: every signal is a simple
combination of observable fields, expressed on a common 0-1 percentile scale so
that fields with different units can be compared.

Only observable fields are read here. Latent ground-truth fields and long-term
outcome fields are never touched - those belong to the Stage 5 calibration step.

Derived signals
---------------
engagement_intensity : breadth / activity of early interaction
    mean percentile of repeat_visits_14d, browse_sessions_14d, product_views_14d
purchase_intent      : depth of purchase intent
    percentile of cart_adds_14d (kept separate from engagement so product views
    are not weighted twice)
discount_dependency  : reliance on the discount itself
    percentile of discount_page_views_14d combined with coupon_used
early_timing         : how early the first purchase landed (purchasers only)
    higher = earlier; NaN for non-purchasers. On its own this is NOT a
    borrowed-demand signal - Stage 3 only treats it as risk when it co-occurs
    with weak engagement and high discount dependency.
"""

from __future__ import annotations

import pandas as pd

# Observable inputs this layer is allowed to read. Listed explicitly so it is
# obvious that no latent or long-term field is ever used.
ENGAGEMENT_FIELDS = ["repeat_visits_14d", "browse_sessions_14d", "product_views_14d"]
INTENT_FIELD = "cart_adds_14d"
DISCOUNT_VIEWS_FIELD = "discount_page_views_14d"
COUPON_USED_FIELD = "coupon_used"
TIMING_FIELD = "days_to_first_purchase"
PURCHASE_FLAG = "made_first_purchase_30d"


def _percentile(series: pd.Series) -> pd.Series:
    """Percentile rank in [0, 1]. Ties share a rank; NaNs stay NaN."""
    return series.rank(pct=True)


def add_derived_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with the four Stage 2 derived signals added.

    Parameters
    ----------
    df : observable-field dataframe, one row per user.

    Returns
    -------
    A copy of ``df`` with these columns added:
    ``engagement_intensity``, ``purchase_intent``, ``discount_dependency``,
    ``early_timing``.
    """
    out = df.copy()

    # 1. engagement_intensity - average percentile across the three breadth
    #    fields, so no single field dominates and the units stay comparable.
    engagement = pd.concat([_percentile(out[c]) for c in ENGAGEMENT_FIELDS], axis=1)
    out["engagement_intensity"] = engagement.mean(axis=1)

    # 2. purchase_intent - cart adds only, deliberately separate from engagement
    #    so product views are not weighted twice.
    out["purchase_intent"] = _percentile(out[INTENT_FIELD])

    # 3. discount_dependency - discount-page browsing (percentile) combined, at
    #    equal weight, with whether a coupon was actually redeemed (0/1, already
    #    on a 0-1 scale). Both halves point at reliance on the discount itself.
    discount_views_pct = _percentile(out[DISCOUNT_VIEWS_FIELD])
    out["discount_dependency"] = 0.5 * discount_views_pct + 0.5 * out[COUPON_USED_FIELD]

    # 4. early_timing - purchasers only. A smaller days_to_first_purchase means
    #    an earlier purchase, so we invert the percentile: higher = earlier.
    #    Non-purchasers have no first-purchase timing, so they stay NaN rather
    #    than being forced to 0.
    is_purchaser = out[PURCHASE_FLAG] == 1
    timing_pct = _percentile(out.loc[is_purchaser, TIMING_FIELD])
    out["early_timing"] = 1.0 - timing_pct  # aligns on index; non-purchasers -> NaN

    return out
