"""
predictions.py – Spending predictions & smart insights.

Uses a trained Ridge/DecisionTree model to predict end-of-month spending,
plus rule-based alerts for budget runway, category drift, and pace warnings.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

MODEL_PATH = Path("spending_predictor.pkl")

# ── Load model bundle ────────────────────────────────────────────────
_bundle = None


def _load_model():
    global _bundle
    if _bundle is not None:
        return _bundle
    if MODEL_PATH.exists():
        try:
            _bundle = joblib.load(MODEL_PATH)
        except Exception:
            _bundle = None
    return _bundle


# ── ML Prediction ────────────────────────────────────────────────────
def predict_eom_spend(df: pd.DataFrame, month: str, budget: float) -> dict | None:
    """
    Predict end-of-month total spending using the trained model.

    Uses ML prediction with sanity bounds: if the model gives a result
    that's wildly different from the simple pace projection, we blend
    them together weighted by how far into the month we are.

    Returns a dict with:
      - predicted_eom: predicted total spend for the month
      - predicted_remaining: how much more the user will likely spend
      - model_used: which model was used
      - confidence: 'high' or 'medium'
      - budget_status: 'on_track', 'warning', 'over'
      - days_until_budget_out: estimated days until budget runs out (or None)
    """
    bundle = _load_model()
    if df.empty:
        return None

    dfm = df.copy()
    dfm["date"] = pd.to_datetime(dfm["date"])
    dfm["month"] = dfm["date"].dt.to_period("M").astype(str)
    dfm = dfm[dfm["month"] == month]

    if dfm.empty:
        return None

    today = datetime.today()
    dom = today.day
    month_period = pd.Period(month, "M")
    days_in_month = month_period.days_in_month

    # Current month stats
    cum_spend = float(dfm["amount_bhd"].sum())
    cum_count = len(dfm)
    days_elapsed = dom
    pct_elapsed = days_elapsed / days_in_month
    daily_rate = cum_spend / max(days_elapsed, 1)
    remaining_days = days_in_month - dom

    # Simple pace projection (always available as baseline)
    simple_projected = daily_rate * days_in_month

    # Previous month total
    prev_month = (month_period - 1).strftime("%Y-%m")
    df_all = df.copy()
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["month"] = df_all["date"].dt.to_period("M").astype(str)
    df_prev = df_all[df_all["month"] == prev_month]
    prev_month_total = float(df_prev["amount_bhd"].sum()) if not df_prev.empty else None

    # If no previous month data, use average of all available months
    if not prev_month_total or prev_month_total == 0:
        monthly_totals = df_all.groupby("month")["amount_bhd"].sum()
        # Exclude current month from the average
        other_months = monthly_totals[monthly_totals.index != month]
        if not other_months.empty:
            prev_month_total = float(other_months.mean())
        else:
            prev_month_total = cum_spend

    # Compute rolling 3-month average from all available data
    all_months_data = df_all.groupby("month")["amount_bhd"].sum()
    other_months = all_months_data[all_months_data.index != month]
    if len(other_months) >= 3:
        # Last 3 months before current
        sorted_m = sorted(other_months.index)
        last_3 = sorted_m[-3:]
        rolling_3m_avg = float(other_months[last_3].mean())
    elif not other_months.empty:
        rolling_3m_avg = float(other_months.mean())
    else:
        rolling_3m_avg = prev_month_total

    # Try ML prediction
    ml_predicted_eom = None
    model_name = "Pace-based"

    if bundle is not None:
        features = bundle["features"]
        feature_values = {
            "dom": dom,
            "cum_spend": cum_spend,
            "cum_count": cum_count,
            "pct_elapsed": pct_elapsed,
            "daily_rate": daily_rate,
            "remaining_days": remaining_days,
            "prev_month_total": prev_month_total,
            "days_in_month": days_in_month,
            "rolling_3m_avg": rolling_3m_avg,
        }
        X = np.array([[feature_values.get(f, 0) for f in features]])

        # Prefer ensemble > ridge > random_forest > decision_tree
        model = bundle.get("ensemble")
        model_name = "Ensemble (Ridge + RF)"
        if model is None:
            model = bundle.get("ridge")
            model_name = "Ridge Regression"
        if model is None:
            model = bundle.get("random_forest")
            model_name = "Random Forest"
        if model is None:
            model = bundle.get("decision_tree")
            model_name = "Decision Tree"

        if model is not None:
            predicted_remaining = float(model.predict(X)[0])
            predicted_remaining = max(predicted_remaining, 0)
            ml_predicted_eom = cum_spend + predicted_remaining

    # Blend ML and simple projection based on confidence
    if ml_predicted_eom is not None and simple_projected > 0:
        ratio = ml_predicted_eom / simple_projected if simple_projected > 0 else 999

        if 0.5 <= ratio <= 2.0:
            # ML and simple agree within 2x — trust ML more as month progresses
            ml_weight = min(pct_elapsed * 1.5, 0.85)  # max 85% ML weight
            predicted_eom = (ml_weight * ml_predicted_eom) + ((1 - ml_weight) * simple_projected)
            model_name = f"{model_name} (blended)"
        else:
            # ML is way off — lean heavily on simple projection
            ml_weight = max(pct_elapsed * 0.3, 0.0)  # much lower ML trust
            predicted_eom = (ml_weight * ml_predicted_eom) + ((1 - ml_weight) * simple_projected)
            model_name = f"{model_name} (pace-adjusted)"
    elif ml_predicted_eom is not None:
        predicted_eom = ml_predicted_eom
    else:
        predicted_eom = simple_projected
        model_name = "Pace-based"

    # Final sanity: prediction can never be less than what's already spent
    predicted_eom = max(predicted_eom, cum_spend)
    predicted_remaining = predicted_eom - cum_spend

    # Budget analysis
    if budget > 0:
        if predicted_eom > budget:
            status = "over"
        elif predicted_eom > budget * 0.8:
            status = "warning"
        else:
            status = "on_track"

        # Days until budget runs out
        if daily_rate > 0 and remaining_days > 0:
            budget_left = budget - cum_spend
            if budget_left > 0:
                days_until_out = budget_left / daily_rate
                days_until_out = min(days_until_out, remaining_days)
            else:
                days_until_out = 0
        else:
            days_until_out = None
    else:
        status = "on_track"
        days_until_out = None

    # Confidence based on how much of the month has passed + data quality
    if pct_elapsed > 0.5 and cum_count >= 10:
        confidence = "high"
    elif pct_elapsed > 0.3 and cum_count >= 5:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "predicted_eom": round(predicted_eom, 3),
        "predicted_remaining": round(predicted_remaining, 3),
        "cum_spend": round(cum_spend, 3),
        "daily_rate": round(daily_rate, 3),
        "model_used": model_name,
        "model_cv_r2": round(bundle.get("ensemble_cv_r2", bundle.get("ridge_cv_r2", 0)), 4) if bundle else 0,
        "confidence": confidence,
        "budget_status": status,
        "days_until_budget_out": round(days_until_out, 1) if days_until_out is not None else None,
        "pct_elapsed": round(pct_elapsed * 100, 1),
        "remaining_days": remaining_days,
    }


# ── Category Drift Detection ────────────────────────────────────────
def detect_category_drift(df: pd.DataFrame, month: str) -> list[dict]:
    """
    Compare this month's category spending against the 3-month rolling average.
    Returns a list of alerts like:
      {'category': 'مطاعم ومقاهي', 'current': 45.0, 'avg': 30.0, 'change_pct': 50.0, 'direction': 'up'}
    """
    if df.empty:
        return []

    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    dfx["month"] = dfx["date"].dt.to_period("M").astype(str)

    current = dfx[dfx["month"] == month]
    if current.empty:
        return []

    # Get the 3 months before this one
    month_period = pd.Period(month, "M")
    prev_months = [(month_period - i).strftime("%Y-%m") for i in range(1, 4)]
    historical = dfx[dfx["month"].isin(prev_months)]

    if historical.empty:
        return []

    # Current month by category
    cur_cats = current.groupby("category")["amount_bhd"].sum()

    # Historical monthly average by category
    hist_monthly = historical.groupby(["month", "category"])["amount_bhd"].sum().reset_index()
    hist_avg = hist_monthly.groupby("category")["amount_bhd"].mean()

    alerts = []
    for cat in cur_cats.index:
        cur_val = float(cur_cats[cat])
        avg_val = float(hist_avg.get(cat, 0))

        if avg_val < 5:  # skip tiny categories
            continue

        change_pct = ((cur_val - avg_val) / avg_val) * 100

        if abs(change_pct) >= 30:  # only alert on 30%+ change
            alerts.append({
                "category": cat,
                "current": round(cur_val, 3),
                "avg_3m": round(avg_val, 3),
                "change_pct": round(change_pct, 1),
                "direction": "up" if change_pct > 0 else "down",
            })

    return sorted(alerts, key=lambda x: abs(x["change_pct"]), reverse=True)


# ── Top Insights Summary ────────────────────────────────────────────
def generate_insights(df: pd.DataFrame, month: str, budget: float) -> list[str]:
    """Generate a list of plain-text insight strings for the dashboard."""
    insights = []

    if df.empty:
        return ["No data yet — add some transactions to see insights."]

    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"])
    dfx["month"] = dfx["date"].dt.to_period("M").astype(str)
    dfm = dfx[dfx["month"] == month]

    if dfm.empty:
        return [f"No transactions for {month} yet."]

    total = float(dfm["amount_bhd"].sum())
    count = len(dfm)
    today = datetime.today()
    dom = today.day
    days_in_month = pd.Period(month, "M").days_in_month

    # 1. Pace insight — uses ML prediction when available, falls back to simple projection
    daily_rate = total / max(dom, 1)
    prediction = predict_eom_spend(dfx, month, budget)
    projected = prediction["predicted_eom"] if prediction else daily_rate * days_in_month

    if budget > 0:
        if projected > budget * 1.1:
            insights.append(
                f"📈 At your current pace ({daily_rate:.1f} BHD/day), "
                f"you're projected to spend ~{projected:.0f} BHD this month — "
                f"that's {projected - budget:.0f} BHD over your {budget:.0f} BHD budget."
            )
        elif projected > budget * 0.9:
            insights.append(
                f"⚠️ You're spending ~{daily_rate:.1f} BHD/day — "
                f"projected {projected:.0f} BHD which is close to your {budget:.0f} BHD budget."
            )
        else:
            insights.append(
                f"✅ Spending pace looks healthy at {daily_rate:.1f} BHD/day — "
                f"projected ~{projected:.0f} BHD against your {budget:.0f} BHD budget."
            )

    # 2. Biggest category
    top_cat = dfm.groupby("category")["amount_bhd"].sum().sort_values(ascending=False)
    if not top_cat.empty:
        pct = (float(top_cat.iloc[0]) / total * 100)
        insights.append(
            f"🏷️ Top category: **{top_cat.index[0]}** "
            f"({float(top_cat.iloc[0]):.1f} BHD, {pct:.0f}% of spending)."
        )

    # 3. Biggest single transaction
    biggest = dfm.loc[dfm["amount_bhd"].idxmax()]
    insights.append(
        f"💰 Biggest transaction: **{biggest['merchant']}** — "
        f"{float(biggest['amount_bhd']):.3f} BHD."
    )

    # 4. Category drift
    drifts = detect_category_drift(dfx, month)
    for d in drifts[:2]:
        emoji = "🔺" if d["direction"] == "up" else "🔻"
        insights.append(
            f"{emoji} **{d['category']}** is {abs(d['change_pct']):.0f}% "
            f"{'higher' if d['direction'] == 'up' else 'lower'} "
            f"than your 3-month average "
            f"({d['current']:.1f} BHD this month vs {d['avg_3m']:.1f} BHD average)."
        )

    return insights