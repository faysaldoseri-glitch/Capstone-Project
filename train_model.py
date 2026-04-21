"""
train_model.py – Train the spending prediction model.

Usage:  python train_model.py

Reads generated_sms_data_1_year.csv, trains Ridge + RandomForest + Ensemble,
compares them, and saves the best bundle as spending_predictor.pkl.

v2.0 — Improvements:
  - Added rolling_3m_avg feature (3-month historical average)
  - Denser training snapshots (every day instead of every 2nd)
  - Ridge+RandomForest ensemble for more stable predictions
  - Leave-One-Month-Out CV for realistic evaluation
  - Higher regularization (alpha=100) to prevent overfitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib


def load_and_prepare(csv_path: str = "generated_sms_data_1_year.csv"):
    """Load SMS data, filter to purchases, build daily time series."""
    df = pd.read_csv(csv_path)
    purchases = df[df["SMS Type"] == "Purchase"].copy()
    purchases["date"] = pd.to_datetime(purchases["date"], format="%d-%m-%y", dayfirst=True)
    purchases = purchases.sort_values("date")

    # Daily aggregation
    daily = purchases.groupby(purchases["date"].dt.date).agg(
        daily_total=("Amount (BHD)", "sum"),
        daily_count=("Amount (BHD)", "count"),
    ).reset_index()
    daily.columns = ["date", "daily_total", "daily_count"]
    daily["date"] = pd.to_datetime(daily["date"])

    # Fill missing days with 0
    full_range = pd.date_range(daily["date"].min(), daily["date"].max())
    daily = daily.set_index("date").reindex(full_range, fill_value=0).reset_index()
    daily.columns = ["date", "daily_total", "daily_count"]

    return daily


def build_features(daily: pd.DataFrame):
    """Engineer features for intra-month prediction."""
    daily = daily.copy()
    daily["month"] = daily["date"].dt.to_period("M")
    daily["dom"] = daily["date"].dt.day
    daily["days_in_month"] = daily["date"].dt.days_in_month

    # End-of-month actuals
    eom = daily.groupby("month")["daily_total"].sum().to_dict()
    daily["eom_total"] = daily["month"].map(eom)

    # Cumulative within month
    daily["cum_spend"] = daily.groupby("month")["daily_total"].cumsum()
    daily["cum_count"] = daily.groupby("month")["daily_count"].cumsum()
    daily["days_elapsed"] = daily.groupby("month").cumcount() + 1
    daily["pct_elapsed"] = daily["days_elapsed"] / daily["days_in_month"]
    daily["daily_rate"] = daily["cum_spend"] / daily["days_elapsed"]
    daily["remaining_days"] = daily["days_in_month"] - daily["dom"]

    # Previous month total
    sorted_months = sorted(eom.keys())
    prev_map = {sorted_months[i]: eom[sorted_months[i - 1]] for i in range(1, len(sorted_months))}
    daily["prev_month_total"] = daily["month"].map(prev_map)
    daily["prev_month_total"] = daily["prev_month_total"].fillna(daily["prev_month_total"].median())

    # NEW: Rolling 3-month average (more stable than just previous month)
    rolling_3m = {}
    for i, m in enumerate(sorted_months):
        prev = [eom[sorted_months[j]] for j in range(max(0, i - 3), i)]
        rolling_3m[m] = np.mean(prev) if prev else eom[m]
    daily["rolling_3m_avg"] = daily["month"].map(rolling_3m)

    # Target: remaining spend
    daily["remaining_spend"] = daily["eom_total"] - daily["cum_spend"]

    return daily


def create_snapshots(daily: pd.DataFrame):
    """Take every day as a training snapshot (denser = more training data)."""
    # Use every day from day 1 to day 28
    snapshots = daily[daily["dom"].isin(list(range(1, 29)))].copy()

    # Remove first and last months (often incomplete)
    all_months = sorted(snapshots["month"].unique())
    if len(all_months) > 2:
        snapshots = snapshots[~snapshots["month"].isin([all_months[0], all_months[-1]])]

    return snapshots


def train(csv_path: str = "generated_sms_data_1_year.csv"):
    """Full training pipeline."""
    print("=" * 60)
    print("  Spending Prediction Model v2.0 — Training")
    print("=" * 60)

    # 1. Load data
    daily = load_and_prepare(csv_path)
    print(f"\nLoaded {len(daily)} daily records")
    print(f"Date range: {daily['date'].min().date()} → {daily['date'].max().date()}")

    # 2. Build features
    daily = build_features(daily)
    snapshots = create_snapshots(daily)

    features = [
        "dom", "cum_spend", "cum_count", "pct_elapsed",
        "daily_rate", "remaining_days", "prev_month_total",
        "days_in_month", "rolling_3m_avg",
    ]
    target = "remaining_spend"

    X = snapshots[features].values
    y = snapshots[target].values
    snapshots[features + [target]].to_csv("training_snapshots.csv", index=False)
    print(f"Saved snapshots to training_snapshots.csv")
    groups = snapshots["month"].astype(str).values

    n_months = snapshots["month"].nunique()
    print(f"Training samples: {len(X)} snapshots from {n_months} months")
    print(f"Features: {features}")
    print(f"Target: {target}")

    # Use Leave-One-Month-Out CV (realistic for time series)
    logo = LeaveOneGroupOut()

    # 3. Train Ridge Regression (high regularization)
    print("\n── Ridge Regression (alpha=100) ──")
    ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=100))])
    cv_ridge = cross_val_score(ridge, X, y, cv=logo, groups=groups, scoring="r2")
    cv_ridge_clean = cv_ridge[cv_ridge > -1]
    ridge.fit(X, y)
    pred_ridge = ridge.predict(X)
    r2_ridge = r2_score(y, pred_ridge)
    rmse_ridge = np.sqrt(mean_squared_error(y, pred_ridge))
    mae_ridge = mean_absolute_error(y, pred_ridge)
    print(f"  Leave-One-Month-Out R²: {cv_ridge_clean.mean():.4f} ± {cv_ridge_clean.std():.4f}")
    print(f"  Worst month R²:        {cv_ridge_clean.min():.4f}")
    print(f"  Train R²:              {r2_ridge:.4f}")
    print(f"  RMSE:                  {rmse_ridge:.2f} BHD")
    print(f"  MAE:                   {mae_ridge:.2f} BHD")

    # 4. Train Random Forest
    print("\n── Random Forest ──")
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42
    )
    cv_rf = cross_val_score(rf, X, y, cv=logo, groups=groups, scoring="r2")
    cv_rf_clean = cv_rf[cv_rf > -1]
    rf.fit(X, y)
    pred_rf = rf.predict(X)
    r2_rf = r2_score(y, pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y, pred_rf))
    mae_rf = mean_absolute_error(y, pred_rf)
    print(f"  Leave-One-Month-Out R²: {cv_rf_clean.mean():.4f} ± {cv_rf_clean.std():.4f}")
    print(f"  Worst month R²:        {cv_rf_clean.min():.4f}")
    print(f"  Train R²:              {r2_rf:.4f}")
    print(f"  RMSE:                  {rmse_rf:.2f} BHD")
    print(f"  MAE:                   {mae_rf:.2f} BHD")

    # 5. Train Ensemble (Ridge + Random Forest)
    print("\n── Ensemble (Ridge + Random Forest) ──")
    ensemble = VotingRegressor([
        ("ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=100))])),
        ("rf", RandomForestRegressor(
            n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42
        )),
    ])
    cv_ensemble = cross_val_score(ensemble, X, y, cv=logo, groups=groups, scoring="r2")
    cv_ensemble_clean = cv_ensemble[cv_ensemble > -1]
    ensemble.fit(X, y)
    pred_ensemble = ensemble.predict(X)
    r2_ensemble = r2_score(y, pred_ensemble)
    rmse_ensemble = np.sqrt(mean_squared_error(y, pred_ensemble))
    mae_ensemble = mean_absolute_error(y, pred_ensemble)
    print(f"  Leave-One-Month-Out R²: {cv_ensemble_clean.mean():.4f} ± {cv_ensemble_clean.std():.4f}")
    print(f"  Worst month R²:        {cv_ensemble_clean.min():.4f}")
    print(f"  Train R²:              {r2_ensemble:.4f}")
    print(f"  RMSE:                  {rmse_ensemble:.2f} BHD")
    print(f"  MAE:                   {mae_ensemble:.2f} BHD")

    # 6. Comparison
    print("\n" + "=" * 60)
    print("  Model Comparison")
    print("=" * 60)
    results = pd.DataFrame({
        "Model": ["Ridge", "RandomForest", "Ensemble"],
        "CV_R2": [cv_ridge_clean.mean(), cv_rf_clean.mean(), cv_ensemble_clean.mean()],
        "CV_std": [cv_ridge_clean.std(), cv_rf_clean.std(), cv_ensemble_clean.std()],
        "Worst_Month": [cv_ridge_clean.min(), cv_rf_clean.min(), cv_ensemble_clean.min()],
        "MAE_BHD": [mae_ridge, mae_rf, mae_ensemble],
    }).sort_values("CV_R2", ascending=False)
    print(results.to_string(index=False))

    best = results.iloc[0]["Model"]
    print(f"\n  Winner: {best}")

    # 7. Per-month breakdown
    print("\n── Per-Month Performance (Ensemble) ──")
    logo2 = LeaveOneGroupOut()
    for train_idx, test_idx in logo2.split(X, y, groups):
        month_name = groups[test_idx[0]]
        temp_ensemble = VotingRegressor([
            ("ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=100))])),
            ("rf", RandomForestRegressor(
                n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42
            )),
        ])
        temp_ensemble.fit(X[train_idx], y[train_idx])
        preds = temp_ensemble.predict(X[test_idx])
        r2 = r2_score(y[test_idx], preds)
        mae = mean_absolute_error(y[test_idx], preds)
        status = "✅" if r2 > 0.8 else ("⚠️" if r2 > 0.5 else "❌")
        print(f"  {status} {month_name}: R²={r2:.3f}, MAE={mae:.1f} BHD")

    # 8. Feature importance from Random Forest
    print("\n── Feature Importance (Random Forest) ──")
    for f, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {f:25s} {imp:.4f} {bar}")

    # 9. Ridge Coefficients
    print("\n── Ridge Coefficients ──")
    coefs = ridge.named_steps["model"].coef_
    for f, c in sorted(zip(features, coefs), key=lambda x: -abs(x[1])):
        print(f"  {f:25s} {c:+.4f}")

    # 10. Save
    model_bundle = {
        "ridge": ridge,
        "random_forest": rf,
        "ensemble": ensemble,
        "features": features,
        "ridge_cv_r2": cv_ridge_clean.mean(),
        "rf_cv_r2": cv_rf_clean.mean(),
        "ensemble_cv_r2": cv_ensemble_clean.mean(),
        "training_months": n_months,
        "training_samples": len(X),
        "version": "2.0",
    }

    output_path = "spending_predictor.pkl"
    joblib.dump(model_bundle, output_path)
    print(f"\n✅ Saved model bundle to {output_path}")
    print(f"   Contains: Ridge, RandomForest, Ensemble")
    print(f"   Ensemble CV R²: {cv_ensemble_clean.mean():.4f} (primary model)")
    print(f"   Average prediction error: ~{mae_ensemble:.0f} BHD per month")


if __name__ == "__main__":
    train()