"""
Gradient Boosted Trees model for Atlanta pollen.

Uses scikit-learn's HistGradientBoostingRegressor which supports:
- Native missing value handling
- Poisson loss (for count data)
- Quantile loss (for uncertainty intervals)
- Monotonic constraints
- Fast training on moderate datasets

This is the first nonlinear model — should capture interactions
that linear regression misses (e.g., temp × wind, dry spell × warming trend).

Compares GBT against V5 (best linear) on same evaluation framework.
Also builds a GBT version of the direct remaining-days model.
"""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, recall_score, precision_score

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "model_output"

REGIMES = ["hot_windy", "hot_calm", "warm_windy", "warm_calm", "cool", "drizzle", "rainy"]


def load_features():
    with open(DATA_DIR / "features_daily.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["year"] = int(r["year"])
        r["day_of_year"] = int(r["day_of_year"])
        r["total_count"] = int(r["total_count"]) if r["total_count"] else None
        r["log_count"] = float(r["log_count"]) if r["log_count"] else None
        r["cumulative_burden"] = float(r["cumulative_burden"])
        r["gdd_cumulative"] = float(r["gdd_cumulative"])
        r["season_progress_pct"] = float(r["season_progress_pct"]) if r.get("season_progress_pct") else 0
        for field in ["gdd_daily", "temp_max", "temp_min", "temp_mean", "precipitation",
                      "precip_yesterday", "precip_2day_sum", "wind_max", "wind_gust"]:
            r[field] = float(r[field]) if r.get(field) and r[field] else None
        r["missing"] = r.get("missing") == "True"
    return rows


def build_index(rows):
    idx = {}
    for r in rows:
        idx[(r["year"], r["day_of_year"])] = r
    return idx


def classify_regime(r):
    if r["temp_mean"] is None or r["precipitation"] is None or r["wind_max"] is None:
        return None
    t, p, w = r["temp_mean"], r["precipitation"], r["wind_max"]
    if p >= 0.25: return 6  # rainy
    elif p >= 0.05: return 5  # drizzle
    elif t >= 65 and w >= 10: return 0  # hot_windy
    elif t >= 65: return 1  # hot_calm
    elif t >= 50 and w >= 10: return 2  # warm_windy
    elif t >= 50: return 3  # warm_calm
    else: return 4  # cool


def compute_window_stats(row, idx, lookback=5):
    yr, doy = row["year"], row["day_of_year"]
    temps, precips, pollens = [], [], []
    for d in range(1, lookback + 1):
        prev = idx.get((yr, doy - d))
        if prev:
            if prev["temp_mean"] is not None: temps.append(prev["temp_mean"])
            if prev["precipitation"] is not None: precips.append(prev["precipitation"])
            if prev["log_count"] is not None: pollens.append(prev["log_count"])

    return {
        "temp_5d_trend": (temps[0] - temps[-1]) if len(temps) >= 2 else np.nan,
        "precip_5d_total": sum(precips) if precips else np.nan,
        "pollen_5d_max": max(pollens) if pollens else np.nan,
        "pollen_5d_trend": (pollens[0] - pollens[-1]) if len(pollens) >= 2 else np.nan,
        "dry_days_5d": sum(1 for p in precips if p < 0.05),
    }


def extract_features_gbt(row, idx, rows_by_doy):
    """Feature extraction for GBT — can use NaN for missing (native handling)."""
    if row["total_count"] is None:
        return None

    prev = idx.get((row["year"], row["day_of_year"] - 1))
    prev2 = idx.get((row["year"], row["day_of_year"] - 2))
    prev_log = prev["log_count"] if prev and prev["log_count"] is not None else np.nan
    prev2_log = prev2["log_count"] if prev2 and prev2["log_count"] is not None else np.nan

    if np.isnan(prev_log):
        return None

    window = compute_window_stats(row, idx, 5)
    regime = classify_regime(row) if row["temp_mean"] is not None else np.nan
    prev_regime = classify_regime(prev) if prev and prev["temp_mean"] is not None else np.nan

    temp = row["temp_mean"] if row["temp_mean"] is not None else np.nan

    # Temperature anomaly
    doy = row["day_of_year"]
    hist_temps = rows_by_doy.get(doy, [])
    temp_anomaly = (temp - statistics.mean(hist_temps)) if hist_temps and not np.isnan(temp) else np.nan

    features = {
        # Autoregressive
        "prev_log_count": prev_log,
        "prev2_log_count": prev2_log,
        "d1_d2_diff": (prev_log - prev2_log) if not np.isnan(prev2_log) else np.nan,
        "pollen_5d_max": window["pollen_5d_max"],
        "pollen_5d_trend": window["pollen_5d_trend"],

        # Temperature
        "temp_mean": temp,
        "temp_max": row["temp_max"] if row["temp_max"] is not None else np.nan,
        "temp_anomaly": temp_anomaly,
        "temp_5d_trend": window["temp_5d_trend"],

        # Precipitation
        "precip_yesterday": row["precip_yesterday"] if row["precip_yesterday"] is not None else np.nan,
        "precip_2day_sum": row["precip_2day_sum"] if row["precip_2day_sum"] is not None else np.nan,
        "precip_5d_total": window["precip_5d_total"],
        "dry_days_5d": float(window["dry_days_5d"]),

        # Wind
        "wind_max": row["wind_max"] if row["wind_max"] is not None else np.nan,

        # GDD
        "gdd_daily": row["gdd_daily"] if row["gdd_daily"] is not None else np.nan,
        "gdd_cumulative": row["gdd_cumulative"],

        # Seasonal
        "day_of_year": float(row["day_of_year"]),
        "season_progress": row["season_progress_pct"],

        # Regime (numeric encoding — GBT handles ordinal fine)
        "regime_today": float(regime) if not np.isnan(regime) else np.nan,
        "regime_yesterday": float(prev_regime) if not np.isnan(prev_regime) else np.nan,

        # Trend
        "year": float(row["year"]),

        # Cumulative burden (season state)
        "cumulative_burden_log": math.log(row["cumulative_burden"] + 1),
    }

    return features, row["log_count"]


def compute_year_stats(rows):
    stats = {}
    for yr in set(r["year"] for r in rows):
        yr_rows = [r for r in rows if r["year"] == yr and r["total_count"] is not None]
        if not yr_rows:
            continue
        total = sum(r["total_count"] for r in yr_rows)
        extreme = sum(1 for r in yr_rows if r["total_count"] >= 1500)
        stats[yr] = {"total_burden": total, "extreme_days": extreme}
    return stats


def compute_remaining(rows, year, from_doy, threshold=100):
    return sum(1 for r in rows if r["year"] == year and r["day_of_year"] > from_doy
               and r["total_count"] is not None and r["total_count"] >= threshold)


def main():
    rows = load_features()
    idx = build_index(rows)
    year_stats = compute_year_stats(rows)

    rows_by_doy = defaultdict(list)
    for r in rows:
        if r["year"] <= 2025 and r["temp_mean"] is not None:
            rows_by_doy[r["day_of_year"]].append(r["temp_mean"])

    test_years = list(range(2015, 2026))

    # ================================================================
    # PART 1: GBT for daily pollen prediction
    # ================================================================
    print("=" * 70)
    print("GRADIENT BOOSTED TREES — DAILY POLLEN PREDICTION")
    print("=" * 70)

    all_actuals, all_preds = [], []
    print(f"\n{'Year':>6} {'N':>4} {'MAE':>8} {'RMSE':>8} {'Acc':>7} {'Recall':>7}")
    print("-" * 42)

    threshold = math.log(91)

    for test_yr in test_years:
        train = [r for r in rows if r["year"] < test_yr and 1 <= r["day_of_year"] <= 180]
        test = [r for r in rows if r["year"] == test_yr and 30 <= r["day_of_year"] <= 150]

        # Build arrays
        tX, ty = [], []
        fn = None
        for r in train:
            result = extract_features_gbt(r, idx, rows_by_doy)
            if result:
                feats, target = result
                if fn is None:
                    fn = list(feats.keys())
                tX.append([feats[f] for f in fn])
                ty.append(target)

        testX, testy, test_actual_counts = [], [], []
        for r in test:
            result = extract_features_gbt(r, idx, rows_by_doy)
            if result:
                feats, target = result
                testX.append([feats[f] for f in fn])
                testy.append(target)
                test_actual_counts.append(r["total_count"])

        if len(tX) < 50 or len(testX) < 10:
            continue

        X_train = np.array(tX)
        y_train = np.array(ty)
        X_test = np.array(testX)
        y_test = np.array(testy)

        # Train GBT
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=200,
            max_depth=6,
            min_samples_leaf=10,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = math.sqrt(mean_squared_error(y_test, preds))

        # Bad-day classification
        actual_bad = (y_test >= threshold).astype(int)
        pred_bad = (preds >= threshold).astype(int)
        acc = accuracy_score(actual_bad, pred_bad)
        rec = recall_score(actual_bad, pred_bad, zero_division=0)

        all_actuals.extend(y_test.tolist())
        all_preds.extend(preds.tolist())

        print(f"{test_yr:>6} {len(y_test):>4} {mae:>8.3f} {rmse:>8.3f} {acc:>6.1%} {rec:>6.1%}")

    # Aggregate
    all_a = np.array(all_actuals)
    all_p = np.array(all_preds)
    total_mae = mean_absolute_error(all_a, all_p)
    total_rmse = math.sqrt(mean_squared_error(all_a, all_p))
    total_bad_a = (all_a >= threshold).astype(int)
    total_bad_p = (all_p >= threshold).astype(int)
    total_acc = accuracy_score(total_bad_a, total_bad_p)
    total_prec = precision_score(total_bad_a, total_bad_p, zero_division=0)
    total_rec = recall_score(total_bad_a, total_bad_p, zero_division=0)

    print(f"\n{'Model':<25} {'MAE':>7} {'RMSE':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7}")
    print("-" * 60)
    print(f"{'GBT (nonlinear)':<25} {total_mae:>7.3f} {total_rmse:>7.3f} {total_acc:>6.1%} {total_prec:>6.1%} {total_rec:>6.1%}")
    print(f"{'V5 (best linear)':<25} {'0.724':>7} {'0.967':>7} {'85.3%':>7} {'86.8%':>7} {'90.3%':>7}")
    print(f"{'V1 (original linear)':<25} {'0.822':>7} {'1.078':>7} {'83.3%':>7} {'86.5%':>7} {'86.9%':>7}")

    # Feature importance (from final model trained on all data)
    print(f"\nTraining final GBT on all data through 2025...")
    all_train_X, all_train_y = [], []
    fn = None
    for r in rows:
        if r["year"] <= 2025 and 1 <= r["day_of_year"] <= 180:
            result = extract_features_gbt(r, idx, rows_by_doy)
            if result:
                feats, target = result
                if fn is None:
                    fn = list(feats.keys())
                all_train_X.append([feats[f] for f in fn])
                all_train_y.append(target)

    final_model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=200, max_depth=6,
        min_samples_leaf=10, learning_rate=0.1, random_state=42,
    )
    final_model.fit(np.array(all_train_X), np.array(all_train_y))

    # Use permutation importance since HistGBT doesn't have feature_importances_
    from sklearn.inspection import permutation_importance
    X_all = np.array(all_train_X)
    y_all = np.array(all_train_y)
    perm_result = permutation_importance(final_model, X_all, y_all, n_repeats=5, random_state=42)
    importances = perm_result.importances_mean
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\nTop 15 features (GBT permutation importance):")
    for i in sorted_idx[:15]:
        bar = "#" * int(importances[i] * 200)
        print(f"  {importances[i]:.4f}  {fn[i]:<25} {bar}")

    # ================================================================
    # PART 2: GBT for direct remaining-days prediction
    # ================================================================
    print(f"\n{'='*70}")
    print(f"GBT — DIRECT REMAINING-DAYS PREDICTION")
    print(f"{'='*70}")

    start_dates = [(32, "Feb 1"), (60, "Mar 1"), (74, "Mar 15"), (91, "Apr 1"), (105, "Apr 15")]

    for start_doy, start_label in start_dates:
        gbt_errors = []
        linear_errors = []

        for test_yr in test_years:
            tX, ty = [], []

            for yr in sorted(year_stats.keys()):
                if yr >= test_yr or yr < 1993:
                    continue
                row = idx.get((yr, start_doy))
                if not row or row["total_count"] is None:
                    continue
                feats_result = extract_features_gbt(row, idx, rows_by_doy)
                if not feats_result:
                    continue
                feats, _ = feats_result
                remaining = compute_remaining(rows, yr, start_doy, 100)
                if fn is None:
                    fn = list(feats.keys())
                tX.append([feats[f] for f in fn])
                ty.append(float(remaining))

            if len(tX) < 8:
                continue

            # GBT
            gbt = HistGradientBoostingRegressor(
                max_iter=100, max_depth=4, min_samples_leaf=3,
                learning_rate=0.1, random_state=42,
            )
            gbt.fit(np.array(tX), np.array(ty))

            # Predict test year
            test_row = idx.get((test_yr, start_doy))
            if not test_row or test_row["total_count"] is None:
                continue
            test_feats = extract_features_gbt(test_row, idx, rows_by_doy)
            if not test_feats:
                continue
            feats, _ = test_feats
            pred = max(0, gbt.predict(np.array([[feats[f] for f in fn]]))[0])
            actual = compute_remaining(rows, test_yr, start_doy, 100)
            gbt_errors.append(abs(pred - actual))

        if gbt_errors:
            print(f"  From {start_label}: GBT MAE = {statistics.mean(gbt_errors):.1f} days >100")


if __name__ == "__main__":
    main()
