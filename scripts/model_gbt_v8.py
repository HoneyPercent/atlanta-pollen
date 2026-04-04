"""
GBT with V8 feature set — can the nonlinear model find interactions
we haven't manually encoded?

Uses same V8-style features but lets HistGradientBoosting handle
the nonlinearities, interactions, and missing values natively.
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "model_output"


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
                      "precip_yesterday", "precip_2day_sum", "wind_max", "wind_gust",
                      "wind_direction", "vpd_max", "solar_radiation"]:
            r[field] = float(r[field]) if r.get(field) and r[field] else None
        r["missing"] = r.get("missing") == "True"
    return rows


def build_index(rows):
    idx = {}
    for r in rows:
        idx[(r["year"], r["day_of_year"])] = r
    return idx


def extract_features_gbt_v8(row, idx, rows_by_doy):
    """All V8 features but using NaN for missing (GBT handles natively)."""
    if row["total_count"] is None:
        return None
    prev = idx.get((row["year"], row["day_of_year"] - 1))
    prev2 = idx.get((row["year"], row["day_of_year"] - 2))
    prev_log = prev["log_count"] if prev and prev["log_count"] is not None else np.nan
    if np.isnan(prev_log):
        return None
    prev2_log = prev2["log_count"] if prev2 and prev2["log_count"] is not None else np.nan

    temp = row["temp_mean"] if row["temp_mean"] is not None else np.nan
    temp_min = row["temp_min"] if row["temp_min"] is not None else np.nan
    temp_max = row["temp_max"] if row["temp_max"] is not None else np.nan
    progress = row["season_progress_pct"] / 100

    # Window stats
    yr, doy = row["year"], row["day_of_year"]
    temps, precips, pollens = [], [], []
    for d in range(1, 6):
        p = idx.get((yr, doy - d))
        if p:
            if p["temp_mean"] is not None: temps.append(p["temp_mean"])
            if p["precipitation"] is not None: precips.append(p["precipitation"])
            if p["log_count"] is not None: pollens.append(p["log_count"])

    hist = rows_by_doy.get(doy, [])
    temp_anomaly = (temp - statistics.mean(hist)) if hist and not np.isnan(temp) else np.nan

    features = {
        "prev_log": prev_log,
        "prev2_log": prev2_log,
        "d1_d2_diff": (prev_log - prev2_log) if not np.isnan(prev2_log) else np.nan,
        "pollen_5d_max": max(pollens) if pollens else np.nan,
        "pollen_5d_trend": (pollens[0] - pollens[-1]) if len(pollens) >= 2 else np.nan,
        "temp_mean": temp,
        "temp_max": temp_max,
        "temp_min": temp_min,  # NEW in GBT
        "temp_anomaly": temp_anomaly,
        "temp_5d_trend": (temps[0] - temps[-1]) if len(temps) >= 2 else np.nan,
        "temp_stability": -(statistics.stdev(temps) if len(temps) >= 3 else np.nan),
        "vpd_max": row.get("vpd_max") or np.nan,
        "solar_radiation": row.get("solar_radiation") or np.nan,
        "precip_yesterday": row["precip_yesterday"] if row["precip_yesterday"] is not None else np.nan,
        "precip_2day": row["precip_2day_sum"] if row["precip_2day_sum"] is not None else np.nan,
        "precip_5d": sum(precips) if precips else np.nan,
        "dry_days_5d": float(sum(1 for p in precips if p < 0.05)),
        "wind_max": row["wind_max"] if row["wind_max"] is not None else np.nan,
        "wind_dir": row.get("wind_direction") or np.nan,
        "wind_dir_ns": math.cos(math.radians(row["wind_direction"])) if row.get("wind_direction") else np.nan,
        "gdd_daily": row["gdd_daily"] if row["gdd_daily"] is not None else np.nan,
        "gdd_cumulative": row["gdd_cumulative"],
        "doy": float(doy),
        "doy_sin": math.sin(2 * math.pi * doy / 365),
        "season_progress": progress,
        "year": float(row["year"]),
        "burden_log": math.log(row["cumulative_burden"] + 1),
        # Let GBT find the phase interaction itself
        "temp_x_progress": (temp - 50) * progress if not np.isnan(temp) else np.nan,
    }

    return list(features.values()), row["log_count"], list(features.keys())


def main():
    rows = load_features()
    idx = build_index(rows)
    rows_by_doy = defaultdict(list)
    for r in rows:
        if r["year"] <= 2025 and r["temp_mean"] is not None:
            rows_by_doy[r["day_of_year"]].append(r["temp_mean"])

    test_years = list(range(2015, 2026))
    all_a, all_p = [], []
    threshold = math.log(91)

    print("=" * 70)
    print("GBT WITH V8 FEATURE SET")
    print("=" * 70)
    print(f"\n{'Year':>6} {'N':>4} {'MAE':>8} {'RMSE':>8} {'Sev Acc':>8}")
    print("-" * 38)

    for test_yr in test_years:
        train = [r for r in rows if r["year"] < test_yr and 1 <= r["day_of_year"] <= 180]
        test = [r for r in rows if r["year"] == test_yr and 30 <= r["day_of_year"] <= 150]

        tX, ty, fn = [], [], None
        for r in train:
            result = extract_features_gbt_v8(r, idx, rows_by_doy)
            if result:
                feats, target, names = result
                fn = names
                tX.append(feats)
                ty.append(target)

        testX, testy = [], []
        for r in test:
            result = extract_features_gbt_v8(r, idx, rows_by_doy)
            if result:
                feats, target, _ = result
                testX.append(feats)
                testy.append(target)

        if len(tX) < 50 or len(testX) < 10:
            continue

        X_train, y_train = np.array(tX), np.array(ty)
        X_test, y_test = np.array(testX), np.array(testy)

        gbt = HistGradientBoostingRegressor(
            max_iter=300, max_depth=6, min_samples_leaf=8,
            learning_rate=0.08, l2_regularization=0.1,
            random_state=42,
        )
        gbt.fit(X_train, y_train)
        preds = gbt.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = math.sqrt(mean_squared_error(y_test, preds))

        # Severity accuracy
        actual_sev = [0 if y < math.log(101) else 1 if y < math.log(501) else 2 if y < math.log(1501) else 3 for y in y_test]
        pred_sev = [0 if p < math.log(101) else 1 if p < math.log(501) else 2 if p < math.log(1501) else 3 for p in preds]
        sev_acc = sum(1 for a, p in zip(actual_sev, pred_sev) if a == p) / len(actual_sev)

        all_a.extend(y_test.tolist())
        all_p.extend(preds.tolist())

        print(f"{test_yr:>6} {len(y_test):>4} {mae:>8.3f} {rmse:>8.3f} {sev_acc:>7.1%}")

    # Aggregate
    aa, ap = np.array(all_a), np.array(all_p)
    total_mae = mean_absolute_error(aa, ap)
    total_rmse = math.sqrt(mean_squared_error(aa, ap))

    actual_bad = (aa >= threshold).astype(int)
    pred_bad = (ap >= threshold).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    bad_acc = accuracy_score(actual_bad, pred_bad)
    bad_prec = precision_score(actual_bad, pred_bad)
    bad_rec = recall_score(actual_bad, pred_bad)

    print(f"\n{'Model':<25} {'MAE':>7} {'RMSE':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7}")
    print("-" * 60)
    print(f"{'GBT + V8 features':<25} {total_mae:>7.3f} {total_rmse:>7.3f} {bad_acc:>6.1%} {bad_prec:>6.1%} {bad_rec:>6.1%}")
    print(f"{'V8 (linear)':<25} {'0.703':>7} {'0.950':>7} {'84.7%':>7} {'87.0%':>7} {'88.9%':>7}")
    print(f"{'GBT + V5 features':<25} {'0.720':>7} {'0.970':>7} {'85.4%':>7} {'89.7%':>7} {'86.6%':>7}")
    print(f"{'V1 (original)':<25} {'0.822':>7} {'1.078':>7} {'83.3%':>7} {'86.5%':>7} {'86.9%':>7}")

    # Feature importance
    print(f"\nTraining final model for feature importance...")
    all_tX, all_ty = [], []
    for r in rows:
        if r["year"] <= 2025 and 1 <= r["day_of_year"] <= 180:
            result = extract_features_gbt_v8(r, idx, rows_by_doy)
            if result:
                feats, target, _ = result
                all_tX.append(feats)
                all_ty.append(target)

    final = HistGradientBoostingRegressor(
        max_iter=300, max_depth=6, min_samples_leaf=8,
        learning_rate=0.08, l2_regularization=0.1, random_state=42,
    )
    final.fit(np.array(all_tX), np.array(all_ty))
    perm = permutation_importance(final, np.array(all_tX), np.array(all_ty), n_repeats=5, random_state=42)

    print(f"\nTop 15 features (permutation importance):")
    sorted_idx = np.argsort(perm.importances_mean)[::-1]
    for i in sorted_idx[:15]:
        bar = "#" * int(perm.importances_mean[i] * 200)
        print(f"  {perm.importances_mean[i]:.4f}  {fn[i]:<25} {bar}")


if __name__ == "__main__":
    main()
