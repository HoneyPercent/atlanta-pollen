"""
Severity band classifier — predict Low/Moderate/High/Extreme directly.

Instead of predicting log(count) and thresholding, predict the severity band
as an ordinal classification problem. GT capstone found classification
outperformed regression — let's see if that holds for our data.

Uses V8 features but targets severity bands.
Also: predict P(bad day) = P(High or Extreme) as a calibrated probability.
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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


def classify_severity(count):
    if count >= 1500: return 3  # Extreme
    elif count >= 500: return 2  # High
    elif count >= 100: return 1  # Moderate
    else: return 0  # Low

SEVERITY_NAMES = ["Low", "Moderate", "High", "Extreme"]


def classify_regime(r):
    if r["temp_mean"] is None or r["precipitation"] is None or r["wind_max"] is None:
        return np.nan
    t, p, w = r["temp_mean"], r["precipitation"], r["wind_max"]
    if p >= 0.25: return 6
    elif p >= 0.05: return 5
    elif t >= 65 and w >= 10: return 0
    elif t >= 65: return 1
    elif t >= 50 and w >= 10: return 2
    elif t >= 50: return 3
    else: return 4


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
        "pollen_5d_max": max(pollens) if pollens else np.nan,
        "pollen_5d_trend": (pollens[0] - pollens[-1]) if len(pollens) >= 2 else np.nan,
        "dry_days_5d": sum(1 for p in precips if p < 0.05),
        "temp_stability": -(statistics.stdev(temps) if len(temps) >= 3 else 5.0),
        "rain_then_dry": 1.0 if (any(p >= 0.25 for p in precips) and
                                   row["precipitation"] is not None and row["precipitation"] < 0.05) else 0.0,
    }


def extract_features(row, idx, rows_by_doy):
    """Same features as V8."""
    if row["total_count"] is None or row["temp_mean"] is None:
        return None
    prev = idx.get((row["year"], row["day_of_year"] - 1))
    prev2 = idx.get((row["year"], row["day_of_year"] - 2))
    prev_log = prev["log_count"] if prev and prev["log_count"] is not None else np.nan
    prev2_log = prev2["log_count"] if prev2 and prev2["log_count"] is not None else np.nan
    if np.isnan(prev_log):
        return None

    temp = row["temp_mean"]
    temp_min = row["temp_min"] if row["temp_min"] is not None else np.nan
    window = compute_window_stats(row, idx, 5)
    progress = row["season_progress_pct"] / 100
    early_phase = max(0, 1 - progress * 3)
    late_phase = max(0, progress * 1.5 - 0.75)

    doy = row["day_of_year"]
    hist = rows_by_doy.get(doy, [])
    temp_anomaly = (temp - statistics.mean(hist)) if hist else np.nan

    features = [
        prev_log, prev2_log,
        (prev_log - prev2_log) if not np.isnan(prev2_log) else np.nan,
        window["pollen_5d_max"], window["pollen_5d_trend"],
        temp, max(0, temp - 50), temp_anomaly, window["temp_5d_trend"],
        temp_min, 1.0 if (not np.isnan(temp_min) and temp_min >= 55) else 0.0,
        max(0, temp - 50) * early_phase, max(0, temp - 50) * late_phase,
        window["temp_stability"],
        math.cos(math.radians(row["wind_direction"])) if row.get("wind_direction") is not None else np.nan,
        row.get("solar_radiation") or np.nan,
        row["precip_yesterday"] if row["precip_yesterday"] is not None else np.nan,
        row["precip_2day_sum"] if row["precip_2day_sum"] is not None else np.nan,
        float(window["dry_days_5d"]), window["rain_then_dry"],
        row["wind_max"] if row["wind_max"] is not None else np.nan,
        row["gdd_daily"] if row["gdd_daily"] is not None else np.nan,
        1.0 if row["gdd_cumulative"] >= 200 else 0.0,
        math.sin(2 * math.pi * doy / 365), math.cos(2 * math.pi * doy / 365),
        progress, float(row["year"]),
        math.log(row["cumulative_burden"] + 1),
        classify_regime(row),
        classify_regime(prev) if prev else np.nan,
    ]

    feature_names = [
        "prev_log", "prev2_log", "d1_d2_diff",
        "pollen_5d_max", "pollen_5d_trend",
        "temp_mean", "temp_above_50", "temp_anomaly", "temp_5d_trend",
        "temp_min", "warm_night", "temp_x_early", "temp_x_late",
        "temp_stability", "wind_dir_ns", "solar_radiation",
        "precip_yesterday", "precip_2day_sum", "dry_days_5d", "rain_then_dry",
        "wind_max", "gdd_daily", "gdd_armed",
        "doy_sin", "doy_cos", "season_progress", "year",
        "cumulative_burden_log", "regime_today", "regime_yesterday",
    ]

    severity = classify_severity(row["total_count"])
    return features, severity, feature_names


def main():
    rows = load_features()
    idx = build_index(rows)
    rows_by_doy = defaultdict(list)
    for r in rows:
        if r["year"] <= 2025 and r["temp_mean"] is not None:
            rows_by_doy[r["day_of_year"]].append(r["temp_mean"])

    test_years = list(range(2015, 2026))

    print("=" * 70)
    print("SEVERITY CLASSIFIER — PREDICT LOW/MODERATE/HIGH/EXTREME DIRECTLY")
    print("=" * 70)

    all_actual, all_pred, all_proba = [], [], []

    print(f"\n{'Year':>6} {'N':>4} {'Accuracy':>9} {'Bad-Day Acc':>12}")
    print("-" * 35)

    for test_yr in test_years:
        train = [r for r in rows if r["year"] < test_yr and 1 <= r["day_of_year"] <= 180]
        test = [r for r in rows if r["year"] == test_yr and 30 <= r["day_of_year"] <= 150]

        tX, ty = [], []
        fn = None
        for r in train:
            result = extract_features(r, idx, rows_by_doy)
            if result:
                feats, severity, feature_names = result
                fn = feature_names
                tX.append(feats)
                ty.append(severity)

        testX, testy = [], []
        for r in test:
            result = extract_features(r, idx, rows_by_doy)
            if result:
                feats, severity, _ = result
                testX.append(feats)
                testy.append(severity)

        if len(tX) < 50 or len(testX) < 10:
            continue

        X_train, y_train = np.array(tX), np.array(ty)
        X_test, y_test = np.array(testX), np.array(testy)

        # Train classifier
        clf = HistGradientBoostingClassifier(
            max_iter=200, max_depth=5, min_samples_leaf=10,
            learning_rate=0.1, random_state=42,
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        probas = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, preds)

        # "Bad day" = High or Extreme (severity >= 2)
        actual_bad = (y_test >= 2).astype(int)
        pred_bad = (preds >= 2).astype(int)
        bad_acc = accuracy_score(actual_bad, pred_bad)

        all_actual.extend(y_test.tolist())
        all_pred.extend(preds.tolist())
        all_proba.extend(probas.tolist())

        print(f"{test_yr:>6} {len(y_test):>4} {acc:>8.1%} {bad_acc:>11.1%}")

    # Aggregate
    all_a = np.array(all_actual)
    all_p = np.array(all_pred)

    print(f"\n{'='*70}")
    print(f"AGGREGATE RESULTS (2015-2025)")
    print(f"{'='*70}")

    total_acc = accuracy_score(all_a, all_p)
    print(f"\n4-class accuracy: {total_acc:.1%}")

    # Bad-day (High+Extreme) accuracy
    actual_bad = (all_a >= 2).astype(int)
    pred_bad = (all_p >= 2).astype(int)
    from sklearn.metrics import precision_score, recall_score
    bad_acc = accuracy_score(actual_bad, pred_bad)
    bad_prec = precision_score(actual_bad, pred_bad)
    bad_rec = recall_score(actual_bad, pred_bad)
    print(f"Bad-day (High+Extreme) accuracy: {bad_acc:.1%}, precision: {bad_prec:.1%}, recall: {bad_rec:.1%}")

    # Extreme-day accuracy
    actual_ext = (all_a >= 3).astype(int)
    pred_ext = (all_p >= 3).astype(int)
    ext_acc = accuracy_score(actual_ext, pred_ext)
    ext_prec = precision_score(actual_ext, pred_ext, zero_division=0)
    ext_rec = recall_score(actual_ext, pred_ext, zero_division=0)
    print(f"Extreme-day accuracy: {ext_acc:.1%}, precision: {ext_prec:.1%}, recall: {ext_rec:.1%}")

    # Confusion matrix
    print(f"\nConfusion matrix:")
    cm = confusion_matrix(all_a, all_p)
    print(f"{'':>12} {'Pred Low':>10} {'Pred Mod':>10} {'Pred High':>10} {'Pred Ext':>10}")
    for i, name in enumerate(SEVERITY_NAMES):
        print(f"{'Act '+name:>12}", end="")
        for j in range(4):
            print(f"{cm[i][j]:>10}", end="")
        print()

    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for i, name in enumerate(SEVERITY_NAMES):
        class_total = sum(cm[i])
        class_correct = cm[i][i]
        if class_total > 0:
            print(f"  {name}: {class_correct}/{class_total} = {100*class_correct/class_total:.0f}%")

    # Comparison with V8 regression thresholded
    print(f"\n{'='*70}")
    print(f"COMPARISON: CLASSIFIER vs V8 REGRESSION (thresholded)")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Classifier':>12} {'V8 Regression':>14}")
    print("-" * 56)
    print(f"{'4-class severity accuracy':<30} {total_acc:>11.1%} {'~50-55%':>14}")
    print(f"{'Bad-day accuracy':<30} {bad_acc:>11.1%} {'84.7%':>14}")
    print(f"{'Bad-day precision':<30} {bad_prec:>11.1%} {'87.0%':>14}")
    print(f"{'Bad-day recall':<30} {bad_rec:>11.1%} {'88.9%':>14}")
    print(f"{'Extreme-day precision':<30} {ext_prec:>11.1%} {'--':>14}")
    print(f"{'Extreme-day recall':<30} {ext_rec:>11.1%} {'--':>14}")

    # Calibration: when classifier says P(bad) > 0.7, how often is it actually bad?
    print(f"\nCalibration check — P(High+Extreme) predictions:")
    all_proba_arr = np.array(all_proba)
    p_bad = all_proba_arr[:, 2] + all_proba_arr[:, 3]  # P(High) + P(Extreme)

    for threshold in [0.3, 0.5, 0.7, 0.9]:
        predicted_bad = p_bad >= threshold
        if predicted_bad.sum() > 0:
            actual_rate = actual_bad[predicted_bad].mean()
            print(f"  When P(bad) >= {threshold:.0%}: predicted {predicted_bad.sum()} days, "
                  f"actually bad {100*actual_rate:.0f}% of the time")


if __name__ == "__main__":
    main()
