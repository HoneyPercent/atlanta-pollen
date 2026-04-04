"""
Comprehensive hindcast validation across multiple years and start dates.

Tests V3 model + analog projections from 6 different start dates across
11 test years (2015-2025) = 66 hindcast scenarios.

For each scenario: predict remaining extreme days, remaining days >100,
and 14-day severity accuracy. Compare V3 vs V1 vs analog-only.
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


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
            r[field] = float(r[field]) if r.get(field) else None
        r["missing"] = r.get("missing") == "True"
    return rows


def build_index(rows):
    idx = {}
    for r in rows:
        idx[(r["year"], r["day_of_year"])] = r
    return idx


REGIMES = ["hot_windy", "hot_calm", "warm_windy", "warm_calm", "cool", "drizzle", "rainy"]

def classify_regime(r):
    if r["temp_mean"] is None or r["precipitation"] is None or r["wind_max"] is None:
        return None
    t, p, w = r["temp_mean"], r["precipitation"], r["wind_max"]
    if p >= 0.25: return "rainy"
    elif p >= 0.05: return "drizzle"
    elif t >= 65 and w >= 10: return "hot_windy"
    elif t >= 65: return "hot_calm"
    elif t >= 50 and w >= 10: return "warm_windy"
    elif t >= 50: return "warm_calm"
    else: return "cool"


def regime_features(regime_name, prefix="regime"):
    return {f"{prefix}_{r}": (1.0 if regime_name == r else 0.0) for r in REGIMES}


def compute_consecutive_dry(row, idx):
    yr, doy = row["year"], row["day_of_year"]
    if row["precipitation"] is not None and row["precipitation"] >= 0.1:
        return 0
    count = 1
    for d in range(1, 10):
        prev = idx.get((yr, doy - d))
        if prev and prev["precipitation"] is not None and prev["precipitation"] < 0.1:
            count += 1
        else:
            break
    return count


def extract_features_v3(row, idx):
    if row["total_count"] is None or row["temp_mean"] is None:
        return None
    prev = idx.get((row["year"], row["day_of_year"] - 1))
    prev2 = idx.get((row["year"], row["day_of_year"] - 2))
    prev_log = prev["log_count"] if prev and prev["log_count"] is not None else None
    prev2_log = prev2["log_count"] if prev2 and prev2["log_count"] is not None else None
    if prev_log is None:
        return None
    regime = classify_regime(row)
    prev_regime = classify_regime(prev) if prev else None
    if regime is None:
        return None
    temp = row["temp_mean"]
    consec_dry = compute_consecutive_dry(row, idx)
    features = {
        "prev_log_count": prev_log,
        "prev2_log_count": prev2_log if prev2_log is not None else prev_log,
        "d1_d2_diff": (prev_log - prev2_log) if prev2_log is not None else 0,
        "temp_mean": temp,
        "temp_above_50": max(0, temp - 50),
        "precip_yesterday": row["precip_yesterday"] or 0,
        "precip_2day_sum": row["precip_2day_sum"] or 0,
        "wind_max": row["wind_max"] or 0,
        "gdd_daily": row["gdd_daily"] or 0,
        "doy_sin": math.sin(2 * math.pi * row["day_of_year"] / 365),
        "doy_cos": math.cos(2 * math.pi * row["day_of_year"] / 365),
        "consec_dry_days": float(min(consec_dry, 7)),
        "year_trend": float(row["year"] - 2000) / 25,
        "season_progress": row["season_progress_pct"] / 100,
    }
    features.update(regime_features(regime, "today"))
    if prev_regime:
        features.update(regime_features(prev_regime, "yest"))
    else:
        features.update(regime_features("cool", "yest"))
    features["yest_was_rainy"] = 1.0 if (prev and prev["precipitation"] is not None and prev["precipitation"] >= 0.25) else 0.0
    return features, row["log_count"]


class WeightedLinearRegression:
    def __init__(self):
        self.weights = self.bias = self.feature_names = self.means = self.stds = None

    def fit(self, X, y, feature_names, sample_weights=None):
        self.feature_names = feature_names
        n, k = len(y), len(feature_names)
        if sample_weights is None:
            sample_weights = [1.0] * n
        self.means = [statistics.mean(X[i][j] for i in range(n)) for j in range(k)]
        self.stds = []
        for j in range(k):
            vals = [X[i][j] for i in range(n)]
            s = statistics.stdev(vals) if len(set(vals)) > 1 else 1
            self.stds.append(s if s > 0 else 1)
        X_std = [[((X[i][j] - self.means[j]) / self.stds[j]) for j in range(k)] for i in range(n)]
        weights = [0.0] * k
        bias = sum(w * yi for w, yi in zip(sample_weights, y)) / sum(sample_weights)
        lr, w_sum = 0.008, sum(sample_weights)
        for _ in range(1000):
            preds = [bias + sum(weights[j] * X_std[i][j] for j in range(k)) for i in range(n)]
            errors = [preds[i] - y[i] for i in range(n)]
            grad_bias = sum(sample_weights[i] * errors[i] for i in range(n)) / w_sum
            grad_w = [sum(sample_weights[i] * errors[i] * X_std[i][j] for i in range(n)) / w_sum for j in range(k)]
            bias -= lr * grad_bias
            for j in range(k):
                weights[j] -= lr * grad_w[j]
        self.weights, self.bias = weights, bias

    def predict(self, x_raw):
        x_std = [(x_raw[j] - self.means[j]) / self.stds[j] for j in range(len(self.feature_names))]
        return self.bias + sum(self.weights[j] * x_std[j] for j in range(len(self.feature_names)))


def classify_severity(count):
    if count >= 1500: return "E"
    elif count >= 500: return "H"
    elif count >= 100: return "M"
    else: return "L"


def doy_to_date(doy, year=2025):
    from datetime import date, timedelta
    return (date(year, 1, 1) + timedelta(days=doy - 1)).strftime("%b %d")


def main():
    rows = load_features()
    idx = build_index(rows)

    test_years = list(range(2015, 2026))
    start_dates = [
        (32, "Feb 1"),
        (46, "Feb 15"),
        (60, "Mar 1"),
        (74, "Mar 15"),
        (91, "Apr 1"),
        (105, "Apr 15"),
    ]

    print("=" * 80)
    print("COMPREHENSIVE HINDCAST: V3 + ANALOG across 11 years x 6 start dates")
    print("=" * 80)

    # Collect results for summary tables
    analog_ext_errors = defaultdict(list)  # keyed by start_label
    analog_100_errors = defaultdict(list)
    v3_sev_accuracy = defaultdict(list)

    for test_yr in test_years:
        # Train V3 on all prior years
        train_rows = [r for r in rows if r["year"] < test_yr and 1 <= r["day_of_year"] <= 180]
        train_X, train_y, train_w = [], [], []
        feature_names = None
        for r in train_rows:
            result = extract_features_v3(r, idx)
            if result:
                feats, target = result
                if feature_names is None:
                    feature_names = list(feats.keys())
                train_X.append([feats[f] for f in feature_names])
                train_y.append(target)
                train_w.append(math.exp(-0.03 * (test_yr - r["year"])))

        if len(train_X) < 50:
            continue

        model = WeightedLinearRegression()
        model.fit(train_X, train_y, feature_names, train_w)

        prior_years = sorted(set(r["year"] for r in rows if r["year"] < test_yr))

        for start_doy, start_label in start_dates:
            # Get test year data at cutoff
            at_cutoff = [r for r in rows if r["year"] == test_yr
                         and r["day_of_year"] <= start_doy and r["total_count"] is not None]
            after_cutoff = [r for r in rows if r["year"] == test_yr
                           and r["day_of_year"] > start_doy and r["total_count"] is not None]

            if not at_cutoff:
                continue

            burden = at_cutoff[-1]["cumulative_burden"]
            last_log = at_cutoff[-1]["log_count"]

            # Actual outcomes
            actual_ext = sum(1 for r in after_cutoff if r["total_count"] >= 1500)
            actual_100 = sum(1 for r in after_cutoff if r["total_count"] >= 100)

            # ANALOG projection
            analogs = []
            for yr in prior_years:
                yr_at = [r for r in rows if r["year"] == yr and r["day_of_year"] <= start_doy]
                yr_after = [r for r in rows if r["year"] == yr and r["day_of_year"] > start_doy
                            and r["total_count"] is not None]
                if not yr_at:
                    continue
                b = yr_at[-1]["cumulative_burden"]
                ext = sum(1 for r in yr_after if r["total_count"] >= 1500)
                o100 = sum(1 for r in yr_after if r["total_count"] >= 100)
                analogs.append({"burden": b, "ext": ext, "o100": o100})

            analogs.sort(key=lambda a: abs(a["burden"] - burden))
            top5 = analogs[:5]
            if top5:
                pred_ext = statistics.mean(a["ext"] for a in top5)
                pred_100 = statistics.mean(a["o100"] for a in top5)
                analog_ext_errors[start_label].append(abs(pred_ext - actual_ext))
                analog_100_errors[start_label].append(abs(pred_100 - actual_100))

            # V3 14-day severity forecast
            prev_pred = last_log
            sev_correct = 0
            sev_total = 0
            for h in range(1, 15):
                tdoy = start_doy + h
                ar = idx.get((test_yr, tdoy))
                if not ar or ar["total_count"] is None or ar["temp_mean"] is None:
                    continue

                result = extract_features_v3(ar, idx)
                if not result:
                    continue
                feats, actual = result
                # Substitute chained prediction for prev counts
                feats["prev_log_count"] = prev_pred
                pred = model.predict([feats[f] for f in feature_names])
                pc = max(0, round(math.exp(pred) - 1))

                if classify_severity(pc) == classify_severity(ar["total_count"]):
                    sev_correct += 1
                sev_total += 1
                prev_pred = pred

            if sev_total > 0:
                v3_sev_accuracy[start_label].append(sev_correct / sev_total * 100)

    # ======================== SUMMARY TABLES ========================
    print(f"\n{'='*80}")
    print(f"ANALOG MODEL: MAE for 'remaining extreme days' by start date")
    print(f"{'='*80}")
    print(f"{'Start Date':<12} {'N scenarios':>12} {'MAE Extreme':>12} {'MAE >100':>10}")
    print("-" * 46)
    for _, label in start_dates:
        if analog_ext_errors[label]:
            n = len(analog_ext_errors[label])
            mae_ext = statistics.mean(analog_ext_errors[label])
            mae_100 = statistics.mean(analog_100_errors[label])
            print(f"{label:<12} {n:>12} {mae_ext:>12.1f} {mae_100:>10.1f}")

    print(f"\n{'='*80}")
    print(f"V3 REGRESSION: 14-day severity accuracy by start date")
    print(f"{'='*80}")
    print(f"{'Start Date':<12} {'N scenarios':>12} {'Mean Accuracy':>14} {'Min':>8} {'Max':>8}")
    print("-" * 54)
    for _, label in start_dates:
        if v3_sev_accuracy[label]:
            accs = v3_sev_accuracy[label]
            print(f"{label:<12} {len(accs):>12} {statistics.mean(accs):>13.1f}% "
                  f"{min(accs):>7.0f}% {max(accs):>7.0f}%")

    # Combined table: how does performance change through the season?
    print(f"\n{'='*80}")
    print(f"PERFORMANCE THROUGH THE SEASON (both models)")
    print(f"{'='*80}")
    print(f"{'Start':<10} {'Analog Ext MAE':>15} {'Analog >100 MAE':>16} {'V3 14d Sev Acc':>15}")
    print("-" * 56)
    for _, label in start_dates:
        ext = f"{statistics.mean(analog_ext_errors[label]):.1f}" if analog_ext_errors[label] else "n/a"
        o100 = f"{statistics.mean(analog_100_errors[label]):.1f}" if analog_100_errors[label] else "n/a"
        v3 = f"{statistics.mean(v3_sev_accuracy[label]):.0f}%" if v3_sev_accuracy[label] else "n/a"
        print(f"{label:<10} {ext:>15} {o100:>16} {v3:>15}")

    # Worst-case analysis
    print(f"\n{'='*80}")
    print(f"WORST-CASE SCENARIOS (largest analog errors)")
    print(f"{'='*80}")
    all_errors = []
    for test_yr in test_years:
        for start_doy, start_label in start_dates:
            at_cutoff = [r for r in rows if r["year"] == test_yr
                         and r["day_of_year"] <= start_doy and r["total_count"] is not None]
            after_cutoff = [r for r in rows if r["year"] == test_yr
                           and r["day_of_year"] > start_doy and r["total_count"] is not None]
            if not at_cutoff:
                continue
            burden = at_cutoff[-1]["cumulative_burden"]
            actual_ext = sum(1 for r in after_cutoff if r["total_count"] >= 1500)

            prior_years = sorted(set(r["year"] for r in rows if r["year"] < test_yr))
            analogs = []
            for yr in prior_years:
                yr_at = [r for r in rows if r["year"] == yr and r["day_of_year"] <= start_doy]
                yr_after = [r for r in rows if r["year"] == yr and r["day_of_year"] > start_doy
                            and r["total_count"] is not None]
                if not yr_at:
                    continue
                b = yr_at[-1]["cumulative_burden"]
                ext = sum(1 for r in yr_after if r["total_count"] >= 1500)
                analogs.append({"burden": b, "ext": ext})
            analogs.sort(key=lambda a: abs(a["burden"] - burden))
            top5 = analogs[:5]
            if top5:
                pred_ext = statistics.mean(a["ext"] for a in top5)
                err = abs(pred_ext - actual_ext)
                all_errors.append((test_yr, start_label, pred_ext, actual_ext, err))

    all_errors.sort(key=lambda x: x[4], reverse=True)
    print(f"{'Year':>6} {'Start':>10} {'Predicted':>10} {'Actual':>8} {'Error':>8}")
    print("-" * 42)
    for yr, label, pred, actual, err in all_errors[:10]:
        print(f"{yr:>6} {label:>10} {pred:>10.1f} {actual:>8} {err:>8.1f}")


if __name__ == "__main__":
    main()
