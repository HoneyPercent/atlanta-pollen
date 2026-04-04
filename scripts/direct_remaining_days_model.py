"""
Direct remaining-days model — predicts "remaining days >100" as the target.

Both ChatGPT responses' #1 recommendation: model the product output directly
instead of deriving it from daily count predictions.

Target: remaining_days_over_100 (from this DOY to end of season)
Features: current season state + weather + prior year metrics

Also adds prior-year features per biology research:
- Prior year's total burden, extreme days, season end DOY
- Prior fall (Sep-Nov) precipitation (predicts pine intensity)
- Prior spring (Mar-Apr) precipitation (predicts oak intensity)
"""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

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
                      "precip_yesterday", "precip_2day_sum", "wind_max", "wind_gust"]:
            r[field] = float(r[field]) if r.get(field) and r[field] else None
        r["missing"] = r.get("missing") == "True"
    return rows


def build_index(rows):
    idx = {}
    for r in rows:
        idx[(r["year"], r["day_of_year"])] = r
    return idx


def compute_year_stats(rows):
    """Pre-compute per-year stats for prior-year features."""
    stats = {}
    for yr in set(r["year"] for r in rows):
        yr_rows = [r for r in rows if r["year"] == yr and r["total_count"] is not None]
        if not yr_rows:
            continue

        total = sum(r["total_count"] for r in yr_rows)
        extreme = sum(1 for r in yr_rows if r["total_count"] >= 1500)
        over100 = sum(1 for r in yr_rows if r["total_count"] >= 100)
        peak = max(r["total_count"] for r in yr_rows)
        last_100 = max((r["day_of_year"] for r in yr_rows if r["total_count"] >= 100), default=0)

        # Jan-Feb temperature
        janfeb = [r for r in rows if r["year"] == yr and r["day_of_year"] <= 59
                  and r.get("temp_mean") is not None]
        janfeb_temp = statistics.mean(r["temp_mean"] for r in janfeb) if janfeb else None

        # Fall precipitation (Sep-Nov = DOY 244-334)
        fall_precip = [r for r in rows if r["year"] == yr and 244 <= r["day_of_year"] <= 334
                       and r.get("precipitation") is not None]
        fall_precip_total = sum(r["precipitation"] for r in fall_precip) if fall_precip else None

        # Spring Mar-Apr precipitation (DOY 60-120)
        spring_precip = [r for r in rows if r["year"] == yr and 60 <= r["day_of_year"] <= 120
                         and r.get("precipitation") is not None]
        spring_precip_total = sum(r["precipitation"] for r in spring_precip) if spring_precip else None

        # GDD at end of Feb
        feb_rows = [r for r in rows if r["year"] == yr and r["day_of_year"] <= 59]
        gdd_feb = feb_rows[-1]["gdd_cumulative"] if feb_rows else 0

        stats[yr] = {
            "total_burden": total, "extreme_days": extreme, "over100_days": over100,
            "peak": peak, "last_100_doy": last_100,
            "janfeb_temp": janfeb_temp, "fall_precip": fall_precip_total,
            "spring_precip": spring_precip_total, "gdd_feb": gdd_feb,
        }
    return stats


def compute_remaining(rows, year, from_doy, threshold=100):
    """Count remaining days >= threshold after from_doy in the given year."""
    return sum(1 for r in rows if r["year"] == year and r["day_of_year"] > from_doy
               and r["total_count"] is not None and r["total_count"] >= threshold)


def extract_direct_features(row, idx, year_stats):
    """Extract features for predicting remaining days >100."""
    yr = row["year"]
    doy = row["day_of_year"]

    if row["total_count"] is None:
        return None

    # Current season state
    burden = row["cumulative_burden"]
    progress = row["season_progress_pct"]
    gdd = row["gdd_cumulative"]
    latest_count = row["total_count"]
    latest_log = row["log_count"] or 0

    # Recent pollen stats
    recent_counts = []
    for d in range(1, 8):
        prev = idx.get((yr, doy - d))
        if prev and prev["total_count"] is not None:
            recent_counts.append(prev["total_count"])

    recent_max = max(recent_counts) if recent_counts else 0
    recent_mean = statistics.mean(recent_counts) if recent_counts else 0
    recent_over100 = sum(1 for c in recent_counts if c >= 100)

    # Current year's Jan-Feb temp and GDD
    cur_stats = year_stats.get(yr, {})
    janfeb_temp = cur_stats.get("janfeb_temp", 45)
    gdd_feb = cur_stats.get("gdd_feb", 150)

    # Prior year features
    prior = year_stats.get(yr - 1, {})
    prior_burden = prior.get("total_burden", 40000)
    prior_extreme = prior.get("extreme_days", 8)
    prior_last_100 = prior.get("last_100_doy", 110)
    prior_fall_precip = prior.get("fall_precip", 10)
    prior_spring_precip = prior.get("spring_precip", 15)

    features = {
        # Current season state
        "cumulative_burden_log": math.log(burden + 1),
        "season_progress_pct": progress,
        "gdd_cumulative": gdd,
        "latest_log_count": latest_log,
        "day_of_year": float(doy),
        "doy_sin": math.sin(2 * math.pi * doy / 365),

        # Recent pollen activity
        "recent_7d_max_log": math.log(recent_max + 1),
        "recent_7d_mean_log": math.log(recent_mean + 1),
        "recent_7d_days_over100": float(recent_over100),

        # Current year preseason
        "janfeb_temp": janfeb_temp or 45,
        "gdd_at_feb_end": gdd_feb,

        # Prior year (biology: oak/pine source strength signals)
        "prior_burden_log": math.log(prior_burden + 1),
        "prior_extreme_days": float(prior_extreme),
        "prior_last_100_doy": float(prior_last_100),
        "prior_fall_precip": prior_fall_precip or 10,
        "prior_spring_precip": prior_spring_precip or 15,

        # Year trend
        "year_trend": float(yr - 2000) / 25,
    }

    return features


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
        lr, w_sum = 0.005, sum(sample_weights)
        for _ in range(1500):
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

    def feature_importance(self):
        imp = [(self.feature_names[j], self.weights[j]) for j in range(len(self.feature_names))]
        imp.sort(key=lambda x: abs(x[1]), reverse=True)
        return imp


def main():
    rows = load_features()
    idx = build_index(rows)
    year_stats = compute_year_stats(rows)

    test_years = list(range(2015, 2026))
    start_dates = [
        (32, "Feb 1"), (46, "Feb 15"), (60, "Mar 1"),
        (74, "Mar 15"), (91, "Apr 1"), (105, "Apr 15"),
    ]

    print("=" * 80)
    print("DIRECT REMAINING-DAYS MODEL")
    print("Target: remaining days with count >= 100 after the given date")
    print("=" * 80)

    # For each start date, build a model and evaluate across test years
    for start_doy, start_label in start_dates:
        direct_errors = []
        analog_errors = []
        direct_ext_errors = []
        analog_ext_errors = []

        for test_yr in test_years:
            # Build training data: for each prior year, at this DOY, what was the remaining count?
            train_X, train_y, train_y_ext, train_w = [], [], [], []
            feature_names = None

            for yr in sorted(year_stats.keys()):
                if yr >= test_yr or yr < 1993:  # need prior year
                    continue

                row_at_doy = idx.get((yr, start_doy))
                if not row_at_doy or row_at_doy["total_count"] is None:
                    # Find nearest observation
                    for offset in range(1, 5):
                        row_at_doy = idx.get((yr, start_doy + offset)) or idx.get((yr, start_doy - offset))
                        if row_at_doy and row_at_doy["total_count"] is not None:
                            break
                    else:
                        continue

                feats = extract_direct_features(row_at_doy, idx, year_stats)
                if feats is None:
                    continue

                if feature_names is None:
                    feature_names = list(feats.keys())

                remaining_100 = compute_remaining(rows, yr, start_doy, 100)
                remaining_ext = compute_remaining(rows, yr, start_doy, 1500)

                train_X.append([feats[f] for f in feature_names])
                train_y.append(float(remaining_100))
                train_y_ext.append(float(remaining_ext))
                train_w.append(math.exp(-0.03 * (test_yr - yr)))

            if len(train_X) < 8:
                continue

            # Train direct model for remaining >100
            model_100 = WeightedLinearRegression()
            model_100.fit(train_X, train_y, feature_names, train_w)

            # Train direct model for remaining extreme
            model_ext = WeightedLinearRegression()
            model_ext.fit(train_X, train_y_ext, feature_names, train_w)

            # Predict for test year
            test_row = idx.get((test_yr, start_doy))
            if not test_row or test_row["total_count"] is None:
                for offset in range(1, 5):
                    test_row = idx.get((test_yr, start_doy + offset)) or idx.get((test_yr, start_doy - offset))
                    if test_row and test_row["total_count"] is not None:
                        break
                else:
                    continue

            feats = extract_direct_features(test_row, idx, year_stats)
            if feats is None:
                continue

            pred_100 = max(0, model_100.predict([feats[f] for f in feature_names]))
            pred_ext = max(0, model_ext.predict([feats[f] for f in feature_names]))

            actual_100 = compute_remaining(rows, test_yr, start_doy, 100)
            actual_ext = compute_remaining(rows, test_yr, start_doy, 1500)

            direct_errors.append(abs(pred_100 - actual_100))
            direct_ext_errors.append(abs(pred_ext - actual_ext))

            # Analog baseline for comparison
            burdens = []
            for yr in sorted(year_stats.keys()):
                if yr >= test_yr or yr < 1993:
                    continue
                yr_row = idx.get((yr, start_doy))
                if yr_row:
                    b = yr_row["cumulative_burden"]
                    rem100 = compute_remaining(rows, yr, start_doy, 100)
                    rem_ext = compute_remaining(rows, yr, start_doy, 1500)
                    burdens.append((yr, b, rem100, rem_ext))

            test_burden = test_row["cumulative_burden"]
            burdens.sort(key=lambda x: abs(x[1] - test_burden))
            top5 = burdens[:5]
            if top5:
                analog_pred_100 = statistics.mean(b[2] for b in top5)
                analog_pred_ext = statistics.mean(b[3] for b in top5)
                analog_errors.append(abs(analog_pred_100 - actual_100))
                analog_ext_errors.append(abs(analog_pred_ext - actual_ext))

        # Report
        if direct_errors:
            print(f"\n--- From {start_label} (DOY {start_doy}) ---")
            print(f"  {'Model':<25} {'MAE >100 days':>14} {'MAE Extreme':>12}")
            print(f"  {'-'*51}")
            print(f"  {'Direct regression':<25} {statistics.mean(direct_errors):>14.1f} {statistics.mean(direct_ext_errors):>12.1f}")
            if analog_errors:
                print(f"  {'Analog (top-5 burden)':<25} {statistics.mean(analog_errors):>14.1f} {statistics.mean(analog_ext_errors):>12.1f}")
            improvement = ((statistics.mean(analog_errors) - statistics.mean(direct_errors)) / statistics.mean(analog_errors) * 100) if analog_errors else 0
            print(f"  Improvement: {improvement:+.1f}%")

    # Feature importance from the Mar 15 model (most relevant)
    # Retrain on all data through 2024 for final weights
    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE (Mar 15 model, trained through 2024)")
    print(f"{'='*80}")

    train_X, train_y, train_w = [], [], []
    feature_names = None
    for yr in sorted(year_stats.keys()):
        if yr > 2024 or yr < 1993:
            continue
        row_at_doy = idx.get((yr, 74))
        if not row_at_doy or row_at_doy["total_count"] is None:
            continue
        feats = extract_direct_features(row_at_doy, idx, year_stats)
        if feats is None:
            continue
        if feature_names is None:
            feature_names = list(feats.keys())
        remaining = compute_remaining(rows, yr, 74, 100)
        train_X.append([feats[f] for f in feature_names])
        train_y.append(float(remaining))
        train_w.append(math.exp(-0.03 * (2025 - yr)))

    if train_X:
        final_model = WeightedLinearRegression()
        final_model.fit(train_X, train_y, feature_names, train_w)
        for fname, weight in final_model.feature_importance():
            bar = "#" * int(abs(weight) * 3)
            sign = "+" if weight > 0 else "-"
            print(f"  {sign} {abs(weight):.1f}  {fname:<25} {bar}")

        # Save weights
        model_data = {
            "feature_names": final_model.feature_names,
            "weights": final_model.weights,
            "bias": final_model.bias,
            "means": final_model.means,
            "stds": final_model.stds,
            "version": "Direct remaining days",
            "target": "remaining_days_over_100",
        }
        with open(OUTPUT_DIR / "direct_remaining_model_weights.json", "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"\nDirect model weights saved.")

    # 2026 prediction
    print(f"\n{'='*80}")
    print(f"2026 PREDICTIONS (from each start date)")
    print(f"{'='*80}")
    for start_doy, start_label in start_dates:
        row_2026 = idx.get((2026, start_doy))
        if not row_2026 or row_2026["total_count"] is None:
            continue

        # Train on all prior years
        tX, ty, ty_ext, tw = [], [], [], []
        fn = None
        for yr in sorted(year_stats.keys()):
            if yr > 2025 or yr < 1993:
                continue
            r = idx.get((yr, start_doy))
            if not r or r["total_count"] is None:
                continue
            feats = extract_direct_features(r, idx, year_stats)
            if feats is None:
                continue
            if fn is None:
                fn = list(feats.keys())
            tX.append([feats[f] for f in fn])
            ty.append(float(compute_remaining(rows, yr, start_doy, 100)))
            ty_ext.append(float(compute_remaining(rows, yr, start_doy, 1500)))
            tw.append(math.exp(-0.03 * (2026 - yr)))

        if len(tX) < 8:
            continue

        m100 = WeightedLinearRegression()
        m100.fit(tX, ty, fn, tw)
        m_ext = WeightedLinearRegression()
        m_ext.fit(tX, ty_ext, fn, tw)

        feats_2026 = extract_direct_features(row_2026, idx, year_stats)
        if feats_2026:
            pred_100 = max(0, m100.predict([feats_2026[f] for f in fn]))
            pred_ext = max(0, m_ext.predict([feats_2026[f] for f in fn]))
            print(f"  From {start_label}: {pred_100:.0f} more days >100, {pred_ext:.0f} more extreme days")


if __name__ == "__main__":
    main()
