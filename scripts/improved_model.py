"""
Improved pollen forecast model incorporating findings from deep analysis.

Changes from v1 weather_forecast_model.py:
1. Add above-50F threshold feature (nonlinear temp response)
2. Add year trend feature (pollen is getting worse — 3.2x over 30 years)
3. Add consecutive dry days feature (more predictive than just yesterday's rain)
4. Add streak feature (currently in an extreme streak? how long?)
5. Weight recent years more in training (recency bias)
6. Compare to v1 model on same evaluation framework.
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


def compute_consecutive_dry(row, idx):
    """Count consecutive dry days ending at this day (0 if today is wet)."""
    yr = row["year"]
    doy = row["day_of_year"]
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


def compute_streak(row, idx):
    """How many consecutive days >= 1500 ending yesterday? (0 if yesterday was not extreme)."""
    yr = row["year"]
    doy = row["day_of_year"]
    count = 0
    for d in range(1, 20):
        prev = idx.get((yr, doy - d))
        if prev and prev["total_count"] is not None and prev["total_count"] >= 1500:
            count += 1
        else:
            break
    return count


def extract_features_v2(row, idx):
    """Enhanced feature vector."""
    if row["total_count"] is None or row["temp_mean"] is None:
        return None

    prev = idx.get((row["year"], row["day_of_year"] - 1))
    prev_log = prev["log_count"] if prev and prev["log_count"] is not None else None
    if prev_log is None:
        return None

    temp = row["temp_mean"]
    consec_dry = compute_consecutive_dry(row, idx)
    streak = compute_streak(row, idx)

    features = {
        "prev_log_count": prev_log,
        "temp_mean": temp,
        "temp_max": row["temp_max"] or temp,
        "above_50": 1.0 if temp >= 50 else 0.0,
        "temp_above_50": max(0, temp - 50),  # piecewise: only temp above 50 matters
        "precip_yesterday": row["precip_yesterday"] or 0,
        "precip_2day_sum": row["precip_2day_sum"] or 0,
        "wind_max": row["wind_max"] or 0,
        "gdd_daily": row["gdd_daily"] or 0,
        "doy_sin": math.sin(2 * math.pi * row["day_of_year"] / 365),
        "doy_cos": math.cos(2 * math.pi * row["day_of_year"] / 365),
        "consec_dry_days": float(consec_dry),
        "extreme_streak": float(min(streak, 7)),  # cap at 7 to avoid outlier influence
        "year_trend": float(row["year"] - 2000) / 25,  # normalized trend: 0 in 2000, 1 in 2025
    }

    target = row["log_count"]
    return features, target


class WeightedLinearRegression:
    """Multiple linear regression with sample weights."""

    def __init__(self):
        self.weights = None
        self.bias = None
        self.feature_names = None
        self.means = None
        self.stds = None

    def fit(self, X, y, feature_names, sample_weights=None):
        self.feature_names = feature_names
        n = len(y)
        k = len(feature_names)

        if sample_weights is None:
            sample_weights = [1.0] * n

        # Standardize
        self.means = [statistics.mean(X[i][j] for i in range(n)) for j in range(k)]
        self.stds = []
        for j in range(k):
            vals = [X[i][j] for i in range(n)]
            s = statistics.stdev(vals) if len(set(vals)) > 1 else 1
            self.stds.append(s if s > 0 else 1)

        X_std = [[((X[i][j] - self.means[j]) / self.stds[j]) for j in range(k)] for i in range(n)]

        # Weighted gradient descent
        weights = [0.0] * k
        bias = sum(w * y_i for w, y_i in zip(sample_weights, y)) / sum(sample_weights)
        lr = 0.01
        w_sum = sum(sample_weights)

        for _ in range(800):  # more epochs for convergence with weights
            preds = [bias + sum(weights[j] * X_std[i][j] for j in range(k)) for i in range(n)]
            errors = [preds[i] - y[i] for i in range(n)]

            grad_bias = sum(sample_weights[i] * errors[i] for i in range(n)) / w_sum
            grad_weights = [
                sum(sample_weights[i] * errors[i] * X_std[i][j] for i in range(n)) / w_sum
                for j in range(k)
            ]

            bias -= lr * grad_bias
            for j in range(k):
                weights[j] -= lr * grad_weights[j]

        self.weights = weights
        self.bias = bias

    def predict(self, x_raw):
        x_std = [(x_raw[j] - self.means[j]) / self.stds[j] for j in range(len(self.feature_names))]
        return self.bias + sum(self.weights[j] * x_std[j] for j in range(len(self.feature_names)))

    def feature_importance(self):
        imp = [(self.feature_names[j], self.weights[j]) for j in range(len(self.feature_names))]
        imp.sort(key=lambda x: abs(x[1]), reverse=True)
        return imp


def evaluate_year(rows, idx, test_year, use_v2=True):
    """Train on all years before test_year, evaluate on test_year."""
    train_rows = [r for r in rows if r["year"] < test_year and 1 <= r["day_of_year"] <= 180]
    test_rows = [r for r in rows if r["year"] == test_year and 30 <= r["day_of_year"] <= 150]

    extract_fn = extract_features_v2 if use_v2 else None
    if extract_fn is None:
        return None

    train_X, train_y, train_weights = [], [], []
    feature_names = None
    for r in train_rows:
        result = extract_fn(r, idx)
        if result:
            feats, target = result
            if feature_names is None:
                feature_names = list(feats.keys())
            train_X.append([feats[f] for f in feature_names])
            train_y.append(target)
            # Recency weight: recent years count more (exponential decay)
            years_ago = test_year - r["year"]
            weight = math.exp(-0.03 * years_ago)  # half-life ~23 years
            train_weights.append(weight)

    if len(train_X) < 50:
        return None

    model = WeightedLinearRegression()
    model.fit(train_X, train_y, feature_names, train_weights)

    # Evaluate
    abs_errors, sq_errors = [], []
    actuals, preds = [], []
    for r in test_rows:
        result = extract_fn(r, idx)
        if not result:
            continue
        feats, actual = result
        pred = model.predict([feats[f] for f in feature_names])
        err = pred - actual
        abs_errors.append(abs(err))
        sq_errors.append(err ** 2)
        actuals.append(actual)
        preds.append(pred)

    if not abs_errors:
        return None

    # Bad day classification
    threshold = math.log(91)
    tp = fp = tn = fn = 0
    for a, p in zip(actuals, preds):
        ab = a >= threshold
        pb = p >= threshold
        if ab and pb: tp += 1
        elif not ab and pb: fp += 1
        elif ab and not pb: fn += 1
        else: tn += 1

    return {
        "mae": statistics.mean(abs_errors),
        "rmse": math.sqrt(statistics.mean(sq_errors)),
        "n": len(abs_errors),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "model": model,
    }


def main():
    rows = load_features()
    idx = build_index(rows)

    test_years = list(range(2015, 2026))

    print("=" * 70)
    print("IMPROVED MODEL (V2) vs ORIGINAL (V1)")
    print("=" * 70)
    print("V2 additions: 50F threshold, piecewise temp, consecutive dry days,")
    print("extreme streak, year trend, recency-weighted training")
    print()

    v2_metrics = []
    v2_bad_day = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    print(f"{'Year':>6} {'N':>4} {'MAE':>8} {'RMSE':>8}")
    print("-" * 30)

    last_model = None
    for yr in test_years:
        result = evaluate_year(rows, idx, yr, use_v2=True)
        if result is None:
            continue
        v2_metrics.append(result)
        for k in ["tp", "fp", "tn", "fn"]:
            v2_bad_day[k] += result[k]
        last_model = result["model"]
        print(f"{yr:>6} {result['n']:>4} {result['mae']:>8.3f} {result['rmse']:>8.3f}")

    # Aggregate
    avg_mae = statistics.mean(m["mae"] for m in v2_metrics)
    avg_rmse = statistics.mean(m["rmse"] for m in v2_metrics)
    total = v2_bad_day["tp"] + v2_bad_day["fp"] + v2_bad_day["tn"] + v2_bad_day["fn"]
    acc = (v2_bad_day["tp"] + v2_bad_day["tn"]) / total if total > 0 else 0
    prec = v2_bad_day["tp"] / (v2_bad_day["tp"] + v2_bad_day["fp"]) if (v2_bad_day["tp"] + v2_bad_day["fp"]) > 0 else 0
    rec = v2_bad_day["tp"] / (v2_bad_day["tp"] + v2_bad_day["fn"]) if (v2_bad_day["tp"] + v2_bad_day["fn"]) > 0 else 0

    print(f"\n{'Model':<15} {'MAE':>8} {'RMSE':>8} {'Bad-Day Acc':>12} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)
    print(f"{'V2 (improved)':<15} {avg_mae:>8.3f} {avg_rmse:>8.3f} {acc:>11.1%} {prec:>10.1%} {rec:>10.1%}")
    print(f"{'V1 (original)':<15} {'0.822':>8} {'1.078':>8} {'83.3%':>12} {'86.5%':>10} {'86.9%':>10}")
    print(f"{'Persistence':<15} {'0.915':>8} {'1.272':>8} {'81.0%':>12} {'85.2%':>10} {'84.3%':>10}")
    print(f"{'Climatology':<15} {'1.127':>8} {'1.448':>8} {'74.2%':>12} {'88.6%':>10} {'67.5%':>10}")

    # Feature importance
    if last_model:
        print(f"\nFeature importance (V2 model):")
        for fname, weight in last_model.feature_importance():
            bar = "#" * int(abs(weight) * 8)
            sign = "+" if weight > 0 else "-"
            print(f"  {sign} {abs(weight):.3f}  {fname:<25} {bar}")

    # Save V2 weights
    if last_model:
        model_data = {
            "feature_names": last_model.feature_names,
            "weights": last_model.weights,
            "bias": last_model.bias,
            "means": last_model.means,
            "stds": last_model.stds,
            "description": "V2 improved model with 50F threshold, trend, streak features. Recency-weighted.",
            "training_years": "1992-2025 (recency weighted)",
            "evaluation_years": "2015-2025",
        }
        with open(OUTPUT_DIR / "weather_model_v2_weights.json", "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"\nV2 weights saved to data/model_output/weather_model_v2_weights.json")


if __name__ == "__main__":
    main()
