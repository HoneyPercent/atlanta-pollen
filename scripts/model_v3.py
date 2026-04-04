"""
Model V3 — incorporating hypothesis battery findings.

New features over V2:
1. D-2 log count (partial autocorrelation signal — mean reversion)
2. Weather regime (categorical: Hot+Windy, Warm+Windy, Cool, Rainy, etc.)
3. Season phase proxy (cumulative burden as % of recent-year average)
4. Dry spell length (consecutive dry days — upward surprise predictor)
5. Prior day's regime (transition signal: Rainy→? is highly predictive)

Also: separate evaluation for ramp-up phase (where V2 is weakest).
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
            r[field] = float(r[field]) if r.get(field) else None
        r["missing"] = r.get("missing") == "True"
    return rows


def build_index(rows):
    idx = {}
    for r in rows:
        idx[(r["year"], r["day_of_year"])] = r
    return idx


def classify_regime(r):
    """Classify a day into a weather regime."""
    if r["temp_mean"] is None or r["precipitation"] is None or r["wind_max"] is None:
        return None
    t, p, w = r["temp_mean"], r["precipitation"], r["wind_max"]
    if p >= 0.25:
        return "rainy"
    elif p >= 0.05:
        return "drizzle"
    elif t >= 65 and w >= 10:
        return "hot_windy"
    elif t >= 65:
        return "hot_calm"
    elif t >= 50 and w >= 10:
        return "warm_windy"
    elif t >= 50:
        return "warm_calm"
    else:
        return "cool"


# Regime encoding: one-hot style as numeric features
REGIMES = ["hot_windy", "hot_calm", "warm_windy", "warm_calm", "cool", "drizzle", "rainy"]


def regime_features(regime_name, prefix="regime"):
    """One-hot encode a regime."""
    feats = {}
    for r in REGIMES:
        feats[f"{prefix}_{r}"] = 1.0 if regime_name == r else 0.0
    return feats


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
        "prev2_log_count": prev2_log if prev2_log is not None else prev_log,  # fallback
        "d1_d2_diff": (prev_log - prev2_log) if prev2_log is not None else 0,  # momentum
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

    # Add regime one-hots (today and yesterday)
    features.update(regime_features(regime, "today"))
    if prev_regime:
        features.update(regime_features(prev_regime, "yest"))
    else:
        features.update(regime_features("cool", "yest"))  # default

    # Key transition: was yesterday rainy?
    features["yest_was_rainy"] = 1.0 if (prev and prev["precipitation"] is not None and prev["precipitation"] >= 0.25) else 0.0

    target = row["log_count"]
    return features, target


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

    def feature_importance(self):
        imp = [(self.feature_names[j], self.weights[j]) for j in range(len(self.feature_names))]
        imp.sort(key=lambda x: abs(x[1]), reverse=True)
        return imp


def evaluate_year(rows, idx, test_year):
    train_rows = [r for r in rows if r["year"] < test_year and 1 <= r["day_of_year"] <= 180]
    test_rows = [r for r in rows if r["year"] == test_year and 30 <= r["day_of_year"] <= 150]

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
            train_w.append(math.exp(-0.03 * (test_year - r["year"])))

    if len(train_X) < 50:
        return None

    model = WeightedLinearRegression()
    model.fit(train_X, train_y, feature_names, train_w)

    # Evaluate
    results = {"all": {"ae": [], "se": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0},
               "rampup": {"ae": [], "se": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}}
    threshold = math.log(91)

    for r in test_rows:
        result = extract_features_v3(r, idx)
        if not result:
            continue
        feats, actual = result
        pred = model.predict([feats[f] for f in feature_names])
        err = pred - actual

        for bucket in ["all"]:
            results[bucket]["ae"].append(abs(err))
            results[bucket]["se"].append(err ** 2)
            ab, pb = actual >= threshold, pred >= threshold
            if ab and pb: results[bucket]["tp"] += 1
            elif not ab and pb: results[bucket]["fp"] += 1
            elif ab and not pb: results[bucket]["fn"] += 1
            else: results[bucket]["tn"] += 1

        # Ramp-up phase
        if 5 <= r["season_progress_pct"] <= 35:
            results["rampup"]["ae"].append(abs(err))
            results["rampup"]["se"].append(err ** 2)
            ab, pb = actual >= threshold, pred >= threshold
            if ab and pb: results["rampup"]["tp"] += 1
            elif not ab and pb: results["rampup"]["fp"] += 1
            elif ab and not pb: results["rampup"]["fn"] += 1
            else: results["rampup"]["tn"] += 1

    return results, model


def main():
    rows = load_features()
    idx = build_index(rows)
    test_years = list(range(2015, 2026))

    print("=" * 70)
    print("MODEL V3 — WEATHER REGIMES + D-2 + MOMENTUM + TRANSITIONS")
    print("=" * 70)
    print()

    all_metrics = {"ae": [], "se": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    ramp_metrics = {"ae": [], "se": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}

    print(f"{'Year':>6} {'N':>4} {'MAE':>8} {'RMSE':>8} {'Ramp MAE':>9}")
    print("-" * 38)

    last_model = None
    for yr in test_years:
        result = evaluate_year(rows, idx, yr)
        if result is None:
            continue
        results, model = result
        last_model = model

        r_all = results["all"]
        mae = statistics.mean(r_all["ae"])
        rmse = math.sqrt(statistics.mean(r_all["se"]))
        all_metrics["ae"].extend(r_all["ae"])
        all_metrics["se"].extend(r_all["se"])
        for k in ["tp", "fp", "tn", "fn"]:
            all_metrics[k] += r_all[k]

        r_ramp = results["rampup"]
        ramp_mae = statistics.mean(r_ramp["ae"]) if r_ramp["ae"] else 0
        ramp_metrics["ae"].extend(r_ramp["ae"])
        for k in ["tp", "fp", "tn", "fn"]:
            ramp_metrics[k] += r_ramp[k]

        print(f"{yr:>6} {len(r_all['ae']):>4} {mae:>8.3f} {rmse:>8.3f} {ramp_mae:>9.3f}")

    # Aggregate
    avg_mae = statistics.mean(all_metrics["ae"])
    avg_rmse = math.sqrt(statistics.mean(all_metrics["se"]))
    total = all_metrics["tp"] + all_metrics["fp"] + all_metrics["tn"] + all_metrics["fn"]
    acc = (all_metrics["tp"] + all_metrics["tn"]) / total if total > 0 else 0
    prec = all_metrics["tp"] / (all_metrics["tp"] + all_metrics["fp"]) if (all_metrics["tp"] + all_metrics["fp"]) > 0 else 0
    rec = all_metrics["tp"] / (all_metrics["tp"] + all_metrics["fn"]) if (all_metrics["tp"] + all_metrics["fn"]) > 0 else 0

    ramp_total = ramp_metrics["tp"] + ramp_metrics["fp"] + ramp_metrics["tn"] + ramp_metrics["fn"]
    ramp_acc = (ramp_metrics["tp"] + ramp_metrics["tn"]) / ramp_total if ramp_total > 0 else 0

    print(f"\n{'Model':<20} {'MAE':>7} {'RMSE':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'Ramp Acc':>9}")
    print("-" * 67)
    print(f"{'V3 (regimes+D2)':<20} {avg_mae:>7.3f} {avg_rmse:>7.3f} {acc:>6.1%} {prec:>6.1%} {rec:>6.1%} {ramp_acc:>8.1%}")
    print(f"{'V2 (improved)':<20} {'0.787':>7} {'1.056':>7} {'84.0%':>7} {'85.0%':>7} {'90.3%':>7} {'~49%':>9}")
    print(f"{'V1 (original)':<20} {'0.822':>7} {'1.078':>7} {'83.3%':>7} {'86.5%':>7} {'86.9%':>7} {'~49%':>9}")
    print(f"{'Persistence':<20} {'0.915':>7} {'1.272':>7} {'81.0%':>7} {'85.2%':>7} {'84.3%':>7} {'~49%':>9}")

    # Feature importance
    if last_model:
        print(f"\nTop 15 features (V3):")
        for fname, weight in last_model.feature_importance()[:15]:
            bar = "#" * int(abs(weight) * 8)
            sign = "+" if weight > 0 else "-"
            print(f"  {sign} {abs(weight):.3f}  {fname:<25} {bar}")

        model_data = {
            "feature_names": last_model.feature_names,
            "weights": last_model.weights,
            "bias": last_model.bias,
            "means": last_model.means,
            "stds": last_model.stds,
            "version": "V3",
            "description": "Weather regimes, D-2 lookback, momentum, transitions, season progress",
        }
        with open(OUTPUT_DIR / "weather_model_v3_weights.json", "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"\nV3 weights saved.")


if __name__ == "__main__":
    main()
