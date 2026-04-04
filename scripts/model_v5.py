"""
Model V5 — incorporating spike forensics findings.

New features over V4:
1. 5-day temperature trend (warming ramp = strongest pre-spike signal)
2. Rain-then-dry pattern (rain in prior 5 days + dry today)
3. GDD readiness zone (GDD > 200 = "armed" for extremes)
4. 5-day pollen momentum (trend over last 5 observations)
5. 5-day precip total (accumulated dryness)
"""

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

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
    if p >= 0.25: return "rainy"
    elif p >= 0.05: return "drizzle"
    elif t >= 65 and w >= 10: return "hot_windy"
    elif t >= 65: return "hot_calm"
    elif t >= 50 and w >= 10: return "warm_windy"
    elif t >= 50: return "warm_calm"
    else: return "cool"


def regime_features(regime_name, prefix):
    return {f"{prefix}_{r}": (1.0 if regime_name == r else 0.0) for r in REGIMES}


def compute_window_stats(row, idx, lookback=5):
    """Compute stats for the N-day window before this day."""
    yr, doy = row["year"], row["day_of_year"]
    temps, precips, pollens = [], [], []
    for d in range(1, lookback + 1):
        prev = idx.get((yr, doy - d))
        if prev:
            if prev["temp_mean"] is not None:
                temps.append(prev["temp_mean"])
            if prev["precipitation"] is not None:
                precips.append(prev["precipitation"])
            if prev["log_count"] is not None:
                pollens.append(prev["log_count"])

    temp_trend = (temps[0] - temps[-1]) if len(temps) >= 2 else 0
    total_precip = sum(precips) if precips else 0
    had_rain = any(p >= 0.25 for p in precips) if precips else False
    is_dry_today = row["precipitation"] is not None and row["precipitation"] < 0.05
    rain_then_dry = had_rain and is_dry_today

    pollen_trend = (pollens[0] - pollens[-1]) if len(pollens) >= 2 else 0
    pollen_max = max(pollens) if pollens else 0

    dry_days = sum(1 for p in precips if p < 0.05)

    return {
        "temp_5d_trend": temp_trend,
        "precip_5d_total": total_precip,
        "rain_then_dry": 1.0 if rain_then_dry else 0.0,
        "pollen_5d_trend": pollen_trend,
        "pollen_5d_max": pollen_max,
        "dry_days_5d": float(dry_days),
    }


def compute_temp_anomaly(row, rows_by_doy):
    doy = row["day_of_year"]
    if doy not in rows_by_doy or row["temp_mean"] is None:
        return 0
    return row["temp_mean"] - statistics.mean(rows_by_doy[doy])


def extract_features_v5(row, idx, rows_by_doy):
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
    window = compute_window_stats(row, idx, 5)
    temp_anomaly = compute_temp_anomaly(row, rows_by_doy)

    features = {
        # Autoregressive core
        "prev_log_count": prev_log,
        "prev2_log_count": prev2_log if prev2_log is not None else prev_log,
        "d1_d2_diff": (prev_log - prev2_log) if prev2_log is not None else 0,
        "pollen_5d_max": window["pollen_5d_max"],
        "pollen_5d_trend": window["pollen_5d_trend"],

        # Temperature
        "temp_mean": temp,
        "temp_above_50": max(0, temp - 50),
        "temp_anomaly": temp_anomaly,
        "temp_5d_trend": window["temp_5d_trend"],  # KEY: warming ramp

        # Precipitation / dryness
        "precip_yesterday": row["precip_yesterday"] or 0,
        "precip_2day_sum": row["precip_2day_sum"] or 0,
        "precip_5d_total": window["precip_5d_total"],
        "dry_days_5d": window["dry_days_5d"],
        "rain_then_dry": window["rain_then_dry"],  # KEY: rain-then-dry pattern

        # Wind
        "wind_max": row["wind_max"] or 0,

        # GDD
        "gdd_daily": row["gdd_daily"] or 0,
        "gdd_armed": 1.0 if row["gdd_cumulative"] >= 200 else 0.0,  # KEY: readiness zone

        # Seasonal
        "doy_sin": math.sin(2 * math.pi * row["day_of_year"] / 365),
        "doy_cos": math.cos(2 * math.pi * row["day_of_year"] / 365),
        "season_progress": row["season_progress_pct"] / 100,

        # Trend
        "year_trend": float(row["year"] - 2000) / 25,

        # Yesterday's conditions
        "yest_was_rainy": 1.0 if (prev and prev["precipitation"] is not None and prev["precipitation"] >= 0.25) else 0.0,
        "yest_was_dry_warm": 1.0 if (prev and prev["precipitation"] is not None and prev["precipitation"] < 0.05
                                      and prev["temp_mean"] is not None and prev["temp_mean"] >= 55) else 0.0,
    }

    # Regime one-hots
    features.update(regime_features(regime, "today"))
    features.update(regime_features(prev_regime or "cool", "yest"))

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

    rows_by_doy = defaultdict(list)
    for r in rows:
        if r["year"] <= 2025 and r["temp_mean"] is not None:
            rows_by_doy[r["day_of_year"]].append(r["temp_mean"])

    test_years = list(range(2015, 2026))
    metrics = {"ae": [], "se": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    ramp = {"ae": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    threshold = math.log(91)

    print("=" * 70)
    print("MODEL V5 — SPIKE FORENSICS: WARMING TREND + RAIN-THEN-DRY + GDD ARMED")
    print("=" * 70)
    print(f"\n{'Year':>6} {'N':>4} {'MAE':>8} {'RMSE':>8}")
    print("-" * 30)

    last_model = None
    for test_yr in test_years:
        train = [r for r in rows if r["year"] < test_yr and 1 <= r["day_of_year"] <= 180]
        test = [r for r in rows if r["year"] == test_yr and 30 <= r["day_of_year"] <= 150]

        # Build DOY lookup excluding test year
        doy_lookup = defaultdict(list)
        for r in rows:
            if r["year"] < test_yr and r["temp_mean"] is not None:
                doy_lookup[r["day_of_year"]].append(r["temp_mean"])

        tX, ty, tw = [], [], []
        fn = None
        for r in train:
            result = extract_features_v5(r, idx, doy_lookup)
            if result:
                feats, target = result
                if fn is None:
                    fn = list(feats.keys())
                tX.append([feats[f] for f in fn])
                ty.append(target)
                tw.append(math.exp(-0.03 * (test_yr - r["year"])))

        if len(tX) < 50:
            continue

        model = WeightedLinearRegression()
        model.fit(tX, ty, fn, tw)
        last_model = model

        year_ae = []
        for r in test:
            result = extract_features_v5(r, idx, doy_lookup)
            if not result:
                continue
            feats, actual = result
            pred = model.predict([feats[f] for f in fn])
            ae = abs(pred - actual)
            year_ae.append(ae)
            metrics["ae"].append(ae)
            metrics["se"].append((pred - actual)**2)

            ab, pb = actual >= threshold, pred >= threshold
            if ab and pb: metrics["tp"] += 1
            elif not ab and pb: metrics["fp"] += 1
            elif ab and not pb: metrics["fn"] += 1
            else: metrics["tn"] += 1

            if 5 <= r["season_progress_pct"] <= 35:
                ramp["ae"].append(ae)
                if ab and pb: ramp["tp"] += 1
                elif not ab and pb: ramp["fp"] += 1
                elif ab and not pb: ramp["fn"] += 1
                else: ramp["tn"] += 1

        if year_ae:
            print(f"{test_yr:>6} {len(year_ae):>4} {statistics.mean(year_ae):>8.3f} "
                  f"{math.sqrt(statistics.mean(e**2 for e in year_ae)):>8.3f}")

    # Aggregate
    avg_mae = statistics.mean(metrics["ae"])
    avg_rmse = math.sqrt(statistics.mean(metrics["se"]))
    total = metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]
    acc = (metrics["tp"] + metrics["tn"]) / total
    prec = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if (metrics["tp"] + metrics["fp"]) > 0 else 0
    rec = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0
    ramp_total = ramp["tp"] + ramp["fp"] + ramp["tn"] + ramp["fn"]
    ramp_acc = (ramp["tp"] + ramp["tn"]) / ramp_total if ramp_total > 0 else 0

    print(f"\n{'Model':<20} {'MAE':>7} {'RMSE':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'Ramp':>7}")
    print("-" * 62)
    print(f"{'V5 (spike forens)':<20} {avg_mae:>7.3f} {avg_rmse:>7.3f} {acc:>6.1%} {prec:>6.1%} {rec:>6.1%} {ramp_acc:>6.1%}")
    print(f"{'V4 (interactions)':<20} {'0.761':>7} {'1.016':>7} {'85.8%':>7} {'87.0%':>7} {'91.0%':>7} {'86.7%':>7}")
    print(f"{'V3 (regimes+D2)':<20} {'0.777':>7} {'1.041':>7} {'85.0%':>7} {'86.0%':>7} {'90.8%':>7} {'86.1%':>7}")
    print(f"{'V1 (original)':<20} {'0.822':>7} {'1.078':>7} {'83.3%':>7} {'86.5%':>7} {'86.9%':>7} {'~49%':>7}")

    if last_model:
        print(f"\nTop 15 features (V5):")
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
            "version": "V5",
        }
        with open(OUTPUT_DIR / "weather_model_v5_weights.json", "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"\nV5 weights saved.")


if __name__ == "__main__":
    main()
