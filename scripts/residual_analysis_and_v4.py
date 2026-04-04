"""
Residual analysis of V3 model + V4 model with new features.

Part 1: Analyze V3 residuals — where does it fail systematically?
Part 2: Test new feature candidates against residuals
Part 3: Build V4 with the winners
"""

import csv
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
                      "precip_yesterday", "precip_2day_sum", "wind_max", "wind_gust",
                      "humidity_mean", "et0"]:
            r[field] = float(r[field]) if r.get(field) and r[field] else None
        r["missing"] = r.get("missing") == "True"
    return rows


def build_index(rows):
    idx = {}
    for r in rows:
        idx[(r["year"], r["day_of_year"])] = r
    return idx


def pearson_r(xs, ys):
    n = len(xs)
    if n < 5:
        return None
    mx, my = sum(xs)/n, sum(ys)/n
    sx = math.sqrt(sum((x-mx)**2 for x in xs)/(n-1))
    sy = math.sqrt(sum((y-my)**2 for y in ys)/(n-1))
    if sx == 0 or sy == 0:
        return 0
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / ((n-1)*sx*sy)


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


def compute_3day_temp_avg(row, idx):
    """3-day rolling average temperature."""
    temps = []
    for d in range(3):
        r = idx.get((row["year"], row["day_of_year"] - d))
        if r and r["temp_mean"] is not None:
            temps.append(r["temp_mean"])
    return statistics.mean(temps) if temps else row["temp_mean"]


def compute_temp_anomaly(row, rows_by_doy):
    """Temperature deviation from historical average for this DOY."""
    doy = row["day_of_year"]
    if doy not in rows_by_doy or row["temp_mean"] is None:
        return 0
    hist_temps = rows_by_doy[doy]
    if not hist_temps:
        return 0
    return row["temp_mean"] - statistics.mean(hist_temps)


def compute_humidity_wind_interaction(row):
    """Low humidity + high wind = more pollen dispersal."""
    h = row.get("humidity_mean")
    w = row.get("wind_max")
    if h is None or w is None:
        return 0
    # Inverse humidity * wind: higher when dry and windy
    return max(0, (100 - h) / 100) * (w / 20)


def compute_prev_3day_max(row, idx):
    """Max pollen in last 3 days — captures recent peak."""
    vals = []
    for d in range(1, 4):
        prev = idx.get((row["year"], row["day_of_year"] - d))
        if prev and prev["log_count"] is not None:
            vals.append(prev["log_count"])
    return max(vals) if vals else (row["log_count"] or 0)


def extract_features_v4(row, idx, rows_by_doy):
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
    temp_3day = compute_3day_temp_avg(row, idx)
    temp_anomaly = compute_temp_anomaly(row, rows_by_doy)
    humid_wind = compute_humidity_wind_interaction(row)
    prev_3day_max = compute_prev_3day_max(row, idx)

    features = {
        # Core autoregressive
        "prev_log_count": prev_log,
        "prev2_log_count": prev2_log if prev2_log is not None else prev_log,
        "d1_d2_diff": (prev_log - prev2_log) if prev2_log is not None else 0,
        "prev_3day_max": prev_3day_max,

        # Temperature features
        "temp_mean": temp,
        "temp_above_50": max(0, temp - 50),
        "temp_3day_avg": temp_3day,
        "temp_anomaly": temp_anomaly,

        # Interaction: temp * wind (when both high, pollen is extreme)
        "temp_wind_interaction": max(0, temp - 50) * (row["wind_max"] or 0) / 100,

        # Precipitation
        "precip_yesterday": row["precip_yesterday"] or 0,
        "precip_2day_sum": row["precip_2day_sum"] or 0,
        "consec_dry_days": float(min(consec_dry, 7)),

        # Humidity-wind dispersal index
        "humid_wind_index": humid_wind,

        # Wind
        "wind_max": row["wind_max"] or 0,

        # Seasonal
        "gdd_daily": row["gdd_daily"] or 0,
        "doy_sin": math.sin(2 * math.pi * row["day_of_year"] / 365),
        "doy_cos": math.cos(2 * math.pi * row["day_of_year"] / 365),
        "season_progress": row["season_progress_pct"] / 100,

        # Trend
        "year_trend": float(row["year"] - 2000) / 25,

        # Yesterday's regime was rainy (strong next-day signal)
        "yest_was_rainy": 1.0 if (prev and prev["precipitation"] is not None and prev["precipitation"] >= 0.25) else 0.0,
        "yest_was_dry_warm": 1.0 if (prev and prev["precipitation"] is not None and prev["precipitation"] < 0.05
                                      and prev["temp_mean"] is not None and prev["temp_mean"] >= 55) else 0.0,
    }

    # Today's regime one-hots
    features.update(regime_features(regime, "today"))
    # Yesterday's regime one-hots
    if prev_regime:
        features.update(regime_features(prev_regime, "yest"))
    else:
        features.update(regime_features("cool", "yest"))

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
        for epoch in range(1500):
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

    # Build DOY → historical temps lookup (for temp anomaly)
    rows_by_doy = defaultdict(list)
    for r in rows:
        if r["year"] <= 2025 and r["temp_mean"] is not None:
            rows_by_doy[r["day_of_year"]].append(r["temp_mean"])

    # ================================================================
    # PART 1: Residual analysis of V3 predictions
    # ================================================================
    print("=" * 70)
    print("PART 1: V3 RESIDUAL ANALYSIS — WHERE DOES THE MODEL FAIL?")
    print("=" * 70)

    # Quick V3 reproduction to get residuals
    # (simplified — train on 1992-2024, test 2025)
    from model_v3 import extract_features_v3 as v3_extract, WeightedLinearRegression as V3Model

    train_rows = [r for r in rows if r["year"] <= 2024 and 1 <= r["day_of_year"] <= 180]
    test_rows = [r for r in rows if r["year"] == 2025 and 30 <= r["day_of_year"] <= 150
                 and r["total_count"] is not None]

    train_X, train_y, train_w = [], [], []
    feature_names_v3 = None
    for r in train_rows:
        result = v3_extract(r, idx)
        if result:
            feats, target = result
            if feature_names_v3 is None:
                feature_names_v3 = list(feats.keys())
            train_X.append([feats[f] for f in feature_names_v3])
            train_y.append(target)
            train_w.append(math.exp(-0.03 * (2025 - r["year"])))

    v3_model = V3Model()
    v3_model.fit(train_X, train_y, feature_names_v3, train_w)

    # Get residuals for 2025
    residuals = []
    for r in test_rows:
        result = v3_extract(r, idx)
        if not result:
            continue
        feats, actual = result
        pred = v3_model.predict([feats[f] for f in feature_names_v3])
        err = actual - pred  # positive = model under-predicted
        residuals.append({
            "doy": r["day_of_year"], "actual": actual, "pred": pred, "error": err,
            "count": r["total_count"], "temp": r["temp_mean"],
            "precip": r["precipitation"], "wind": r["wind_max"],
            "humidity": r.get("humidity_mean"),
            "regime": classify_regime(r),
            "season_pct": r["season_progress_pct"],
        })

    # Analyze residuals by regime
    print(f"\nResidual by weather regime (positive = model under-predicted):")
    print(f"{'Regime':<15} {'N':>4} {'Mean Err':>10} {'Median Err':>11} {'Std':>8}")
    print("-" * 48)
    regime_resids = defaultdict(list)
    for r in residuals:
        regime_resids[r["regime"]].append(r["error"])
    for regime in ["hot_windy", "hot_calm", "warm_windy", "warm_calm", "cool", "drizzle", "rainy"]:
        errs = regime_resids.get(regime, [])
        if errs:
            print(f"{regime:<15} {len(errs):>4} {statistics.mean(errs):>+10.3f} "
                  f"{sorted(errs)[len(errs)//2]:>+11.3f} {statistics.stdev(errs) if len(errs) > 1 else 0:>8.3f}")

    # Analyze residuals by season phase
    print(f"\nResidual by season phase:")
    print(f"{'Phase':<25} {'N':>4} {'Mean Err':>10} {'Std':>8}")
    print("-" * 47)
    phases = [("Pre-season (0-10%)", 0, 10), ("Ramp-up (10-30%)", 10, 30),
              ("Peak (30-70%)", 30, 70), ("Decline (70-90%)", 70, 90), ("Tail (90-100%)", 90, 100)]
    for name, lo, hi in phases:
        errs = [r["error"] for r in residuals if lo <= r["season_pct"] < hi]
        if errs:
            print(f"{name:<25} {len(errs):>4} {statistics.mean(errs):>+10.3f} "
                  f"{statistics.stdev(errs) if len(errs) > 1 else 0:>8.3f}")

    # Do residuals correlate with any unused features?
    print(f"\nCorrelation of RESIDUALS with candidate new features:")
    err_vals = [r["error"] for r in residuals]

    candidates = {
        "humidity": [r["humidity"] for r in residuals],
        "temp_anomaly": [compute_temp_anomaly(
            idx.get((2025, r["doy"])), rows_by_doy) if idx.get((2025, r["doy"])) else 0 for r in residuals],
        "temp_3day_avg": [compute_3day_temp_avg(
            idx.get((2025, r["doy"])), idx) if idx.get((2025, r["doy"])) else 0 for r in residuals],
    }

    for name, vals in candidates.items():
        clean = [(v, e) for v, e in zip(vals, err_vals) if v is not None]
        if len(clean) > 10:
            r_val = pearson_r([c[0] for c in clean], [c[1] for c in clean])
            print(f"  {name}: r = {r_val:.3f}" if r_val else f"  {name}: insufficient data")

    # ================================================================
    # PART 2: V4 MODEL — incorporate new features
    # ================================================================
    print()
    print("=" * 70)
    print("PART 2: V4 MODEL — TEMP ANOMALY + INTERACTIONS + 3-DAY LOOKBACK")
    print("=" * 70)

    test_years = list(range(2015, 2026))
    v4_metrics = {"ae": [], "se": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    v4_ramp = {"ae": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    threshold = math.log(91)

    print(f"\n{'Year':>6} {'N':>4} {'MAE':>8} {'RMSE':>8}")
    print("-" * 30)

    last_model = None
    for test_yr in test_years:
        train = [r for r in rows if r["year"] < test_yr and 1 <= r["day_of_year"] <= 180]
        test = [r for r in rows if r["year"] == test_yr and 30 <= r["day_of_year"] <= 150]

        tX, ty, tw = [], [], []
        fn = None
        for r in train:
            result = extract_features_v4(r, idx, rows_by_doy)
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
            result = extract_features_v4(r, idx, rows_by_doy)
            if not result:
                continue
            feats, actual = result
            pred = model.predict([feats[f] for f in fn])
            err = pred - actual
            ae = abs(err)
            year_ae.append(ae)
            v4_metrics["ae"].append(ae)
            v4_metrics["se"].append(err**2)

            ab, pb = actual >= threshold, pred >= threshold
            if ab and pb: v4_metrics["tp"] += 1
            elif not ab and pb: v4_metrics["fp"] += 1
            elif ab and not pb: v4_metrics["fn"] += 1
            else: v4_metrics["tn"] += 1

            if 5 <= r["season_progress_pct"] <= 35:
                v4_ramp["ae"].append(ae)
                if ab and pb: v4_ramp["tp"] += 1
                elif not ab and pb: v4_ramp["fp"] += 1
                elif ab and not pb: v4_ramp["fn"] += 1
                else: v4_ramp["tn"] += 1

        if year_ae:
            mae = statistics.mean(year_ae)
            rmse = math.sqrt(statistics.mean(e**2 for e in year_ae))
            print(f"{test_yr:>6} {len(year_ae):>4} {mae:>8.3f} {rmse:>8.3f}")

    # Aggregate
    avg_mae = statistics.mean(v4_metrics["ae"])
    avg_rmse = math.sqrt(statistics.mean(v4_metrics["se"]))
    total = v4_metrics["tp"] + v4_metrics["fp"] + v4_metrics["tn"] + v4_metrics["fn"]
    acc = (v4_metrics["tp"] + v4_metrics["tn"]) / total if total > 0 else 0
    prec = v4_metrics["tp"] / (v4_metrics["tp"] + v4_metrics["fp"]) if (v4_metrics["tp"] + v4_metrics["fp"]) > 0 else 0
    rec = v4_metrics["tp"] / (v4_metrics["tp"] + v4_metrics["fn"]) if (v4_metrics["tp"] + v4_metrics["fn"]) > 0 else 0

    ramp_total = v4_ramp["tp"] + v4_ramp["fp"] + v4_ramp["tn"] + v4_ramp["fn"]
    ramp_acc = (v4_ramp["tp"] + v4_ramp["tn"]) / ramp_total if ramp_total > 0 else 0

    print(f"\n{'Model':<20} {'MAE':>7} {'RMSE':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'Ramp Acc':>9}")
    print("-" * 67)
    print(f"{'V4 (interactions)':<20} {avg_mae:>7.3f} {avg_rmse:>7.3f} {acc:>6.1%} {prec:>6.1%} {rec:>6.1%} {ramp_acc:>8.1%}")
    print(f"{'V3 (regimes+D2)':<20} {'0.777':>7} {'1.041':>7} {'85.0%':>7} {'86.0%':>7} {'90.8%':>7} {'86.1%':>9}")
    print(f"{'V2 (improved)':<20} {'0.787':>7} {'1.056':>7} {'84.0%':>7} {'85.0%':>7} {'90.3%':>7} {'~49%':>9}")
    print(f"{'V1 (original)':<20} {'0.822':>7} {'1.078':>7} {'83.3%':>7} {'86.5%':>7} {'86.9%':>7} {'~49%':>9}")

    # Feature importance
    if last_model:
        print(f"\nTop 20 features (V4):")
        for fname, weight in last_model.feature_importance()[:20]:
            bar = "#" * int(abs(weight) * 8)
            sign = "+" if weight > 0 else "-"
            print(f"  {sign} {abs(weight):.3f}  {fname:<30} {bar}")

    # Save weights
    if last_model:
        import json
        model_data = {
            "feature_names": last_model.feature_names,
            "weights": last_model.weights,
            "bias": last_model.bias,
            "means": last_model.means,
            "stds": last_model.stds,
            "version": "V4",
            "description": "V4: temp anomaly, temp-wind interaction, 3-day max, humidity-wind index, 3-day temp avg",
        }
        with open(OUTPUT_DIR / "weather_model_v4_weights.json", "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"\nV4 weights saved.")


if __name__ == "__main__":
    main()
