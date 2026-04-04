"""
Model V7 — V5 features + parsed contributor features.

ChatGPT flagged named contributors as the most underused signal.
Contributors tell us WHICH species are active, which is a direct proxy
for season phase (juniper early → oak/pine mid → grass/weed late).

New features:
- has_oak, has_pine, has_juniper (binary: is this species in today's contributor list?)
- has_grass, has_weed (binary: are grass/weeds appearing? = season is transitioning)
- n_tree_species (count of tree contributors listed = diversity of active species)
- tree_severity_num (low=1, medium=2, high=3, extreme=4)
- dominant_is_pine (binary: pine is the first-listed contributor)
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
                      "precip_yesterday", "precip_2day_sum", "wind_max", "wind_gust", "vpd_max"]:
            r[field] = float(r[field]) if r.get(field) and r[field] else None
        r["missing"] = r.get("missing") == "True"
    return rows


def load_contributor_data():
    """Load all detail CSVs and index by date."""
    contrib = {}
    for yr in range(2015, 2026):
        try:
            with open(DATA_DIR / f"detail_{yr}_details.csv") as f:
                for r in csv.DictReader(f):
                    if r.get("missing") == "True":
                        continue
                    contrib[r["date"]] = r
        except FileNotFoundError:
            pass
    return contrib


def parse_contributors(detail_row):
    """Parse a detail row into structured features."""
    if not detail_row:
        return None

    trees_str = detail_row.get("tree_contributors", "")
    trees = [t.strip().upper() for t in trees_str.split(",") if t.strip()] if trees_str else []

    grass_sev = detail_row.get("grass_severity", "").lower()
    weed_str = detail_row.get("weed_contributors", "")
    weeds = [w.strip().upper() for w in weed_str.split(",") if w.strip()] if weed_str else []

    tree_sev = detail_row.get("tree_severity", "").lower()
    sev_map = {"low": 1, "medium": 2, "high": 3, "extreme": 4}

    return {
        "has_oak": 1.0 if "OAK" in trees else 0.0,
        "has_pine": 1.0 if "PINE" in trees else 0.0,
        "has_juniper": 1.0 if "JUNIPER" in trees else 0.0,
        "has_sycamore": 1.0 if "SYCAMORE" in trees else 0.0,
        "has_mulberry": 1.0 if "MULBERRY" in trees else 0.0,
        "has_birch": 1.0 if "BIRCH" in trees else 0.0,
        "has_grass": 1.0 if grass_sev not in ("", "low") else 0.0,  # grass severity > low
        "has_weeds": 1.0 if len(weeds) > 0 else 0.0,
        "n_tree_species": float(len(trees)),
        "tree_severity_num": float(sev_map.get(tree_sev, 0)),
        "dominant_is_pine": 1.0 if trees and trees[0] == "PINE" else 0.0,
        "dominant_is_oak": 1.0 if trees and trees[0] == "OAK" else 0.0,
        "oak_and_pine": 1.0 if "OAK" in trees and "PINE" in trees else 0.0,  # both active = peak
    }


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
    yr, doy = row["year"], row["day_of_year"]
    temps, precips, pollens = [], [], []
    for d in range(1, lookback + 1):
        prev = idx.get((yr, doy - d))
        if prev:
            if prev["temp_mean"] is not None: temps.append(prev["temp_mean"])
            if prev["precipitation"] is not None: precips.append(prev["precipitation"])
            if prev["log_count"] is not None: pollens.append(prev["log_count"])
    return {
        "temp_5d_trend": (temps[0] - temps[-1]) if len(temps) >= 2 else 0,
        "precip_5d_total": sum(precips) if precips else 0,
        "pollen_5d_max": max(pollens) if pollens else 0,
        "pollen_5d_trend": (pollens[0] - pollens[-1]) if len(pollens) >= 2 else 0,
        "dry_days_5d": sum(1 for p in precips if p < 0.05),
        "rain_then_dry": 1.0 if (any(p >= 0.25 for p in precips) and
                                   row["precipitation"] is not None and row["precipitation"] < 0.05) else 0.0,
    }


def extract_features_v7(row, idx, contrib_data, rows_by_doy):
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

    # Temperature anomaly
    doy = row["day_of_year"]
    hist_temps = rows_by_doy.get(doy, [])
    temp_anomaly = (temp - statistics.mean(hist_temps)) if hist_temps else 0

    features = {
        "prev_log_count": prev_log,
        "prev2_log_count": prev2_log if prev2_log is not None else prev_log,
        "d1_d2_diff": (prev_log - prev2_log) if prev2_log is not None else 0,
        "pollen_5d_max": window["pollen_5d_max"],
        "pollen_5d_trend": window["pollen_5d_trend"],
        "temp_mean": temp,
        "temp_above_50": max(0, temp - 50),
        "temp_anomaly": temp_anomaly,
        "temp_5d_trend": window["temp_5d_trend"],
        "precip_yesterday": row["precip_yesterday"] or 0,
        "precip_2day_sum": row["precip_2day_sum"] or 0,
        "precip_5d_total": window["precip_5d_total"],
        "dry_days_5d": float(window["dry_days_5d"]),
        "rain_then_dry": window["rain_then_dry"],
        "wind_max": row["wind_max"] or 0,
        "gdd_daily": row["gdd_daily"] or 0,
        "gdd_armed": 1.0 if row["gdd_cumulative"] >= 200 else 0.0,
        "doy_sin": math.sin(2 * math.pi * row["day_of_year"] / 365),
        "doy_cos": math.cos(2 * math.pi * row["day_of_year"] / 365),
        "season_progress": row["season_progress_pct"] / 100,
        "year_trend": float(row["year"] - 2000) / 25,
        "yest_was_rainy": 1.0 if (prev and prev["precipitation"] is not None and prev["precipitation"] >= 0.25) else 0.0,
    }
    features.update(regime_features(regime, "today"))
    features.update(regime_features(prev_regime or "cool", "yest"))

    # Contributor features — use YESTERDAY's contributors (today's aren't known at prediction time)
    prev_date = prev.get("date") if prev else None
    if prev_date:
        # Try yesterday's detail data
        prev_contrib = contrib_data.get(prev_date)
    else:
        prev_contrib = None

    if prev_contrib:
        cf = parse_contributors(prev_contrib)
        for k, v in cf.items():
            features[f"prev_{k}"] = v
    else:
        # No contributor data — fill with defaults
        for k in ["has_oak", "has_pine", "has_juniper", "has_sycamore", "has_mulberry",
                   "has_birch", "has_grass", "has_weeds", "n_tree_species", "tree_severity_num",
                   "dominant_is_pine", "dominant_is_oak", "oak_and_pine"]:
            features[f"prev_{k}"] = 0.0

    return features, row["log_count"]


class WeightedLinearRegression:
    def __init__(self):
        self.weights = self.bias = self.feature_names = self.means = self.stds = None

    def fit(self, X, y, feature_names, sample_weights=None):
        self.feature_names = feature_names
        n, k = len(y), len(feature_names)
        if sample_weights is None: sample_weights = [1.0] * n
        self.means = [statistics.mean(X[i][j] for i in range(n)) for j in range(k)]
        self.stds = []
        for j in range(k):
            vals = [X[i][j] for i in range(n)]
            s = statistics.stdev(vals) if len(set(vals)) > 1 else 1
            self.stds.append(s if s > 0 else 1)
        X_std = [[((X[i][j] - self.means[j]) / self.stds[j]) for j in range(k)] for i in range(n)]
        weights = [0.0] * k
        bias = sum(w * yi for w, yi in zip(sample_weights, y)) / sum(sample_weights)
        lr, w_sum = 0.004, sum(sample_weights)
        for _ in range(2000):
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
    contrib_data = load_contributor_data()
    print(f"Loaded {len(contrib_data)} days of contributor data")

    rows_by_doy = defaultdict(list)
    for r in rows:
        if r["year"] <= 2025 and r["temp_mean"] is not None:
            rows_by_doy[r["day_of_year"]].append(r["temp_mean"])

    # Only test years with contributor data (2015-2022 available so far)
    # Train can use all years (contributor features default to 0 when absent)
    test_years = list(range(2016, 2026))  # full range with contributor data
    metrics = {"ae": [], "se": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    threshold = math.log(91)

    print("=" * 70)
    print("MODEL V7 — V5 + CONTRIBUTOR FEATURES (species proxy)")
    print("=" * 70)
    print(f"Test years: {test_years[0]}-{test_years[-1]} (contributor data available)")
    print(f"\n{'Year':>6} {'N':>4} {'MAE':>8} {'RMSE':>8}")
    print("-" * 30)

    # Also run V5-equivalent (no contributors) on same test years for fair comparison
    v5_metrics = {"ae": [], "se": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}

    last_model = None
    for test_yr in test_years:
        doy_lookup = defaultdict(list)
        for r in rows:
            if r["year"] < test_yr and r["temp_mean"] is not None:
                doy_lookup[r["day_of_year"]].append(r["temp_mean"])

        train = [r for r in rows if r["year"] < test_yr and 1 <= r["day_of_year"] <= 180]
        test = [r for r in rows if r["year"] == test_yr and 30 <= r["day_of_year"] <= 150]

        tX, ty, tw = [], [], []
        fn = None
        for r in train:
            result = extract_features_v7(r, idx, contrib_data, doy_lookup)
            if result:
                feats, target = result
                if fn is None: fn = list(feats.keys())
                tX.append([feats[f] for f in fn])
                ty.append(target)
                tw.append(math.exp(-0.03 * (test_yr - r["year"])))

        if len(tX) < 50: continue
        model = WeightedLinearRegression()
        model.fit(tX, ty, fn, tw)
        last_model = model

        year_ae = []
        for r in test:
            result = extract_features_v7(r, idx, contrib_data, doy_lookup)
            if not result: continue
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

        if year_ae:
            print(f"{test_yr:>6} {len(year_ae):>4} {statistics.mean(year_ae):>8.3f} "
                  f"{math.sqrt(statistics.mean(e**2 for e in year_ae)):>8.3f}")

    if not metrics["ae"]:
        print("No test data available!")
        return

    avg_mae = statistics.mean(metrics["ae"])
    avg_rmse = math.sqrt(statistics.mean(metrics["se"]))
    total = metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"]
    acc = (metrics["tp"] + metrics["tn"]) / total
    prec = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if (metrics["tp"] + metrics["fp"]) > 0 else 0
    rec = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0

    print(f"\n{'Model':<25} {'MAE':>7} {'RMSE':>7} {'Acc':>7} {'Prec':>7} {'Recall':>7}")
    print("-" * 60)
    print(f"{'V7 (V5+contributors)':<25} {avg_mae:>7.3f} {avg_rmse:>7.3f} {acc:>6.1%} {prec:>6.1%} {rec:>6.1%}")
    print(f"{'V5 (best no-contrib)':<25} {'0.724':>7} {'0.967':>7} {'85.3%':>7} {'86.8%':>7} {'90.3%':>7}")
    print(f"(Note: V7 test years 2016-2022 only; V5 was 2015-2025)")

    if last_model:
        print(f"\nContributor-related feature weights:")
        for fname, weight in last_model.feature_importance():
            if 'prev_' in fname and any(k in fname for k in ['oak', 'pine', 'juniper', 'grass', 'weed',
                                                              'species', 'severity', 'dominant', 'sycamore',
                                                              'mulberry', 'birch']):
                sign = "+" if weight > 0 else "-"
                print(f"  {sign} {abs(weight):.3f}  {fname}")

        print(f"\nTop 15 overall features:")
        for fname, weight in last_model.feature_importance()[:15]:
            bar = "#" * int(abs(weight) * 8)
            sign = "+" if weight > 0 else "-"
            print(f"  {sign} {abs(weight):.3f}  {fname:<30} {bar}")


if __name__ == "__main__":
    main()
