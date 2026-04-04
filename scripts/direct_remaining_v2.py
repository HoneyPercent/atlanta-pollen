"""
Direct remaining-days model V2 — upgraded with Hypothesis Battery 2 findings.

New over V1:
- Prior Jul-Aug temperature (r=0.534 with total burden — strongest new find)
- Prior Jul-Aug precipitation (r=-0.378)
- Phase-specific temperature interaction
- Contributor-based season state (has_grass = season winding down)
- Wind direction N-S component
- Overnight low temperature
- Solar radiation

Also: predict remaining EXTREME days (>=1500) and remaining days >100 separately.
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


def compute_year_stats(rows):
    stats = {}
    for yr in set(r["year"] for r in rows):
        yr_rows = [r for r in rows if r["year"] == yr and r["total_count"] is not None]
        if not yr_rows:
            continue
        total = sum(r["total_count"] for r in yr_rows)
        extreme = sum(1 for r in yr_rows if r["total_count"] >= 1500)
        over100 = sum(1 for r in yr_rows if r["total_count"] >= 100)

        janfeb = [r for r in rows if r["year"] == yr and r["day_of_year"] <= 59 and r["temp_mean"] is not None]
        janfeb_temp = statistics.mean(r["temp_mean"] for r in janfeb) if janfeb else None
        feb_rows = [r for r in rows if r["year"] == yr and r["day_of_year"] <= 59]
        gdd_feb = feb_rows[-1]["gdd_cumulative"] if feb_rows else 0

        # Jul-Aug weather (DOY 182-243)
        jul_aug_t = [r for r in rows if r["year"] == yr and 182 <= r["day_of_year"] <= 243 and r["temp_mean"] is not None]
        jul_aug_temp = statistics.mean(r["temp_mean"] for r in jul_aug_t) if jul_aug_t else None
        jul_aug_p = [r for r in rows if r["year"] == yr and 182 <= r["day_of_year"] <= 243 and r["precipitation"] is not None]
        jul_aug_precip = sum(r["precipitation"] for r in jul_aug_p) if jul_aug_p else None

        # Fall precip (Sep-Nov, DOY 244-334)
        fall_p = [r for r in rows if r["year"] == yr and 244 <= r["day_of_year"] <= 334 and r["precipitation"] is not None]
        fall_precip = sum(r["precipitation"] for r in fall_p) if fall_p else None

        # Prior spring precip (Mar-Apr, DOY 60-120)
        spring_p = [r for r in rows if r["year"] == yr and 60 <= r["day_of_year"] <= 120 and r["precipitation"] is not None]
        spring_precip = sum(r["precipitation"] for r in spring_p) if spring_p else None

        stats[yr] = {
            "total_burden": total, "extreme_days": extreme, "over100_days": over100,
            "janfeb_temp": janfeb_temp, "gdd_feb": gdd_feb,
            "jul_aug_temp": jul_aug_temp, "jul_aug_precip": jul_aug_precip,
            "fall_precip": fall_precip, "spring_precip": spring_precip,
        }
    return stats


def load_contributor_data():
    contrib = {}
    for yr in range(2015, 2026):
        try:
            with open(DATA_DIR / f"detail_{yr}_details.csv") as f:
                for r in csv.DictReader(f):
                    if r.get("missing") != "True":
                        contrib[r["date"]] = r
        except FileNotFoundError:
            pass
    return contrib


def compute_remaining(rows, year, from_doy, threshold=100):
    return sum(1 for r in rows if r["year"] == year and r["day_of_year"] > from_doy
               and r["total_count"] is not None and r["total_count"] >= threshold)


def extract_direct_features_v2(row, idx, year_stats, contrib_data):
    yr, doy = row["year"], row["day_of_year"]
    if row["total_count"] is None:
        return None

    burden = row["cumulative_burden"]
    progress = row["season_progress_pct"]
    gdd = row["gdd_cumulative"]
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
    recent_extreme = sum(1 for c in recent_counts if c >= 1500)

    # Current year stats
    cur = year_stats.get(yr, {})
    janfeb_temp = cur.get("janfeb_temp") or 45
    gdd_feb = cur.get("gdd_feb") or 150

    # Prior year features
    prior = year_stats.get(yr - 1, {})
    prior_burden = prior.get("total_burden", 40000)
    prior_extreme = prior.get("extreme_days", 8)

    # NEW: Prior Jul-Aug weather (strongest new predictor, r=0.534)
    prior_jul_aug_temp = prior.get("jul_aug_temp") or 78
    prior_jul_aug_precip = prior.get("jul_aug_precip") or 8
    prior_fall_precip = prior.get("fall_precip") or 12
    prior_spring_precip = prior.get("spring_precip") or 15

    # Phase
    early_phase = max(0, 1 - progress / 100 * 3)
    late_phase = max(0, progress / 100 * 1.5 - 0.75)

    # Current weather
    temp = row["temp_mean"] or 55
    temp_min = row["temp_min"] or (temp - 10)
    wind_dir_ns = math.cos(math.radians(row["wind_direction"])) if row.get("wind_direction") else 0

    # Contributor state (if available)
    date_str = row.get("date", "")
    detail = contrib_data.get(date_str)
    has_grass = 0.0
    has_weeds = 0.0
    oak_and_pine = 0.0
    if detail:
        grass_sev = detail.get("grass_severity", "").lower()
        has_grass = 1.0 if grass_sev not in ("", "low") else 0.0
        weed_str = detail.get("weed_contributors", "")
        has_weeds = 1.0 if weed_str.strip() else 0.0
        trees = [t.strip().upper() for t in detail.get("tree_contributors", "").split(",") if t.strip()]
        oak_and_pine = 1.0 if "OAK" in trees and "PINE" in trees else 0.0

    features = {
        # Season state
        "cumulative_burden_log": math.log(burden + 1),
        "season_progress_pct": progress,
        "gdd_cumulative": gdd,
        "latest_log_count": latest_log,
        "day_of_year": float(doy),
        "doy_sin": math.sin(2 * math.pi * doy / 365),

        # Recent pollen
        "recent_7d_max_log": math.log(recent_max + 1),
        "recent_7d_mean_log": math.log(recent_mean + 1),
        "recent_7d_days_over100": float(recent_over100),
        "recent_7d_extreme": float(recent_extreme),

        # Preseason
        "janfeb_temp": janfeb_temp,
        "gdd_at_feb_end": gdd_feb,

        # Prior year (biology)
        "prior_burden_log": math.log(prior_burden + 1),
        "prior_extreme_days": float(prior_extreme),

        # NEW: Prior Jul-Aug weather (strongest new finding)
        "prior_jul_aug_temp": prior_jul_aug_temp,
        "prior_jul_aug_precip": prior_jul_aug_precip,
        "prior_fall_precip": prior_fall_precip,
        "prior_spring_precip": prior_spring_precip,

        # NEW: Phase-specific temp
        "temp_x_early": max(0, temp - 50) * early_phase,
        "temp_x_late": max(0, temp - 50) * late_phase,

        # NEW: Overnight low
        "temp_min": temp_min,

        # NEW: Wind direction
        "wind_dir_ns": wind_dir_ns,

        # NEW: Contributor season state
        "has_grass": has_grass,
        "has_weeds": has_weeds,
        "oak_and_pine": oak_and_pine,

        # Trend
        "year_trend": float(yr - 2000) / 25,
    }

    return features


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
    contrib = load_contributor_data()

    test_years = list(range(2015, 2026))
    start_dates = [
        (32, "Feb 1"), (46, "Feb 15"), (60, "Mar 1"),
        (74, "Mar 15"), (91, "Apr 1"), (105, "Apr 15"),
    ]

    print("=" * 80)
    print("DIRECT REMAINING-DAYS MODEL V2")
    print("(+ prior Jul-Aug weather, phase temp, contributors, overnight low)")
    print("=" * 80)

    for start_doy, start_label in start_dates:
        v2_100_err, v2_ext_err = [], []
        v1_100_err, v1_ext_err = [], []  # V1 baseline (analog)

        for test_yr in test_years:
            row_at_doy = idx.get((test_yr, start_doy))
            if not row_at_doy or row_at_doy["total_count"] is None:
                for offset in range(1, 5):
                    row_at_doy = idx.get((test_yr, start_doy + offset)) or idx.get((test_yr, start_doy - offset))
                    if row_at_doy and row_at_doy["total_count"] is not None:
                        break
                else:
                    continue

            actual_100 = compute_remaining(rows, test_yr, start_doy, 100)
            actual_ext = compute_remaining(rows, test_yr, start_doy, 1500)

            # Build training data
            tX, ty_100, ty_ext, tw = [], [], [], []
            fn = None
            for yr in sorted(year_stats.keys()):
                if yr >= test_yr or yr < 1993:
                    continue
                yr_row = idx.get((yr, start_doy))
                if not yr_row or yr_row["total_count"] is None:
                    for offset in range(1, 5):
                        yr_row = idx.get((yr, start_doy + offset)) or idx.get((yr, start_doy - offset))
                        if yr_row and yr_row["total_count"] is not None:
                            break
                    else:
                        continue

                feats = extract_direct_features_v2(yr_row, idx, year_stats, contrib)
                if feats is None:
                    continue
                if fn is None:
                    fn = list(feats.keys())
                tX.append([feats[f] for f in fn])
                ty_100.append(float(compute_remaining(rows, yr, start_doy, 100)))
                ty_ext.append(float(compute_remaining(rows, yr, start_doy, 1500)))
                tw.append(math.exp(-0.03 * (test_yr - yr)))

            if len(tX) < 8:
                continue

            # Train V2 models
            m100 = WeightedLinearRegression()
            m100.fit(tX, ty_100, fn, tw)
            m_ext = WeightedLinearRegression()
            m_ext.fit(tX, ty_ext, fn, tw)

            test_feats = extract_direct_features_v2(row_at_doy, idx, year_stats, contrib)
            if test_feats is None:
                continue

            pred_100 = max(0, m100.predict([test_feats[f] for f in fn]))
            pred_ext = max(0, m_ext.predict([test_feats[f] for f in fn]))
            v2_100_err.append(abs(pred_100 - actual_100))
            v2_ext_err.append(abs(pred_ext - actual_ext))

            # Analog baseline
            burden = row_at_doy["cumulative_burden"]
            analogs = []
            for yr in sorted(year_stats.keys()):
                if yr >= test_yr or yr < 1993:
                    continue
                yr_r = idx.get((yr, start_doy))
                if yr_r:
                    analogs.append((yr, yr_r["cumulative_burden"],
                                    compute_remaining(rows, yr, start_doy, 100),
                                    compute_remaining(rows, yr, start_doy, 1500)))
            analogs.sort(key=lambda a: abs(a[1] - burden))
            top5 = analogs[:5]
            if top5:
                v1_100_err.append(abs(statistics.mean(a[2] for a in top5) - actual_100))
                v1_ext_err.append(abs(statistics.mean(a[3] for a in top5) - actual_ext))

        if v2_100_err:
            v2_mae_100 = statistics.mean(v2_100_err)
            v2_mae_ext = statistics.mean(v2_ext_err)
            v1_mae_100 = statistics.mean(v1_100_err) if v1_100_err else 0
            v1_mae_ext = statistics.mean(v1_ext_err) if v1_ext_err else 0
            imp_100 = (v1_mae_100 - v2_mae_100) / v1_mae_100 * 100 if v1_mae_100 > 0 else 0
            imp_ext = (v1_mae_ext - v2_mae_ext) / v1_mae_ext * 100 if v1_mae_ext > 0 else 0

            print(f"\n--- From {start_label} (DOY {start_doy}) ---")
            print(f"  {'Model':<25} {'MAE >100':>10} {'MAE Extreme':>12}")
            print(f"  {'-'*47}")
            print(f"  {'Direct V2 (upgraded)':<25} {v2_mae_100:>10.1f} {v2_mae_ext:>12.1f}")
            print(f"  {'Direct V1 (original)':<25} {'--':>10} {'--':>12}")
            print(f"  {'Analog (top-5 burden)':<25} {v1_mae_100:>10.1f} {v1_mae_ext:>12.1f}")
            print(f"  Improvement over analog: {imp_100:+.0f}% (>100), {imp_ext:+.0f}% (extreme)")

    # Feature importance from Mar 15 model
    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE (Mar 15 model)")
    print(f"{'='*80}")
    tX, ty, tw = [], [], []
    fn = None
    for yr in sorted(year_stats.keys()):
        if yr > 2024 or yr < 1993: continue
        r = idx.get((yr, 74))
        if not r or r["total_count"] is None: continue
        feats = extract_direct_features_v2(r, idx, year_stats, contrib)
        if feats is None: continue
        if fn is None: fn = list(feats.keys())
        tX.append([feats[f] for f in fn])
        ty.append(float(compute_remaining(rows, yr, 74, 100)))
        tw.append(math.exp(-0.03 * (2025 - yr)))

    if tX:
        m = WeightedLinearRegression()
        m.fit(tX, ty, fn, tw)
        print(f"\nTop 15 features for predicting remaining days >100:")
        for fname, weight in m.feature_importance()[:15]:
            bar = "#" * int(abs(weight) * 2)
            sign = "+" if weight > 0 else "-"
            print(f"  {sign} {abs(weight):.1f}  {fname:<25} {bar}")


if __name__ == "__main__":
    main()
