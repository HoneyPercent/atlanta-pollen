"""
January Early-Season Forecast — predict season severity from January data alone.

By February 1, we know:
- January pollen activity (total, days with pollen)
- January weather (temp, GDD, precip)
- Prior year's season metrics (total burden, extreme days, timing)
- Prior summer/fall weather (Jul-Aug temp/precip, fall precip)

Can we predict:
1. Total season burden (how bad will this year be?)
2. Number of extreme days (how many really terrible days?)
3. Season onset date (when will it start getting bad?)
4. Season archetype (front-loaded, slow-build, back-loaded?)

This is a "pre-season outlook" product — issue on Feb 1.
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
        r["cumulative_burden"] = float(r["cumulative_burden"])
        r["gdd_cumulative"] = float(r["gdd_cumulative"])
        for field in ["temp_mean", "precipitation", "wind_max", "vpd_max", "solar_radiation"]:
            r[field] = float(r[field]) if r.get(field) and r[field] else None
    return rows


def compute_year_stats(rows):
    stats = {}
    for yr in set(r["year"] for r in rows):
        yr_rows = [r for r in rows if r["year"] == yr and r["total_count"] is not None]
        if not yr_rows:
            continue
        total = sum(r["total_count"] for r in yr_rows)
        extreme = sum(1 for r in yr_rows if r["total_count"] >= 1500)
        over100 = sum(1 for r in yr_rows if r["total_count"] >= 100)
        peak = max(r["total_count"] for r in yr_rows)
        over100_list = [r for r in yr_rows if r["total_count"] > 100]
        onset_doy = over100_list[0]["day_of_year"] if over100_list else None
        last_100_doy = over100_list[-1]["day_of_year"] if over100_list else None

        # January stats
        jan = [r for r in rows if r["year"] == yr and r["day_of_year"] <= 31]
        jan_pollen = [r for r in jan if r["total_count"] is not None and r["total_count"] > 0]
        jan_pollen_total = sum(r["total_count"] for r in jan_pollen) if jan_pollen else 0
        jan_pollen_days = len(jan_pollen)
        jan_temps = [r["temp_mean"] for r in jan if r["temp_mean"] is not None]
        jan_temp = statistics.mean(jan_temps) if jan_temps else None
        jan_precip = [r for r in jan if r["precipitation"] is not None]
        jan_precip_total = sum(r["precipitation"] for r in jan_precip) if jan_precip else None
        jan_gdd = jan[-1]["gdd_cumulative"] if jan else 0

        # Jan-Feb (for models evaluated at Feb 1, we only have January)
        janfeb = [r for r in rows if r["year"] == yr and r["day_of_year"] <= 59]
        janfeb_temps = [r["temp_mean"] for r in janfeb if r["temp_mean"] is not None]
        janfeb_temp = statistics.mean(janfeb_temps) if janfeb_temps else None

        # Prior year stats
        prev_yr = yr - 1
        jul_aug = [r for r in rows if r["year"] == prev_yr and 182 <= r["day_of_year"] <= 243]
        jul_aug_t = [r["temp_mean"] for r in jul_aug if r["temp_mean"] is not None]
        jul_aug_temp = statistics.mean(jul_aug_t) if jul_aug_t else None
        jul_aug_p = [r for r in jul_aug if r["precipitation"] is not None]
        jul_aug_precip = sum(r["precipitation"] for r in jul_aug_p) if jul_aug_p else None

        fall_p = [r for r in rows if r["year"] == prev_yr and 244 <= r["day_of_year"] <= 334
                  and r["precipitation"] is not None]
        fall_precip = sum(r["precipitation"] for r in fall_p) if fall_p else None

        prev_rows = [r for r in rows if r["year"] == prev_yr and r["total_count"] is not None]
        prev_total = sum(r["total_count"] for r in prev_rows) if prev_rows else None
        prev_extreme = sum(1 for r in prev_rows if r["total_count"] >= 1500) if prev_rows else None

        stats[yr] = {
            "total": total, "extreme": extreme, "over100": over100,
            "peak": peak, "onset_doy": onset_doy, "last_100_doy": last_100_doy,
            "jan_pollen_total": jan_pollen_total, "jan_pollen_days": jan_pollen_days,
            "jan_temp": jan_temp, "jan_precip": jan_precip_total, "jan_gdd": jan_gdd,
            "janfeb_temp": janfeb_temp,
            "prior_jul_aug_temp": jul_aug_temp, "prior_jul_aug_precip": jul_aug_precip,
            "prior_fall_precip": fall_precip,
            "prior_total": prev_total, "prior_extreme": prev_extreme,
        }
    return stats


def extract_january_features(year_stats_entry):
    """Features available by February 1."""
    s = year_stats_entry
    return {
        "jan_pollen_total": s.get("jan_pollen_total", 0),
        "jan_pollen_days": float(s.get("jan_pollen_days", 0)),
        "jan_temp": s.get("jan_temp") or 42,
        "jan_precip": s.get("jan_precip") or 4,
        "jan_gdd": s.get("jan_gdd") or 0,
        "prior_total_log": math.log((s.get("prior_total") or 40000) + 1),
        "prior_extreme": float(s.get("prior_extreme") or 8),
        "prior_jul_aug_temp": s.get("prior_jul_aug_temp") or 78,
        "prior_jul_aug_precip": s.get("prior_jul_aug_precip") or 8,
        "prior_fall_precip": s.get("prior_fall_precip") or 12,
    }


class SimpleRegression:
    def __init__(self):
        self.weights = self.bias = self.feature_names = self.means = self.stds = None

    def fit(self, X, y, feature_names):
        self.feature_names = feature_names
        n, k = len(y), len(feature_names)
        self.means = [statistics.mean(X[i][j] for i in range(n)) for j in range(k)]
        self.stds = []
        for j in range(k):
            vals = [X[i][j] for i in range(n)]
            s = statistics.stdev(vals) if len(set(vals)) > 1 else 1
            self.stds.append(s if s > 0 else 1)
        X_std = [[((X[i][j] - self.means[j]) / self.stds[j]) for j in range(k)] for i in range(n)]
        weights = [0.0] * k
        bias = statistics.mean(y)
        lr = 0.01
        for _ in range(1000):
            preds = [bias + sum(weights[j] * X_std[i][j] for j in range(k)) for i in range(n)]
            errors = [preds[i] - y[i] for i in range(n)]
            grad_b = sum(errors) / n
            grad_w = [sum(errors[i] * X_std[i][j] for i in range(n)) / n for j in range(k)]
            bias -= lr * grad_b
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
    year_stats = compute_year_stats(rows)

    test_years = list(range(2010, 2026))
    complete_years = sorted(yr for yr in year_stats.keys() if yr <= 2025 and yr >= 1993)

    print("=" * 80)
    print("JANUARY EARLY-SEASON FORECAST")
    print("Issue date: February 1 | Predicts: total burden, extreme days, onset")
    print("=" * 80)

    # ================================================================
    # TEST 1: Predict total season burden
    # ================================================================
    print(f"\n--- TOTAL SEASON BURDEN ---")
    print(f"{'Year':>6} {'Predicted':>12} {'Actual':>12} {'Error':>10} {'% Error':>10}")
    print("-" * 50)

    burden_errors = []
    for test_yr in test_years:
        if test_yr not in year_stats:
            continue
        tX, ty = [], []
        fn = None
        for yr in complete_years:
            if yr >= test_yr or yr not in year_stats:
                continue
            feats = extract_january_features(year_stats[yr])
            if fn is None: fn = list(feats.keys())
            tX.append([feats[f] for f in fn])
            ty.append(float(year_stats[yr]["total"]))

        if len(tX) < 8:
            continue

        model = SimpleRegression()
        model.fit(tX, ty, fn)

        test_feats = extract_january_features(year_stats[test_yr])
        pred = max(0, model.predict([test_feats[f] for f in fn]))
        actual = year_stats[test_yr]["total"]
        err = abs(pred - actual)
        pct_err = err / actual * 100 if actual > 0 else 0
        burden_errors.append(pct_err)

        print(f"{test_yr:>6} {pred:>12,.0f} {actual:>12,} {err:>10,.0f} {pct_err:>9.0f}%")

    if burden_errors:
        print(f"\nMedian % error: {sorted(burden_errors)[len(burden_errors)//2]:.0f}%")
        print(f"Mean % error: {statistics.mean(burden_errors):.0f}%")

    # ================================================================
    # TEST 2: Predict extreme days
    # ================================================================
    print(f"\n--- EXTREME DAYS (>=1500) ---")
    print(f"{'Year':>6} {'Predicted':>10} {'Actual':>8} {'Error':>8}")
    print("-" * 32)

    ext_errors = []
    for test_yr in test_years:
        if test_yr not in year_stats:
            continue
        tX, ty = [], []
        fn = None
        for yr in complete_years:
            if yr >= test_yr or yr not in year_stats:
                continue
            feats = extract_january_features(year_stats[yr])
            if fn is None: fn = list(feats.keys())
            tX.append([feats[f] for f in fn])
            ty.append(float(year_stats[yr]["extreme"]))

        if len(tX) < 8:
            continue

        model = SimpleRegression()
        model.fit(tX, ty, fn)

        test_feats = extract_january_features(year_stats[test_yr])
        pred = max(0, model.predict([test_feats[f] for f in fn]))
        actual = year_stats[test_yr]["extreme"]
        err = abs(pred - actual)
        ext_errors.append(err)

        print(f"{test_yr:>6} {pred:>10.1f} {actual:>8} {err:>8.1f}")

    if ext_errors:
        print(f"\nMAE: {statistics.mean(ext_errors):.1f} extreme days")
        print(f"Median error: {sorted(ext_errors)[len(ext_errors)//2]:.1f} extreme days")

    # ================================================================
    # TEST 3: Predict season onset
    # ================================================================
    print(f"\n--- SEASON ONSET (first day >100) ---")
    print(f"{'Year':>6} {'Predicted':>10} {'Actual':>8} {'Error':>8}")
    print("-" * 32)

    onset_errors = []
    for test_yr in test_years:
        if test_yr not in year_stats or year_stats[test_yr].get("onset_doy") is None:
            continue
        tX, ty = [], []
        fn = None
        for yr in complete_years:
            if yr >= test_yr or yr not in year_stats or year_stats[yr].get("onset_doy") is None:
                continue
            feats = extract_january_features(year_stats[yr])
            if fn is None: fn = list(feats.keys())
            tX.append([feats[f] for f in fn])
            ty.append(float(year_stats[yr]["onset_doy"]))

        if len(tX) < 8:
            continue

        model = SimpleRegression()
        model.fit(tX, ty, fn)

        test_feats = extract_january_features(year_stats[test_yr])
        pred = model.predict([test_feats[f] for f in fn])
        actual = year_stats[test_yr]["onset_doy"]
        err = abs(pred - actual)
        onset_errors.append(err)

        from datetime import date, timedelta
        pred_date = (date(test_yr, 1, 1) + timedelta(days=int(pred) - 1)).strftime("%b %d")
        actual_date = (date(test_yr, 1, 1) + timedelta(days=actual - 1)).strftime("%b %d")
        print(f"{test_yr:>6} {pred_date:>10} {actual_date:>8} {err:>7.0f}d")

    if onset_errors:
        print(f"\nMAE: {statistics.mean(onset_errors):.0f} days")
        print(f"Median error: {sorted(onset_errors)[len(onset_errors)//2]:.0f} days")

    # ================================================================
    # Feature importance
    # ================================================================
    print(f"\n{'='*80}")
    print(f"FEATURE IMPORTANCE (burden model, trained through 2024)")
    print(f"{'='*80}")
    tX, ty = [], []
    fn = None
    for yr in complete_years:
        if yr > 2024 or yr not in year_stats:
            continue
        feats = extract_january_features(year_stats[yr])
        if fn is None: fn = list(feats.keys())
        tX.append([feats[f] for f in fn])
        ty.append(float(year_stats[yr]["total"]))

    if tX:
        m = SimpleRegression()
        m.fit(tX, ty, fn)
        print(f"\nWhat predicts total season burden from January data:")
        for fname, weight in m.feature_importance():
            bar = "#" * int(abs(weight) / 1000)
            sign = "+" if weight > 0 else "-"
            print(f"  {sign} {abs(weight):>8,.0f}  {fname}")

    # ================================================================
    # 2026 FORECAST
    # ================================================================
    if 2026 in year_stats:
        print(f"\n{'='*80}")
        print(f"2026 SEASON FORECAST (issued Feb 1, 2026)")
        print(f"{'='*80}")
        tX_all, ty_burden, ty_ext, ty_onset = [], [], [], []
        fn = None
        for yr in complete_years:
            if yr > 2025 or yr not in year_stats:
                continue
            feats = extract_january_features(year_stats[yr])
            if fn is None: fn = list(feats.keys())
            tX_all.append([feats[f] for f in fn])
            ty_burden.append(float(year_stats[yr]["total"]))
            ty_ext.append(float(year_stats[yr]["extreme"]))
            if year_stats[yr].get("onset_doy"):
                ty_onset.append(float(year_stats[yr]["onset_doy"]))

        feats_2026 = extract_january_features(year_stats[2026])
        x = [feats_2026[f] for f in fn]

        m_b = SimpleRegression()
        m_b.fit(tX_all, ty_burden, fn)
        pred_burden = max(0, m_b.predict(x))

        m_e = SimpleRegression()
        m_e.fit(tX_all, ty_ext, fn)
        pred_ext = max(0, m_e.predict(x))

        # Onset needs filtered training data
        tX_onset = [tX_all[i] for i in range(len(tX_all)) if i < len(ty_onset)]
        if tX_onset and len(tX_onset) == len(ty_onset):
            m_o = SimpleRegression()
            m_o.fit(tX_onset, ty_onset, fn)
            pred_onset = m_o.predict(x)
            from datetime import date, timedelta
            onset_date = (date(2026, 1, 1) + timedelta(days=int(pred_onset) - 1)).strftime("%b %d")
        else:
            onset_date = "unknown"

        print(f"\n2026 January data:")
        print(f"  January pollen total: {year_stats[2026]['jan_pollen_total']}")
        print(f"  January pollen days: {year_stats[2026]['jan_pollen_days']}")
        print(f"  January temp: {year_stats[2026].get('jan_temp', 'N/A')}")
        print(f"  January GDD: {year_stats[2026]['jan_gdd']:.0f}")
        print(f"  Prior year (2025) burden: {year_stats.get(2025, {}).get('total', 'N/A'):,}")
        print(f"  Prior year extreme days: {year_stats.get(2025, {}).get('extreme', 'N/A')}")

        print(f"\n2026 PREDICTED SEASON:")
        print(f"  Total burden: {pred_burden:,.0f}")
        print(f"  Extreme days: {pred_ext:.0f}")
        print(f"  Season onset: ~{onset_date}")

        # Context
        avg_burden = statistics.mean(year_stats[yr]["total"] for yr in complete_years if yr >= 2015)
        if pred_burden > avg_burden * 1.2:
            print(f"  Assessment: ABOVE AVERAGE season (>{avg_burden*1.2:,.0f})")
        elif pred_burden < avg_burden * 0.8:
            print(f"  Assessment: BELOW AVERAGE season (<{avg_burden*0.8:,.0f})")
        else:
            print(f"  Assessment: NEAR AVERAGE season")


if __name__ == "__main__":
    main()
