"""
Hindcast validation: Pretend it's March 15, 2025.

What the model knows:
- All pollen history through 2024
- 2025 pollen data through March 15
- Actual weather through March 15
- Weather forecast for the next 14 days (we use actual weather as proxy, but note this)

What the model predicts:
- D+1 through D+14 daily pollen (weather regression, chained)
- Season progress: how far through the S-curve are we?
- Remaining extreme days this season
- Estimated season end date

Then we compare against what actually happened in 2025.
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
        r["season_progress_pct"] = float(r["season_progress_pct"]) if r.get("season_progress_pct") else 0
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


class SimpleLinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.feature_names = None
        self.means = None
        self.stds = None

    def fit(self, X, y, feature_names):
        self.feature_names = feature_names
        n = len(y)
        k = len(feature_names)
        self.means = [statistics.mean(X[i][j] for i in range(n)) for j in range(k)]
        self.stds = [statistics.stdev(X[i][j] for i in range(n)) if statistics.stdev(X[i][j] for i in range(n)) > 0 else 1 for j in range(k)]
        X_std = [[((X[i][j] - self.means[j]) / self.stds[j]) for j in range(k)] for i in range(n)]

        weights = [0.0] * k
        bias = statistics.mean(y)
        lr = 0.01
        for _ in range(500):
            preds = [bias + sum(weights[j] * X_std[i][j] for j in range(k)) for i in range(n)]
            errors = [preds[i] - y[i] for i in range(n)]
            grad_bias = sum(errors) / n
            grad_weights = [sum(errors[i] * X_std[i][j] for i in range(n)) / n for j in range(k)]
            bias -= lr * grad_bias
            for j in range(k):
                weights[j] -= lr * grad_weights[j]
        self.weights = weights
        self.bias = bias

    def predict(self, x_raw):
        x_std = [(x_raw[j] - self.means[j]) / self.stds[j] for j in range(len(self.feature_names))]
        return self.bias + sum(self.weights[j] * x_std[j] for j in range(len(self.feature_names)))


def extract_features(row, idx):
    if row["total_count"] is None or row["temp_mean"] is None:
        return None
    prev = idx.get((row["year"], row["day_of_year"] - 1))
    prev_log = prev["log_count"] if prev and prev["log_count"] is not None else None
    if prev_log is None:
        return None
    features = {
        "prev_log_count": prev_log,
        "temp_mean": row["temp_mean"],
        "temp_max": row["temp_max"] or row["temp_mean"],
        "precip_yesterday": row["precip_yesterday"] or 0,
        "precip_2day_sum": row["precip_2day_sum"] or 0,
        "wind_max": row["wind_max"] or 0,
        "gdd_daily": row["gdd_daily"] or 0,
        "day_of_year": float(row["day_of_year"]),
        "doy_sin": math.sin(2 * math.pi * row["day_of_year"] / 365),
        "doy_cos": math.cos(2 * math.pi * row["day_of_year"] / 365),
    }
    target = row["log_count"]
    return features, target


def classify_severity(count):
    if count >= 1500:
        return "EXTREME"
    elif count >= 500:
        return "HIGH"
    elif count >= 100:
        return "MODERATE"
    else:
        return "LOW"


def doy_to_date(doy, year=2025):
    from datetime import date, timedelta
    return (date(year, 1, 1) + timedelta(days=doy - 1)).strftime("%b %d")


def main():
    rows = load_features()
    idx = build_index(rows)

    # === SETUP: Pretend it's March 15, 2025 (DOY 74) ===
    CUTOFF_YEAR = 2025
    CUTOFF_DOY = 74  # March 15

    print("=" * 70)
    print("HINDCAST VALIDATION: Pretending it's March 15, 2025")
    print("=" * 70)
    print(f"Model knows: all pollen data through 2024 + 2025 through DOY {CUTOFF_DOY}")
    print(f"Model predicts: the rest of spring 2025")
    print()

    # === TRAIN REGRESSION on all data through 2024 ===
    train_rows = [r for r in rows if r["year"] <= 2024 and 1 <= r["day_of_year"] <= 180]
    train_X = []
    train_y = []
    feature_names = None
    for r in train_rows:
        result = extract_features(r, idx)
        if result:
            feats, target = result
            if feature_names is None:
                feature_names = list(feats.keys())
            train_X.append([feats[f] for f in feature_names])
            train_y.append(target)

    model = SimpleLinearRegression()
    model.fit(train_X, train_y, feature_names)
    print(f"Regression trained on {len(train_X)} days (1992-2024)")

    # === ANALOG-YEAR PROJECTION ===
    # Cumulative burden at cutoff DOY in 2025
    rows_2025_to_cutoff = [r for r in rows if r["year"] == 2025
                           and r["day_of_year"] <= CUTOFF_DOY
                           and r["total_count"] is not None]
    if rows_2025_to_cutoff:
        burden_at_cutoff = rows_2025_to_cutoff[-1]["cumulative_burden"]
        last_count = rows_2025_to_cutoff[-1]["total_count"]
        last_log = rows_2025_to_cutoff[-1]["log_count"]
    else:
        print("No 2025 data found through cutoff!")
        return

    print(f"2025 burden at DOY {CUTOFF_DOY}: {burden_at_cutoff:,.0f}")
    print(f"Last observed count (DOY {CUTOFF_DOY}): {last_count:,}")

    # Find analog years
    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2024))
    analogs = []
    for yr in complete_years:
        yr_at_cutoff = [r for r in rows if r["year"] == yr and r["day_of_year"] <= CUTOFF_DOY]
        yr_after = [r for r in rows if r["year"] == yr and r["day_of_year"] > CUTOFF_DOY
                    and r["total_count"] is not None]
        if not yr_at_cutoff:
            continue
        b = yr_at_cutoff[-1]["cumulative_burden"]
        total = sum(r["total_count"] for r in rows if r["year"] == yr and r["total_count"] is not None)
        remaining_extreme = sum(1 for r in yr_after if r["total_count"] >= 1500)
        remaining_high = sum(1 for r in yr_after if r["total_count"] >= 500)
        remaining_over100 = sum(1 for r in yr_after if r["total_count"] >= 100)
        last_ext = [r for r in yr_after if r["total_count"] >= 1500]
        last_100 = [r for r in yr_after if r["total_count"] >= 100]
        analogs.append({
            "year": yr,
            "burden_at_cutoff": b,
            "total_burden": total,
            "pct_done": b / total * 100 if total > 0 else 0,
            "remaining_extreme": remaining_extreme,
            "remaining_high": remaining_high,
            "remaining_over100": remaining_over100,
            "last_extreme_doy": last_ext[-1]["day_of_year"] if last_ext else CUTOFF_DOY,
            "last_100_doy": last_100[-1]["day_of_year"] if last_100 else CUTOFF_DOY,
        })

    analogs.sort(key=lambda a: abs(a["burden_at_cutoff"] - burden_at_cutoff))
    top5 = analogs[:5]

    print(f"\n--- ANALOG-YEAR PROJECTIONS (top 5 most similar at DOY {CUTOFF_DOY}) ---")
    print(f"{'Year':>6} {'Burden@DOY':>12} {'% Done':>8} {'Rem Ext':>8} {'Rem >100':>9} {'Last Ext':>9} {'Last >100':>10}")
    for a in top5:
        print(f"{a['year']:>6} {a['burden_at_cutoff']:>12,.0f} {a['pct_done']:>7.1f}% {a['remaining_extreme']:>8} {a['remaining_over100']:>9} {doy_to_date(a['last_extreme_doy']):>9} {doy_to_date(a['last_100_doy']):>10}")

    avg_rem_ext = statistics.mean(a["remaining_extreme"] for a in top5)
    avg_rem_100 = statistics.mean(a["remaining_over100"] for a in top5)
    avg_last_ext = statistics.mean(a["last_extreme_doy"] for a in top5)
    avg_last_100 = statistics.mean(a["last_100_doy"] for a in top5)

    print(f"\nAnalog projection:")
    print(f"  Remaining extreme days: {avg_rem_ext:.1f} (range {min(a['remaining_extreme'] for a in top5)}-{max(a['remaining_extreme'] for a in top5)})")
    print(f"  Remaining days > 100: {avg_rem_100:.1f}")
    print(f"  Last extreme day: ~{doy_to_date(int(avg_last_ext))}")
    print(f"  Season end (last >100): ~{doy_to_date(int(avg_last_100))}")

    # === WHAT ACTUALLY HAPPENED in 2025 after DOY 74 ===
    actual_after = [r for r in rows if r["year"] == 2025 and r["day_of_year"] > CUTOFF_DOY
                    and r["total_count"] is not None]

    actual_extreme = sum(1 for r in actual_after if r["total_count"] >= 1500)
    actual_high = sum(1 for r in actual_after if r["total_count"] >= 500)
    actual_over100 = sum(1 for r in actual_after if r["total_count"] >= 100)
    last_ext_actual = [r for r in actual_after if r["total_count"] >= 1500]
    last_100_actual = [r for r in actual_after if r["total_count"] >= 100]

    print(f"\n--- WHAT ACTUALLY HAPPENED (2025 after DOY {CUTOFF_DOY}) ---")
    print(f"  Actual remaining extreme days: {actual_extreme}")
    print(f"  Actual remaining days > 100: {actual_over100}")
    if last_ext_actual:
        print(f"  Actual last extreme day: {last_ext_actual[-1]['date']} (DOY {last_ext_actual[-1]['day_of_year']})")
    if last_100_actual:
        print(f"  Actual last day > 100: {last_100_actual[-1]['date']} (DOY {last_100_actual[-1]['day_of_year']})")

    # === REGRESSION: 14-DAY CHAINED FORECAST from March 15 ===
    print(f"\n--- 14-DAY WEATHER REGRESSION FORECAST (chained from DOY {CUTOFF_DOY}) ---")
    print(f"{'Day':>4} {'Date':>8} {'Predicted':>10} {'Pred Sev':>10} {'Actual':>8} {'Act Sev':>10} {'Error':>8} {'Match':>6}")
    print("-" * 72)

    prev_pred_log = last_log
    correct_severity = 0
    total_compared = 0
    abs_errors_log = []

    for horizon in range(1, 15):
        target_doy = CUTOFF_DOY + horizon
        actual_row = idx.get((2025, target_doy))
        if not actual_row or actual_row["total_count"] is None or actual_row["temp_mean"] is None:
            continue

        # Build features using actual weather but PREDICTED previous pollen
        feats = {
            "prev_log_count": prev_pred_log,
            "temp_mean": actual_row["temp_mean"],
            "temp_max": actual_row["temp_max"] or actual_row["temp_mean"],
            "precip_yesterday": actual_row["precip_yesterday"] or 0,
            "precip_2day_sum": actual_row["precip_2day_sum"] or 0,
            "wind_max": actual_row["wind_max"] or 0,
            "gdd_daily": actual_row["gdd_daily"] or 0,
            "day_of_year": float(target_doy),
            "doy_sin": math.sin(2 * math.pi * target_doy / 365),
            "doy_cos": math.cos(2 * math.pi * target_doy / 365),
        }

        pred_log = model.predict([feats[f] for f in feature_names])
        pred_count = max(0, round(math.exp(pred_log) - 1))
        actual_count = actual_row["total_count"]
        actual_log = actual_row["log_count"]

        pred_sev = classify_severity(pred_count)
        act_sev = classify_severity(actual_count)
        match = "YES" if pred_sev == act_sev else "no"

        err_log = pred_log - actual_log if actual_log else 0
        abs_errors_log.append(abs(err_log))

        if pred_sev == act_sev:
            correct_severity += 1
        total_compared += 1

        print(f"D+{horizon:<2} {doy_to_date(target_doy):>8} {pred_count:>10,} {pred_sev:>10} {actual_count:>8,} {act_sev:>10} {err_log:>+8.2f} {match:>6}")

        prev_pred_log = pred_log  # Chain forward

    if total_compared > 0:
        print(f"\nSeverity accuracy: {correct_severity}/{total_compared} = {100*correct_severity/total_compared:.0f}%")
        print(f"MAE (log scale): {statistics.mean(abs_errors_log):.3f}")

    # === ALSO RUN FROM MULTIPLE START DATES ===
    print(f"\n{'='*70}")
    print("HINDCAST FROM MULTIPLE START DATES IN 2025")
    print(f"{'='*70}")

    for start_doy, start_label in [(32, "Feb 1"), (46, "Feb 15"), (60, "Mar 1"), (74, "Mar 15"), (91, "Apr 1"), (105, "Apr 15")]:
        rows_to_start = [r for r in rows if r["year"] == 2025
                         and r["day_of_year"] <= start_doy and r["total_count"] is not None]
        if not rows_to_start:
            continue

        burden = rows_to_start[-1]["cumulative_burden"]
        last_log_val = rows_to_start[-1]["log_count"]

        # Analog projection
        analogs_local = []
        for yr in complete_years:
            yr_at = [r for r in rows if r["year"] == yr and r["day_of_year"] <= start_doy]
            yr_after = [r for r in rows if r["year"] == yr and r["day_of_year"] > start_doy
                        and r["total_count"] is not None]
            if not yr_at:
                continue
            b = yr_at[-1]["cumulative_burden"]
            rem_ext = sum(1 for r in yr_after if r["total_count"] >= 1500)
            rem_100 = sum(1 for r in yr_after if r["total_count"] >= 100)
            analogs_local.append({"year": yr, "burden": b, "rem_ext": rem_ext, "rem_100": rem_100})

        analogs_local.sort(key=lambda a: abs(a["burden"] - burden))
        t5 = analogs_local[:5]

        # 14-day regression
        prev_p = last_log_val
        sev_correct = 0
        sev_total = 0
        for h in range(1, 15):
            tdoy = start_doy + h
            ar = idx.get((2025, tdoy))
            if not ar or ar["total_count"] is None or ar["temp_mean"] is None:
                continue
            f = {
                "prev_log_count": prev_p,
                "temp_mean": ar["temp_mean"],
                "temp_max": ar["temp_max"] or ar["temp_mean"],
                "precip_yesterday": ar["precip_yesterday"] or 0,
                "precip_2day_sum": ar["precip_2day_sum"] or 0,
                "wind_max": ar["wind_max"] or 0,
                "gdd_daily": ar["gdd_daily"] or 0,
                "day_of_year": float(tdoy),
                "doy_sin": math.sin(2 * math.pi * tdoy / 365),
                "doy_cos": math.cos(2 * math.pi * tdoy / 365),
            }
            pl = model.predict([f[fn] for fn in feature_names])
            pc = max(0, round(math.exp(pl) - 1))
            if classify_severity(pc) == classify_severity(ar["total_count"]):
                sev_correct += 1
            sev_total += 1
            prev_p = pl

        # Actual remaining
        act_after = [r for r in rows if r["year"] == 2025 and r["day_of_year"] > start_doy
                     and r["total_count"] is not None]
        a_ext = sum(1 for r in act_after if r["total_count"] >= 1500)
        a_100 = sum(1 for r in act_after if r["total_count"] >= 100)

        proj_ext = statistics.mean(a["rem_ext"] for a in t5) if t5 else 0
        proj_100 = statistics.mean(a["rem_100"] for a in t5) if t5 else 0

        sev_pct = f"{100*sev_correct/sev_total:.0f}%" if sev_total > 0 else "n/a"

        print(f"\nFrom {start_label} (DOY {start_doy}):")
        print(f"  Burden so far: {burden:,.0f}")
        print(f"  Analog projection: {proj_ext:.1f} extreme days remaining (actual: {a_ext})")
        print(f"  Analog projection: {proj_100:.1f} days > 100 remaining (actual: {a_100})")
        print(f"  14-day severity accuracy: {sev_pct}")
        err_ext = abs(proj_ext - a_ext)
        err_100 = abs(proj_100 - a_100)
        print(f"  Analog error: ±{err_ext:.1f} extreme days, ±{err_100:.1f} days > 100")


if __name__ == "__main__":
    main()
