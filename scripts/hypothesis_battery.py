"""
Hypothesis battery — five rounds of analysis, each building on prior findings.

Round 1: Species succession model — do named top contributors predict what's coming next?
Round 2: Weather regime classification — are there distinct "pollen weather types"?
Round 3: Season shape clustering — do seasons fall into distinct archetypes?
Round 4: Predictive power by season phase — is the model better early or late?
Round 5: "Surprise" days analysis — when does the model fail and why?
"""

import csv
import math
import statistics
from collections import defaultdict, Counter
from pathlib import Path
from datetime import date, timedelta

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


def pearson_r(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / (n - 1))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / (n - 1))
    if sx == 0 or sy == 0:
        return 0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / ((n - 1) * sx * sy)


# ============================================================
# ROUND 1: Weather regime classification
# ============================================================
def round1_weather_regimes(rows):
    print("=" * 70)
    print("ROUND 1: WEATHER REGIME CLASSIFICATION")
    print("=" * 70)
    print("Hypothesis: There are distinct 'pollen weather types' that predict")
    print("severity better than individual variables.")
    print()

    spring = [r for r in rows if 60 <= r["day_of_year"] <= 120
              and r["total_count"] is not None and r["year"] <= 2025
              and r["temp_mean"] is not None and r["precipitation"] is not None
              and r["wind_max"] is not None]

    # Classify each day into a weather regime
    for r in spring:
        t = r["temp_mean"]
        p = r["precipitation"]
        w = r["wind_max"]

        if p >= 0.25:
            r["regime"] = "Rainy"
        elif p >= 0.05:
            r["regime"] = "Drizzle"
        elif t >= 65 and w >= 10:
            r["regime"] = "Hot+Windy"
        elif t >= 65 and w < 10:
            r["regime"] = "Hot+Calm"
        elif 50 <= t < 65 and w >= 10:
            r["regime"] = "Warm+Windy"
        elif 50 <= t < 65 and w < 10:
            r["regime"] = "Warm+Calm"
        elif t < 50:
            r["regime"] = "Cool"
        else:
            r["regime"] = "Other"

    regime_stats = defaultdict(list)
    for r in spring:
        regime_stats[r["regime"]].append(r["total_count"])

    print(f"{'Regime':<15} {'N':>6} {'Median':>8} {'Mean':>8} {'P90':>8} {'% Extreme':>10} {'% Low':>8}")
    print("-" * 64)
    for regime in ["Hot+Windy", "Hot+Calm", "Warm+Windy", "Warm+Calm", "Cool", "Drizzle", "Rainy"]:
        counts = sorted(regime_stats[regime])
        if not counts:
            continue
        n = len(counts)
        ext_pct = 100 * sum(1 for c in counts if c >= 1500) / n
        low_pct = 100 * sum(1 for c in counts if c < 100) / n
        print(f"{regime:<15} {n:>6} {counts[n//2]:>8} {sum(counts)//n:>8} "
              f"{counts[int(n*0.90)]:>8} {ext_pct:>9.1f}% {low_pct:>7.1f}%")

    # Regime transitions: what follows what?
    print(f"\nREGIME TRANSITIONS: What happens the day after each regime?")
    print(f"(Median next-day pollen count)")
    transitions = defaultdict(list)
    idx = build_index(rows)
    for r in spring:
        nxt = idx.get((r["year"], r["day_of_year"] + 1))
        if nxt and nxt["total_count"] is not None:
            transitions[r["regime"]].append(nxt["total_count"])

    print(f"\n{'Todays Regime':<15} {'Next-Day Med':>12} {'Next-Day Mean':>13} {'Next % Ext':>10}")
    print("-" * 50)
    for regime in ["Hot+Windy", "Hot+Calm", "Warm+Windy", "Warm+Calm", "Cool", "Drizzle", "Rainy"]:
        if transitions[regime]:
            counts = sorted(transitions[regime])
            n = len(counts)
            ext_pct = 100 * sum(1 for c in counts if c >= 1500) / n
            print(f"{regime:<15} {counts[n//2]:>12} {sum(counts)//n:>13} {ext_pct:>9.1f}%")

    # Key finding: Rainy -> next day
    if transitions["Rainy"]:
        rainy_next = sorted(transitions["Rainy"])
        n = len(rainy_next)
        print(f"\nAfter a Rainy day, next day is Low (<100) {100*sum(1 for c in rainy_next if c < 100)/n:.0f}% of the time")


# ============================================================
# ROUND 2: Season shape clustering
# ============================================================
def round2_season_shapes(rows):
    print()
    print("=" * 70)
    print("ROUND 2: SEASON SHAPE ARCHETYPES")
    print("=" * 70)
    print("Hypothesis: Seasons fall into distinct shapes (early-peak, late-peak,")
    print("double-peak, gradual, explosive) and the shape predicts severity.")
    print()

    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))
    year_profiles = []

    for yr in complete_years:
        yr_rows = [r for r in rows if r["year"] == yr and r["total_count"] is not None]
        if len(yr_rows) < 20:
            continue

        total = sum(r["total_count"] for r in yr_rows)
        if total == 0:
            continue

        # Peak timing (DOY of max count)
        peak_row = max(yr_rows, key=lambda r: r["total_count"])
        peak_doy = peak_row["day_of_year"]
        peak_count = peak_row["total_count"]

        # How much of the season burden is in the top 5 days? (concentration)
        top5 = sorted([r["total_count"] for r in yr_rows], reverse=True)[:5]
        top5_share = sum(top5) / total * 100

        # How much is in the first half vs second half of the season?
        spring = [r for r in yr_rows if 30 <= r["day_of_year"] <= 150]
        if not spring:
            continue
        mid_doy = (spring[0]["day_of_year"] + spring[-1]["day_of_year"]) // 2
        first_half = sum(r["total_count"] for r in spring if r["day_of_year"] <= mid_doy)
        second_half = sum(r["total_count"] for r in spring if r["day_of_year"] > mid_doy)
        skewness = first_half / (first_half + second_half) * 100 if (first_half + second_half) > 0 else 50

        # Number of distinct peaks (days > 2x the seasonal mean)
        mean_count = total / len(yr_rows)
        peak_days = sum(1 for r in yr_rows if r["total_count"] > 2 * mean_count)

        # Season ramp-up speed: days from first >100 to first >1500
        over100 = [r for r in yr_rows if r["total_count"] > 100]
        over1500 = [r for r in yr_rows if r["total_count"] >= 1500]
        ramp_days = (over1500[0]["day_of_year"] - over100[0]["day_of_year"]) if over100 and over1500 else None

        # Extreme days
        extreme = sum(1 for r in yr_rows if r["total_count"] >= 1500)

        # Classify shape
        if skewness > 65:
            shape = "Front-loaded"
        elif skewness < 35:
            shape = "Back-loaded"
        elif top5_share > 50:
            shape = "Explosive (concentrated)"
        elif ramp_days is not None and ramp_days <= 7:
            shape = "Fast ramp"
        elif ramp_days is not None and ramp_days > 21:
            shape = "Slow build"
        else:
            shape = "Balanced"

        year_profiles.append({
            "year": yr, "total": total, "peak_doy": peak_doy, "peak_count": peak_count,
            "top5_share": top5_share, "skewness": skewness, "peak_days": peak_days,
            "ramp_days": ramp_days, "extreme": extreme, "shape": shape
        })

    # Show all years
    print(f"{'Year':>6} {'Shape':<25} {'Total':>10} {'Peak DOY':>9} {'Top5 %':>8} {'Skew %':>8} {'Ramp':>6} {'Ext':>5}")
    print("-" * 77)
    for p in year_profiles:
        rd = f"{p['ramp_days']}d" if p["ramp_days"] is not None else "n/a"
        print(f"{p['year']:>6} {p['shape']:<25} {p['total']:>10,} {p['peak_doy']:>9} "
              f"{p['top5_share']:>7.1f}% {p['skewness']:>7.1f}% {rd:>6} {p['extreme']:>5}")

    # Aggregate by shape
    print(f"\nAGGREGATE BY SHAPE:")
    shapes = defaultdict(list)
    for p in year_profiles:
        shapes[p["shape"]].append(p)

    print(f"{'Shape':<25} {'N':>4} {'Avg Burden':>12} {'Avg Extreme':>12} {'Avg Peak':>10}")
    print("-" * 63)
    for shape, profiles in sorted(shapes.items(), key=lambda x: -statistics.mean(p["total"] for p in x[1])):
        print(f"{shape:<25} {len(profiles):>4} {statistics.mean(p['total'] for p in profiles):>12,.0f} "
              f"{statistics.mean(p['extreme'] for p in profiles):>12.1f} "
              f"{statistics.mean(p['peak_count'] for p in profiles):>10,.0f}")

    # Can we predict shape from preseason data?
    print(f"\nDoes Jan-Feb warmth predict season shape?")
    for p in year_profiles:
        yr_rows = [r for r in rows if r["year"] == p["year"] and r["day_of_year"] <= 59
                   and r.get("temp_mean") is not None]
        p["janfeb_temp"] = statistics.mean(r["temp_mean"] for r in yr_rows) if yr_rows else None

    for shape, profiles in shapes.items():
        temps = [p["janfeb_temp"] for p in profiles if p["janfeb_temp"] is not None]
        if temps:
            print(f"  {shape}: avg Jan-Feb temp = {statistics.mean(temps):.1f}F (n={len(temps)})")


# ============================================================
# ROUND 3: Predictive power by season phase
# ============================================================
def round3_phase_accuracy(rows):
    print()
    print("=" * 70)
    print("ROUND 3: MODEL ACCURACY BY SEASON PHASE")
    print("=" * 70)
    print("Hypothesis: The model is better during some phases than others.")
    print("Understanding this tells us where to focus improvement efforts.")
    print()

    idx = build_index(rows)

    # Define phases by season progress %
    phases = [
        ("Pre-season (0-10%)", 0, 10),
        ("Ramp-up (10-30%)", 10, 30),
        ("Peak (30-70%)", 30, 70),
        ("Decline (70-90%)", 70, 90),
        ("Tail (90-100%)", 90, 100),
    ]

    # Simple next-day prediction errors by phase
    phase_errors = {name: {"abs_errors": [], "sev_correct": 0, "sev_total": 0} for name, _, _ in phases}

    spring = [r for r in rows if 30 <= r["day_of_year"] <= 150
              and r["total_count"] is not None and r["year"] <= 2025 and r["year"] >= 2000]

    for r in spring:
        prev = idx.get((r["year"], r["day_of_year"] - 1))
        if not prev or prev["log_count"] is None or r["log_count"] is None:
            continue

        # Simple persistence error
        err = abs(prev["log_count"] - r["log_count"])
        pct = r["season_progress_pct"]

        # Severity match
        def sev(c):
            if c >= 1500: return "E"
            elif c >= 500: return "H"
            elif c >= 100: return "M"
            else: return "L"

        prev_sev = sev(prev["total_count"]) if prev["total_count"] is not None else "L"
        act_sev = sev(r["total_count"])

        for name, lo, hi in phases:
            if lo <= pct < hi:
                phase_errors[name]["abs_errors"].append(err)
                phase_errors[name]["sev_total"] += 1
                if prev_sev == act_sev:
                    phase_errors[name]["sev_correct"] += 1
                break

    print(f"Persistence model error by season phase:")
    print(f"{'Phase':<25} {'N':>6} {'MAE (log)':>10} {'Sev Accuracy':>13}")
    print("-" * 54)
    for name, _, _ in phases:
        pe = phase_errors[name]
        if pe["abs_errors"]:
            mae = statistics.mean(pe["abs_errors"])
            sev_acc = pe["sev_correct"] / pe["sev_total"] * 100 if pe["sev_total"] > 0 else 0
            print(f"{name:<25} {len(pe['abs_errors']):>6} {mae:>10.3f} {sev_acc:>12.1f}%")

    # Day-to-day volatility by phase
    print(f"\nDay-to-day count volatility (std of log-count changes):")
    for name, lo, hi in phases:
        changes = []
        for r in spring:
            prev = idx.get((r["year"], r["day_of_year"] - 1))
            if prev and prev["log_count"] is not None and r["log_count"] is not None:
                if lo <= r["season_progress_pct"] < hi:
                    changes.append(r["log_count"] - prev["log_count"])
        if len(changes) > 5:
            print(f"  {name}: std={statistics.stdev(changes):.3f}, "
                  f"mean change={statistics.mean(changes):+.3f}")


# ============================================================
# ROUND 4: "Surprise" days — when does the model fail?
# ============================================================
def round4_surprise_days(rows):
    print()
    print("=" * 70)
    print("ROUND 4: SURPRISE DAYS — WHEN DOES PERSISTENCE FAIL?")
    print("=" * 70)
    print("Hypothesis: Model failures cluster around specific weather transitions.")
    print()

    idx = build_index(rows)
    spring = [r for r in rows if 60 <= r["day_of_year"] <= 120
              and r["total_count"] is not None and r["year"] <= 2025]

    surprises_up = []   # Predicted low, actual high
    surprises_down = []  # Predicted high, actual low

    for r in spring:
        prev = idx.get((r["year"], r["day_of_year"] - 1))
        if not prev or prev["total_count"] is None:
            continue

        log_err = (r["log_count"] or 0) - (prev["log_count"] or 0)

        if log_err > 2.0:  # Massive upward surprise (>7x increase)
            surprises_up.append({
                "year": r["year"], "doy": r["day_of_year"],
                "prev_count": prev["total_count"], "actual": r["total_count"],
                "ratio": r["total_count"] / max(prev["total_count"], 1),
                "temp": r["temp_mean"], "precip_yest": r["precip_yesterday"],
                "wind": r["wind_max"],
                "prev_precip": prev["precipitation"],
            })
        elif log_err < -2.0:  # Massive downward surprise
            surprises_down.append({
                "year": r["year"], "doy": r["day_of_year"],
                "prev_count": prev["total_count"], "actual": r["total_count"],
                "ratio": r["total_count"] / max(prev["total_count"], 1),
                "temp": r["temp_mean"], "precip_yest": r["precip_yesterday"],
                "wind": r["wind_max"],
                "today_precip": r["precipitation"],
            })

    print(f"Upward surprises (actual >> predicted): {len(surprises_up)}")
    print(f"Downward surprises (actual << predicted): {len(surprises_down)}")

    # Analyze upward surprises
    if surprises_up:
        print(f"\nUPWARD SURPRISES (pollen jumped unexpectedly):")
        print(f"{'Year':>6} {'DOY':>5} {'Prev':>8} {'Actual':>8} {'Ratio':>7} {'Temp':>6} {'Precip Yest':>12} {'Wind':>6}")
        print("-" * 60)
        for s in sorted(surprises_up, key=lambda x: x["ratio"], reverse=True)[:15]:
            py = f"{s['precip_yest']:.2f}\"" if s["precip_yest"] is not None else "n/a"
            print(f"{s['year']:>6} {s['doy']:>5} {s['prev_count']:>8,} {s['actual']:>8,} "
                  f"{s['ratio']:>6.1f}x {s['temp'] or 0:>5.0f}F {py:>12} {s['wind'] or 0:>5.0f}")

        # What do upward surprises have in common?
        temps = [s["temp"] for s in surprises_up if s["temp"] is not None]
        winds = [s["wind"] for s in surprises_up if s["wind"] is not None]
        precip_yest = [s["precip_yest"] for s in surprises_up if s["precip_yest"] is not None]

        print(f"\nUpward surprise weather profile:")
        print(f"  Median temp: {statistics.median(temps):.0f}F" if temps else "  No temp data")
        print(f"  Median wind: {statistics.median(winds):.0f} mph" if winds else "  No wind data")
        print(f"  % with rain yesterday: {100*sum(1 for p in precip_yest if p >= 0.1)/len(precip_yest):.0f}%" if precip_yest else "")
        print(f"  % following a dry spell (precip_yest < 0.01): {100*sum(1 for p in precip_yest if p < 0.01)/len(precip_yest):.0f}%" if precip_yest else "")

    # Analyze downward surprises
    if surprises_down:
        print(f"\nDOWNWARD SURPRISES (pollen dropped unexpectedly):")
        today_precip = [s["today_precip"] for s in surprises_down if s["today_precip"] is not None]
        temps_d = [s["temp"] for s in surprises_down if s["temp"] is not None]

        print(f"  N = {len(surprises_down)}")
        print(f"  % with rain today: {100*sum(1 for p in today_precip if p >= 0.1)/len(today_precip):.0f}%" if today_precip else "")
        print(f"  % with heavy rain today (>0.5\"): {100*sum(1 for p in today_precip if p >= 0.5)/len(today_precip):.0f}%" if today_precip else "")
        print(f"  Median temp: {statistics.median(temps_d):.0f}F" if temps_d else "")
        print(f"  % temp < 50F: {100*sum(1 for t in temps_d if t < 50)/len(temps_d):.0f}%" if temps_d else "")


# ============================================================
# ROUND 5: Pollen persistence — how many days does a reading "echo"?
# ============================================================
def round5_autocorrelation_decay(rows):
    print()
    print("=" * 70)
    print("ROUND 5: AUTOCORRELATION DECAY — HOW FAR BACK MATTERS?")
    print("=" * 70)
    print("Hypothesis: We only use yesterday's count, but maybe 2-3 day lookback")
    print("would improve predictions (captures the 'echo' of pollen staying airborne).")
    print()

    idx = build_index(rows)
    spring = [r for r in rows if 30 <= r["day_of_year"] <= 150
              and r["total_count"] is not None and r["year"] <= 2025
              and r["log_count"] is not None]

    # Compute autocorrelation at lags 1-7
    print(f"{'Lag':>5} {'r (log)':>10} {'r (raw)':>10} {'Notes':>30}")
    print("-" * 55)
    for lag in range(1, 8):
        pairs_log = []
        pairs_raw = []
        for r in spring:
            prev = idx.get((r["year"], r["day_of_year"] - lag))
            if prev and prev["log_count"] is not None and prev["total_count"] is not None:
                pairs_log.append((prev["log_count"], r["log_count"]))
                pairs_raw.append((float(prev["total_count"]), float(r["total_count"])))

        if len(pairs_log) > 10:
            r_log = pearson_r([p[0] for p in pairs_log], [p[1] for p in pairs_log])
            r_raw = pearson_r([p[0] for p in pairs_raw], [p[1] for p in pairs_raw])
            note = ""
            if lag == 1:
                note = "*** Currently used"
            elif r_log > 0.5:
                note = "** Strong — worth adding"
            elif r_log > 0.3:
                note = "* Moderate"
            print(f"  D-{lag} {r_log:>10.3f} {r_raw:>10.3f} {note:>30}")

    # Partial autocorrelation: after controlling for D-1, does D-2 add value?
    print(f"\nPARTIAL AUTOCORRELATION TEST:")
    print(f"Does D-2 add value AFTER controlling for D-1?")

    # Simple: compute residual of D-1 prediction, then correlate residual with D-2
    residuals = []
    d2_vals = []
    for r in spring:
        d1 = idx.get((r["year"], r["day_of_year"] - 1))
        d2 = idx.get((r["year"], r["day_of_year"] - 2))
        if d1 and d2 and d1["log_count"] is not None and d2["log_count"] is not None:
            residual = r["log_count"] - d1["log_count"]  # what D-1 didn't predict
            residuals.append(residual)
            d2_vals.append(d2["log_count"])

    if len(residuals) > 10:
        r_partial = pearson_r(d2_vals, residuals)
        print(f"  Correlation of D-2 with residual after D-1: r = {r_partial:.3f}")
        if abs(r_partial) > 0.1:
            print(f"  -> YES, D-2 has independent predictive value (partial r = {r_partial:.3f})")
            print(f"     Consider adding D-2 log count as a feature")
        else:
            print(f"  -> D-2 adds negligible value after D-1 is accounted for")

    # Same for D-3
    residuals3, d3_vals = [], []
    for r in spring:
        d1 = idx.get((r["year"], r["day_of_year"] - 1))
        d3 = idx.get((r["year"], r["day_of_year"] - 3))
        if d1 and d3 and d1["log_count"] is not None and d3["log_count"] is not None:
            residual = r["log_count"] - d1["log_count"]
            residuals3.append(residual)
            d3_vals.append(d3["log_count"])

    if len(residuals3) > 10:
        r_partial3 = pearson_r(d3_vals, residuals3)
        print(f"  Correlation of D-3 with residual after D-1: r = {r_partial3:.3f}")


# ============================================================
# ROUND 6 (BONUS): Pollen vs temperature derivative
# ============================================================
def round6_temp_derivative(rows):
    print()
    print("=" * 70)
    print("ROUND 6: TEMPERATURE CHANGE (DERIVATIVE) AS PREDICTOR")
    print("=" * 70)
    print("Hypothesis: It's not the absolute temperature that matters most,")
    print("but the CHANGE in temperature. A warming trend triggers pollen release.")
    print()

    idx = build_index(rows)
    spring = [r for r in rows if 30 <= r["day_of_year"] <= 150
              and r["total_count"] is not None and r["year"] <= 2025
              and r["temp_mean"] is not None]

    # Compute temperature derivatives
    temp_changes = []
    pollen_logs = []
    for r in spring:
        prev = idx.get((r["year"], r["day_of_year"] - 1))
        if prev and prev["temp_mean"] is not None and r["log_count"] is not None:
            delta_t = r["temp_mean"] - prev["temp_mean"]
            temp_changes.append(delta_t)
            pollen_logs.append(r["log_count"])

    if temp_changes:
        r_val = pearson_r(temp_changes, pollen_logs)
        print(f"Correlation of 1-day temp change with log(pollen): r = {r_val:.3f}")

    # 3-day warming trend
    warming_3d = []
    pollen_3d = []
    for r in spring:
        d3 = idx.get((r["year"], r["day_of_year"] - 3))
        if d3 and d3["temp_mean"] is not None and r["log_count"] is not None:
            delta = r["temp_mean"] - d3["temp_mean"]
            warming_3d.append(delta)
            pollen_3d.append(r["log_count"])

    if warming_3d:
        r_3d = pearson_r(warming_3d, pollen_3d)
        print(f"Correlation of 3-day temp change with log(pollen): r = {r_3d:.3f}")

    # Bin analysis
    print(f"\nPollen by temperature CHANGE (1-day):")
    print(f"{'Temp Change':>15} {'N':>6} {'Median Pollen':>14} {'% Extreme':>10}")
    print("-" * 45)
    bins = [(-100, -10, "Big drop (>10F)"), (-10, -5, "Drop (5-10F)"), (-5, -1, "Slight drop"),
            (-1, 1, "Stable"), (1, 5, "Slight warm"), (5, 10, "Warm (5-10F)"), (10, 100, "Big warm (>10F)")]
    for lo, hi, label in bins:
        group = [(tc, pl) for tc, pl in zip(temp_changes, pollen_logs) if lo <= tc < hi]
        if len(group) >= 5:
            counts = [math.exp(p) - 1 for _, p in group]
            ext_pct = 100 * sum(1 for c in counts if c >= 1500) / len(counts)
            med = sorted(counts)[len(counts)//2]
            print(f"{label:>15} {len(group):>6} {med:>14,.0f} {ext_pct:>9.1f}%")


# ============================================================
# ROUND 7: Weekend gap imputation test
# ============================================================
def round7_gap_dynamics(rows):
    print()
    print("=" * 70)
    print("ROUND 7: WHAT HAPPENS DURING DATA GAPS?")
    print("=" * 70)
    print("Hypothesis: The model can learn from what happens around gaps")
    print("(weekends/holidays) to infer what we missed and improve predictions.")
    print()

    idx = build_index(rows)

    # Find all gaps: a day with data followed by 1+ missing days followed by data
    gaps = []
    for yr in range(1992, 2026):
        for doy in range(30, 150):
            before = idx.get((yr, doy))
            if not before or before["total_count"] is None:
                continue
            # Count gap length
            gap_len = 0
            for d in range(1, 5):
                mid = idx.get((yr, doy + d))
                if mid and mid["total_count"] is not None:
                    break
                gap_len += 1
            else:
                continue

            if gap_len == 0:
                continue

            after = idx.get((yr, doy + gap_len + 1))
            if not after or after["total_count"] is None:
                continue

            gaps.append({
                "year": yr, "before_doy": doy, "gap_len": gap_len,
                "before_count": before["total_count"],
                "after_count": after["total_count"],
                "ratio": after["total_count"] / max(before["total_count"], 1),
            })

    # Analyze by gap length
    print(f"{'Gap Length':>10} {'N':>6} {'Med Before':>10} {'Med After':>10} {'Med Ratio':>10}")
    print("-" * 46)
    by_gap = defaultdict(list)
    for g in gaps:
        by_gap[g["gap_len"]].append(g)

    for gl in sorted(by_gap.keys()):
        group = by_gap[gl]
        before = sorted(g["before_count"] for g in group)
        after = sorted(g["after_count"] for g in group)
        ratios = sorted(g["ratio"] for g in group)
        n = len(group)
        print(f"{gl:>10}d {n:>6} {before[n//2]:>10,} {after[n//2]:>10,} {ratios[n//2]:>10.2f}x")

    # Does a gap followed by high pollen suggest pollen was building during the gap?
    print(f"\nAfter 2-day gaps, how often does pollen jump vs drop?")
    twod = by_gap.get(2, [])
    if twod:
        jumps = sum(1 for g in twod if g["ratio"] > 1.5)
        drops = sum(1 for g in twod if g["ratio"] < 0.67)
        stable = len(twod) - jumps - drops
        print(f"  Jumped (>1.5x): {jumps} ({100*jumps/len(twod):.0f}%)")
        print(f"  Dropped (<0.67x): {drops} ({100*drops/len(twod):.0f}%)")
        print(f"  Stable: {stable} ({100*stable/len(twod):.0f}%)")


if __name__ == "__main__":
    rows = load_features()
    round1_weather_regimes(rows)
    round2_season_shapes(rows)
    round3_phase_accuracy(rows)
    round4_surprise_days(rows)
    round5_autocorrelation_decay(rows)
    round6_temp_derivative(rows)
    round7_gap_dynamics(rows)
