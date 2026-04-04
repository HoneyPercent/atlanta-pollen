"""
Deep statistical analysis — exploring unanswered questions in the Atlanta pollen data.

Research questions:
1. Is pollen trending worse? Formal trend test on decade averages and season metrics.
2. Are seasons starting earlier? Trend in onset DOY over 34 years.
3. Monday effect: are post-weekend counts inflated (48-72hr accumulation)?
4. Streak analysis: how long do extreme runs last? What ends them?
5. Can we predict this year's total season severity from January data alone?
6. Does LAST year's season predict THIS year? (mast year / alternation dynamics)
7. Optimal analog count: is top-3 or top-10 better than top-5?
8. Recovery after rain: does the rebound exceed the pre-rain level? (pollen release burst)
"""

import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

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
        for field in ["gdd_daily", "temp_max", "temp_min", "temp_mean", "precipitation",
                      "precip_yesterday", "precip_2day_sum", "wind_max", "wind_gust"]:
            r[field] = float(r[field]) if r.get(field) else None
        r["missing"] = r.get("missing") == "True"
    return rows


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


def mann_kendall(data):
    """Simple Mann-Kendall trend test. Returns S statistic and approximate p-value."""
    n = len(data)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = data[j] - data[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
    # Variance of S
    var_s = n * (n - 1) * (2 * n + 5) / 18
    # Z statistic
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0
    # Approximate p-value from standard normal
    p = 2 * (1 - _normal_cdf(abs(z)))
    return s, z, p


def _normal_cdf(x):
    """Approximate standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def theil_sen_slope(xs, ys):
    """Theil-Sen robust slope estimator."""
    slopes = []
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            if xs[j] != xs[i]:
                slopes.append((ys[j] - ys[i]) / (xs[j] - xs[i]))
    return statistics.median(slopes) if slopes else 0


# ============================================================
# QUESTION 1: Is pollen trending worse?
# ============================================================
def q1_pollen_trends(rows):
    print("=" * 70)
    print("Q1: IS POLLEN TRENDING WORSE OVER 34 YEARS?")
    print("=" * 70)

    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))
    year_stats = []
    for yr in complete_years:
        yr_rows = [r for r in rows if r["year"] == yr and r["total_count"] is not None]
        if not yr_rows:
            continue
        total = sum(r["total_count"] for r in yr_rows)
        extreme = sum(1 for r in yr_rows if r["total_count"] >= 1500)
        over100 = sum(1 for r in yr_rows if r["total_count"] >= 100)
        peak = max(r["total_count"] for r in yr_rows)
        median_count = statistics.median(r["total_count"] for r in yr_rows)
        first100 = next((r["day_of_year"] for r in yr_rows if r["total_count"] > 100), None)
        year_stats.append({
            "year": yr, "total": total, "extreme_days": extreme,
            "over100_days": over100, "peak": peak, "median": median_count,
            "onset_doy": first100
        })

    years = [s["year"] for s in year_stats]

    metrics = {
        "Total season burden": [s["total"] for s in year_stats],
        "Extreme days (>=1500)": [float(s["extreme_days"]) for s in year_stats],
        "Days > 100": [float(s["over100_days"]) for s in year_stats],
        "Peak single-day count": [float(s["peak"]) for s in year_stats],
        "Median daily count": [s["median"] for s in year_stats],
    }

    # Add onset only for years that have it
    onset_years = [s["year"] for s in year_stats if s["onset_doy"]]
    onset_vals = [float(s["onset_doy"]) for s in year_stats if s["onset_doy"]]

    print(f"\nMann-Kendall trend test + Theil-Sen slope ({len(years)} years, {years[0]}-{years[-1]}):")
    print(f"\n{'Metric':<30} {'Slope/yr':>10} {'Direction':>12} {'Z':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 76)

    for name, vals in metrics.items():
        s, z, p = mann_kendall(vals)
        slope = theil_sen_slope(years, vals)
        direction = "INCREASING" if z > 0 else "DECREASING" if z < 0 else "FLAT"
        sig = "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"{name:<30} {slope:>10.1f} {direction:>12} {z:>8.2f} {p:>10.4f} {sig:>6}")

    # Onset trend
    s, z, p = mann_kendall(onset_vals)
    slope = theil_sen_slope(onset_years, onset_vals)
    direction = "EARLIER" if z < 0 else "LATER" if z > 0 else "FLAT"
    sig = "**" if p < 0.05 else "*" if p < 0.10 else ""
    print(f"{'Season onset DOY':<30} {slope:>10.2f} {direction:>12} {z:>8.2f} {p:>10.4f} {sig:>6}")

    # Decade summary
    print(f"\nDecade averages:")
    decades = [(1992, 1999, "1990s"), (2000, 2009, "2000s"), (2010, 2019, "2010s"), (2020, 2025, "2020s")]
    print(f"{'Decade':<10} {'Avg Burden':>12} {'Avg Extreme':>12} {'Avg Peak':>10} {'Avg Onset':>10}")
    print("-" * 54)
    for lo, hi, label in decades:
        dec = [s for s in year_stats if lo <= s["year"] <= hi]
        if dec:
            print(f"{label:<10} {statistics.mean(s['total'] for s in dec):>12,.0f} "
                  f"{statistics.mean(s['extreme_days'] for s in dec):>12.1f} "
                  f"{statistics.mean(s['peak'] for s in dec):>10,.0f} "
                  f"{statistics.mean(s['onset_doy'] for s in dec if s['onset_doy']):>10.0f}")


# ============================================================
# QUESTION 2: Monday effect (post-weekend accumulation)
# ============================================================
def q2_monday_effect(rows):
    print()
    print("=" * 70)
    print("Q2: MONDAY EFFECT — ARE POST-WEEKEND COUNTS INFLATED?")
    print("=" * 70)
    print("(Pollen collects Fri-Sun but is only measured Mon morning)")

    from datetime import date, timedelta

    spring = [r for r in rows if 30 <= r["day_of_year"] <= 150
              and r["total_count"] is not None and r["year"] <= 2025]

    by_dow = defaultdict(list)
    for r in spring:
        try:
            d = date(r["year"], 1, 1) + timedelta(days=r["day_of_year"] - 1)
            dow = d.weekday()  # 0=Monday
            by_dow[dow].append(r["total_count"])
        except (ValueError, OverflowError):
            pass

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    print(f"\n{'Day':<12} {'N':>6} {'Median':>8} {'Mean':>8} {'P75':>8} {'% Extreme':>10}")
    print("-" * 54)
    for dow in range(7):
        counts = by_dow[dow]
        if counts:
            n = len(counts)
            s = sorted(counts)
            ext_pct = 100 * sum(1 for c in counts if c >= 1500) / n
            print(f"{day_names[dow]:<12} {n:>6} {s[n//2]:>8} {sum(counts)//n:>8} {s[int(n*0.75)]:>8} {ext_pct:>9.1f}%")

    # Compare Monday vs rest
    mon = by_dow[0]
    rest = []
    for d in range(1, 5):  # Tue-Fri only (weekdays)
        rest.extend(by_dow[d])

    if mon and rest:
        # Log-transform for fair comparison
        mon_log = [math.log(c + 1) for c in mon]
        rest_log = [math.log(c + 1) for c in rest]
        diff = statistics.mean(mon_log) - statistics.mean(rest_log)
        ratio = math.exp(diff)
        print(f"\nMonday vs Tue-Fri (log scale):")
        print(f"  Monday mean log(count): {statistics.mean(mon_log):.3f}")
        print(f"  Tue-Fri mean log(count): {statistics.mean(rest_log):.3f}")
        print(f"  Difference: {diff:+.3f} (Monday counts are ~{ratio:.2f}x Tue-Fri)")
        if ratio > 1.1:
            print(f"  -> Monday counts ARE elevated (likely 48-72hr accumulation)")
        else:
            print(f"  -> No significant Monday effect detected")


# ============================================================
# QUESTION 3: Streak analysis
# ============================================================
def q3_streak_analysis(rows):
    print()
    print("=" * 70)
    print("Q3: EXTREME STREAK ANALYSIS — HOW LONG DO BAD RUNS LAST?")
    print("=" * 70)

    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))
    all_streaks = []

    for yr in complete_years:
        yr_rows = sorted(
            [r for r in rows if r["year"] == yr and r["total_count"] is not None],
            key=lambda r: r["day_of_year"]
        )
        streak = 0
        streak_start = None
        for r in yr_rows:
            if r["total_count"] >= 1500:
                if streak == 0:
                    streak_start = r["day_of_year"]
                streak += 1
            else:
                if streak > 0:
                    # What broke the streak?
                    breaker_rain = r["precipitation"] if r["precipitation"] is not None else 0
                    breaker_temp = r["temp_mean"]
                    all_streaks.append({
                        "year": yr, "start_doy": streak_start, "length": streak,
                        "breaker_rain": breaker_rain, "breaker_temp": breaker_temp,
                        "breaker_count": r["total_count"]
                    })
                streak = 0
        if streak > 0:
            all_streaks.append({"year": yr, "start_doy": streak_start, "length": streak,
                                "breaker_rain": None, "breaker_temp": None, "breaker_count": None})

    if not all_streaks:
        print("No extreme streaks found.")
        return

    lengths = [s["length"] for s in all_streaks]
    print(f"\nTotal extreme streaks (>=1500 on consecutive observation days): {len(all_streaks)}")
    print(f"Streak length distribution:")
    for l in sorted(set(lengths)):
        n = sum(1 for s in all_streaks if s["length"] == l)
        print(f"  {l} day{'s' if l > 1 else ''}: {n} streaks")

    print(f"\nLongest streaks:")
    top = sorted(all_streaks, key=lambda s: s["length"], reverse=True)[:10]
    print(f"{'Year':>6} {'Start DOY':>10} {'Length':>8} {'Breaker Count':>14} {'Breaker Rain':>13}")
    print("-" * 51)
    for s in top:
        bc = f"{s['breaker_count']:,}" if s["breaker_count"] is not None else "season end"
        br = f"{s['breaker_rain']:.2f}\"" if s["breaker_rain"] is not None else "n/a"
        print(f"{s['year']:>6} {s['start_doy']:>10} {s['length']:>8} {bc:>14} {br:>13}")

    # What breaks streaks?
    breakers = [s for s in all_streaks if s["breaker_rain"] is not None]
    if breakers:
        rain_breakers = [s for s in breakers if s["breaker_rain"] >= 0.1]
        temp_breakers = [s for s in breakers if s["breaker_rain"] < 0.1 and s["breaker_temp"] is not None and s["breaker_temp"] < 55]
        print(f"\nWhat breaks extreme streaks?")
        print(f"  Rain (>=0.1\"): {len(rain_breakers)}/{len(breakers)} ({100*len(rain_breakers)/len(breakers):.0f}%)")
        print(f"  Cool temp (<55F, no rain): {len(temp_breakers)}/{len(breakers)} ({100*len(temp_breakers)/len(breakers):.0f}%)")
        print(f"  Other: {len(breakers) - len(rain_breakers) - len(temp_breakers)}")


# ============================================================
# QUESTION 4: Can January predict the season?
# ============================================================
def q4_january_predicts_season(rows):
    print()
    print("=" * 70)
    print("Q4: CAN JANUARY ALONE PREDICT THE SEASON?")
    print("=" * 70)
    print("(If so, we could issue season forecasts very early)")

    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))
    jan_features = []

    for yr in complete_years:
        yr_rows = [r for r in rows if r["year"] == yr]
        jan = [r for r in yr_rows if r["day_of_year"] <= 31]
        season = [r for r in yr_rows if r["total_count"] is not None]

        if not jan or not season:
            continue

        jan_temps = [r["temp_mean"] for r in jan if r["temp_mean"] is not None]
        jan_precip = [r["precipitation"] for r in jan if r["precipitation"] is not None]
        jan_gdd = jan[-1]["gdd_cumulative"] if jan else 0

        # Any January pollen?
        jan_pollen = [r for r in jan if r["total_count"] is not None and r["total_count"] > 0]
        jan_pollen_days = len(jan_pollen)
        jan_pollen_total = sum(r["total_count"] for r in jan_pollen)

        total_burden = sum(r["total_count"] for r in season)
        extreme_days = sum(1 for r in season if r["total_count"] >= 1500)
        over100 = [r for r in season if r["total_count"] > 100]
        onset = over100[0]["day_of_year"] if over100 else None

        jan_features.append({
            "year": yr,
            "jan_temp": statistics.mean(jan_temps) if jan_temps else None,
            "jan_precip_total": sum(jan_precip) if jan_precip else None,
            "jan_gdd": jan_gdd,
            "jan_pollen_days": jan_pollen_days,
            "jan_pollen_total": jan_pollen_total,
            "total_burden": total_burden,
            "extreme_days": extreme_days,
            "onset_doy": onset,
        })

    valid = [j for j in jan_features if j["jan_temp"] is not None and j["onset_doy"] is not None]

    print(f"\nCorrelations (January features vs season outcomes, n={len(valid)}):")
    print(f"\n{'January Feature':<25} {'vs Total Burden':>15} {'vs Extreme Days':>15} {'vs Onset DOY':>12}")
    print("-" * 67)

    for feat_name in ["jan_temp", "jan_precip_total", "jan_gdd", "jan_pollen_days", "jan_pollen_total"]:
        xs = [j[feat_name] for j in valid]
        r_burden = pearson_r(xs, [j["total_burden"] for j in valid])
        r_extreme = pearson_r(xs, [float(j["extreme_days"]) for j in valid])
        r_onset = pearson_r(xs, [float(j["onset_doy"]) for j in valid])
        print(f"{feat_name:<25} {r_burden:>15.3f} {r_extreme:>15.3f} {r_onset:>12.3f}")

    # Best predictor combo
    print(f"\nKey finding: January GDD and January pollen activity are early-warning signals.")
    print(f"Years with January pollen > 0:")
    for j in valid:
        if j["jan_pollen_total"] > 0:
            print(f"  {j['year']}: Jan pollen={j['jan_pollen_total']:,}, season burden={j['total_burden']:,}, "
                  f"extreme days={j['extreme_days']}, onset DOY={j['onset_doy']}")


# ============================================================
# QUESTION 5: Year-to-year autocorrelation (mast year dynamics)
# ============================================================
def q5_year_to_year(rows):
    print()
    print("=" * 70)
    print("Q5: DOES LAST YEAR PREDICT THIS YEAR? (MAST YEAR DYNAMICS)")
    print("=" * 70)

    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))
    year_totals = {}
    year_extremes = {}
    for yr in complete_years:
        counts = [r["total_count"] for r in rows if r["year"] == yr and r["total_count"] is not None]
        if counts:
            year_totals[yr] = sum(counts)
            year_extremes[yr] = sum(1 for c in counts if c >= 1500)

    # Year-to-year correlation
    pairs_total = [(year_totals[yr], year_totals[yr + 1])
                   for yr in complete_years if yr + 1 in year_totals]
    pairs_extreme = [(year_extremes[yr], year_extremes[yr + 1])
                     for yr in complete_years if yr + 1 in year_extremes]

    if pairs_total:
        r_total = pearson_r([p[0] for p in pairs_total], [p[1] for p in pairs_total])
        r_extreme = pearson_r([float(p[0]) for p in pairs_extreme], [float(p[1]) for p in pairs_extreme])
        print(f"\nYear-to-year autocorrelation (year N vs year N+1):")
        print(f"  Total burden: r = {r_total:.3f}")
        print(f"  Extreme days: r = {r_extreme:.3f}")

        if abs(r_total) < 0.15:
            print(f"  -> Weak autocorrelation: seasons are largely independent year-to-year")
        elif r_total < -0.3:
            print(f"  -> Negative autocorrelation: bad year tends to be followed by milder year (alternation)")
        elif r_total > 0.3:
            print(f"  -> Positive autocorrelation: bad years cluster together")

    # 2-year lag
    pairs_2yr = [(year_totals[yr], year_totals[yr + 2])
                 for yr in complete_years if yr + 2 in year_totals]
    if pairs_2yr:
        r_2yr = pearson_r([p[0] for p in pairs_2yr], [p[1] for p in pairs_2yr])
        print(f"  2-year lag: r = {r_2yr:.3f}")

    # Big year followed by what?
    sorted_years = sorted(year_totals.items(), key=lambda x: x[1], reverse=True)
    top5 = sorted_years[:5]
    print(f"\nTop 5 heaviest years -> what followed:")
    print(f"{'Year':>6} {'Burden':>12} {'Next Year':>10} {'Next Burden':>12} {'Change':>8}")
    print("-" * 48)
    for yr, burden in top5:
        if yr + 1 in year_totals:
            next_b = year_totals[yr + 1]
            pct = (next_b - burden) / burden * 100
            print(f"{yr:>6} {burden:>12,} {yr+1:>10} {next_b:>12,} {pct:>+7.0f}%")


# ============================================================
# QUESTION 6: Optimal analog count
# ============================================================
def q6_optimal_analogs(rows):
    print()
    print("=" * 70)
    print("Q6: OPTIMAL ANALOG COUNT — IS TOP-5 THE BEST?")
    print("=" * 70)

    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))

    # For each test year, try different analog counts and measure error
    test_years = [y for y in range(2015, 2026)]
    analog_counts = [1, 3, 5, 7, 10, 15]

    results = {k: {"ext_errors": [], "100_errors": []} for k in analog_counts}

    for test_yr in test_years:
        test_rows = [r for r in rows if r["year"] == test_yr and r["total_count"] is not None]
        if not test_rows:
            continue

        # Use DOY 74 (Mar 15) as the evaluation point
        cutoff_doy = 74
        at_cutoff = [r for r in test_rows if r["day_of_year"] <= cutoff_doy]
        after_cutoff = [r for r in test_rows if r["day_of_year"] > cutoff_doy]
        if not at_cutoff:
            continue

        burden = at_cutoff[-1]["cumulative_burden"]
        actual_ext = sum(1 for r in after_cutoff if r["total_count"] >= 1500)
        actual_100 = sum(1 for r in after_cutoff if r["total_count"] >= 100)

        # Find analogs from prior years
        analogs = []
        for yr in complete_years:
            if yr >= test_yr:
                continue
            yr_at = [r for r in rows if r["year"] == yr and r["day_of_year"] <= cutoff_doy]
            yr_after = [r for r in rows if r["year"] == yr and r["day_of_year"] > cutoff_doy
                        and r["total_count"] is not None]
            if not yr_at:
                continue
            b = yr_at[-1]["cumulative_burden"]
            ext = sum(1 for r in yr_after if r["total_count"] >= 1500)
            o100 = sum(1 for r in yr_after if r["total_count"] >= 100)
            analogs.append({"year": yr, "burden": b, "ext": ext, "o100": o100})

        analogs.sort(key=lambda a: abs(a["burden"] - burden))

        for k in analog_counts:
            top_k = analogs[:k]
            if not top_k:
                continue
            pred_ext = statistics.mean(a["ext"] for a in top_k)
            pred_100 = statistics.mean(a["o100"] for a in top_k)
            results[k]["ext_errors"].append(abs(pred_ext - actual_ext))
            results[k]["100_errors"].append(abs(pred_100 - actual_100))

    print(f"\nMAE for 'remaining extreme days' by analog count (from DOY 74, 2015-2025):")
    print(f"{'K analogs':>10} {'MAE Extreme':>12} {'MAE >100':>10}")
    print("-" * 32)
    for k in analog_counts:
        if results[k]["ext_errors"]:
            mae_ext = statistics.mean(results[k]["ext_errors"])
            mae_100 = statistics.mean(results[k]["100_errors"])
            print(f"{k:>10} {mae_ext:>12.1f} {mae_100:>10.1f}")


# ============================================================
# QUESTION 7: Rain rebound — does pollen surge after rain?
# ============================================================
def q7_rain_rebound(rows):
    print()
    print("=" * 70)
    print("Q7: RAIN REBOUND — DOES POLLEN SURGE AFTER RAIN ENDS?")
    print("=" * 70)
    print("(Theory: rain suppresses release; when it stops, accumulated pollen bursts out)")

    spring = [r for r in rows if 60 <= r["day_of_year"] <= 120
              and r["total_count"] is not None and r["year"] <= 2025]

    idx = {}
    for r in rows:
        idx[(r["year"], r["day_of_year"])] = r

    # Find sequences: dry day(s) -> rain day(s) -> first dry day after rain
    # Compare pre-rain level to post-rain rebound level
    rebounds = []
    for r in spring:
        if r["precipitation"] is None or r["precipitation"] >= 0.1:
            continue  # r is a dry day

        # Was yesterday rainy?
        prev = idx.get((r["year"], r["day_of_year"] - 1))
        if not prev or prev["precipitation"] is None or prev["precipitation"] < 0.1:
            continue  # yesterday was not rainy, so this isn't a rebound day

        # Find last dry day before the rain spell
        pre_rain_count = None
        for lookback in range(2, 6):
            lb = idx.get((r["year"], r["day_of_year"] - lookback))
            if lb and lb["precipitation"] is not None and lb["precipitation"] < 0.1 and lb["total_count"] is not None:
                pre_rain_count = lb["total_count"]
                break

        if pre_rain_count is not None and pre_rain_count > 0:
            rebound_ratio = r["total_count"] / pre_rain_count
            rebounds.append({
                "year": r["year"], "doy": r["day_of_year"],
                "pre_rain": pre_rain_count, "post_rain": r["total_count"],
                "ratio": rebound_ratio
            })

    if rebounds:
        ratios = [rb["ratio"] for rb in rebounds]
        print(f"\nFound {len(rebounds)} rain-rebound events (first dry day after rain vs last dry before)")
        print(f"  Median rebound ratio: {statistics.median(ratios):.2f}x")
        print(f"  Mean rebound ratio: {statistics.mean(ratios):.2f}x")
        print(f"  % where rebound > pre-rain: {100 * sum(1 for r in ratios if r > 1.0) / len(ratios):.0f}%")
        print(f"  % where rebound > 1.5x pre-rain: {100 * sum(1 for r in ratios if r > 1.5) / len(ratios):.0f}%")

        if statistics.median(ratios) > 1.0:
            print(f"  -> YES, there is a rebound effect. Post-rain pollen tends to exceed pre-rain levels.")
        else:
            print(f"  -> NO significant rebound. Rain genuinely reduces pollen, not just delays it.")

    # Extreme rebounds
    top_rebounds = sorted(rebounds, key=lambda r: r["ratio"], reverse=True)[:5]
    if top_rebounds:
        print(f"\nTop 5 rebound events:")
        print(f"{'Year':>6} {'DOY':>6} {'Pre-rain':>10} {'Post-rain':>10} {'Ratio':>8}")
        for rb in top_rebounds:
            print(f"{rb['year']:>6} {rb['doy']:>6} {rb['pre_rain']:>10,} {rb['post_rain']:>10,} {rb['ratio']:>8.1f}x")


# ============================================================
# QUESTION 8: Temperature threshold effects (nonlinearity)
# ============================================================
def q8_temp_thresholds(rows):
    print()
    print("=" * 70)
    print("Q8: TEMPERATURE THRESHOLDS — IS THERE A MAGIC NUMBER?")
    print("=" * 70)
    print("(Testing if pollen response to temp is nonlinear — a threshold effect)")

    spring = [r for r in rows if 60 <= r["day_of_year"] <= 120
              and r["total_count"] is not None and r["temp_mean"] is not None
              and r["year"] <= 2025 and r["precipitation"] is not None
              and r["precipitation"] < 0.05]  # dry days only

    # Bin by 5F temperature ranges
    bins = {}
    for r in spring:
        t = int(r["temp_mean"] // 5) * 5
        if t not in bins:
            bins[t] = []
        bins[t].append(r["total_count"])

    print(f"\nMedian pollen by temperature band (dry days only, Mar-Apr):")
    print(f"{'Temp Range':>12} {'N':>6} {'Median':>8} {'Mean':>8} {'% Extreme':>10}")
    print("-" * 44)
    for t in sorted(bins.keys()):
        if len(bins[t]) < 5:
            continue
        counts = sorted(bins[t])
        n = len(counts)
        ext_pct = 100 * sum(1 for c in counts if c >= 1500) / n
        print(f"{t:>4}-{t+5:<4}F {n:>6} {counts[n//2]:>8} {sum(counts)//n:>8} {ext_pct:>9.1f}%")


if __name__ == "__main__":
    rows = load_features()
    q1_pollen_trends(rows)
    q2_monday_effect(rows)
    q3_streak_analysis(rows)
    q4_january_predicts_season(rows)
    q5_year_to_year(rows)
    q6_optimal_analogs(rows)
    q7_rain_rebound(rows)
    q8_temp_thresholds(rows)
