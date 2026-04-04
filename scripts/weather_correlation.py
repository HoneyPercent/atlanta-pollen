"""
Weather-pollen correlation analysis for Atlanta.

Questions to answer:
1. Which weather variables best predict next-day pollen count?
2. How strong is the rain wash-out effect? How many days does it last?
3. Does warm+dry+windy = high pollen?
4. Does January-February warmth predict earlier/worse seasons?
5. Can a simple weather regression beat climatology for daily prediction?
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
        r["gdd_daily"] = float(r["gdd_daily"]) if r["gdd_daily"] else 0
        r["temp_max"] = float(r["temp_max"]) if r["temp_max"] else None
        r["temp_min"] = float(r["temp_min"]) if r["temp_min"] else None
        r["temp_mean"] = float(r["temp_mean"]) if r["temp_mean"] else None
        r["precipitation"] = float(r["precipitation"]) if r["precipitation"] else None
        r["precip_yesterday"] = float(r["precip_yesterday"]) if r["precip_yesterday"] else None
        r["precip_2day_sum"] = float(r["precip_2day_sum"]) if r["precip_2day_sum"] else None
        r["wind_max"] = float(r["wind_max"]) if r["wind_max"] else None
        r["wind_gust"] = float(r["wind_gust"]) if r["wind_gust"] else None
        r["missing"] = r["missing"] == "True"
    return rows


def pearson_r(xs, ys):
    """Simple Pearson correlation coefficient."""
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


def analyze_correlations(rows):
    """Compute correlations between weather variables and pollen count."""
    print("=" * 60)
    print("WEATHER-POLLEN CORRELATIONS")
    print("=" * 60)

    # Filter to spring (DOY 30-150, ~Feb-May) with data
    spring = [r for r in rows if 30 <= r["day_of_year"] <= 150
              and r["total_count"] is not None and r["year"] <= 2025]

    log_counts = [r["log_count"] for r in spring]
    raw_counts = [r["total_count"] for r in spring]

    # Build lagged features: previous day's count
    # Index all rows by (year, doy) for lookups
    by_yr_doy = {}
    for r in rows:
        by_yr_doy[(r["year"], r["day_of_year"])] = r

    prev_counts = []
    prev_log_counts = []
    for r in spring:
        prev = by_yr_doy.get((r["year"], r["day_of_year"] - 1))
        if prev and prev["total_count"] is not None:
            prev_counts.append(prev["total_count"])
            prev_log_counts.append(prev["log_count"])
        else:
            prev_counts.append(None)
            prev_log_counts.append(None)

    features = {
        "temp_max": [r["temp_max"] for r in spring],
        "temp_min": [r["temp_min"] for r in spring],
        "temp_mean": [r["temp_mean"] for r in spring],
        "precipitation (today)": [r["precipitation"] for r in spring],
        "precip_yesterday": [r["precip_yesterday"] for r in spring],
        "precip_2day_sum": [r["precip_2day_sum"] for r in spring],
        "wind_max": [r["wind_max"] for r in spring],
        "wind_gust": [r["wind_gust"] for r in spring],
        "gdd_daily": [r["gdd_daily"] for r in spring],
        "gdd_cumulative": [r["gdd_cumulative"] for r in spring],
        "day_of_year": [float(r["day_of_year"]) for r in spring],
        "cumulative_burden": [r["cumulative_burden"] for r in spring],
        "prev_day_count": prev_counts,
        "prev_day_log_count": prev_log_counts,
    }

    print(f"\nCorrelation with log(pollen count) during spring (n={len(spring)}):\n")
    print(f"{'Feature':<25} {'r (log count)':>15} {'r (raw count)':>15}")
    print("-" * 55)

    correlations = []
    for name, vals in features.items():
        # Filter to pairs where both are non-None
        pairs_log = [(v, lc) for v, lc in zip(vals, log_counts) if v is not None and lc is not None]
        pairs_raw = [(v, rc) for v, rc in zip(vals, raw_counts) if v is not None and rc is not None]

        if len(pairs_log) >= 10:
            r_log = pearson_r([p[0] for p in pairs_log], [p[1] for p in pairs_log])
            r_raw = pearson_r([p[0] for p in pairs_raw], [p[1] for p in pairs_raw])
            correlations.append((name, r_log, r_raw, len(pairs_log)))
            print(f"{name:<25} {r_log:>15.3f} {r_raw:>15.3f}")

    correlations.sort(key=lambda x: abs(x[1] or 0), reverse=True)
    print(f"\nRanked by |r| with log(count):")
    for name, r_log, r_raw, n in correlations:
        bar = "#" * int(abs(r_log or 0) * 40)
        sign = "+" if (r_log or 0) > 0 else "-"
        print(f"  {sign} {abs(r_log or 0):.3f}  {name:<25} {bar}")

    return correlations


def analyze_rain_washout(rows):
    """Quantify the rain wash-out effect on pollen."""
    print()
    print("=" * 60)
    print("RAIN WASH-OUT EFFECT")
    print("=" * 60)

    spring = [r for r in rows if 30 <= r["day_of_year"] <= 150
              and r["total_count"] is not None and r["year"] <= 2025]

    # Group by precipitation bins
    bins = {
        "No rain (0)": (0, 0.001),
        "Trace (0-0.1 in)": (0.001, 0.1),
        "Light (0.1-0.25 in)": (0.1, 0.25),
        "Moderate (0.25-0.5 in)": (0.25, 0.5),
        "Heavy (0.5+ in)": (0.5, 100),
    }

    print(f"\nPollen count by same-day precipitation:")
    print(f"{'Rain category':<25} {'N':>6} {'Median':>8} {'Mean':>8} {'P75':>8}")
    print("-" * 55)
    for name, (lo, hi) in bins.items():
        group = [r for r in spring if r["precipitation"] is not None and lo <= r["precipitation"] < hi]
        if group:
            counts = sorted(r["total_count"] for r in group)
            n = len(counts)
            print(f"{name:<25} {n:>6} {counts[n//2]:>8} {sum(counts)//n:>8} {counts[int(n*0.75)]:>8}")

    # Same analysis for YESTERDAY's rain -> today's count
    print(f"\nPollen count by YESTERDAY's precipitation:")
    print(f"{'Rain yesterday':<25} {'N':>6} {'Median':>8} {'Mean':>8} {'P75':>8}")
    print("-" * 55)
    for name, (lo, hi) in bins.items():
        group = [r for r in spring if r["precip_yesterday"] is not None and lo <= r["precip_yesterday"] < hi]
        if group:
            counts = sorted(r["total_count"] for r in group)
            n = len(counts)
            print(f"{name:<25} {n:>6} {counts[n//2]:>8} {sum(counts)//n:>8} {counts[int(n*0.75)]:>8}")

    # Days after rain: how long does suppression last?
    print(f"\nDays since last rain (>= 0.1 in) and pollen count:")
    by_yr_doy = {}
    for r in rows:
        by_yr_doy[(r["year"], r["day_of_year"])] = r

    days_since_bins = defaultdict(list)
    for r in spring:
        # Count days since last rain >= 0.1"
        days_back = 0
        for d in range(1, 8):
            prev = by_yr_doy.get((r["year"], r["day_of_year"] - d))
            if prev and prev["precipitation"] is not None and prev["precipitation"] >= 0.1:
                days_back = d
                break
        else:
            days_back = 7  # 7+ days since rain

        days_since_bins[days_back].append(r["total_count"])

    print(f"{'Days since rain':<20} {'N':>6} {'Median':>8} {'Mean':>8}")
    print("-" * 42)
    for d in sorted(days_since_bins.keys()):
        counts = days_since_bins[d]
        n = len(counts)
        label = f"{d} days" if d < 7 else "7+ days"
        print(f"{label:<20} {n:>6} {sorted(counts)[n//2]:>8} {sum(counts)//n:>8}")


def analyze_temp_wind_interaction(rows):
    """Warm + dry + windy = high pollen?"""
    print()
    print("=" * 60)
    print("TEMPERATURE-WIND-RAIN INTERACTION")
    print("=" * 60)

    spring = [r for r in rows if 60 <= r["day_of_year"] <= 120  # Mar-Apr peak
              and r["total_count"] is not None and r["year"] <= 2025
              and r["temp_mean"] is not None and r["precipitation"] is not None
              and r["wind_max"] is not None]

    # Define conditions
    warm = [r for r in spring if r["temp_mean"] >= 60]
    cool = [r for r in spring if r["temp_mean"] < 60]
    dry = [r for r in spring if r["precipitation"] < 0.05]
    wet = [r for r in spring if r["precipitation"] >= 0.05]
    windy = [r for r in spring if r["wind_max"] >= 10]
    calm = [r for r in spring if r["wind_max"] < 10]

    combos = {
        "Warm + Dry + Windy": [r for r in spring if r["temp_mean"] >= 60 and r["precipitation"] < 0.05 and r["wind_max"] >= 10],
        "Warm + Dry + Calm": [r for r in spring if r["temp_mean"] >= 60 and r["precipitation"] < 0.05 and r["wind_max"] < 10],
        "Warm + Wet": [r for r in spring if r["temp_mean"] >= 60 and r["precipitation"] >= 0.05],
        "Cool + Dry": [r for r in spring if r["temp_mean"] < 60 and r["precipitation"] < 0.05],
        "Cool + Wet": [r for r in spring if r["temp_mean"] < 60 and r["precipitation"] >= 0.05],
    }

    print(f"\nMedian pollen by weather condition (Mar-Apr peak season):")
    print(f"{'Condition':<25} {'N':>6} {'Median':>8} {'Mean':>8} {'P90':>8} {'% Extreme':>10}")
    print("-" * 65)
    for name, group in combos.items():
        if group:
            counts = sorted(r["total_count"] for r in group)
            n = len(counts)
            extreme_pct = 100 * sum(1 for c in counts if c >= 1500) / n
            print(f"{name:<25} {n:>6} {counts[n//2]:>8} {sum(counts)//n:>8} {counts[int(n*0.90)]:>8} {extreme_pct:>9.1f}%")


def analyze_preseason_warmth(rows):
    """Does January-February warmth predict earlier/worse seasons?"""
    print()
    print("=" * 60)
    print("PRESEASON WARMTH -> SEASON SEVERITY")
    print("=" * 60)

    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))

    year_stats = []
    for yr in complete_years:
        yr_rows = [r for r in rows if r["year"] == yr]

        # Jan-Feb average temp
        janfeb = [r for r in yr_rows if r["day_of_year"] <= 59 and r["temp_mean"] is not None]
        if not janfeb:
            continue
        janfeb_temp = statistics.mean(r["temp_mean"] for r in janfeb)

        # GDD at end of Feb
        feb_end = [r for r in yr_rows if r["day_of_year"] <= 59]
        gdd_feb = feb_end[-1]["gdd_cumulative"] if feb_end else 0

        # Season stats
        total_burden = sum(r["total_count"] for r in yr_rows if r["total_count"] is not None)
        extreme_days = sum(1 for r in yr_rows if r["total_count"] is not None and r["total_count"] >= 1500)

        # First day > 100
        over100 = [r for r in yr_rows if r["total_count"] is not None and r["total_count"] > 100]
        first_100_doy = over100[0]["day_of_year"] if over100 else None

        # Peak count
        peak = max((r["total_count"] for r in yr_rows if r["total_count"] is not None), default=0)

        year_stats.append({
            "year": yr,
            "janfeb_temp": janfeb_temp,
            "gdd_at_feb_end": gdd_feb,
            "total_burden": total_burden,
            "extreme_days": extreme_days,
            "first_100_doy": first_100_doy,
            "peak_count": peak,
        })

    # Correlations
    temps = [s["janfeb_temp"] for s in year_stats]
    gdds = [s["gdd_at_feb_end"] for s in year_stats]
    burdens = [s["total_burden"] for s in year_stats]
    extremes = [float(s["extreme_days"]) for s in year_stats]
    onsets = [float(s["first_100_doy"]) for s in year_stats if s["first_100_doy"]]
    temps_for_onset = [s["janfeb_temp"] for s in year_stats if s["first_100_doy"]]
    peaks = [float(s["peak_count"]) for s in year_stats]

    print(f"\nJan-Feb mean temperature correlations (n={len(year_stats)} years):")
    print(f"  vs. Total burden:    r = {pearson_r(temps, burdens):.3f}")
    print(f"  vs. Extreme days:    r = {pearson_r(temps, extremes):.3f}")
    print(f"  vs. Season onset:    r = {pearson_r(temps_for_onset, onsets):.3f} (negative = earlier)")
    print(f"  vs. Peak count:      r = {pearson_r(temps, peaks):.3f}")

    print(f"\nGDD at end of Feb correlations:")
    print(f"  vs. Total burden:    r = {pearson_r(gdds, burdens):.3f}")
    print(f"  vs. Extreme days:    r = {pearson_r(gdds, extremes):.3f}")

    # Show warm vs cool winters
    median_temp = sorted(temps)[len(temps) // 2]
    warm_yrs = [s for s in year_stats if s["janfeb_temp"] >= median_temp]
    cool_yrs = [s for s in year_stats if s["janfeb_temp"] < median_temp]

    print(f"\nWarm vs Cool winters (split at Jan-Feb median = {median_temp:.1f}F):")
    print(f"  {'Metric':<25} {'Warm Winters':>15} {'Cool Winters':>15}")
    print(f"  {'-'*55}")
    print(f"  {'Avg total burden':<25} {statistics.mean(s['total_burden'] for s in warm_yrs):>15,.0f} {statistics.mean(s['total_burden'] for s in cool_yrs):>15,.0f}")
    print(f"  {'Avg extreme days':<25} {statistics.mean(s['extreme_days'] for s in warm_yrs):>15.1f} {statistics.mean(s['extreme_days'] for s in cool_yrs):>15.1f}")
    print(f"  {'Avg season onset (DOY)':<25} {statistics.mean(s['first_100_doy'] for s in warm_yrs if s['first_100_doy']):>15.0f} {statistics.mean(s['first_100_doy'] for s in cool_yrs if s['first_100_doy']):>15.0f}")

    # 2026 preseason
    yr2026 = [r for r in rows if r["year"] == 2026]
    janfeb_2026 = [r for r in yr2026 if r["day_of_year"] <= 59 and r["temp_mean"] is not None]
    if janfeb_2026:
        temp_2026 = statistics.mean(r["temp_mean"] for r in janfeb_2026)
        gdd_2026 = janfeb_2026[-1]["gdd_cumulative"]
        print(f"\n2026 Jan-Feb temp: {temp_2026:.1f}F (rank: ", end="")
        rank = sum(1 for t in temps if t >= temp_2026) + 1
        print(f"#{rank} of {len(temps) + 1} years)")
        print(f"2026 GDD at end of Feb: {gdd_2026:.0f}")


if __name__ == "__main__":
    rows = load_features()
    analyze_correlations(rows)
    analyze_rain_washout(rows)
    analyze_temp_wind_interaction(rows)
    analyze_preseason_warmth(rows)
