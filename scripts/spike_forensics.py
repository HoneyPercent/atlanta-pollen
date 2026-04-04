"""
Spike forensics — deep investigation of what causes unexpected extreme days.

Goal: Find signals that precede spikes which our model currently misses.
Approach: Look at the 3-5 day weather pattern BEFORE every extreme day,
compare to the pattern before non-extreme days in the same season phase.
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
    if n < 5: return None
    mx, my = sum(xs)/n, sum(ys)/n
    sx = math.sqrt(sum((x-mx)**2 for x in xs)/(n-1))
    sy = math.sqrt(sum((y-my)**2 for y in ys)/(n-1))
    if sx == 0 or sy == 0: return 0
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / ((n-1)*sx*sy)


def get_weather_window(idx, year, doy, lookback=5):
    """Get the weather pattern for the N days leading up to (but not including) this day."""
    window = []
    for d in range(1, lookback + 1):
        r = idx.get((year, doy - d))
        if r:
            window.append(r)
    return window


def summarize_window(window):
    """Compute summary stats for a weather window."""
    if not window:
        return None

    temps = [r["temp_mean"] for r in window if r["temp_mean"] is not None]
    temp_maxes = [r["temp_max"] for r in window if r["temp_max"] is not None]
    precips = [r["precipitation"] for r in window if r["precipitation"] is not None]
    winds = [r["wind_max"] for r in window if r["wind_max"] is not None]
    humids = [r["humidity_mean"] for r in window if r.get("humidity_mean") is not None]
    counts = [r["total_count"] for r in window if r["total_count"] is not None]
    gdds = [r["gdd_daily"] for r in window if r.get("gdd_daily") is not None]

    total_precip = sum(precips) if precips else 0
    dry_days = sum(1 for p in precips if p < 0.05)
    had_rain = any(p >= 0.25 for p in precips)
    rain_then_dry = had_rain and (precips[0] < 0.05 if precips else False)  # rain earlier, dry recently

    return {
        "temp_avg": statistics.mean(temps) if temps else None,
        "temp_max_avg": statistics.mean(temp_maxes) if temp_maxes else None,
        "temp_trend": (temps[0] - temps[-1]) if len(temps) >= 2 else 0,  # recent - earlier
        "total_precip": total_precip,
        "dry_days": dry_days,
        "had_rain": had_rain,
        "rain_then_dry": rain_then_dry,
        "wind_avg": statistics.mean(winds) if winds else None,
        "wind_max": max(winds) if winds else None,
        "humidity_avg": statistics.mean(humids) if humids else None,
        "pollen_trend": (counts[0] - counts[-1]) if len(counts) >= 2 else 0,
        "pollen_max": max(counts) if counts else 0,
        "gdd_total": sum(gdds) if gdds else 0,
    }


def main():
    rows = load_features()
    idx = build_index(rows)

    # Focus on Mar-Apr (DOY 60-120) during peak season, 2000-2025
    peak_days = [r for r in rows if 60 <= r["day_of_year"] <= 120
                 and r["total_count"] is not None and r["year"] >= 2000 and r["year"] <= 2025]

    extreme_days = [r for r in peak_days if r["total_count"] >= 1500]
    non_extreme = [r for r in peak_days if r["total_count"] < 1500]

    print("=" * 70)
    print("SPIKE FORENSICS: What weather patterns precede extreme days?")
    print("=" * 70)
    print(f"Extreme days (>=1500): {len(extreme_days)}")
    print(f"Non-extreme days: {len(non_extreme)}")
    print()

    # Get 5-day weather windows before each day
    extreme_windows = [summarize_window(get_weather_window(idx, r["year"], r["day_of_year"], 5))
                       for r in extreme_days]
    extreme_windows = [w for w in extreme_windows if w is not None]

    nonext_windows = [summarize_window(get_weather_window(idx, r["year"], r["day_of_year"], 5))
                      for r in non_extreme]
    nonext_windows = [w for w in nonext_windows if w is not None]

    # Compare the two groups
    print("5-DAY WEATHER WINDOW BEFORE EXTREME vs NON-EXTREME DAYS:")
    print(f"{'Feature':<25} {'Before Extreme':>15} {'Before Non-Ext':>15} {'Difference':>12}")
    print("-" * 67)

    comparisons = [
        ("Avg temp (F)", "temp_avg"),
        ("Avg temp max (F)", "temp_max_avg"),
        ("Temp trend (F)", "temp_trend"),
        ("Total precip (in)", "total_precip"),
        ("Dry days (of 5)", "dry_days"),
        ("% with rain event", "had_rain"),
        ("% rain-then-dry", "rain_then_dry"),
        ("Avg wind (mph)", "wind_avg"),
        ("Max wind (mph)", "wind_max"),
        ("Avg humidity (%)", "humidity_avg"),
        ("Pollen trend", "pollen_trend"),
        ("Pollen max (prior 5d)", "pollen_max"),
        ("GDD total (5d)", "gdd_total"),
    ]

    for label, key in comparisons:
        ext_vals = [w[key] for w in extreme_windows if w[key] is not None]
        non_vals = [w[key] for w in nonext_windows if w[key] is not None]

        if key in ["had_rain", "rain_then_dry"]:
            ext_avg = sum(ext_vals) / len(ext_vals) * 100 if ext_vals else 0
            non_avg = sum(non_vals) / len(non_vals) * 100 if non_vals else 0
            diff = ext_avg - non_avg
            print(f"{label:<25} {ext_avg:>14.1f}% {non_avg:>14.1f}% {diff:>+11.1f}pp")
        else:
            ext_avg = statistics.mean(ext_vals) if ext_vals else 0
            non_avg = statistics.mean(non_vals) if non_vals else 0
            diff = ext_avg - non_avg
            print(f"{label:<25} {ext_avg:>15.1f} {non_avg:>15.1f} {diff:>+12.1f}")

    # ================================================================
    # FIRST-SPIKE ANALYSIS: What precedes the FIRST extreme day of the season?
    # ================================================================
    print()
    print("=" * 70)
    print("FIRST-SPIKE ANALYSIS: What precedes the season's first extreme day?")
    print("=" * 70)
    print("(This is the hardest prediction — the transition from non-extreme to extreme)")
    print()

    first_extremes = []
    for yr in range(2000, 2026):
        yr_extreme = [r for r in extreme_days if r["year"] == yr]
        if yr_extreme:
            first = min(yr_extreme, key=lambda r: r["day_of_year"])
            first_extremes.append(first)

    print(f"Found {len(first_extremes)} first-extreme-day events (2000-2025)")

    first_windows = [summarize_window(get_weather_window(idx, r["year"], r["day_of_year"], 5))
                     for r in first_extremes]
    first_windows = [w for w in first_windows if w is not None]

    print(f"\n5-day weather window before the FIRST extreme day of each season:")
    print(f"{'Feature':<25} {'Before 1st Extreme':>18} {'Before Typical Day':>18}")
    print("-" * 61)
    for label, key in comparisons:
        first_vals = [w[key] for w in first_windows if w[key] is not None]
        non_vals = [w[key] for w in nonext_windows if w[key] is not None]
        if key in ["had_rain", "rain_then_dry"]:
            f_avg = sum(first_vals) / len(first_vals) * 100 if first_vals else 0
            n_avg = sum(non_vals) / len(non_vals) * 100 if non_vals else 0
            print(f"{label:<25} {f_avg:>17.1f}% {n_avg:>17.1f}%")
        else:
            f_avg = statistics.mean(first_vals) if first_vals else 0
            n_avg = statistics.mean(non_vals) if non_vals else 0
            print(f"{label:<25} {f_avg:>18.1f} {n_avg:>18.1f}")

    # ================================================================
    # GDD THRESHOLD ANALYSIS: Is there a cumulative GDD "trigger point"?
    # ================================================================
    print()
    print("=" * 70)
    print("GDD TRIGGER: Is there a cumulative GDD threshold that triggers extremes?")
    print("=" * 70)

    print(f"\nCumulative GDD at the time of first extreme day each year:")
    print(f"{'Year':>6} {'First Ext DOY':>14} {'GDD at that point':>18} {'Count':>8}")
    print("-" * 46)
    gdds_at_first = []
    for r in first_extremes:
        gdd = r["gdd_cumulative"]
        gdds_at_first.append(gdd)
        print(f"{r['year']:>6} {r['day_of_year']:>14} {gdd:>18.0f} {r['total_count']:>8,}")

    if gdds_at_first:
        print(f"\nGDD at first extreme: median={statistics.median(gdds_at_first):.0f}, "
              f"mean={statistics.mean(gdds_at_first):.0f}, "
              f"min={min(gdds_at_first):.0f}, max={max(gdds_at_first):.0f}")
        print(f"P25={sorted(gdds_at_first)[len(gdds_at_first)//4]:.0f}, "
              f"P75={sorted(gdds_at_first)[3*len(gdds_at_first)//4]:.0f}")

    # How often does an extreme day occur before GDD reaches 200?
    below_200 = sum(1 for g in gdds_at_first if g < 200)
    below_300 = sum(1 for g in gdds_at_first if g < 300)
    print(f"\nFirst extreme before GDD 200: {below_200}/{len(gdds_at_first)} ({100*below_200/len(gdds_at_first):.0f}%)")
    print(f"First extreme before GDD 300: {below_300}/{len(gdds_at_first)} ({100*below_300/len(gdds_at_first):.0f}%)")

    # ================================================================
    # HUMIDITY DEEP DIVE: Does humidity predict extremes?
    # ================================================================
    print()
    print("=" * 70)
    print("HUMIDITY DEEP DIVE: Does humidity predict extreme pollen?")
    print("=" * 70)

    humid_days = [r for r in peak_days if r.get("humidity_mean") is not None]
    if humid_days:
        # Bin by humidity
        bins = [(0, 40, "Very dry (<40%)"), (40, 55, "Dry (40-55%)"),
                (55, 70, "Moderate (55-70%)"), (70, 85, "Humid (70-85%)"),
                (85, 100, "Very humid (>85%)")]

        print(f"\n{'Humidity Range':<25} {'N':>6} {'Median Pollen':>14} {'% Extreme':>10}")
        print("-" * 55)
        for lo, hi, label in bins:
            group = [r for r in humid_days if lo <= r["humidity_mean"] < hi]
            if group:
                counts = sorted(r["total_count"] for r in group)
                n = len(counts)
                ext_pct = 100 * sum(1 for c in counts if c >= 1500) / n
                print(f"{label:<25} {n:>6} {counts[n//2]:>14,} {ext_pct:>9.1f}%")

        # Humidity change (drying trend)
        print(f"\nHumidity CHANGE (today vs yesterday):")
        humid_changes = []
        for r in humid_days:
            prev = idx.get((r["year"], r["day_of_year"] - 1))
            if prev and prev.get("humidity_mean") is not None:
                delta = r["humidity_mean"] - prev["humidity_mean"]
                humid_changes.append((delta, r["total_count"], r["log_count"]))

        if humid_changes:
            # Correlation
            r_val = pearson_r([h[0] for h in humid_changes], [h[2] for h in humid_changes])
            print(f"  Correlation of humidity change with log(pollen): r = {r_val:.3f}")

            # Bin
            drying = [h for h in humid_changes if h[0] < -10]
            stable = [h for h in humid_changes if -10 <= h[0] <= 10]
            moistening = [h for h in humid_changes if h[0] > 10]

            for label, group in [("Drying (>10% drop)", drying), ("Stable", stable), ("Moistening (>10% rise)", moistening)]:
                if group:
                    counts = sorted(h[1] for h in group)
                    n = len(counts)
                    ext_pct = 100 * sum(1 for c in counts if c >= 1500) / n
                    print(f"  {label}: N={n}, median={counts[n//2]:,}, extreme={ext_pct:.1f}%")

    # ================================================================
    # DIURNAL TEMPERATURE RANGE: Does a big temp swing predict pollen?
    # ================================================================
    print()
    print("=" * 70)
    print("DIURNAL RANGE: Does a big temp max-min swing predict pollen?")
    print("=" * 70)

    diurnal_days = [r for r in peak_days if r["temp_max"] is not None and r["temp_min"] is not None]
    if diurnal_days:
        for r in diurnal_days:
            r["diurnal_range"] = r["temp_max"] - r["temp_min"]

        # Correlation
        r_val = pearson_r([r["diurnal_range"] for r in diurnal_days],
                          [r["log_count"] for r in diurnal_days if r["log_count"] is not None])
        print(f"Correlation of diurnal range with log(pollen): r = {r_val:.3f}")

        # Bin
        print(f"\n{'Diurnal Range':>15} {'N':>6} {'Median Pollen':>14} {'% Extreme':>10}")
        print("-" * 45)
        for lo, hi, label in [(0, 15, "<15F"), (15, 20, "15-20F"), (20, 25, "20-25F"),
                               (25, 30, "25-30F"), (30, 50, ">30F")]:
            group = [r for r in diurnal_days if lo <= r["diurnal_range"] < hi]
            if group:
                counts = sorted(r["total_count"] for r in group)
                n = len(counts)
                ext_pct = 100 * sum(1 for c in counts if c >= 1500) / n
                print(f"{label:>15} {n:>6} {counts[n//2]:>14,} {ext_pct:>9.1f}%")


if __name__ == "__main__":
    main()
