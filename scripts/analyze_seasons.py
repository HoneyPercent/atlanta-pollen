"""
Season analysis for Atlanta pollen data.

Analyzes:
1. S-curve consistency: do cumulative burden curves follow similar shapes?
2. Season timing: when does pollen start, peak, and end?
3. Remaining bad days: once you've hit X% of burden, how many extreme days remain?
4. "Bad day" threshold definition
5. 2026 season position relative to historical patterns
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
    # Convert types
    for r in rows:
        r["year"] = int(r["year"])
        r["day_of_year"] = int(r["day_of_year"])
        r["total_count"] = int(r["total_count"]) if r["total_count"] else None
        r["cumulative_burden"] = float(r["cumulative_burden"])
        r["season_progress_pct"] = float(r["season_progress_pct"])
        r["gdd_cumulative"] = float(r["gdd_cumulative"])
        r["missing"] = r["missing"] == "True"
    return rows


def analyze_bad_day_threshold(rows):
    """Define 'bad pollen day' empirically from total count distribution."""
    counts = sorted(r["total_count"] for r in rows if r["total_count"] is not None)
    n = len(counts)

    print("=" * 60)
    print("BAD POLLEN DAY THRESHOLD ANALYSIS")
    print("=" * 60)
    print(f"Total observation days: {n}")
    print(f"Min: {counts[0]}, Max: {counts[-1]}")
    print(f"Median: {counts[n // 2]}")
    print(f"Mean: {sum(counts) / n:.0f}")
    print()

    # Percentiles
    for pct in [50, 75, 80, 85, 90, 95, 99]:
        idx = int(n * pct / 100)
        print(f"  P{pct}: {counts[min(idx, n-1)]}")

    # Proposed thresholds
    print()
    print("Proposed severity thresholds (total count):")
    print("  Low:      0-99")
    print("  Moderate: 100-499")
    print("  High:     500-1499")
    print("  Extreme:  1500+")
    print()

    # Count days in each band
    bands = {"low (0-99)": 0, "moderate (100-499)": 0, "high (500-1499)": 0, "extreme (1500+)": 0}
    for c in counts:
        if c < 100:
            bands["low (0-99)"] += 1
        elif c < 500:
            bands["moderate (100-499)"] += 1
        elif c < 1500:
            bands["high (500-1499)"] += 1
        else:
            bands["extreme (1500+)"] += 1

    for band, count in bands.items():
        print(f"  {band}: {count} days ({100 * count / n:.1f}%)")

    return 1500  # threshold for "bad day"


def analyze_season_timing(rows):
    """When does the season start, peak, and end each year?"""
    print()
    print("=" * 60)
    print("SEASON TIMING ANALYSIS")
    print("=" * 60)

    years = sorted(set(r["year"] for r in rows))
    # Exclude 2026 (incomplete)
    years = [y for y in years if y <= 2025]

    print(f"\n{'Year':>6} {'First>100':>10} {'First>1500':>12} {'Peak DOY':>10} {'Peak Count':>12} {'Last>1500':>12} {'Last>100':>10} {'Season Len':>12}")

    first_100_doys = []
    peak_doys = []
    last_100_doys = []
    season_lens = []

    for yr in years:
        yr_rows = [r for r in rows if r["year"] == yr and r["total_count"] is not None]
        if not yr_rows:
            continue

        # First day > 100
        over_100 = [r for r in yr_rows if r["total_count"] > 100]
        first_100 = over_100[0]["day_of_year"] if over_100 else None

        # First day > 1500
        over_1500 = [r for r in yr_rows if r["total_count"] >= 1500]
        first_1500 = over_1500[0]["day_of_year"] if over_1500 else None

        # Peak day
        peak_row = max(yr_rows, key=lambda r: r["total_count"])
        peak_doy = peak_row["day_of_year"]
        peak_count = peak_row["total_count"]

        # Last day > 1500
        last_1500 = over_1500[-1]["day_of_year"] if over_1500 else None

        # Last day > 100
        last_100 = over_100[-1]["day_of_year"] if over_100 else None

        # Season length (first >100 to last >100)
        season_len = (last_100 - first_100 + 1) if first_100 and last_100 else None

        if first_100:
            first_100_doys.append(first_100)
        if last_100:
            last_100_doys.append(last_100)
        peak_doys.append(peak_doy)
        if season_len:
            season_lens.append(season_len)

        print(f"{yr:>6} {_doy_str(first_100):>10} {_doy_str(first_1500):>12} {_doy_str(peak_doy):>10} {peak_count:>12,} {_doy_str(last_1500):>12} {_doy_str(last_100):>10} {season_len or '':>12}")

    print()
    print("Averages:")
    print(f"  First day > 100:  DOY {statistics.mean(first_100_doys):.0f} (~{_doy_to_date(statistics.mean(first_100_doys))})")
    print(f"  Peak day:         DOY {statistics.mean(peak_doys):.0f} (~{_doy_to_date(statistics.mean(peak_doys))})")
    print(f"  Last day > 100:   DOY {statistics.mean(last_100_doys):.0f} (~{_doy_to_date(statistics.mean(last_100_doys))})")
    print(f"  Season length:    {statistics.mean(season_lens):.0f} days (std: {statistics.stdev(season_lens):.0f})")


def analyze_scurve_consistency(rows, bad_threshold=1500):
    """Do cumulative burden curves follow consistent S-shapes?"""
    print()
    print("=" * 60)
    print("S-CURVE / CUMULATIVE BURDEN ANALYSIS")
    print("=" * 60)

    years = sorted(set(r["year"] for r in rows))
    complete_years = [y for y in years if y <= 2025]

    # At what % of season progress do various milestones happen?
    print(f"\n{'Year':>6} {'25% DOY':>10} {'50% DOY':>10} {'75% DOY':>10} {'90% DOY':>10} {'95% DOY':>10}")

    milestone_doys = {25: [], 50: [], 75: [], 90: [], 95: []}

    for yr in complete_years:
        yr_rows = [r for r in rows if r["year"] == yr]
        milestones = {}
        for pct in [25, 50, 75, 90, 95]:
            hit = [r for r in yr_rows if r["season_progress_pct"] >= pct]
            if hit:
                milestones[pct] = hit[0]["day_of_year"]
                milestone_doys[pct].append(hit[0]["day_of_year"])
            else:
                milestones[pct] = None

        print(f"{yr:>6} {_doy_str(milestones.get(25)):>10} {_doy_str(milestones.get(50)):>10} {_doy_str(milestones.get(75)):>10} {_doy_str(milestones.get(90)):>10} {_doy_str(milestones.get(95)):>10}")

    print()
    print("Average milestone dates:")
    for pct in [25, 50, 75, 90, 95]:
        vals = milestone_doys[pct]
        if vals:
            avg = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"  {pct}% of burden: DOY {avg:.0f} (~{_doy_to_date(avg)}), std: {std:.0f} days")

    # Remaining extreme days after each milestone
    print()
    print("REMAINING EXTREME DAYS (count >= 1500) after reaching % of season burden:")
    print(f"{'Year':>6} {'After 50%':>10} {'After 75%':>10} {'After 90%':>10}")

    for yr in complete_years:
        yr_rows = [r for r in rows if r["year"] == yr]
        results = {}
        for pct in [50, 75, 90]:
            hit = [r for r in yr_rows if r["season_progress_pct"] >= pct]
            if hit:
                cutoff_doy = hit[0]["day_of_year"]
                remaining = sum(
                    1 for r in yr_rows
                    if r["day_of_year"] > cutoff_doy
                    and r["total_count"] is not None
                    and r["total_count"] >= bad_threshold
                )
                results[pct] = remaining
            else:
                results[pct] = None

        print(f"{yr:>6} {results.get(50, ''):>10} {results.get(75, ''):>10} {results.get(90, ''):>10}")

    # Averages
    print()
    for pct in [50, 75, 90]:
        vals = []
        for yr in complete_years:
            yr_rows = [r for r in rows if r["year"] == yr]
            hit = [r for r in yr_rows if r["season_progress_pct"] >= pct]
            if hit:
                cutoff_doy = hit[0]["day_of_year"]
                remaining = sum(
                    1 for r in yr_rows
                    if r["day_of_year"] > cutoff_doy
                    and r["total_count"] is not None
                    and r["total_count"] >= bad_threshold
                )
                vals.append(remaining)
        if vals:
            print(f"  After {pct}% burden: avg {statistics.mean(vals):.1f} extreme days remaining (max {max(vals)})")


def analyze_2026_position(rows):
    """Where is 2026 relative to historical patterns?"""
    print()
    print("=" * 60)
    print("2026 SEASON POSITION")
    print("=" * 60)

    rows_2026 = [r for r in rows if r["year"] == 2026 and r["total_count"] is not None]
    if not rows_2026:
        print("No 2026 data with counts found.")
        return

    latest = rows_2026[-1]
    latest_doy = latest["day_of_year"]
    burden_2026 = latest["cumulative_burden"]

    print(f"Latest 2026 data: {latest['date']} (DOY {latest_doy})")
    print(f"Cumulative burden so far: {burden_2026:,.0f}")
    print(f"Latest count: {latest['total_count']:,}")

    # Compare to same DOY in other years
    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))
    print(f"\nComparison: burden at DOY {latest_doy} across all years:")

    burden_at_doy = []
    for yr in complete_years:
        yr_rows = [r for r in rows if r["year"] == yr and r["day_of_year"] <= latest_doy]
        if yr_rows:
            b = yr_rows[-1]["cumulative_burden"]
            burden_at_doy.append((yr, b))

    burden_at_doy.sort(key=lambda x: x[1], reverse=True)
    for yr, b in burden_at_doy[:10]:
        # What % of that year's total was reached by this DOY?
        total = sum(r["total_count"] for r in rows if r["year"] == yr and r["total_count"] is not None)
        pct = 100 * b / total if total > 0 else 0
        print(f"  {yr}: {b:>10,.0f} ({pct:.0f}% of season total)")

    print(f"\n  2026: {burden_2026:>10,.0f} (season in progress)")

    # Rank
    all_burdens = [b for _, b in burden_at_doy]
    rank = sum(1 for b in all_burdens if b >= burden_2026) + 1
    print(f"\n2026 ranks #{rank} of {len(all_burdens) + 1} years at DOY {latest_doy}")

    # Estimate: if 2026 follows a typical year's trajectory from this point
    print(f"\nProjections based on analog years:")
    # Find years with similar burden at this DOY
    similar = [(yr, b) for yr, b in burden_at_doy if 0.6 * burden_2026 <= b <= 1.5 * burden_2026]
    if similar:
        print(f"Years with similar burden at DOY {latest_doy} (±50%):")
        for yr, b in similar:
            total = sum(r["total_count"] for r in rows if r["year"] == yr and r["total_count"] is not None)
            remaining_extreme = sum(
                1 for r in rows
                if r["year"] == yr and r["day_of_year"] > latest_doy
                and r["total_count"] is not None and r["total_count"] >= 1500
            )
            remaining_over100 = sum(
                1 for r in rows
                if r["year"] == yr and r["day_of_year"] > latest_doy
                and r["total_count"] is not None and r["total_count"] >= 100
            )
            pct_at_doy = 100 * b / total if total > 0 else 0
            print(f"  {yr}: burden {b:,.0f} at DOY {latest_doy} ({pct_at_doy:.0f}% done) -> {remaining_extreme} more extreme days, {remaining_over100} more days > 100")

        # Average projection
        ext_remaining = [
            sum(1 for r in rows if r["year"] == yr and r["day_of_year"] > latest_doy
                and r["total_count"] is not None and r["total_count"] >= 1500)
            for yr, _ in similar
        ]
        over100_remaining = [
            sum(1 for r in rows if r["year"] == yr and r["day_of_year"] > latest_doy
                and r["total_count"] is not None and r["total_count"] >= 100)
            for yr, _ in similar
        ]
        print(f"\n  -> Average projection: {statistics.mean(ext_remaining):.0f} more extreme days (range: {min(ext_remaining)}–{max(ext_remaining)})")
        print(f"  -> Average projection: {statistics.mean(over100_remaining):.0f} more days > 100 (range: {min(over100_remaining)}–{max(over100_remaining)})")


def _doy_str(doy):
    if doy is None:
        return "—"
    return str(int(doy))


def _doy_to_date(doy):
    """Approximate DOY to month-day string."""
    from datetime import date, timedelta
    d = date(2024, 1, 1) + timedelta(days=int(doy) - 1)  # 2024 is a leap year
    return d.strftime("%b %d")


if __name__ == "__main__":
    rows = load_features()
    bad_threshold = analyze_bad_day_threshold(rows)
    analyze_season_timing(rows)
    analyze_scurve_consistency(rows, bad_threshold)
    analyze_2026_position(rows)
