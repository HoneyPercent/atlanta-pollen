"""
Baseline models for Atlanta pollen prediction.

Three baselines that the website can use without any ML framework:

Baseline A: Day-of-year climatology
  - For any date, what's the historical average/median/percentile range?
  - "Today is X% worse than the March 25 average"

Baseline B: Analog-year model
  - Given cumulative burden so far + DOY, find similar past years
  - Project remaining extreme days, remaining days > 100, estimated season end

Baseline C: Season progress model
  - Given cumulative burden, estimate % through the season
  - Estimate remaining extreme days based on S-curve position

All output JSON that the website can consume directly.
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
        r["season_progress_pct"] = float(r["season_progress_pct"])
        r["gdd_cumulative"] = float(r["gdd_cumulative"])
        r["missing"] = r["missing"] == "True"
        r["temp_mean"] = float(r["temp_mean"]) if r["temp_mean"] else None
        r["precipitation"] = float(r["precipitation"]) if r["precipitation"] else None
    return rows


def baseline_a_climatology(rows):
    """
    Day-of-year climatology: for each DOY, compute historical stats.
    Output: JSON with percentiles for each DOY.
    """
    complete_years = [y for y in set(r["year"] for r in rows) if y <= 2025]

    # Group counts by DOY (only complete years, only days with data)
    doy_counts = defaultdict(list)
    for r in rows:
        if r["year"] in complete_years and r["total_count"] is not None:
            doy_counts[r["day_of_year"]].append(r["total_count"])

    climatology = {}
    for doy in sorted(doy_counts.keys()):
        counts = sorted(doy_counts[doy])
        n = len(counts)
        if n < 3:
            continue
        climatology[doy] = {
            "doy": doy,
            "n_years": n,
            "mean": round(statistics.mean(counts), 1),
            "median": round(statistics.median(counts), 1),
            "p10": counts[int(n * 0.10)],
            "p25": counts[int(n * 0.25)],
            "p75": counts[int(n * 0.75)],
            "p90": counts[int(n * 0.90)],
            "p95": counts[min(int(n * 0.95), n - 1)],
            "max": counts[-1],
            "min": counts[0],
        }

    return climatology


def baseline_b_analog_years(rows, target_year=2026):
    """
    Analog-year model: find years with similar cumulative burden at the same DOY.
    Returns projections based on what happened in those years after that point.
    """
    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))

    # Get latest data point for target year
    target_rows = [r for r in rows if r["year"] == target_year and r["total_count"] is not None]
    if not target_rows:
        return None
    latest = target_rows[-1]
    target_doy = latest["day_of_year"]
    target_burden = latest["cumulative_burden"]

    # For each historical year, get burden at this DOY and what happened after
    analogs = []
    for yr in complete_years:
        yr_rows = [r for r in rows if r["year"] == yr]
        at_doy = [r for r in yr_rows if r["day_of_year"] <= target_doy]
        after_doy = [r for r in yr_rows if r["day_of_year"] > target_doy]

        if not at_doy:
            continue

        burden_at_doy = at_doy[-1]["cumulative_burden"]
        total_season = sum(r["total_count"] for r in yr_rows if r["total_count"] is not None)

        # Remaining stats
        remaining_extreme = sum(1 for r in after_doy if r["total_count"] is not None and r["total_count"] >= 1500)
        remaining_high = sum(1 for r in after_doy if r["total_count"] is not None and r["total_count"] >= 500)
        remaining_over100 = sum(1 for r in after_doy if r["total_count"] is not None and r["total_count"] >= 100)

        # Last day with count > 100
        last_over100 = [r for r in after_doy if r["total_count"] is not None and r["total_count"] >= 100]
        last_over100_doy = last_over100[-1]["day_of_year"] if last_over100 else target_doy

        # Last day with count >= 1500
        last_extreme = [r for r in after_doy if r["total_count"] is not None and r["total_count"] >= 1500]
        last_extreme_doy = last_extreme[-1]["day_of_year"] if last_extreme else target_doy

        pct_done = (burden_at_doy / total_season * 100) if total_season > 0 else 0

        analogs.append({
            "year": yr,
            "burden_at_doy": round(burden_at_doy),
            "total_season_burden": round(total_season),
            "pct_done_at_doy": round(pct_done, 1),
            "remaining_extreme_days": remaining_extreme,
            "remaining_high_days": remaining_high,
            "remaining_over100_days": remaining_over100,
            "last_extreme_doy": last_extreme_doy,
            "last_over100_doy": last_over100_doy,
            "similarity": round(1.0 - abs(burden_at_doy - target_burden) / max(target_burden, 1), 3),
        })

    # Sort by similarity (closest burden)
    analogs.sort(key=lambda x: abs(x["burden_at_doy"] - target_burden))

    # Top 5 most similar years
    top_analogs = analogs[:5]

    # Weighted projections (weight by similarity)
    if top_analogs:
        weights = [a["similarity"] for a in top_analogs]
        w_sum = sum(weights)

        proj = {
            "target_year": target_year,
            "target_doy": target_doy,
            "target_date": latest["date"],
            "target_burden": round(target_burden),
            "analog_years": top_analogs,
            "projection": {
                "remaining_extreme_days": {
                    "weighted_avg": round(sum(a["remaining_extreme_days"] * w for a, w in zip(top_analogs, weights)) / w_sum, 1),
                    "min": min(a["remaining_extreme_days"] for a in top_analogs),
                    "max": max(a["remaining_extreme_days"] for a in top_analogs),
                },
                "remaining_over100_days": {
                    "weighted_avg": round(sum(a["remaining_over100_days"] * w for a, w in zip(top_analogs, weights)) / w_sum, 1),
                    "min": min(a["remaining_over100_days"] for a in top_analogs),
                    "max": max(a["remaining_over100_days"] for a in top_analogs),
                },
                "estimated_season_end_doy": {
                    "weighted_avg": round(sum(a["last_over100_doy"] * w for a, w in zip(top_analogs, weights)) / w_sum),
                    "min": min(a["last_over100_doy"] for a in top_analogs),
                    "max": max(a["last_over100_doy"] for a in top_analogs),
                },
                "estimated_last_extreme_doy": {
                    "weighted_avg": round(sum(a["last_extreme_doy"] * w for a, w in zip(top_analogs, weights)) / w_sum),
                    "min": min(a["last_extreme_doy"] for a in top_analogs),
                    "max": max(a["last_extreme_doy"] for a in top_analogs),
                },
            },
        }
    else:
        proj = None

    return proj


def baseline_c_season_progress(rows, target_year=2026):
    """
    Season progress model based on cumulative burden S-curve.
    Estimates where we are on the S-curve and what typically follows.
    """
    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))

    # Get target year's latest data
    target_rows = [r for r in rows if r["year"] == target_year and r["total_count"] is not None]
    if not target_rows:
        return None
    latest = target_rows[-1]
    target_doy = latest["day_of_year"]
    target_burden = latest["cumulative_burden"]

    # Historical S-curves: for each year, compute burden at each DOY as % of total
    # Then find where target_burden falls on the average curve
    year_totals = {}
    for yr in complete_years:
        total = sum(r["total_count"] for r in rows if r["year"] == yr and r["total_count"] is not None)
        if total > 0:
            year_totals[yr] = total

    # What % of the season is typically done by target_doy?
    pct_done_at_doy = []
    for yr in complete_years:
        if yr not in year_totals:
            continue
        yr_at_doy = [r for r in rows if r["year"] == yr and r["day_of_year"] <= target_doy]
        if yr_at_doy:
            burden = yr_at_doy[-1]["cumulative_burden"]
            pct_done_at_doy.append(burden / year_totals[yr] * 100)

    # After reaching various % completion, how many extreme days remain?
    pct_remaining = {}
    for pct_threshold in [25, 50, 60, 70, 75, 80, 85, 90, 95]:
        remaining_counts = []
        for yr in complete_years:
            if yr not in year_totals:
                continue
            yr_rows = [r for r in rows if r["year"] == yr]
            hit = [r for r in yr_rows if r["season_progress_pct"] >= pct_threshold]
            if hit:
                cutoff_doy = hit[0]["day_of_year"]
                remaining = sum(
                    1 for r in yr_rows
                    if r["day_of_year"] > cutoff_doy
                    and r["total_count"] is not None
                    and r["total_count"] >= 1500
                )
                remaining_counts.append(remaining)

        if remaining_counts:
            pct_remaining[pct_threshold] = {
                "avg_extreme_remaining": round(statistics.mean(remaining_counts), 1),
                "max_extreme_remaining": max(remaining_counts),
                "median_extreme_remaining": round(statistics.median(remaining_counts), 1),
            }

    # Estimate total season burden for target year
    # Use the historical average % done at this DOY to project total
    if pct_done_at_doy:
        avg_pct = statistics.mean(pct_done_at_doy)
        median_pct = statistics.median(pct_done_at_doy)

        # Recent years (2015+) weight more since pollen is trending up
        recent_pct = [p for yr, p in zip(complete_years, pct_done_at_doy) if yr >= 2015]
        recent_avg_pct = statistics.mean(recent_pct) if recent_pct else avg_pct

        projected_total_avg = target_burden / (avg_pct / 100) if avg_pct > 0 else 0
        projected_total_recent = target_burden / (recent_avg_pct / 100) if recent_avg_pct > 0 else 0

    result = {
        "target_year": target_year,
        "target_doy": target_doy,
        "target_date": latest["date"],
        "cumulative_burden": round(target_burden),
        "latest_count": latest["total_count"],
        "historical_pct_done_at_doy": {
            "mean": round(statistics.mean(pct_done_at_doy), 1) if pct_done_at_doy else None,
            "median": round(statistics.median(pct_done_at_doy), 1) if pct_done_at_doy else None,
            "recent_mean": round(recent_avg_pct, 1) if recent_pct else None,
        },
        "projected_total_burden": {
            "from_historical_avg": round(projected_total_avg),
            "from_recent_avg": round(projected_total_recent),
        },
        "remaining_extreme_by_pct": pct_remaining,
        "season_trend": {
            "1990s_avg_total": round(statistics.mean([year_totals[y] for y in year_totals if 1992 <= y <= 1999])),
            "2000s_avg_total": round(statistics.mean([year_totals[y] for y in year_totals if 2000 <= y <= 2009])),
            "2010s_avg_total": round(statistics.mean([year_totals[y] for y in year_totals if 2010 <= y <= 2019])),
            "2020s_avg_total": round(statistics.mean([year_totals[y] for y in year_totals if 2020 <= y <= 2025])),
        },
    }

    return result


def build_cumulative_curves(rows):
    """
    Build the S-curve data for the website visualization.
    Returns curves for each year + percentile envelope.
    """
    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025))

    # For each year, build cumulative burden by DOY
    curves = {}
    for yr in set(r["year"] for r in rows):
        yr_rows = sorted(
            [r for r in rows if r["year"] == yr],
            key=lambda r: r["day_of_year"]
        )
        curve = []
        for r in yr_rows:
            curve.append({
                "doy": r["day_of_year"],
                "date": r["date"],
                "burden": round(r["cumulative_burden"]),
                "count": r["total_count"],
            })
        curves[yr] = curve

    # Percentile envelope across complete years
    doy_burdens = defaultdict(list)
    for yr in complete_years:
        yr_total = sum(r["total_count"] for r in rows if r["year"] == yr and r["total_count"] is not None)
        if yr_total == 0:
            continue
        for r in rows:
            if r["year"] == yr:
                # Normalize to % of that year's total
                pct = r["cumulative_burden"] / yr_total * 100
                doy_burdens[r["day_of_year"]].append(pct)

    envelope = []
    for doy in sorted(doy_burdens.keys()):
        vals = sorted(doy_burdens[doy])
        n = len(vals)
        if n < 3:
            continue
        envelope.append({
            "doy": doy,
            "p10": round(vals[int(n * 0.10)], 1),
            "p25": round(vals[int(n * 0.25)], 1),
            "median": round(statistics.median(vals), 1),
            "p75": round(vals[int(n * 0.75)], 1),
            "p90": round(vals[int(n * 0.90)], 1),
        })

    return {"curves": {str(k): v for k, v in curves.items()}, "envelope": envelope}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_features()

    # Baseline A: Climatology
    print("Building Baseline A: Day-of-year climatology...")
    clim = baseline_a_climatology(rows)
    with open(OUTPUT_DIR / "climatology.json", "w", encoding="utf-8") as f:
        json.dump(clim, f, indent=2)
    print(f"  Saved {len(clim)} DOY entries")

    # Baseline B: Analog years
    print("Building Baseline B: Analog-year projections for 2026...")
    analog = baseline_b_analog_years(rows, 2026)
    with open(OUTPUT_DIR / "analog_projection.json", "w", encoding="utf-8") as f:
        json.dump(analog, f, indent=2)
    if analog:
        proj = analog["projection"]
        print(f"  Top analog years: {[a['year'] for a in analog['analog_years']]}")
        print(f"  Remaining extreme days: {proj['remaining_extreme_days']['weighted_avg']} (range: {proj['remaining_extreme_days']['min']}-{proj['remaining_extreme_days']['max']})")
        print(f"  Remaining days > 100: {proj['remaining_over100_days']['weighted_avg']} (range: {proj['remaining_over100_days']['min']}-{proj['remaining_over100_days']['max']})")
        print(f"  Season end (last > 100): ~DOY {proj['estimated_season_end_doy']['weighted_avg']}")

    # Baseline C: Season progress
    print("Building Baseline C: Season progress model...")
    progress = baseline_c_season_progress(rows, 2026)
    with open(OUTPUT_DIR / "season_progress.json", "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    if progress:
        hp = progress["historical_pct_done_at_doy"]
        print(f"  At DOY {progress['target_doy']}, historically {hp['mean']}% done (recent: {hp['recent_mean']}%)")
        pt = progress["projected_total_burden"]
        print(f"  Projected total: {pt['from_historical_avg']:,} (hist avg) / {pt['from_recent_avg']:,} (recent avg)")
        st = progress["season_trend"]
        print(f"  Decade totals: 1990s={st['1990s_avg_total']:,} | 2000s={st['2000s_avg_total']:,} | 2010s={st['2010s_avg_total']:,} | 2020s={st['2020s_avg_total']:,}")

    # S-curve data for visualization
    print("Building S-curve visualization data...")
    curves = build_cumulative_curves(rows)
    with open(OUTPUT_DIR / "scurve_data.json", "w", encoding="utf-8") as f:
        json.dump(curves, f)  # No indent — this file is large
    print(f"  Saved {len(curves['curves'])} year curves + envelope ({len(curves['envelope'])} DOY points)")

    print("\nAll model outputs saved to data/model_output/")


if __name__ == "__main__":
    main()
