"""
Compute derived features for Atlanta pollen analysis.

Merges pollen observations with weather data and computes:
- Growing Degree Days (GDD, base 50°F) — daily and cumulative since Jan 1
- Cumulative pollen burden since Jan 1
- Lagged precipitation (yesterday, 2-day sum)
- Day of year
- Weekend flag
- Season progress (cumulative burden as % of year's eventual total)
"""

import csv
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_csv(filename):
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    # Load datasets
    pollen_rows = load_csv("full_history_calendar.csv")
    weather_rows = load_csv("weather_complete_v2.csv")

    # Index weather by date
    weather = {}
    for r in weather_rows:
        weather[r["date"]] = r

    # Index pollen by date
    pollen = {}
    for r in pollen_rows:
        pollen[r["date"]] = r

    # Get all years
    pollen_years = sorted(set(r["date"][:4] for r in pollen_rows))

    # Build features for each day
    features = []

    for year_str in pollen_years:
        year = int(year_str)

        # Iterate Jan 1 through Jun 30
        try:
            start = date(year, 1, 1)
            end = date(year, 6, 30)
        except ValueError:
            continue

        gdd_cum = 0.0
        burden_cum = 0.0
        prev_precip = 0.0
        prev_prev_precip = 0.0

        current = start
        while current <= end:
            d = current.isoformat()
            p = pollen.get(d, {})
            w = weather.get(d, {})

            # Pollen data
            total_count = None
            missing = True
            severity = None
            if p:
                if p.get("total_count"):
                    try:
                        total_count = int(p["total_count"])
                        missing = False
                    except (ValueError, TypeError):
                        pass
                severity = p.get("severity_class") or None
                if p.get("missing") == "True":
                    missing = True

            # Weather data
            temp_max = _float(w.get("temperature_2m_max"))
            temp_min = _float(w.get("temperature_2m_min"))
            temp_mean = _float(w.get("temperature_2m_mean"))
            precip = _float(w.get("precipitation_sum"))
            rain = _float(w.get("rain_sum"))
            wind_max = _float(w.get("wind_speed_10m_max"))
            wind_gust = _float(w.get("wind_gusts_10m_max"))
            wind_dir = _float(w.get("wind_direction_10m_dominant"))
            et0 = _float(w.get("et0_fao_evapotranspiration"))
            vpd_max = _float(w.get("vapour_pressure_deficit_max"))
            solar_radiation = _float(w.get("shortwave_radiation_sum"))
            sunshine = _float(w.get("sunshine_duration"))

            # GDD (base 50°F)
            gdd_daily = 0.0
            if temp_mean is not None:
                gdd_daily = max(0.0, temp_mean - 50.0)
            gdd_cum += gdd_daily

            # Cumulative burden
            if total_count is not None:
                burden_cum += total_count

            # Log count
            import math
            log_count = None
            if total_count is not None:
                log_count = round(math.log(total_count + 1), 4)

            # Day of year
            doy = current.timetuple().tm_yday

            # Weekend flag
            is_weekend = current.weekday() >= 5

            # Lagged precip
            precip_yesterday = prev_precip
            precip_2day = (prev_precip or 0) + (prev_prev_precip or 0)

            row = {
                "date": d,
                "year": year,
                "month": current.month,
                "day_of_year": doy,
                "is_weekend": is_weekend,
                # Pollen
                "total_count": total_count,
                "log_count": log_count,
                "severity_class": severity,
                "missing": missing,
                "cumulative_burden": round(burden_cum, 1),
                # Weather
                "temp_max": temp_max,
                "temp_min": temp_min,
                "temp_mean": temp_mean,
                "precipitation": precip,
                "rain": rain,
                "wind_max": wind_max,
                "wind_gust": wind_gust,
                "wind_direction": wind_dir,
                "et0": et0,
                "vpd_max": vpd_max,
                "solar_radiation": solar_radiation,
                "sunshine_duration": sunshine,
                # Derived
                "gdd_daily": round(gdd_daily, 2),
                "gdd_cumulative": round(gdd_cum, 2),
                "precip_yesterday": precip_yesterday,
                "precip_2day_sum": round(precip_2day, 4),
            }
            features.append(row)

            # Shift precip for tomorrow's lagged features
            prev_prev_precip = prev_precip
            prev_precip = precip

            current += timedelta(days=1)

    # Add season_progress: cumulative burden as % of that year's total
    year_totals = defaultdict(float)
    for r in features:
        if r["total_count"] is not None:
            year_totals[r["year"]] += r["total_count"]

    for r in features:
        yr_total = year_totals.get(r["year"], 0)
        if yr_total > 0:
            r["season_progress_pct"] = round(100.0 * r["cumulative_burden"] / yr_total, 2)
        else:
            r["season_progress_pct"] = 0.0

    # Save
    if features:
        keys = list(features[0].keys())
        outpath = DATA_DIR / "features_daily.csv"
        with open(outpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(features)
        print(f"Saved {len(features)} feature rows to {outpath}")

        # Also save year-level summary
        print(f"\nYear-level summary:")
        print(f"{'Year':>6} {'Total Burden':>14} {'Peak Count':>12} {'Extreme Days':>14} {'GDD at Jun30':>14}")
        for yr in sorted(year_totals.keys()):
            yr_rows = [r for r in features if r["year"] == yr]
            total = year_totals[yr]
            peak = max((r["total_count"] or 0) for r in yr_rows)
            extreme = sum(1 for r in yr_rows if r["severity_class"] == "extreme")
            max_gdd = max(r["gdd_cumulative"] for r in yr_rows)
            print(f"{yr:>6} {total:>14,.0f} {peak:>12,} {extreme:>14} {max_gdd:>14,.1f}")


def _float(val):
    if val is None or val == "" or val == "None":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    main()
