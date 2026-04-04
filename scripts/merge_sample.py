"""
Merge pollen calendar data with weather data into a single sample CSV.
Quick utility to demonstrate the joined dataset.
"""

import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def merge():
    # Load pollen data
    pollen = {}
    with open(DATA_DIR / "test_2026_calendar.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pollen[row["date"]] = row

    # Load weather data
    weather = {}
    with open(DATA_DIR / "test_weather_2026_03.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            weather[row["date"]] = row

    # Merge on date
    merged = []
    all_dates = sorted(set(list(pollen.keys()) + list(weather.keys())))

    for d in all_dates:
        row = {"date": d}
        if d in pollen:
            row["total_count"] = pollen[d].get("total_count", "")
            row["severity_class"] = pollen[d].get("severity_class", "")
            row["missing"] = pollen[d].get("missing", "")
        if d in weather:
            w = weather[d]
            for k, v in w.items():
                if k != "date":
                    row[k] = v
        merged.append(row)

    # Write merged CSV
    if merged:
        keys = list(merged[0].keys())
        # Ensure all keys from all rows
        for row in merged:
            for k in row.keys():
                if k not in keys:
                    keys.append(k)

        outpath = DATA_DIR / "sample_pollen_weather_2026_03.csv"
        with open(outpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(merged)

        print(f"Saved {len(merged)} rows to {outpath}")


if __name__ == "__main__":
    merge()
