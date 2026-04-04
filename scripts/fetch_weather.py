"""
Open-Meteo Weather Fetcher for Atlanta

Fetches historical weather data for Atlanta (33.749, -84.388) from Open-Meteo's
free archive API. No API key needed.

Variables fetched (daily):
- temperature_2m_max, temperature_2m_min, temperature_2m_mean
- precipitation_sum
- rain_sum
- relative_humidity_2m_mean (from hourly, averaged)
- wind_speed_10m_max
- wind_gusts_10m_max
- wind_direction_10m_dominant
- et0_fao_evapotranspiration (reference evapotranspiration)

Usage:
    python fetch_weather.py --start-date 2026-03-01 --end-date 2026-03-25
    python fetch_weather.py --start-date 1992-01-01 --end-date 2026-03-25 --output weather_history
"""

import argparse
import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import requests

# Atlanta coordinates
LAT = 33.749
LON = -84.388

# Open-Meteo archive API (free, no key)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Daily variables to fetch
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "et0_fao_evapotranspiration",
    "vapour_pressure_deficit_max",
    "shortwave_radiation_sum",
    "sunshine_duration",
]


def fetch_weather_chunk(start_date, end_date, max_retries=3, delay=5):
    """
    Fetch weather data for a date range from Open-Meteo.
    Open-Meteo can handle large ranges but let's chunk by year for safety.
    Includes retry logic for rate limiting (429 errors).
    """
    import time

    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
    }

    for attempt in range(max_retries):
        resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
        if resp.status_code == 429:
            wait = delay * (2 ** attempt)  # exponential backoff: 5, 10, 20 sec
            print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise ValueError(f"Open-Meteo error: {data['reason']}")
        return data

    raise Exception(f"Failed after {max_retries} retries for {start_date} to {end_date}")


def parse_response(data):
    """Convert Open-Meteo JSON response to list of dicts."""
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    records = []

    for i, d in enumerate(dates):
        record = {"date": d}
        for var in DAILY_VARS:
            record[var] = daily.get(var, [None])[i] if i < len(daily.get(var, [])) else None
        records.append(record)

    return records


def fetch_weather(start_date, end_date):
    """
    Fetch weather data, chunked by year to avoid API limits.
    """
    all_records = []
    current_start = start_date

    while current_start <= end_date:
        # Chunk by year
        chunk_end = min(
            date(current_start.year, 12, 31),
            end_date
        )

        print(f"Fetching weather {current_start} to {chunk_end} ...", end=" ", flush=True)

        try:
            data = fetch_weather_chunk(current_start, chunk_end)
            records = parse_response(data)
            all_records.extend(records)
            print(f"OK — {len(records)} days")
        except Exception as e:
            print(f"Error: {e}")

        current_start = date(current_start.year + 1, 1, 1)

    return all_records


def save_csv(records, filename):
    if not records:
        print("No records to save.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename

    keys = list(records[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved {len(records)} records to {filepath}")


def save_json(records, filename):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = DATA_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Fetch Atlanta weather from Open-Meteo")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    parser.add_argument("--output", type=str, default="weather_data")
    args = parser.parse_args()

    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    records = fetch_weather(start, end)
    save_csv(records, f"{args.output}.csv")
    save_json(records, f"{args.output}.json")


if __name__ == "__main__":
    main()
