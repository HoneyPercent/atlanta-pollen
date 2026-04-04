"""
Atlanta Allergy & Asthma Pollen Count Scraper

Scrapes the calendar pages at atlantaallergy.com/pollen_counts/index/YYYY/MM
to extract daily total pollen counts.

For detail pages (/YYYY/MM/DD), extracts:
- Total count
- Tree/Grass/Weed severity levels (categorical: low/medium/high/extreme)
- Top contributors
- Mold activity level

Usage:
    # Scrape calendar data for a range of years/months
    python scrape_atlanta_allergy.py --calendar --start-year 2009 --end-year 2026 --months 1,2,3,4,5,6

    # Scrape detail pages for specific dates
    python scrape_atlanta_allergy.py --details --start-date 2026-03-01 --end-date 2026-03-25

    # Quick test: scrape one month
    python scrape_atlanta_allergy.py --calendar --start-year 2026 --end-year 2026 --months 3
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.atlantaallergy.com/pollen_counts/index"
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"

# Polite scraping: identify ourselves, respect rate limits
HEADERS = {
    "User-Agent": "AtlantaPollenTracker/1.0 (personal research project; contact: andrewkstein@gmail.com)"
}
REQUEST_DELAY = 1.5  # seconds between requests


def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_page(url, save_raw=True):
    """Fetch a page and optionally save the raw HTML."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    if save_raw:
        # Save raw HTML for audit trail
        safe_name = url.replace(BASE_URL, "").strip("/").replace("/", "_") or "index"
        raw_path = RAW_DIR / f"{safe_name}.html"
        raw_path.write_text(resp.text, encoding="utf-8")

    return resp.text


def parse_calendar_page(html, year, month):
    """
    Parse a monthly calendar page to extract daily pollen counts.

    Returns list of dicts: [{date, total_count, severity_class, source_url}]
    """
    soup = BeautifulSoup(html, "html.parser")
    records = []

    # Find all calendar day divs
    for day_div in soup.select("div.calendar-day.current"):
        # Skip padding days (belong to other months)
        if "padding" in day_div.get("class", []):
            continue

        # Extract severity from CSS class
        classes = day_div.get("class", [])
        severity = None
        for level in ["extreme", "high", "medium", "low"]:
            if level in classes:
                severity = level
                break

        # Find the link with the count
        link = day_div.select_one("a[href*='pollen_counts']")
        if not link:
            continue

        # Extract day number from the URL
        href = link.get("href", "")
        day_match = re.search(r"/(\d{1,2})$", href)
        if not day_match:
            continue
        day = int(day_match.group(1))

        # Extract count from link text
        count_text = link.get_text(strip=True)
        total_count = None
        if count_text:
            try:
                total_count = int(count_text)
            except ValueError:
                total_count = None  # "no data" or other text

        try:
            obs_date = date(year, month, day)
        except ValueError:
            continue  # invalid date

        records.append({
            "date": obs_date.isoformat(),
            "total_count": total_count,
            "severity_class": severity,
            "missing": total_count is None,
            "source": "atlanta_allergy_calendar",
            "source_url": f"{BASE_URL}/{year}/{month:02d}",
        })

    return records


def parse_detail_page(html, year, month, day):
    """
    Parse a daily detail page to extract full pollen breakdown.

    Returns dict with: total_count, tree/grass/weed severity + contributors, mold level.
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "date": f"{year}-{month:02d}-{day:02d}",
        "source": "atlanta_allergy_detail",
        "source_url": f"{BASE_URL}/{year}/{month:02d}/{day:02d}",
    }

    # Check for "no pollen data" message
    no_data = soup.find(string=re.compile(r"There is no pollen data", re.I))
    if no_data:
        result["missing"] = True
        result["total_count"] = None
        return result

    result["missing"] = False

    # Total count
    pollen_num = soup.select_one("span.pollen-num")
    if pollen_num:
        count_text = pollen_num.get_text(strip=True)
        try:
            result["total_count"] = int(count_text)
        except ValueError:
            result["total_count"] = None
    else:
        result["total_count"] = None

    # Tree/Grass/Weed gauges
    gauges = soup.select("div.gauge")
    for gauge in gauges:
        h3 = gauge.select_one("h3")
        if not h3:
            # Check for h4 (Mold)
            h4 = gauge.find_previous_sibling("h4") or gauge.parent.select_one("h4")
            if h4 and "Mold" in h4.get_text():
                active = gauge.select_one("span.active, span[class*='active']")
                if active:
                    result["mold_level"] = active.get_text(strip=True)
                # Also get needle position for mold
                needle = gauge.select_one("span.needle")
                if needle:
                    style = needle.get("style", "")
                    left_match = re.search(r"left:\s*([\d.]+)%", style)
                    if left_match:
                        result["mold_needle_pct"] = float(left_match.group(1))
            continue

        title = h3.get_text(strip=True)

        # Determine category
        if "Trees" in title:
            prefix = "tree"
        elif "Grass" in title:
            prefix = "grass"
        elif "Weeds" in title:
            prefix = "weed"
        else:
            continue

        # Contributors
        p = gauge.select_one("p")
        if p:
            contributors = p.get_text(strip=True).strip("\xa0").strip()
            if contributors:
                result[f"{prefix}_contributors"] = contributors

        # Severity level (which segment is active)
        active = gauge.select_one("span.active, span[class*='active']")
        if active:
            classes = active.get("class", [])
            for level in ["extreme", "high", "medium", "low"]:
                if level in classes:
                    result[f"{prefix}_severity"] = level
                    break

        # Needle position (proxy for relative level within the category)
        needle = gauge.select_one("span.needle")
        if needle:
            style = needle.get("style", "")
            left_match = re.search(r"left:\s*([\d.]+)%", style)
            if left_match:
                result[f"{prefix}_needle_pct"] = float(left_match.group(1))

    # Mold section (separate structure)
    mold_h4 = soup.find("h4", string=re.compile(r"Mold Activity"))
    if mold_h4:
        mold_gauge = mold_h4.find_next("div", class_="gauge")
        if mold_gauge:
            active = mold_gauge.select_one("span.active, span[class*='active']")
            if active:
                # Get the text but filter out non-active spans
                classes = active.get("class", [])
                for level in ["extreme", "high", "medium", "low"]:
                    if level in classes:
                        result["mold_level"] = active.get_text(strip=True)
                        break
            needle = mold_gauge.select_one("span.needle")
            if needle:
                style = needle.get("style", "")
                left_match = re.search(r"left:\s*([\d.]+)%", style)
                if left_match:
                    result["mold_needle_pct"] = float(left_match.group(1))

    return result


def scrape_calendar(start_year, end_year, months):
    """Scrape calendar pages for a range of years and months."""
    ensure_dirs()
    all_records = []

    for year in range(start_year, end_year + 1):
        for month in months:
            url = f"{BASE_URL}/{year}/{month:02d}"
            print(f"Fetching {url} ...", end=" ", flush=True)

            try:
                html = fetch_page(url)
                records = parse_calendar_page(html, year, month)
                all_records.extend(records)
                data_days = sum(1 for r in records if not r["missing"])
                missing_days = sum(1 for r in records if r["missing"])
                print(f"OK — {data_days} days with data, {missing_days} missing")
            except requests.HTTPError as e:
                print(f"HTTP error: {e}")
            except Exception as e:
                print(f"Error: {e}")

            time.sleep(REQUEST_DELAY)

    return all_records


def scrape_details(start_date, end_date):
    """Scrape detail pages for a date range."""
    ensure_dirs()
    all_records = []
    current = start_date

    while current <= end_date:
        url = f"{BASE_URL}/{current.year}/{current.month:02d}/{current.day:02d}"
        print(f"Fetching {url} ...", end=" ", flush=True)

        try:
            html = fetch_page(url)
            record = parse_detail_page(html, current.year, current.month, current.day)
            all_records.append(record)
            if record.get("missing"):
                print("no data")
            else:
                count = record.get("total_count", "?")
                tree = record.get("tree_severity", "?")
                print(f"count={count}, trees={tree}")
        except requests.HTTPError as e:
            print(f"HTTP error: {e}")
        except Exception as e:
            print(f"Error: {e}")

        current += timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    return all_records


def save_csv(records, filename):
    """Save records to CSV."""
    if not records:
        print("No records to save.")
        return

    filepath = DATA_DIR / "processed" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Get all unique keys across records
    all_keys = []
    for r in records:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(records)

    print(f"\nSaved {len(records)} records to {filepath}")


def save_json(records, filename):
    """Save records to JSON."""
    filepath = DATA_DIR / "processed" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Saved {len(records)} records to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Scrape Atlanta Allergy pollen data")
    parser.add_argument("--calendar", action="store_true", help="Scrape monthly calendar pages")
    parser.add_argument("--details", action="store_true", help="Scrape daily detail pages")
    parser.add_argument("--start-year", type=int, default=2009)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--months", type=str, default="1,2,3,4,5,6",
                        help="Comma-separated months to scrape")
    parser.add_argument("--start-date", type=str, help="Start date for detail scraping (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date for detail scraping (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="pollen_data",
                        help="Output filename (without extension)")

    args = parser.parse_args()

    if args.calendar:
        months = [int(m) for m in args.months.split(",")]
        records = scrape_calendar(args.start_year, args.end_year, months)
        save_csv(records, f"{args.output}_calendar.csv")
        save_json(records, f"{args.output}_calendar.json")

    elif args.details:
        if not args.start_date or not args.end_date:
            print("--start-date and --end-date required for detail scraping")
            sys.exit(1)
        start = date.fromisoformat(args.start_date)
        end = date.fromisoformat(args.end_date)
        records = scrape_details(start, end)
        save_csv(records, f"{args.output}_details.csv")
        save_json(records, f"{args.output}_details.json")

    else:
        print("Specify --calendar or --details")
        sys.exit(1)


if __name__ == "__main__":
    main()
