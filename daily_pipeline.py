#!/usr/bin/env python3
"""
Atlanta Pollen Tracker — Daily Pipeline

Produces dashboard.json for the frontend by:
1. Scraping today's pollen count from Atlanta Allergy & Asthma
2. Fetching weather from Open-Meteo forecast API (7-day history + 2-day forecast)
3. Scoring the V8 model for today and tomorrow
4. Looking up climatology percentiles
5. Computing season progress and analog year projections
6. Writing dashboard.json

Self-bootstrapping: reads the previous dashboard.json for recent history and
season accumulators. No large CSVs needed.

Usage:
    python daily_pipeline.py              # normal daily run
    python daily_pipeline.py --date 2026-04-04   # backfill a specific date
    python daily_pipeline.py --dry-run    # print dashboard.json to stdout
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ───────────────────────────────────────────────
# PATHS
# ───────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "model_output"
DASHBOARD_PATH = ROOT / "dashboard.json"

# ───────────────────────────────────────────────
# CONSTANTS
# ───────────────────────────────────────────────
SEVERITY_THRESHOLDS = [(1500, "extreme"), (500, "high"), (100, "moderate"), (0, "low")]
SCRAPER_URL = "https://www.atlantaallergy.com/pollen_counts/index"
SCRAPER_HEADERS = {
    "User-Agent": "AtlantaPollenTracker/1.0 (personal research; andrewkstein@gmail.com)"
}
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_PARAMS = {
    "latitude": 33.749,
    "longitude": -84.388,
    "daily": ",".join([
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "precipitation_sum", "wind_speed_10m_max", "wind_gusts_10m_max",
        "wind_direction_10m_dominant", "et0_fao_evapotranspiration",
        "vapour_pressure_deficit_max", "shortwave_radiation_sum"
    ]),
    "timezone": "America/New_York",
    "past_days": 7,
    "forecast_days": 2,
}


def classify_severity(count):
    if count is None:
        return None
    for threshold, label in SEVERITY_THRESHOLDS:
        if count >= threshold:
            return label
    return "low"


# ───────────────────────────────────────────────
# STEP 1: SCRAPE TODAY
# ───────────────────────────────────────────────
def scrape_today(target_date):
    """Scrape the detail page for target_date. Returns dict or None."""
    y, m, d = target_date.year, target_date.month, target_date.day
    url = f"{SCRAPER_URL}/{y}/{m:02d}/{d:02d}"

    try:
        resp = requests.get(url, headers=SCRAPER_HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[WARN] Scraper HTTP error for {target_date}: {e}", file=sys.stderr)
        return None

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # Structural check: does the page have the expected format?
    no_data = soup.find(string=re.compile(r"There is no pollen data", re.I))
    if no_data:
        return {"date": str(target_date), "missing": True, "total_count": None}

    pollen_num = soup.select_one("span.pollen-num")
    if not pollen_num:
        print(f"[WARN] Structural change: span.pollen-num not found on {target_date}", file=sys.stderr)
        return {"date": str(target_date), "missing": True, "total_count": None,
                "structural_error": True}

    count_text = pollen_num.get_text(strip=True)
    try:
        count = int(count_text)
    except ValueError:
        print(f"[WARN] Could not parse count '{count_text}' on {target_date}", file=sys.stderr)
        return {"date": str(target_date), "missing": True, "total_count": None}

    result = {
        "date": str(target_date),
        "missing": False,
        "total_count": count,
        "severity": classify_severity(count),
    }

    # Extract contributors
    for gauge in soup.select("div.gauge"):
        h3 = gauge.select_one("h3")
        if not h3:
            continue
        title = h3.get_text(strip=True)
        if "Trees" in title:
            prefix = "tree"
        elif "Grass" in title:
            prefix = "grass"
        elif "Weeds" in title:
            prefix = "weed"
        else:
            continue
        p = gauge.select_one("p")
        if p:
            contributors = p.get_text(strip=True).strip("\xa0").strip()
            if contributors:
                result[f"{prefix}_contributors"] = contributors
        active = gauge.select_one("span.active, span[class*='active']")
        if active:
            for level in ["extreme", "high", "medium", "low"]:
                if level in active.get("class", []):
                    result[f"{prefix}_severity"] = level
                    break

    return result


# ───────────────────────────────────────────────
# STEP 2: FETCH WEATHER
# ───────────────────────────────────────────────
def fetch_weather():
    """Fetch 9-day weather window from Open-Meteo forecast API.
    Returns list of dicts with keys: date, temp_max, temp_min, temp_mean,
    precipitation, wind_max, wind_gust, wind_direction, solar_radiation, vpd_max.
    Units: Fahrenheit, inches, mph.
    """
    try:
        resp = requests.get(WEATHER_URL, params=WEATHER_PARAMS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"[ERROR] Weather fetch failed: {e}", file=sys.stderr)
        return None

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    rows = []
    for i, dt in enumerate(dates):
        def val(key):
            arr = daily.get(key, [])
            return arr[i] if i < len(arr) else None

        # Open-Meteo forecast API returns Celsius, mm, km/h — convert
        temp_max_c = val("temperature_2m_max")
        temp_min_c = val("temperature_2m_min")
        temp_mean_c = val("temperature_2m_mean")
        precip_mm = val("precipitation_sum")
        wind_kmh = val("wind_speed_10m_max")
        gust_kmh = val("wind_gusts_10m_max")

        rows.append({
            "date": dt,
            "temp_max": round(temp_max_c * 9 / 5 + 32, 1) if temp_max_c is not None else None,
            "temp_min": round(temp_min_c * 9 / 5 + 32, 1) if temp_min_c is not None else None,
            "temp_mean": round(temp_mean_c * 9 / 5 + 32, 1) if temp_mean_c is not None else None,
            "precipitation": round(precip_mm / 25.4, 3) if precip_mm is not None else None,
            "wind_max": round(wind_kmh / 1.609, 1) if wind_kmh is not None else None,
            "wind_gust": round(gust_kmh / 1.609, 1) if gust_kmh is not None else None,
            "wind_direction": val("wind_direction_10m_dominant"),
            "solar_radiation": val("shortwave_radiation_sum"),
            "vpd_max": val("vapour_pressure_deficit_max"),
        })

    return rows


# ───────────────────────────────────────────────
# STEP 3: V8 MODEL SCORING
# ───────────────────────────────────────────────
def load_v8_weights():
    path = MODEL_DIR / "weather_model_v8_weights.json"
    with open(path) as f:
        return json.load(f)


def classify_regime(temp_mean, wind_max, precip):
    """Classify weather into regime one-hot (7 categories)."""
    hot = (temp_mean or 0) >= 65
    warm = (temp_mean or 0) >= 50
    windy = (wind_max or 0) >= 10
    rainy = (precip or 0) >= 0.25
    drizzle = (precip or 0) >= 0.05

    if rainy:
        return "rainy"
    if drizzle:
        return "drizzle"
    if not warm:
        return "cool"
    if hot and windy:
        return "hot_windy"
    if hot:
        return "hot_calm"
    if windy:
        return "warm_windy"
    return "warm_calm"


def assemble_v8_features(today_weather, yesterday_weather, history, prev_dashboard, target_date):
    """Assemble the 44-feature vector for V8 scoring.

    Args:
        today_weather: dict with today's weather
        yesterday_weather: dict with yesterday's weather
        history: list of recent observation dicts (from dashboard.recent_history)
        prev_dashboard: previous dashboard.json content (or None)
        target_date: date object

    Returns: list of 44 floats in feature_names order, or None if insufficient data.
    """
    doy = target_date.timetuple().tm_yday

    # --- Autoregressive features ---
    # Get recent pollen counts from history
    recent_counts = []
    for h in reversed(history[-7:]):
        c = h.get("count")
        if c is not None:
            recent_counts.append(c)
    if len(recent_counts) < 1:
        # No history at all — can't score
        return None

    prev_count = recent_counts[0]
    prev2_count = recent_counts[1] if len(recent_counts) >= 2 else prev_count
    prev_log = math.log(prev_count + 1)
    prev2_log = math.log(prev2_count + 1)
    d1_d2_diff = prev_log - prev2_log

    # 5-day pollen stats
    recent_logs = [math.log(c + 1) for c in recent_counts[:5]]
    pollen_5d_max = max(recent_logs) if recent_logs else prev_log
    pollen_5d_trend = (recent_logs[0] - recent_logs[-1]) / len(recent_logs) if len(recent_logs) > 1 else 0

    # --- Weather features ---
    tw = today_weather
    yw = yesterday_weather
    temp_mean = tw.get("temp_mean", 58)
    temp_min = tw.get("temp_min", 48)
    temp_max = tw.get("temp_max", 68)
    temp_above_50 = max(0, temp_mean - 50)
    precip_today = tw.get("precipitation", 0)
    wind_max = tw.get("wind_max", 10)
    solar_radiation = tw.get("solar_radiation", 17)

    # Temperature anomaly (deviation from ~58F spring average)
    temp_anomaly = temp_mean - 58

    # 5-day temp trend: approximate from today vs yesterday
    yw_temp = yw.get("temp_mean", temp_mean)
    temp_5d_trend = temp_mean - yw_temp

    # Warm night
    warm_night = 1 if temp_min >= 55 else 0

    # Phase interactions
    temp_x_early = temp_above_50 if doy < 90 else 0
    temp_x_late = temp_above_50 if doy >= 90 else 0

    # Temperature stability (absolute diff from yesterday)
    temp_stability = yw.get("temp_mean", temp_mean) - temp_mean

    # Wind direction N/S component
    wind_dir = tw.get("wind_direction", 180)
    wind_dir_ns = math.cos(math.radians(wind_dir or 180))

    # Precipitation features
    precip_yesterday = yw.get("precipitation", 0) or 0
    precip_2day_sum = precip_today + precip_yesterday

    # Estimate 5-day precip total from available weather window
    precip_5d_total = precip_yesterday + precip_today  # approximate

    # Dry days in 5 (approximate: check yesterday)
    dry_days_5d = 3  # default mid-range
    if precip_yesterday < 0.05 and precip_today < 0.05:
        dry_days_5d = 5
    elif precip_yesterday >= 0.05 or precip_today >= 0.05:
        dry_days_5d = 2

    # Rain then dry
    rain_then_dry = 1 if precip_yesterday >= 0.1 and precip_today < 0.05 else 0

    # --- GDD ---
    gdd_daily = max(0, temp_mean - 50)

    # GDD armed: was GDD threshold crossed?
    prev_gdd = 0
    if prev_dashboard and "season" in prev_dashboard:
        prev_gdd = prev_dashboard["season"].get("gdd_cumulative", 0)
    gdd_armed = 1 if (prev_gdd + gdd_daily) >= 100 else 0

    # --- Seasonal features ---
    doy_sin = math.sin(2 * math.pi * doy / 365)
    doy_cos = math.cos(2 * math.pi * doy / 365)

    # Season progress (fraction of season complete)
    cumulative_burden = 0
    if prev_dashboard and "season" in prev_dashboard:
        cumulative_burden = prev_dashboard["season"].get("cumulative_burden", 0)
    avg_total = 173060  # from_recent_avg
    season_progress = min(cumulative_burden / avg_total, 1.0) if avg_total > 0 else 0.5

    # Year trend (normalized, 2026 is ~0.34 on the training scale)
    year_trend = 0.34

    # --- Yesterday flags ---
    yest_was_rainy = 1 if precip_yesterday >= 0.1 else 0
    yest_was_dry_warm = 1 if precip_yesterday < 0.05 and yw_temp >= 55 else 0

    # --- Regime one-hots ---
    today_regime = classify_regime(temp_mean, wind_max, precip_today)
    yest_regime = classify_regime(yw_temp, yw.get("wind_max", 10), precip_yesterday)

    regime_names = ["hot_windy", "hot_calm", "warm_windy", "warm_calm", "cool", "drizzle", "rainy"]
    today_regimes = [1 if today_regime == r else 0 for r in regime_names]
    yest_regimes = [1 if yest_regime == r else 0 for r in regime_names]

    # Assemble in feature_names order
    features = [
        prev_log,           # prev_log_count
        prev2_log,          # prev2_log_count
        d1_d2_diff,         # d1_d2_diff
        pollen_5d_max,      # pollen_5d_max
        pollen_5d_trend,    # pollen_5d_trend
        temp_mean,          # temp_mean
        temp_above_50,      # temp_above_50
        temp_anomaly,       # temp_anomaly
        temp_5d_trend,      # temp_5d_trend
        temp_min,           # temp_min
        warm_night,         # warm_night
        temp_x_early,       # temp_x_early
        temp_x_late,        # temp_x_late
        temp_stability,     # temp_stability
        wind_dir_ns,        # wind_dir_ns
        solar_radiation,    # solar_radiation
        precip_yesterday,   # precip_yesterday
        precip_2day_sum,    # precip_2day_sum
        precip_5d_total,    # precip_5d_total
        dry_days_5d,        # dry_days_5d
        rain_then_dry,      # rain_then_dry
        wind_max,           # wind_max
        gdd_daily,          # gdd_daily
        gdd_armed,          # gdd_armed
        doy_sin,            # doy_sin
        doy_cos,            # doy_cos
        season_progress,    # season_progress
        year_trend,         # year_trend
        yest_was_rainy,     # yest_was_rainy
        yest_was_dry_warm,  # yest_was_dry_warm
    ] + today_regimes + yest_regimes

    return features


def score_v8(features, weights_data):
    """Score a 44-feature vector using V8 weights.
    Returns (predicted_count, contributions) where contributions is a list of
    (feature_name, contribution, direction) sorted by |contribution|.
    """
    if features is None:
        return None, []

    names = weights_data["feature_names"]
    means = weights_data["means"]
    stds = weights_data["stds"]
    w = weights_data["weights"]
    bias = weights_data["bias"]

    # Standardize and compute per-feature contributions
    contributions = []
    z = []
    for i in range(len(features)):
        zi = (features[i] - means[i]) / stds[i] if stds[i] > 0 else 0
        z.append(zi)
        contrib = w[i] * zi
        contributions.append((names[i], contrib))

    log_pred = bias + sum(w[i] * z[i] for i in range(len(w)))
    predicted_count = max(0, math.exp(log_pred) - 1)

    # Sort by absolute contribution
    contributions.sort(key=lambda x: -abs(x[1]))

    return round(predicted_count), contributions


# Human-readable labels for features
FEATURE_LABELS = {
    "prev_log_count": "yesterday's count",
    "prev2_log_count": "the day before's count",
    "d1_d2_diff": "the day-over-day trend",
    "pollen_5d_max": "the 5-day peak",
    "pollen_5d_trend": "the recent pollen trend",
    "temp_mean": "today's temperature",
    "temp_above_50": "warmth above 50\u00b0F",
    "temp_anomaly": "unusually warm temperatures",
    "temp_5d_trend": "the warming trend",
    "temp_min": "the overnight low",
    "warm_night": "a warm overnight low (above 55\u00b0F)",
    "temp_x_early": "early-season warmth",
    "temp_x_late": "late-season warmth",
    "temp_stability": "temperature stability",
    "wind_dir_ns": "the wind direction",
    "solar_radiation": "solar radiation",
    "precip_yesterday": "yesterday's rain",
    "precip_2day_sum": "recent rainfall",
    "precip_5d_total": "rain over the past 5 days",
    "dry_days_5d": "dry conditions",
    "rain_then_dry": "post-rain drying",
    "wind_max": "wind speed",
    "gdd_daily": "growing degree days",
    "gdd_armed": "accumulated heat",
    "doy_sin": "time of year",
    "doy_cos": "time of year",
    "season_progress": "how far into the season we are",
    "year_trend": "the long-term trend in pollen",
    "yest_was_rainy": "yesterday's rain",
    "yest_was_dry_warm": "yesterday being dry and warm",
    "today_hot_windy": "hot and windy conditions",
    "today_hot_calm": "hot and calm conditions",
    "today_warm_windy": "warm and breezy conditions",
    "today_warm_calm": "warm and calm conditions",
    "today_cool": "cool temperatures",
    "today_drizzle": "drizzle",
    "today_rainy": "rain today",
    "yest_hot_windy": "yesterday being hot and windy",
    "yest_hot_calm": "yesterday being hot and calm",
    "yest_warm_windy": "yesterday being warm and breezy",
    "yest_warm_calm": "yesterday being warm and calm",
    "yest_cool": "yesterday being cool",
    "yest_drizzle": "yesterday's drizzle",
    "yest_rainy": "yesterday's rain",
}


def build_causal_explanation(contributions, predicted_count, actual_count=None, weather=None):
    """Build a human-readable causal explanation from feature contributions."""
    if not contributions:
        return ""

    # Get top 3 positive and top 1 negative contributors (skip near-zero)
    pushers_up = [(n, c) for n, c in contributions if c > 0.05][:3]
    pushers_down = [(n, c) for n, c in contributions if c < -0.05][:2]

    parts = []

    # Lead with the weather context
    if weather:
        temp = weather.get("temp_mean")
        precip = weather.get("precipitation", 0)
        wind = weather.get("wind_max")
        pieces = []
        if temp is not None:
            pieces.append(f"{temp:.0f}\u00b0F")
        if wind is not None:
            pieces.append(f"winds up to {wind:.0f} mph")
        if precip and precip > 0.05:
            pieces.append(f"{precip:.1f}\" of rain")
        elif precip is not None:
            pieces.append("no rain")
        if pieces:
            parts.append("With " + ", ".join(pieces) + ".")

    # Main drivers
    if pushers_up:
        driver_labels = [FEATURE_LABELS.get(n, n) for n, _ in pushers_up]
        # Deduplicate (e.g., two "yesterday's rain" features)
        seen = set()
        unique = []
        for lbl in driver_labels:
            if lbl not in seen:
                seen.add(lbl)
                unique.append(lbl)
        if len(unique) == 1:
            parts.append(f"The biggest factor pushing counts up: {unique[0]}.")
        else:
            parts.append(f"Key factors pushing counts up: {', '.join(unique[:-1])} and {unique[-1]}.")

    if pushers_down:
        down_labels = list(dict.fromkeys(FEATURE_LABELS.get(n, n) for n, _ in pushers_down))
        parts.append(f"Working against that: {', '.join(down_labels)}.")

    # Compare to actual if available
    if actual_count is not None and predicted_count is not None:
        diff = actual_count - predicted_count
        pct = abs(diff) / predicted_count * 100 if predicted_count > 0 else 0
        if pct > 30 and diff > 0:
            parts.append(f"The actual count came in {pct:.0f}% higher than the model expected \u2014 a sign of factors beyond what weather alone explains.")
        elif pct > 30 and diff < 0:
            parts.append(f"The actual count came in {pct:.0f}% lower than predicted \u2014 possibly localized rain or wind patterns.")

    return " ".join(parts)


# ───────────────────────────────────────────────
# STEP 4: CLIMATOLOGY
# ───────────────────────────────────────────────
def load_climatology():
    path = MODEL_DIR / "climatology.json"
    with open(path) as f:
        return json.load(f)


def lookup_climatology(clim_data, doy, actual_count=None):
    """Look up percentile info for a DOY. Optionally compute percentile rank."""
    key = str(doy)
    entry = clim_data.get(key, {})
    result = {
        "doy": doy,
        "n_years": entry.get("n_years", 0),
        "mean": round(entry.get("mean", 0)),
        "median": entry.get("median", 0),
        "p10": entry.get("p10", 0),
        "p25": entry.get("p25", 0),
        "p75": entry.get("p75", 0),
        "p90": entry.get("p90", 0),
        "p95": entry.get("p95", 0),
        "max": entry.get("max", 0),
    }
    if actual_count is not None:
        # Approximate percentile rank
        percentiles = [
            (entry.get("p10", 0), 10), (entry.get("p25", 0), 25),
            (entry.get("median", 0), 50), (entry.get("p75", 0), 75),
            (entry.get("p90", 0), 90), (entry.get("p95", 0), 95),
        ]
        rank = 50  # default
        for val, pct in percentiles:
            if actual_count <= val:
                rank = pct
                break
        else:
            rank = 99
        result["percentile_rank"] = rank

    return result


# ───────────────────────────────────────────────
# STEP 5: SEASON PROGRESS
# ───────────────────────────────────────────────
def load_analog_data():
    path = MODEL_DIR / "analog_projection.json"
    with open(path) as f:
        return json.load(f)


def load_season_progress_ref():
    path = MODEL_DIR / "season_progress.json"
    with open(path) as f:
        return json.load(f)


def compute_season(obs, prev_dashboard, target_date):
    """Compute season progress fields."""
    doy = target_date.timetuple().tm_yday

    # Get previous values
    prev_season = {}
    if prev_dashboard and "season" in prev_dashboard:
        prev_season = prev_dashboard["season"]

    prev_burden = prev_season.get("cumulative_burden", 0)
    prev_extreme = prev_season.get("extreme_days_ytd", 0)
    prev_high = prev_season.get("high_days_ytd", 0)
    prev_gdd = prev_season.get("gdd_cumulative", 0)

    today_count = 0
    if obs and not obs.get("missing") and obs.get("total_count") is not None:
        today_count = obs["total_count"]

    cumulative_burden = prev_burden + today_count
    extreme_days = prev_extreme + (1 if today_count >= 1500 else 0)
    high_days = prev_high + (1 if today_count >= 500 else 0)

    # GDD
    gdd_daily = 0
    if prev_dashboard and "today" in prev_dashboard and "weather" in prev_dashboard["today"]:
        temp_mean = prev_dashboard["today"]["weather"].get("temp_mean", 50)
        gdd_daily = max(0, temp_mean - 50)
    gdd_cumulative = prev_gdd + gdd_daily

    # Analog projection (use cached data, re-weight by current burden)
    ref = load_season_progress_ref()
    analog_data = load_analog_data()
    analog_years = []
    remaining_extreme_list = []
    season_end_list = []
    projected_totals = []

    for ay in analog_data.get("analog_years", []):
        burden_at_doy = ay.get("burden_at_doy", 1)
        ratio = cumulative_burden / burden_at_doy if burden_at_doy > 0 else 1
        # Similarity decays as our burden diverges from the analog's burden at same DOY
        sim = max(0, 1 - abs(ratio - 1))
        total = ay.get("total_season_burden", 50000)
        analog_years.append({
            "year": ay["year"],
            "similarity": round(sim, 3),
            "remaining_extreme_days": ay.get("remaining_extreme_days", 0),
            "last_extreme_doy": ay.get("last_extreme_doy", 100),
            "total_season_burden": total,
        })
        remaining_extreme_list.append((sim, ay.get("remaining_extreme_days", 0)))
        season_end_list.append((sim, ay.get("last_over100_doy", 130)))
        projected_totals.append((sim, total))

    # Season progress: use analog-weighted projected total instead of stale historical avg
    total_sim = sum(s for s, _ in projected_totals) or 1
    analog_projected_total = sum(s * t for s, t in projected_totals) / total_sim
    progress_pct = round(cumulative_burden / analog_projected_total * 100, 1) if analog_projected_total > 0 else 0
    progress_pct = min(progress_pct, 100.0)  # cap at 100%

    # Weighted averages
    total_weight = sum(s for s, _ in remaining_extreme_list) or 1
    remaining_extreme_avg = sum(s * v for s, v in remaining_extreme_list) / total_weight
    season_end_avg = sum(s * v for s, v in season_end_list) / total_weight

    all_remaining = [v for _, v in remaining_extreme_list]
    all_ends = [v for _, v in season_end_list]

    # Convert DOY to date string
    def doy_to_date(d):
        return (date(target_date.year, 1, 1) + timedelta(days=d - 1)).isoformat()

    decade_trend = ref.get("season_trend", {
        "1990s_avg_total": 22107, "2000s_avg_total": 30651,
        "2010s_avg_total": 51593, "2020s_avg_total": 71037
    })

    return {
        "as_of_date": str(target_date),
        "as_of_doy": doy,
        "cumulative_burden": cumulative_burden,
        "extreme_days_ytd": extreme_days,
        "high_days_ytd": high_days,
        "gdd_cumulative": round(gdd_cumulative, 1),
        "season_progress_pct": progress_pct,
        "remaining_extreme_days": {
            "weighted_avg": round(remaining_extreme_avg, 1),
            "min": min(all_remaining) if all_remaining else 0,
            "max": max(all_remaining) if all_remaining else 0,
        },
        "estimated_season_end": {
            "date": doy_to_date(round(season_end_avg)),
            "doy_weighted_avg": round(season_end_avg),
            "doy_min": min(all_ends) if all_ends else 118,
            "doy_max": max(all_ends) if all_ends else 145,
        },
        "analog_years": sorted(analog_years, key=lambda a: -a["similarity"])[:5],
        "decade_trend": {
            "1990s": decade_trend.get("1990s_avg_total", 22107),
            "2000s": decade_trend.get("2000s_avg_total", 30651),
            "2010s": decade_trend.get("2010s_avg_total", 51593),
            "2020s": decade_trend.get("2020s_avg_total", 71037),
        },
    }


# ───────────────────────────────────────────────
# STEP 6: BUILD DASHBOARD
# ───────────────────────────────────────────────
def build_dashboard(target_date, obs, weather_rows, prev_dashboard, dry_run=False):
    """Build the complete dashboard.json."""
    doy = target_date.timetuple().tm_yday
    is_weekend = target_date.weekday() >= 5

    # Load static data
    weights = load_v8_weights()
    clim_data = load_climatology()

    # Find today's and yesterday's weather
    target_str = str(target_date)
    yesterday_str = str(target_date - timedelta(days=1))

    today_weather = {}
    yesterday_weather = {}
    if weather_rows:
        for w in weather_rows:
            if w["date"] == target_str:
                today_weather = w
            elif w["date"] == yesterday_str:
                yesterday_weather = w

    # Get recent history from previous dashboard
    history = []
    if prev_dashboard and "recent_history" in prev_dashboard:
        history = prev_dashboard["recent_history"]

    # Determine data status
    has_observation = obs and not obs.get("missing") and obs.get("total_count") is not None
    structural_error = obs and obs.get("structural_error")

    if structural_error:
        data_status = "scraper_error"
    elif has_observation:
        data_status = "observation_posted"
    elif is_weekend:
        data_status = "weekend_no_data"
    else:
        data_status = "awaiting_observation"

    # Observation block
    observation = None
    observation_status = "not_posted"
    if has_observation:
        observation = {
            "count": obs["total_count"],
            "severity": obs.get("severity"),
            "contributors": obs.get("tree_contributors", ""),
        }
        observation_status = "posted"
    elif is_weekend:
        observation_status = "weekend"

    # Score V8 for today
    today_count = obs.get("total_count") if has_observation else None
    forecast_data_source = "observation" if has_observation else "prior_prediction"

    features_today = assemble_v8_features(
        today_weather, yesterday_weather, history, prev_dashboard, target_date
    )
    today_pred, today_contribs = score_v8(features_today, weights)

    # Build causal explanation for today
    today_explanation = build_causal_explanation(
        today_contribs, today_pred,
        actual_count=today_count if has_observation else None,
        weather=today_weather
    )

    # Score V8 for tomorrow
    tomorrow_date = target_date + timedelta(days=1)
    tomorrow_str = str(tomorrow_date)
    tomorrow_weather = {}
    if weather_rows:
        for w in weather_rows:
            if w["date"] == tomorrow_str:
                tomorrow_weather = w

    # For tomorrow, use today's observation (or prediction) as prev_log_count
    tomorrow_history = list(history)
    if has_observation:
        tomorrow_history.append({"date": target_str, "count": today_count})
    elif today_pred is not None:
        tomorrow_history.append({"date": target_str, "count": today_pred})

    features_tomorrow = assemble_v8_features(
        tomorrow_weather, today_weather, tomorrow_history, None, tomorrow_date
    )
    tomorrow_pred, tomorrow_contribs = score_v8(features_tomorrow, weights)

    # Build causal explanation for tomorrow
    tomorrow_explanation = build_causal_explanation(
        tomorrow_contribs, tomorrow_pred, weather=tomorrow_weather
    )

    # Climatology
    today_actual = today_count if has_observation else today_pred
    today_clim = lookup_climatology(clim_data, doy, today_actual)
    tomorrow_doy = tomorrow_date.timetuple().tm_yday
    tomorrow_clim = lookup_climatology(clim_data, tomorrow_doy)

    # Season progress
    season = compute_season(obs if has_observation else None, prev_dashboard, target_date)

    # Weather regime
    today_regime = classify_regime(
        today_weather.get("temp_mean"), today_weather.get("wind_max"),
        today_weather.get("precipitation")
    )
    tomorrow_regime = classify_regime(
        tomorrow_weather.get("temp_mean"), tomorrow_weather.get("wind_max"),
        tomorrow_weather.get("precipitation")
    )

    # Update recent history (keep last 45 days)
    new_entry = {
        "date": target_str,
        "count": today_count,
        "severity": classify_severity(today_count) if today_count else None,
        "predicted": today_pred,
    }
    updated_history = [h for h in history if h.get("date") != target_str]
    updated_history.append(new_entry)
    updated_history = updated_history[-45:]

    # Current year S-curve
    prev_scurve = []
    if prev_dashboard and "current_year_scurve" in prev_dashboard:
        prev_scurve = prev_dashboard["current_year_scurve"]
    # Append today's point
    new_scurve = [p for p in prev_scurve if p[0] < doy]
    new_scurve.append([doy, season["cumulative_burden"]])

    # Build final dashboard
    now = datetime.now().astimezone()
    dashboard = {
        "_meta": {
            "generated_at": now.isoformat(),
            "data_status": data_status,
            "stale_hours": 0,
        },
        "today": {
            "date": target_str,
            "doy": doy,
            "is_weekend": is_weekend,
            "observation": observation,
            "observation_status": observation_status,
            "forecast": {
                "count": today_pred,
                "severity": classify_severity(today_pred),
                "data_source": forecast_data_source,
                "explanation": today_explanation,
            },
            "climatology": today_clim,
            "weather": {
                "temp_max": today_weather.get("temp_max"),
                "temp_min": today_weather.get("temp_min"),
                "temp_mean": today_weather.get("temp_mean"),
                "precipitation": today_weather.get("precipitation"),
                "wind_max": today_weather.get("wind_max"),
                "solar_radiation": today_weather.get("solar_radiation"),
                "regime": today_regime,
            },
        },
        "tomorrow": {
            "date": tomorrow_str,
            "doy": tomorrow_doy,
            "forecast": {
                "count": tomorrow_pred,
                "severity": classify_severity(tomorrow_pred),
                "explanation": tomorrow_explanation,
            },
            "climatology": {
                "doy": tomorrow_doy,
                "median": tomorrow_clim.get("median"),
                "p75": tomorrow_clim.get("p75"),
                "p90": tomorrow_clim.get("p90"),
            },
            "weather_forecast": {
                "temp_max": tomorrow_weather.get("temp_max"),
                "temp_min": tomorrow_weather.get("temp_min"),
                "temp_mean": tomorrow_weather.get("temp_mean"),
                "precipitation": tomorrow_weather.get("precipitation"),
                "wind_max": tomorrow_weather.get("wind_max"),
                "regime": tomorrow_regime,
            },
        },
        "season": season,
        "recent_history": updated_history,
        "current_year_scurve": new_scurve,
    }

    return dashboard


# ───────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Atlanta Pollen Tracker daily pipeline")
    parser.add_argument("--date", help="Target date (YYYY-MM-DD), default: today")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON to stdout instead of writing file")
    args = parser.parse_args()

    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        target_date = date.today()

    print(f"[INFO] Running pipeline for {target_date}", file=sys.stderr)

    # Load previous dashboard for state
    prev_dashboard = None
    if DASHBOARD_PATH.exists():
        try:
            with open(DASHBOARD_PATH) as f:
                prev_dashboard = json.load(f)
        except (json.JSONDecodeError, IOError):
            print("[WARN] Could not read previous dashboard.json", file=sys.stderr)

    # Step 1: Scrape
    print("[INFO] Step 1: Scraping pollen data...", file=sys.stderr)
    obs = scrape_today(target_date)
    if obs and obs.get("structural_error"):
        print("[ERROR] Scraper structural change detected!", file=sys.stderr)

    # Step 2: Weather
    print("[INFO] Step 2: Fetching weather...", file=sys.stderr)
    weather_rows = fetch_weather()
    if not weather_rows:
        print("[ERROR] Weather fetch failed — cannot score model", file=sys.stderr)
        sys.exit(1)

    # Steps 3-6: Build dashboard
    print("[INFO] Steps 3-6: Scoring model, computing season progress...", file=sys.stderr)
    dashboard = build_dashboard(target_date, obs, weather_rows, prev_dashboard)

    # Output
    output = json.dumps(dashboard, indent=2, ensure_ascii=False)

    if args.dry_run:
        print(output)
    else:
        with open(DASHBOARD_PATH, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"[INFO] Wrote {DASHBOARD_PATH}", file=sys.stderr)

    # Exit with error if structural scraper issue
    if obs and obs.get("structural_error"):
        sys.exit(1)

    print("[INFO] Pipeline complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
