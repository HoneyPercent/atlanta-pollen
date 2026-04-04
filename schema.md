# Canonical Data Schema — Atlanta Pollen Tracker

## Layer 1: observations_raw

Raw HTML files saved by scraper for audit trail.

**Storage:** `data/raw/{YYYY}_{MM}.html` (calendar pages), `data/raw/{YYYY}_{MM}_{DD}.html` (detail pages)

---

## Layer 2: observations_daily

Cleaned daily pollen observations from calendar scraping.

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `date` | ISO date (YYYY-MM-DD) | URL path | Observation date (count covers previous 24 hours) |
| `total_count` | integer or null | Calendar page link text | Grains per cubic meter. Null = no data. |
| `severity_class` | string or null | CSS class on calendar-day div | One of: `low`, `medium`, `high`, `extreme`. Null if no data. |
| `missing` | boolean | Derived | True if no count posted for this date |
| `source` | string | Fixed | `atlanta_allergy_calendar` |
| `source_url` | string | Constructed | Full URL of the calendar page |

**File:** `data/processed/pollen_calendar.csv`

---

## Layer 2b: observations_detail

Enriched observations from detail page scraping (subset of dates, higher cost to scrape).

| Field | Type | Source | Notes |
|-------|------|--------|-------|
| `date` | ISO date | URL path | |
| `total_count` | integer or null | `span.pollen-num` | |
| `missing` | boolean | "no pollen data" message | |
| `tree_severity` | string or null | `span.active` in Trees gauge | `low`/`medium`/`high`/`extreme` |
| `tree_needle_pct` | float or null | `span.needle` style `left: X%` | 0-100, proxy for relative level |
| `tree_contributors` | string or null | `<p>` under Trees `<h3>` | Comma-separated species names |
| `grass_severity` | string or null | Same pattern | |
| `grass_needle_pct` | float or null | | |
| `weed_severity` | string or null | | |
| `weed_needle_pct` | float or null | | |
| `weed_contributors` | string or null | | |
| `mold_level` | string or null | `span.active` in Mold gauge | Text label: "Low", "Moderate", "High", "Extremely High" |
| `mold_needle_pct` | float or null | Mold gauge needle | |
| `source` | string | Fixed | `atlanta_allergy_detail` |
| `source_url` | string | | |

**File:** `data/processed/pollen_details.csv`

---

## Layer 3: weather_actual_daily

Historical weather actuals from Open-Meteo Archive API.

| Field | Type | Unit | Notes |
|-------|------|------|-------|
| `date` | ISO date | | |
| `temperature_2m_max` | float | °F | Daily max temperature |
| `temperature_2m_min` | float | °F | Daily min temperature |
| `temperature_2m_mean` | float | °F | Daily mean temperature |
| `precipitation_sum` | float | inches | Total precipitation |
| `rain_sum` | float | inches | Rain only (no snow) |
| `wind_speed_10m_max` | float | mph | Max wind speed |
| `wind_gusts_10m_max` | float | mph | Max wind gusts |
| `wind_direction_10m_dominant` | integer | degrees | Dominant wind direction |
| `et0_fao_evapotranspiration` | float | inches | Reference evapotranspiration |

**File:** `data/processed/weather_actual.csv`

---

## Layer 4: weather_forecast_archived

Archived forecast weather for honest backtesting. Same fields as Layer 3, plus:

| Field | Type | Notes |
|-------|------|-------|
| `issue_date` | ISO date | Date the forecast was made |
| `target_date` | ISO date | Date being forecast |
| `horizon_days` | integer | target_date - issue_date |

**Source:** Open-Meteo Previous Runs API (to be implemented)

**File:** `data/processed/weather_forecast_archive.csv`

---

## Layer 5: features_daily

Computed features for modeling. Derived from Layers 2 + 3.

| Field | Type | Notes |
|-------|------|-------|
| `date` | ISO date | |
| `total_count` | integer or null | From observations |
| `log_count` | float or null | log(total_count + 1) for modeling |
| `severity_class` | string or null | From observations |
| `missing` | boolean | |
| `gdd_daily` | float | Growing Degree Days for this day (base 50°F) = max(0, mean_temp - 50) |
| `gdd_cumulative` | float | Cumulative GDD since Jan 1 |
| `precip_today` | float | |
| `precip_yesterday` | float | |
| `precip_2day_sum` | float | Sum of today + yesterday precipitation |
| `temp_max` | float | |
| `temp_min` | float | |
| `temp_mean` | float | |
| `wind_max` | float | |
| `day_of_year` | integer | 1-366 |
| `is_weekend` | boolean | Structural missing-data indicator |
| `cumulative_burden` | float | Running sum of total_count since Jan 1 (for season progress) |

**File:** `data/processed/features_daily.csv`

---

## Layer 6: qa_events

Data quality flags and anomalies.

| Field | Type | Notes |
|-------|------|-------|
| `date` | ISO date | |
| `event_type` | string | `missing_weekday`, `missing_holiday`, `extreme_outlier`, `scraper_error`, `stale_data` |
| `description` | string | Human-readable note |
| `auto_detected` | boolean | True if found by QA script, False if manual |

**File:** `data/processed/qa_events.csv`

---

## Key Design Decisions

1. **Total count is the primary signal.** Tree/grass/weed are categorical only (not numeric sub-counts), so the total count drives the model.

2. **Missing days are preserved, not imputed.** The GT capstone used linear interpolation — we intentionally do NOT, because weekend gaps are structural (station doesn't count), not measurement failures.

3. **Severity thresholds are per-type, not per-total.** The Atlanta Allergy site defines L/M/H/E for trees, grass, and weeds separately. For total count, we define our own empirical "bad day" threshold from historical data.

4. **Dates are observation dates.** The count covers the previous 24 hours but is labeled by the date it represents (the date on the calendar page).
