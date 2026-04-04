# Source Audit — Atlanta Pollen Tracker

## Evaluation Criteria

Each source is evaluated on: observed vs. modeled, unit, geography, history depth, update timing, storage limits, cost, failure mode.

---

## 1. Atlanta Allergy & Asthma (atlantaallergy.com)

| Criterion | Value |
|-----------|-------|
| **Type** | Observed (Burkard spore trap, manual microscopy) |
| **Unit** | Grains per cubic meter (total pollen count) |
| **Geography** | Single station in Atlanta (rooftop, NAB-certified) |
| **History** | June 1991 – present (35 years, confirmed by scraper test) |
| **Update timing** | Previous 24-hour count, typically posted next business day by mid-morning |
| **Granularity** | Calendar page: total count + severity class. Detail page: total count + tree/grass/weed severity (categorical L/M/H/E) + top contributors (named species) + mold level |
| **NOT available** | Numeric tree/grass/weed sub-counts — only categorical severity and needle position (%) |
| **Missing days** | Weekends, holidays, equipment issues. ~5 days/week historically, daily during peak spring since ~2011 |
| **Storage limits** | None (public website) |
| **Cost** | Free (web scraping) |
| **Failure mode** | Site structure change breaks scraper; weekend/holiday gaps; count not posted yet today |
| **Role** | **PRIMARY observed pollen source** |

### Scraper test results
- **Earliest data:** June 1991 (1991/03 has 0 days; 1991/06 has 13 days)
- **Consistent data from:** 1992 onward (Jan 1992 has 22 days)
- **Recent month (March 2026):** 20 days with data, 11 missing (weekends + future dates)
- **Detail page verified:** Extracts total count, tree/grass/weed severity, contributors, mold level
- **HTML structure stable** across all tested years (1991–2026)

---

## 2. Open-Meteo (open-meteo.com)

| Criterion | Value |
|-----------|-------|
| **Type** | Reanalysis weather data (ERA5 + high-resolution models) |
| **Unit** | Standard weather units (°F, inches, mph as configured) |
| **Geography** | Global, ~10km grid resolution. Atlanta coordinates: 33.749, -84.388 |
| **History** | 1940 – present (85+ years) |
| **Update timing** | Historical data available with ~5 day lag. Forecast: 16-day ahead |
| **Variables** | temp max/min/mean, precipitation, rain, wind speed/gusts/direction, humidity, evapotranspiration |
| **Forecast archive** | Previous model runs available for honest backtesting |
| **Storage limits** | None for processed results |
| **Cost** | Free (no API key needed) |
| **Failure mode** | API downtime; data lag for recent days |
| **Role** | **PRIMARY weather source** (both historical actuals and forecasts) |

### Fetch test results
- Successfully fetched March 2026 weather for Atlanta (25 days)
- All 9 daily variables returned with complete data
- Fast response, no rate limiting observed

---

## 3. Google Pollen API

| Criterion | Value |
|-----------|-------|
| **Type** | Modeled forecast (proprietary model) |
| **Unit** | Universal Pollen Index (UPI), 0–5 scale — NOT grains/m³ |
| **Geography** | Global, 1x1 km resolution |
| **History** | None — forecast only (up to 5 days ahead) |
| **Update timing** | Daily forecasts |
| **Storage limits** | **Prohibits prefetching, caching, or storage** of API results |
| **Cost** | Pay-per-call (free tier: 10,000 calls/month) |
| **Failure mode** | API quota exceeded; billing required |
| **Role** | **Secondary benchmark only** — cannot compare UPI directly to Atlanta Allergy grains/m³ |

### Key issue
UPI (0–5 index) and Atlanta Allergy counts (0–10,000+ grains/m³) are fundamentally different units. Source comparison requires mapping both to ordinal severity levels, not raw number correlation. Storage restrictions also conflict with our "JSON in repo" approach.

---

## 4. Ambee Pollen API

| Criterion | Value |
|-----------|-------|
| **Type** | Modeled (proprietary model + satellite data) |
| **Unit** | Modeled pollen counts + risk levels |
| **Geography** | Global, hyperlocal |
| **History** | Limited historical access on free tier |
| **Update timing** | Real-time + short-term forecast |
| **Storage limits** | Terms vary by plan |
| **Cost** | Free tier: 30-day trial / limited calls. Paid for bulk historical |
| **Failure mode** | API quota; model accuracy may diverge from local observed counts |
| **Role** | **Optional comparison source** — useful for forecast supplement when Atlanta Allergy hasn't posted yet |

---

## 5. AAAAI / National Allergy Bureau (NAB)

| Criterion | Value |
|-----------|-------|
| **Type** | Observed (aggregates data from certified stations including Atlanta Allergy) |
| **Unit** | Grains per cubic meter (same as Atlanta Allergy) |
| **Geography** | Station-level (Atlanta Allergy is the certified Atlanta station) |
| **History** | Excel-format data available from 2003 onward via formal request. Published research used 1992–2018 |
| **Update timing** | Formal request process, up to 12 weeks |
| **Storage limits** | Per agreement |
| **Cost** | Free for research/non-commercial |
| **Failure mode** | Slow approval; data may not add much beyond what we scrape directly |
| **Role** | **Future deep-history archive** — could provide richer type-level data if formal request is approved |

---

## Recommended V1 Source Stack

| Role | Source | Rationale |
|------|--------|-----------|
| **Primary observed pollen** | Atlanta Allergy (scraped) | Only NAB-certified Atlanta station. 35 years of data. Scraper proven working. |
| **Weather actuals** | Open-Meteo Archive API | Free, 85+ years, no key needed, all needed variables. |
| **Weather forecast** | Open-Meteo Forecast API | Same source for consistency. 16-day forecast. Forecast archive for backtesting. |
| **Forward pollen benchmark** | Google Pollen API (optional) | UPI-based severity comparison only. Do NOT store or cache results. |
| **Forecast fill-in** | Ambee (optional) | When Atlanta Allergy hasn't posted today's count, Ambee can estimate current conditions. |

### What we confirmed today
1. Atlanta Allergy data goes back to June 1991 — much deeper than the 2009 estimate
2. Calendar pages provide total count + severity class for every day
3. Detail pages provide total count + tree/grass/weed categorical severity + top contributors + mold
4. Missing days are clearly identifiable (empty links, "no pollen data" messages)
5. HTML structure has been stable for 35 years of data
6. Open-Meteo provides complete, fast weather data with no authentication
7. Tree/grass/weed numeric sub-counts are NOT available — only categorical L/M/H/E levels
