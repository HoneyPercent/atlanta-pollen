# Atlanta Pollen Tracker

Personal pollen forecast for Atlanta. Live site reads `dashboard.json` + `scurve_data.json` (repo root) and renders via `index.html`. GitHub Pages serves it.

## Cron architecture (two separate workflows)

| Workflow | File | Schedule | Output |
|---|---|---|---|
| Daily | `.github/workflows/daily_update.yml` | 3×/weekday at 10am, 1pm, 4pm ET | `dashboard.json` |
| Weekly rebuild | `.github/workflows/weekly_rebuild.yml` | Fridays 9am ET | `data/model_output/*.json` (climatology, season_progress, analog_projection, scurve_data) |

The daily pipeline is **self-bootstrapping** — reads the previous `dashboard.json` for state, scores the V8 model live, writes the new `dashboard.json`. It does not touch the baseline JSONs.

The weekly rebuild **regenerates the baseline JSONs** from scratch by re-running the full data + features chain. Without this, the season-context layer (% complete, remaining bad days, analog projections) drifts each week.

## Data flow

```
historical_seed.csv (1992-2008, frozen, COMMITTED)
                +
scrape 2009-current (gitignored scratch)
                ↓
full_history_calendar.csv (gitignored, regenerated weekly)
                +
weather (Open-Meteo 1992-current, gitignored)
                ↓
features_daily.csv (gitignored)
                ↓
data/model_output/*.json (COMMITTED — refreshed Fridays)
                ↓
dashboard.json (COMMITTED — refreshed 3×/weekday by daily_pipeline.py)
```

## What's tracked vs gitignored

**Tracked:**
- `dashboard.json`, `scurve_data.json` (repo root) — what the website reads
- `data/model_output/*.json` — baseline outputs
- `data/processed/historical_seed.csv` — irreplaceable seed (see "Pre-2009 data" below)
- `index.html`, `requirements.txt`, scripts, workflows

**Gitignored** (regenerated each weekly run):
- All other files in `data/processed/` — full_history_calendar, weather_complete_v2, features_daily, scraper scratch
- All of `data/raw/`

## Pre-2009 data is irreplaceable

Atlanta Allergy & Asthma's website (`atlantaallergy.com/pollen_counts/`) only exposes data from **2009 onward**. The 1992-2008 history (3,082 rows including 2,183 days with real counts) was sourced from elsewhere — the original source is undocumented and the data isn't re-fetchable from any public API.

`data/processed/historical_seed.csv` is **the only authoritative copy** of pre-2009 data. The committed copy on GitHub is the backup. If the file is lost from both places, the decade-comparison fields in `season_progress.json` (`1990s_avg_total`, `2000s_avg_total`) cannot be regenerated.

`baseline_models.py` requires this data — it crashes with `mean requires at least one data point` if rows for years 1992-1999 are missing.

## Working on this project

- **Always `git pull` before reading local files.** The bot pushes 3-4 times every weekday; local checkouts go stale fast. The `/pollen` skill enforces this in its Step 0.
- **Don't run `scrape_atlanta_allergy.py --output full_history`** — that overwrites `full_history_calendar.csv` with 2009-onward only, losing the seed. Use `--output recent_only` (the workflow's pattern) and stitch with the seed.
- **Trigger workflows manually** via `gh workflow run daily_update.yml` or `gh workflow run weekly_rebuild.yml` — both support `workflow_dispatch`.
