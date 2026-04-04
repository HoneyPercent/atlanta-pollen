"""
Hypothesis Battery 2 — Deeper investigation of unexplained variance.

Now that we understand the basics, what environmental signals are we STILL missing?

Round 1: Wind direction — does pollen come from specific compass directions?
         (Atlanta is surrounded by forest to the N/NW, urban to the S)
Round 2: Prior fall/summer weather -> this year's pollen (the biology says oak/pine
         set their reproductive structures months before release)
Round 3: Temperature VOLATILITY — do big swings stress-trigger pollen release?
Round 4: Solar radiation — does sunshine independently predict pollen beyond temp?
Round 5: Interaction: temp × season_progress — does the same temp have different
         effects early vs. late in the season?
Round 6: Multi-species succession timing — can we detect the oak->pine->grass handoff
         from the contributor data and use it to predict the decline?
Round 7: "Pollen memory" — does cumulative exposure in the last 7/14 days predict
         whether the next extreme day is likely? (reservoir depletion signal)
Round 8: Overnight low temperature — do cold nights suppress next-day release
         even when the high is warm?
"""

import csv
import math
import statistics
from collections import defaultdict, Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_features():
    with open(DATA_DIR / "features_daily.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["year"] = int(r["year"])
        r["day_of_year"] = int(r["day_of_year"])
        r["total_count"] = int(r["total_count"]) if r["total_count"] else None
        r["log_count"] = float(r["log_count"]) if r["log_count"] else None
        r["cumulative_burden"] = float(r["cumulative_burden"])
        r["gdd_cumulative"] = float(r["gdd_cumulative"])
        r["season_progress_pct"] = float(r["season_progress_pct"]) if r.get("season_progress_pct") else 0
        for field in ["gdd_daily", "temp_max", "temp_min", "temp_mean", "precipitation",
                      "precip_yesterday", "precip_2day_sum", "wind_max", "wind_gust",
                      "wind_direction", "vpd_max", "solar_radiation", "sunshine_duration"]:
            r[field] = float(r[field]) if r.get(field) and r[field] else None
        r["missing"] = r.get("missing") == "True"
    return rows


def build_index(rows):
    idx = {}
    for r in rows:
        idx[(r["year"], r["day_of_year"])] = r
    return idx


def pearson_r(xs, ys):
    n = len(xs)
    if n < 5: return None
    mx, my = sum(xs)/n, sum(ys)/n
    sx = math.sqrt(sum((x-mx)**2 for x in xs)/(n-1))
    sy = math.sqrt(sum((y-my)**2 for y in ys)/(n-1))
    if sx == 0 or sy == 0: return 0
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys)) / ((n-1)*sx*sy)


def load_contributor_data():
    contrib = {}
    for yr in range(2015, 2026):
        try:
            with open(DATA_DIR / f"detail_{yr}_details.csv") as f:
                for r in csv.DictReader(f):
                    if r.get("missing") != "True":
                        contrib[r["date"]] = r
        except FileNotFoundError:
            pass
    return contrib


# ============================================================
# ROUND 1: Wind direction
# ============================================================
def round1_wind_direction(rows):
    print("=" * 70)
    print("ROUND 1: DOES WIND DIRECTION PREDICT POLLEN?")
    print("=" * 70)
    print("(Atlanta: forest to N/NW, urban center to S, piedmont/foothills to N)")

    spring = [r for r in rows if 60 <= r["day_of_year"] <= 120
              and r["total_count"] is not None and r["year"] <= 2025
              and r.get("wind_direction") is not None]

    # Bin wind direction into 8 compass sectors
    sectors = {"N": (337.5, 22.5), "NE": (22.5, 67.5), "E": (67.5, 112.5),
               "SE": (112.5, 157.5), "S": (157.5, 202.5), "SW": (202.5, 247.5),
               "W": (247.5, 292.5), "NW": (292.5, 337.5)}

    def get_sector(deg):
        for name, (lo, hi) in sectors.items():
            if name == "N":
                if deg >= 337.5 or deg < 22.5:
                    return "N"
            elif lo <= deg < hi:
                return name
        return "N"

    sector_counts = defaultdict(list)
    for r in spring:
        sector = get_sector(r["wind_direction"])
        sector_counts[sector].append(r["total_count"])

    print(f"\n{'Direction':<10} {'N':>6} {'Median':>8} {'Mean':>8} {'% Extreme':>10}")
    print("-" * 42)
    for sector in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
        counts = sorted(sector_counts[sector])
        if not counts:
            continue
        n = len(counts)
        ext_pct = 100 * sum(1 for c in counts if c >= 1500) / n
        print(f"{sector:<10} {n:>6} {counts[n//2]:>8} {sum(counts)//n:>8} {ext_pct:>9.1f}%")

    # Correlation with log count
    dirs = [r["wind_direction"] for r in spring if r["log_count"] is not None]
    logs = [r["log_count"] for r in spring if r["log_count"] is not None]

    # Sin/cos decomposition of wind direction (circular variable)
    import math as m
    dir_sin = [m.sin(m.radians(d)) for d in dirs]
    dir_cos = [m.cos(m.radians(d)) for d in dirs]
    r_sin = pearson_r(dir_sin, logs)
    r_cos = pearson_r(dir_cos, logs)
    print(f"\nWind direction correlation with log(pollen):")
    print(f"  sin(dir): r = {r_sin:.3f} (E-W component)")
    print(f"  cos(dir): r = {r_cos:.3f} (N-S component)")


# ============================================================
# ROUND 2: Prior fall/summer weather -> this year's pollen
# ============================================================
def round2_prior_season_weather(rows):
    print()
    print("=" * 70)
    print("ROUND 2: DOES LAST FALL/SUMMER WEATHER PREDICT THIS SPRING'S POLLEN?")
    print("=" * 70)
    print("(Biology: oak sets inflorescences in prior spring, pine in prior July-Aug)")

    complete_years = sorted(set(r["year"] for r in rows if r["year"] <= 2025 and r["year"] >= 1993))
    year_features = []

    for yr in complete_years:
        # This year's pollen outcomes
        yr_rows = [r for r in rows if r["year"] == yr and r["total_count"] is not None]
        if not yr_rows:
            continue
        total = sum(r["total_count"] for r in yr_rows)
        extreme = sum(1 for r in yr_rows if r["total_count"] >= 1500)

        # PRIOR YEAR weather features
        prev_yr = yr - 1

        # Prior July-Aug temp (pine strobili formation)
        jul_aug = [r for r in rows if r["year"] == prev_yr and 182 <= r["day_of_year"] <= 243
                   and r["temp_mean"] is not None]
        jul_aug_temp = statistics.mean(r["temp_mean"] for r in jul_aug) if jul_aug else None

        # Prior July-Aug precip
        jul_aug_precip = [r for r in rows if r["year"] == prev_yr and 182 <= r["day_of_year"] <= 243
                          and r["precipitation"] is not None]
        jul_aug_precip_total = sum(r["precipitation"] for r in jul_aug_precip) if jul_aug_precip else None

        # Prior Sep-Nov precip (fall moisture -> pine intensity per Atlanta study)
        fall_precip = [r for r in rows if r["year"] == prev_yr and 244 <= r["day_of_year"] <= 334
                       and r["precipitation"] is not None]
        fall_precip_total = sum(r["precipitation"] for r in fall_precip) if fall_precip else None

        # Prior Mar-Apr precip (oak inflorescence development)
        spring_precip = [r for r in rows if r["year"] == prev_yr and 60 <= r["day_of_year"] <= 120
                         and r["precipitation"] is not None]
        spring_precip_total = sum(r["precipitation"] for r in spring_precip) if spring_precip else None

        # Prior year's total pollen (carryover signal)
        prev_pollen = [r for r in rows if r["year"] == prev_yr and r["total_count"] is not None]
        prev_total = sum(r["total_count"] for r in prev_pollen) if prev_pollen else None

        # Prior annual temp
        prev_temps = [r for r in rows if r["year"] == prev_yr and r["temp_mean"] is not None]
        prev_annual_temp = statistics.mean(r["temp_mean"] for r in prev_temps) if prev_temps else None

        # Prior year GDD at June 30
        prev_jun30 = [r for r in rows if r["year"] == prev_yr and r["day_of_year"] <= 181]
        prev_gdd_jun = prev_jun30[-1]["gdd_cumulative"] if prev_jun30 else None

        year_features.append({
            "year": yr, "total": total, "extreme": extreme,
            "prev_jul_aug_temp": jul_aug_temp,
            "prev_jul_aug_precip": jul_aug_precip_total,
            "prev_fall_precip": fall_precip_total,
            "prev_spring_precip": spring_precip_total,
            "prev_total_pollen": prev_total,
            "prev_annual_temp": prev_annual_temp,
            "prev_gdd_jun": prev_gdd_jun,
        })

    # Correlate each prior-year feature with this year's pollen outcomes
    print(f"\nCorrelation of prior-year weather with THIS year's pollen (n={len(year_features)}):")
    print(f"\n{'Prior-Year Feature':<30} {'vs Total Burden':>15} {'vs Extreme Days':>15}")
    print("-" * 60)

    features_to_test = [
        ("Jul-Aug temp", "prev_jul_aug_temp"),
        ("Jul-Aug precip", "prev_jul_aug_precip"),
        ("Fall (Sep-Nov) precip", "prev_fall_precip"),
        ("Spring (Mar-Apr) precip", "prev_spring_precip"),
        ("Prior year total pollen", "prev_total_pollen"),
        ("Prior annual mean temp", "prev_annual_temp"),
        ("Prior GDD at June 30", "prev_gdd_jun"),
    ]

    for label, key in features_to_test:
        valid = [yf for yf in year_features if yf[key] is not None]
        if len(valid) < 8:
            continue
        xs = [yf[key] for yf in valid]
        r_total = pearson_r(xs, [yf["total"] for yf in valid])
        r_extreme = pearson_r(xs, [float(yf["extreme"]) for yf in valid])
        flag_t = " **" if r_total and abs(r_total) > 0.3 else ""
        flag_e = " **" if r_extreme and abs(r_extreme) > 0.3 else ""
        print(f"{label:<30} {r_total:>15.3f}{flag_t} {r_extreme:>15.3f}{flag_e}")


# ============================================================
# ROUND 3: Temperature volatility
# ============================================================
def round3_temp_volatility(rows):
    print()
    print("=" * 70)
    print("ROUND 3: DOES TEMPERATURE VOLATILITY TRIGGER POLLEN?")
    print("=" * 70)
    print("(Hypothesis: big day-to-day swings stress trees into releasing)")

    idx = build_index(rows)
    spring = [r for r in rows if 60 <= r["day_of_year"] <= 120
              and r["total_count"] is not None and r["year"] <= 2025
              and r["temp_mean"] is not None]

    # Compute 5-day temperature standard deviation
    volatilities = []
    for r in spring:
        temps = []
        for d in range(5):
            prev = idx.get((r["year"], r["day_of_year"] - d))
            if prev and prev["temp_mean"] is not None:
                temps.append(prev["temp_mean"])
        if len(temps) >= 3:
            vol = statistics.stdev(temps)
            volatilities.append((vol, r["total_count"], r["log_count"]))

    if volatilities:
        vols = [v[0] for v in volatilities]
        logs = [v[2] for v in volatilities]
        r_val = pearson_r(vols, logs)
        print(f"\nCorrelation of 5-day temp stdev with log(pollen): r = {r_val:.3f}")

        # Bin
        print(f"\n{'Volatility':<15} {'N':>6} {'Median Pollen':>14} {'% Extreme':>10}")
        print("-" * 45)
        for lo, hi, label in [(0, 3, "Very stable"), (3, 6, "Mild swing"),
                               (6, 10, "Moderate swing"), (10, 20, "Wild swing")]:
            group = [v for v in volatilities if lo <= v[0] < hi]
            if group:
                counts = sorted(v[1] for v in group)
                n = len(counts)
                ext = 100 * sum(1 for c in counts if c >= 1500) / n
                print(f"{label:<15} {n:>6} {counts[n//2]:>14,} {ext:>9.1f}%")


# ============================================================
# ROUND 4: Solar radiation
# ============================================================
def round4_solar_radiation(rows):
    print()
    print("=" * 70)
    print("ROUND 4: DOES SOLAR RADIATION PREDICT POLLEN BEYOND TEMPERATURE?")
    print("=" * 70)

    spring = [r for r in rows if 60 <= r["day_of_year"] <= 120
              and r["total_count"] is not None and r["year"] <= 2025
              and r.get("solar_radiation") is not None and r["temp_mean"] is not None]

    if spring:
        solar = [r["solar_radiation"] for r in spring]
        logs = [r["log_count"] for r in spring if r["log_count"] is not None]
        temps = [r["temp_mean"] for r in spring]
        solar_f = solar[:len(logs)]

        r_solar = pearson_r(solar_f, logs)
        r_temp = pearson_r(temps[:len(logs)], logs)
        print(f"\nCorrelation with log(pollen):")
        print(f"  Solar radiation: r = {r_solar:.3f}")
        print(f"  Temperature:     r = {r_temp:.3f}")

        # Partial: does solar add value after controlling for temp?
        # Compute residuals of temp->pollen, then correlate with solar
        n = len(logs)
        mt = statistics.mean(temps[:n])
        ml = statistics.mean(logs)
        st = statistics.stdev(temps[:n])
        sl = statistics.stdev(logs)
        r_tl = pearson_r(temps[:n], logs)

        # Residual = actual - predicted from temp
        residuals = [logs[i] - (ml + r_tl * sl / st * (temps[i] - mt)) for i in range(n)]
        r_partial = pearson_r(solar_f[:n], residuals)
        print(f"  Solar (after removing temp effect): r = {r_partial:.3f}")

        # Bin by sunshine duration
        sunshine_days = [r for r in spring if r.get("sunshine_duration") is not None]
        if sunshine_days:
            print(f"\nPollen by sunshine duration:")
            print(f"{'Sunshine Hours':<18} {'N':>6} {'Median Pollen':>14} {'% Extreme':>10}")
            print("-" * 48)
            for lo, hi, label in [(0, 4, "<4 hrs (cloudy)"), (4, 8, "4-8 hrs (mixed)"),
                                   (8, 12, "8-12 hrs (sunny)"), (12, 24, ">12 hrs (full sun)")]:
                group = [r for r in sunshine_days if lo <= r["sunshine_duration"] / 3600 < hi]
                if group:
                    counts = sorted(r["total_count"] for r in group)
                    n = len(counts)
                    ext = 100 * sum(1 for c in counts if c >= 1500) / n
                    print(f"{label:<18} {n:>6} {counts[n//2]:>14,} {ext:>9.1f}%")


# ============================================================
# ROUND 5: Temperature × season progress interaction
# ============================================================
def round5_temp_season_interaction(rows):
    print()
    print("=" * 70)
    print("ROUND 5: DOES TEMPERATURE HAVE DIFFERENT EFFECTS EARLY vs LATE SEASON?")
    print("=" * 70)

    spring = [r for r in rows if 30 <= r["day_of_year"] <= 150
              and r["total_count"] is not None and r["year"] <= 2025
              and r["temp_mean"] is not None]

    # Split into early (pre-peak, progress < 30%) and late (post-peak, progress > 70%)
    early = [r for r in spring if r["season_progress_pct"] < 30]
    late = [r for r in spring if r["season_progress_pct"] > 70]

    for label, group in [("Early season (<30% progress)", early), ("Late season (>70% progress)", late)]:
        if not group:
            continue
        warm = [r for r in group if r["temp_mean"] >= 60]
        cool = [r for r in group if r["temp_mean"] < 50]

        print(f"\n{label}:")
        if warm:
            wc = sorted(r["total_count"] for r in warm)
            print(f"  Warm (>=60F): N={len(warm)}, median={wc[len(wc)//2]:,}, "
                  f"extreme={100*sum(1 for c in wc if c>=1500)/len(wc):.0f}%")
        if cool:
            cc = sorted(r["total_count"] for r in cool)
            print(f"  Cool (<50F):  N={len(cool)}, median={cc[len(cc)//2]:,}, "
                  f"extreme={100*sum(1 for c in cc if c>=1500)/len(cc):.0f}%")

        # Correlation of temp with pollen in each phase
        temps = [r["temp_mean"] for r in group]
        logs = [r["log_count"] for r in group if r["log_count"] is not None]
        r_val = pearson_r(temps[:len(logs)], logs)
        print(f"  Temp-pollen correlation: r = {r_val:.3f}")


# ============================================================
# ROUND 6: Species succession from contributor data
# ============================================================
def round6_species_succession(rows, contrib_data):
    print()
    print("=" * 70)
    print("ROUND 6: SPECIES SUCCESSION — WHEN DO CONTRIBUTORS SHIFT?")
    print("=" * 70)
    print("(Can we detect oak->pine->grass handoff and use it to predict decline?)")

    # For each year with contributor data, track when key species appear/disappear
    for yr in range(2020, 2026):
        yr_data = [(d, contrib_data[d]) for d in sorted(contrib_data.keys())
                   if d.startswith(str(yr)) and contrib_data[d].get("tree_contributors")]
        if not yr_data:
            continue

        first_oak = first_pine = first_grass = last_oak = last_pine = None
        for date_str, detail in yr_data:
            trees = [t.strip().upper() for t in detail.get("tree_contributors", "").split(",") if t.strip()]
            grass_sev = detail.get("grass_severity", "").lower()

            if "OAK" in trees:
                if first_oak is None: first_oak = date_str
                last_oak = date_str
            if "PINE" in trees:
                if first_pine is None: first_pine = date_str
                last_pine = date_str
            if grass_sev not in ("", "low"):
                if first_grass is None: first_grass = date_str

        print(f"\n{yr}:")
        if first_pine: print(f"  Pine:  {first_pine} to {last_pine}")
        if first_oak: print(f"  Oak:   {first_oak} to {last_oak}")
        if first_grass: print(f"  Grass: first moderate+ on {first_grass}")

    # Key question: does grass appearance predict remaining extreme days?
    print(f"\nDoes grass emergence predict remaining extreme days?")
    for yr in range(2015, 2026):
        yr_data = {d: contrib_data[d] for d in contrib_data if d.startswith(str(yr))}
        if not yr_data:
            continue
        first_grass = None
        for d in sorted(yr_data.keys()):
            if yr_data[d].get("grass_severity", "").lower() not in ("", "low"):
                first_grass = d
                break
        if first_grass:
            from datetime import date
            parts = first_grass.split("-")
            grass_doy = date(int(parts[0]), int(parts[1]), int(parts[2])).timetuple().tm_yday
            remaining_ext = sum(1 for r in rows if r["year"] == yr and r["day_of_year"] > grass_doy
                                and r["total_count"] is not None and r["total_count"] >= 1500)
            print(f"  {yr}: grass emerged DOY {grass_doy} ({first_grass}), {remaining_ext} extreme days after")


# ============================================================
# ROUND 7: Pollen "reservoir depletion"
# ============================================================
def round7_reservoir_depletion(rows):
    print()
    print("=" * 70)
    print("ROUND 7: DOES HIGH RECENT BURDEN PREDICT IMMINENT DECLINE?")
    print("=" * 70)
    print("(If cumulative burden in last 14 days is very high, is the next")
    print(" extreme day less likely? i.e., local depletion of pollen reservoir)")

    idx = build_index(rows)
    spring = [r for r in rows if 60 <= r["day_of_year"] <= 120
              and r["total_count"] is not None and r["year"] <= 2025]

    events = []
    for r in spring:
        # Compute 14-day rolling burden
        burden_14d = 0
        for d in range(1, 15):
            prev = idx.get((r["year"], r["day_of_year"] - d))
            if prev and prev["total_count"] is not None:
                burden_14d += prev["total_count"]

        events.append({
            "burden_14d": burden_14d,
            "count": r["total_count"],
            "is_extreme": r["total_count"] >= 1500,
        })

    # Bin by 14-day rolling burden
    print(f"\n{'14-day Burden':<20} {'N':>6} {'% Next Extreme':>15} {'Median Next':>12}")
    print("-" * 53)
    for lo, hi, label in [(0, 5000, "<5K"), (5000, 15000, "5-15K"), (15000, 30000, "15-30K"),
                           (30000, 50000, "30-50K"), (50000, 200000, ">50K")]:
        group = [e for e in events if lo <= e["burden_14d"] < hi]
        if group:
            ext_rate = 100 * sum(1 for e in group if e["is_extreme"]) / len(group)
            counts = sorted(e["count"] for e in group)
            print(f"{label:<20} {len(group):>6} {ext_rate:>14.1f}% {counts[len(counts)//2]:>12,}")

    # Is there depletion? After a very heavy 14 days, is the NEXT day less extreme?
    print(f"\nAfter a heavy 14-day period (>30K burden), what happens next?")
    heavy = [e for e in events if e["burden_14d"] >= 30000]
    not_heavy = [e for e in events if e["burden_14d"] < 15000]
    if heavy:
        print(f"  Heavy recent burden (>30K): {100*sum(1 for e in heavy if e['is_extreme'])/len(heavy):.0f}% extreme next")
    if not_heavy:
        print(f"  Light recent burden (<15K): {100*sum(1 for e in not_heavy if e['is_extreme'])/len(not_heavy):.0f}% extreme next")


# ============================================================
# ROUND 8: Overnight low temperature
# ============================================================
def round8_overnight_low(rows):
    print()
    print("=" * 70)
    print("ROUND 8: DO COLD NIGHTS SUPPRESS NEXT-DAY POLLEN?")
    print("=" * 70)
    print("(Even when daytime is warm, a cold overnight might prevent release)")

    spring = [r for r in rows if 60 <= r["day_of_year"] <= 120
              and r["total_count"] is not None and r["year"] <= 2025
              and r["temp_min"] is not None and r["temp_max"] is not None
              and r["precipitation"] is not None and r["precipitation"] < 0.05]  # dry days only

    if not spring:
        print("No data")
        return

    # Look at temp_min (overnight low) independent of temp_max
    # Control for daytime temp by looking at warm days only (max >= 65)
    warm_days = [r for r in spring if r["temp_max"] >= 65]

    if warm_days:
        print(f"\nAmong warm days (max >= 65F), effect of overnight low:")
        print(f"{'Overnight Low':<18} {'N':>6} {'Median Pollen':>14} {'% Extreme':>10}")
        print("-" * 48)
        for lo, hi, label in [(20, 35, "Frost (<35F)"), (35, 45, "Cold (35-45F)"),
                               (45, 55, "Cool (45-55F)"), (55, 70, "Warm (55-70F)")]:
            group = [r for r in warm_days if lo <= r["temp_min"] < hi]
            if group:
                counts = sorted(r["total_count"] for r in group)
                n = len(counts)
                ext = 100 * sum(1 for c in counts if c >= 1500) / n
                print(f"{label:<18} {n:>6} {counts[n//2]:>14,} {ext:>9.1f}%")

        # Correlation of temp_min with pollen on warm days
        mins = [r["temp_min"] for r in warm_days]
        logs = [r["log_count"] for r in warm_days if r["log_count"] is not None]
        r_val = pearson_r(mins[:len(logs)], logs)
        print(f"\n  Correlation of overnight low with log(pollen) on warm days: r = {r_val:.3f}")

        # Also: temp spread (max - min) on warm days
        spreads = [r["temp_max"] - r["temp_min"] for r in warm_days]
        r_spread = pearson_r(spreads[:len(logs)], logs)
        print(f"  Correlation of diurnal spread with log(pollen) on warm days: r = {r_spread:.3f}")


if __name__ == "__main__":
    rows = load_features()
    contrib = load_contributor_data()
    round1_wind_direction(rows)
    round2_prior_season_weather(rows)
    round3_temp_volatility(rows)
    round4_solar_radiation(rows)
    round5_temp_season_interaction(rows)
    round6_species_succession(rows, contrib)
    round7_reservoir_depletion(rows)
    round8_overnight_low(rows)
