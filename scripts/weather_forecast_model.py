"""
Weather-based pollen forecast model for Atlanta.

Builds a simple multiple linear regression to predict log(pollen + 1)
from weather features + previous day's count. Compares against:
  - Baseline A: Day-of-year climatology (historical median for this DOY)
  - Baseline B: Analog-year model (from baseline_models.py)
  - Baseline C: This weather regression

Uses rolling-origin evaluation: train on all years before the test year,
predict each day of the test year using only information available at forecast time.
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
        r["log_count"] = float(r["log_count"]) if r["log_count"] else None
        for field in ["cumulative_burden", "gdd_cumulative", "gdd_daily",
                      "temp_max", "temp_min", "temp_mean", "precipitation",
                      "precip_yesterday", "precip_2day_sum", "wind_max", "wind_gust"]:
            r[field] = float(r[field]) if r.get(field) else None
        r["missing"] = r.get("missing") == "True"
    return rows


def build_index(rows):
    """Index rows by (year, day_of_year) for quick lookups."""
    idx = {}
    for r in rows:
        idx[(r["year"], r["day_of_year"])] = r
    return idx


def extract_features(row, idx):
    """Extract the feature vector for one day. Returns (features_dict, target) or None."""
    if row["total_count"] is None or row["temp_mean"] is None:
        return None

    prev = idx.get((row["year"], row["day_of_year"] - 1))
    prev_log = prev["log_count"] if prev and prev["log_count"] is not None else None

    if prev_log is None:
        return None  # Need previous day's count

    features = {
        "prev_log_count": prev_log,
        "temp_mean": row["temp_mean"],
        "temp_max": row["temp_max"] or row["temp_mean"],
        "precip_yesterday": row["precip_yesterday"] or 0,
        "precip_2day_sum": row["precip_2day_sum"] or 0,
        "wind_max": row["wind_max"] or 0,
        "gdd_daily": row["gdd_daily"] or 0,
        "day_of_year": float(row["day_of_year"]),
        # Seasonal position as sine/cosine (captures cyclical nature)
        "doy_sin": math.sin(2 * math.pi * row["day_of_year"] / 365),
        "doy_cos": math.cos(2 * math.pi * row["day_of_year"] / 365),
    }

    target = row["log_count"]
    return features, target


class SimpleLinearRegression:
    """Multiple linear regression using normal equations (no numpy needed)."""

    def __init__(self):
        self.weights = None
        self.bias = None
        self.feature_names = None
        self.means = None
        self.stds = None

    def fit(self, X, y, feature_names):
        """Fit using gradient descent (simpler than normal equations without numpy)."""
        self.feature_names = feature_names
        n = len(y)
        k = len(feature_names)

        # Standardize features for stable gradient descent
        self.means = [statistics.mean(X[i][j] for i in range(n)) for j in range(k)]
        self.stds = [statistics.stdev(X[i][j] for i in range(n)) if statistics.stdev(X[i][j] for i in range(n)) > 0 else 1 for j in range(k)]

        X_std = [[((X[i][j] - self.means[j]) / self.stds[j]) for j in range(k)] for i in range(n)]

        # Gradient descent
        weights = [0.0] * k
        bias = statistics.mean(y)
        lr = 0.01
        epochs = 500

        for _ in range(epochs):
            # Predictions
            preds = [bias + sum(weights[j] * X_std[i][j] for j in range(k)) for i in range(n)]

            # Gradients
            errors = [preds[i] - y[i] for i in range(n)]
            grad_bias = sum(errors) / n
            grad_weights = [sum(errors[i] * X_std[i][j] for i in range(n)) / n for j in range(k)]

            # Update
            bias -= lr * grad_bias
            for j in range(k):
                weights[j] -= lr * grad_weights[j]

        self.weights = weights
        self.bias = bias

    def predict(self, x_raw):
        """Predict for a single sample (raw, unstandardized features)."""
        x_std = [(x_raw[j] - self.means[j]) / self.stds[j] for j in range(len(self.feature_names))]
        return self.bias + sum(self.weights[j] * x_std[j] for j in range(len(self.feature_names)))

    def feature_importance(self):
        """Return feature importances (absolute standardized weights)."""
        imp = [(self.feature_names[j], self.weights[j]) for j in range(len(self.feature_names))]
        imp.sort(key=lambda x: abs(x[1]), reverse=True)
        return imp


def climatology_baseline(rows, test_year):
    """Day-of-year median from all prior years."""
    train = [r for r in rows if r["year"] < test_year and r["total_count"] is not None]
    by_doy = defaultdict(list)
    for r in train:
        by_doy[r["day_of_year"]].append(r["log_count"])

    medians = {}
    for doy, vals in by_doy.items():
        medians[doy] = statistics.median(vals)
    return medians


def evaluate_year(rows, idx, test_year):
    """Train on all years before test_year, predict test_year, return metrics."""
    train_rows = [r for r in rows if r["year"] < test_year
                  and 1 <= r["day_of_year"] <= 180]
    test_rows = [r for r in rows if r["year"] == test_year
                 and 30 <= r["day_of_year"] <= 150]

    # Build training data
    train_X = []
    train_y = []
    feature_names = None
    for r in train_rows:
        result = extract_features(r, idx)
        if result:
            feats, target = result
            if feature_names is None:
                feature_names = list(feats.keys())
            train_X.append([feats[f] for f in feature_names])
            train_y.append(target)

    if len(train_X) < 50 or not feature_names:
        return None

    # Train regression
    model = SimpleLinearRegression()
    model.fit(train_X, train_y, feature_names)

    # Climatology baseline
    clim = climatology_baseline(rows, test_year)

    # Evaluate on test year
    results = {
        "regression": {"errors": [], "abs_errors": [], "sq_errors": [], "actuals": [], "preds": []},
        "climatology": {"errors": [], "abs_errors": [], "sq_errors": [], "actuals": [], "preds": []},
        "persistence": {"errors": [], "abs_errors": [], "sq_errors": [], "actuals": [], "preds": []},
    }

    for r in test_rows:
        result = extract_features(r, idx)
        if not result:
            continue
        feats, actual = result

        # Regression prediction
        pred_reg = model.predict([feats[f] for f in feature_names])
        err = pred_reg - actual
        results["regression"]["errors"].append(err)
        results["regression"]["abs_errors"].append(abs(err))
        results["regression"]["sq_errors"].append(err ** 2)
        results["regression"]["actuals"].append(actual)
        results["regression"]["preds"].append(pred_reg)

        # Climatology prediction
        pred_clim = clim.get(r["day_of_year"], 0)
        err_c = pred_clim - actual
        results["climatology"]["errors"].append(err_c)
        results["climatology"]["abs_errors"].append(abs(err_c))
        results["climatology"]["sq_errors"].append(err_c ** 2)
        results["climatology"]["actuals"].append(actual)
        results["climatology"]["preds"].append(pred_clim)

        # Persistence (yesterday's count = today's prediction)
        pred_pers = feats["prev_log_count"]
        err_p = pred_pers - actual
        results["persistence"]["errors"].append(err_p)
        results["persistence"]["abs_errors"].append(abs(err_p))
        results["persistence"]["sq_errors"].append(err_p ** 2)
        results["persistence"]["actuals"].append(actual)
        results["persistence"]["preds"].append(pred_pers)

    return results, model


def classify_severity(count):
    """Classify raw pollen count into severity band."""
    if count >= 1500:
        return "extreme"
    elif count >= 90:
        return "high"
    elif count >= 15:
        return "moderate"
    else:
        return "low"


def bad_day_accuracy(actuals_log, preds_log, threshold_log=None):
    """How well does the model predict 'bad pollen days' (>= 90 count, i.e. high+extreme)?"""
    # log(90 + 1) ≈ 4.51
    if threshold_log is None:
        threshold_log = math.log(91)

    tp = fp = tn = fn = 0
    for a, p in zip(actuals_log, preds_log):
        actual_bad = a >= threshold_log
        pred_bad = p >= threshold_log
        if actual_bad and pred_bad:
            tp += 1
        elif not actual_bad and pred_bad:
            fp += 1
        elif actual_bad and not pred_bad:
            fn += 1
        else:
            tn += 1

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def main():
    rows = load_features()
    idx = build_index(rows)

    # Test years: evaluate on each year from 2015-2025
    test_years = [y for y in range(2015, 2026)]

    all_metrics = {"regression": [], "climatology": [], "persistence": []}
    all_bad_day = {"regression": [], "climatology": [], "persistence": []}

    print("=" * 70)
    print("WEATHER-BASED FORECAST MODEL — ROLLING-ORIGIN EVALUATION")
    print("=" * 70)
    print(f"\nTraining: all years before test year | Testing: spring (DOY 30-150)")
    print(f"Target: log(pollen_count + 1)")
    print()

    last_model = None
    for yr in test_years:
        result = evaluate_year(rows, idx, yr)
        if result is None:
            print(f"  {yr}: insufficient data, skipping")
            continue

        results, model = result
        last_model = model
        n = len(results["regression"]["errors"])

        print(f"  {yr} (n={n:>3})  ", end="")
        for name in ["regression", "climatology", "persistence"]:
            mae = statistics.mean(results[name]["abs_errors"])
            rmse = math.sqrt(statistics.mean(results[name]["sq_errors"]))
            all_metrics[name].append({"year": yr, "mae": mae, "rmse": rmse, "n": n})
            print(f"  {name[:5]}: MAE={mae:.3f} RMSE={rmse:.3f}", end="")

            bd = bad_day_accuracy(results[name]["actuals"], results[name]["preds"])
            all_bad_day[name].append({"year": yr, **bd})

        print()

    # Aggregate metrics
    print()
    print("=" * 70)
    print("AGGREGATE METRICS (2015-2025)")
    print("=" * 70)
    print(f"\n{'Model':<15} {'MAE':>8} {'RMSE':>8} {'Bad-Day Acc':>12} {'Precision':>10} {'Recall':>10}")
    print("-" * 65)

    for name in ["regression", "climatology", "persistence"]:
        avg_mae = statistics.mean(m["mae"] for m in all_metrics[name])
        avg_rmse = statistics.mean(m["rmse"] for m in all_metrics[name])
        total_tp = sum(b["tp"] for b in all_bad_day[name])
        total_fp = sum(b["fp"] for b in all_bad_day[name])
        total_tn = sum(b["tn"] for b in all_bad_day[name])
        total_fn = sum(b["fn"] for b in all_bad_day[name])
        acc = (total_tp + total_tn) / (total_tp + total_fp + total_tn + total_fn)
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        print(f"{name:<15} {avg_mae:>8.3f} {avg_rmse:>8.3f} {acc:>11.1%} {prec:>10.1%} {rec:>10.1%}")

    # Feature importance from final model
    if last_model:
        print()
        print("=" * 70)
        print("FEATURE IMPORTANCE (final model, trained on all data through 2025)")
        print("=" * 70)
        for fname, weight in last_model.feature_importance():
            bar = "#" * int(abs(weight) * 10)
            sign = "+" if weight > 0 else "-"
            print(f"  {sign} {abs(weight):.3f}  {fname:<25} {bar}")

    # Multi-day horizon analysis
    print()
    print("=" * 70)
    print("MULTI-DAY HORIZON: ERROR GROWTH")
    print("=" * 70)
    print("\nSimulating D+1 through D+7 by chaining predictions (using own output as next input):")

    # For each test year, chain predictions and measure how error grows
    horizon_errors = defaultdict(list)
    for yr in test_years:
        yr_rows = [r for r in rows if r["year"] == yr and 60 <= r["day_of_year"] <= 120
                   and r["total_count"] is not None]
        if len(yr_rows) < 30:
            continue

        # Train on prior years
        train_rows = [r for r in rows if r["year"] < yr and 1 <= r["day_of_year"] <= 180]
        train_X = []
        train_y = []
        feature_names = None
        for r in train_rows:
            result = extract_features(r, idx)
            if result:
                feats, target = result
                if feature_names is None:
                    feature_names = list(feats.keys())
                train_X.append([feats[f] for f in feature_names])
                train_y.append(target)

        if len(train_X) < 50:
            continue
        model = SimpleLinearRegression()
        model.fit(train_X, train_y, feature_names)

        # Chain predictions from each start day
        for start_idx in range(0, len(yr_rows) - 7, 3):  # every 3rd day
            r0 = yr_rows[start_idx]
            result = extract_features(r0, idx)
            if not result:
                continue

            prev_pred = result[0]["prev_log_count"]  # actual D-1
            for horizon in range(1, 8):
                day_row = yr_rows[start_idx + horizon] if start_idx + horizon < len(yr_rows) else None
                if not day_row or day_row["total_count"] is None or day_row["temp_mean"] is None:
                    break

                # Use actual weather but predicted previous pollen
                feats_raw = extract_features(day_row, idx)
                if not feats_raw:
                    break
                feats, actual = feats_raw
                feats["prev_log_count"] = prev_pred  # substitute chained prediction

                pred = model.predict([feats[f] for f in feature_names])
                err = abs(pred - actual)
                horizon_errors[horizon].append(err)
                prev_pred = pred  # chain forward

    print(f"\n{'Horizon':<12} {'N':>6} {'MAE':>8} {'Median AE':>10}")
    print("-" * 36)
    for h in range(1, 8):
        if horizon_errors[h]:
            errs = horizon_errors[h]
            mae = statistics.mean(errs)
            med = statistics.median(errs)
            print(f"D+{h:<10} {len(errs):>6} {mae:>8.3f} {med:>10.3f}")

    # Save model weights for production use
    if last_model:
        model_data = {
            "feature_names": last_model.feature_names,
            "weights": last_model.weights,
            "bias": last_model.bias,
            "means": last_model.means,
            "stds": last_model.stds,
            "description": "Multiple linear regression on log(pollen+1). Standardized features.",
            "training_years": "1992-2025",
            "evaluation_years": "2015-2025",
        }
        with open(OUTPUT_DIR / "weather_model_weights.json", "w") as f:
            json.dump(model_data, f, indent=2)
        print(f"\nModel weights saved to {OUTPUT_DIR / 'weather_model_weights.json'}")


if __name__ == "__main__":
    main()
