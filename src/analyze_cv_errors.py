
import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREDICTION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "cv",
    "pinn_group_oof_predictions.csv"
)

AGING_FILE = os.path.join(
    BASE_DIR,
    "results",
    "aging_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "cv",
    "error_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("           PINN CROSS-VALIDATION ERROR ANALYSIS")
print("=" * 65)

predictions = pd.read_csv(PREDICTION_FILE)
aging = pd.read_csv(AGING_FILE)

print("\nPrediction dataset:", predictions.shape)
print("Aging dataset:", aging.shape)

print("\nPrediction columns:")
print(predictions.columns.tolist())

print("\nAging columns:")
print(aging.columns.tolist())


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_prediction_columns = [
    "battery_id",
    "mission_id",
    "soh",
    "predicted_soh"
]

required_aging_columns = [
    "battery_id",
    "mission_id",
    "aging_ah"
]

for column in required_prediction_columns:
    if column not in predictions.columns:
        raise ValueError(
            f"Missing prediction column: {column}"
        )

for column in required_aging_columns:
    if column not in aging.columns:
        raise ValueError(
            f"Missing aging column: {column}"
        )


# ============================================================
# MERGE AGING INFORMATION
# ============================================================

df = predictions.merge(
    aging[
        ["battery_id", "mission_id", "aging_ah"]
    ],
    on=["battery_id", "mission_id"],
    how="left",
    validate="one_to_one"
)

if df["aging_ah"].isna().any():
    missing = df[df["aging_ah"].isna()]

    print("\nWARNING: Missing aging values:")
    print(missing[["battery_id", "mission_id"]])

    raise ValueError(
        "Some predictions could not be matched with aging features."
    )


# ============================================================
# CALCULATE ERRORS
# ============================================================

df["error"] = (
    df["predicted_soh"] - df["soh"]
)

df["absolute_error"] = df["error"].abs()

df["squared_error"] = df["error"] ** 2

df["percentage_error"] = (
    df["absolute_error"] / df["soh"].abs()
) * 100

df = df.sort_values(
    ["battery_id", "aging_ah"]
).reset_index(drop=True)


# ============================================================
# OVERALL METRICS
# ============================================================

actual = df["soh"]
predicted = df["predicted_soh"]

overall_mae = mean_absolute_error(
    actual,
    predicted
)

overall_rmse = np.sqrt(
    mean_squared_error(actual, predicted)
)

overall_r2 = r2_score(
    actual,
    predicted
)

print("\n" + "=" * 65)
print("OVERALL PERFORMANCE")
print("=" * 65)

print(f"Records: {len(df)}")
print(f"Batteries: {df['battery_id'].nunique()}")
print(f"MAE:  {overall_mae:.4f}")
print(f"RMSE: {overall_rmse:.4f}")
print(f"R2:   {overall_r2:.4f}")


# ============================================================
# BATTERY-WISE METRICS
# ============================================================

battery_results = []

for battery_id, group in df.groupby("battery_id"):

    y_true = group["soh"]
    y_pred = group["predicted_soh"]

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = (
        r2_score(y_true, y_pred)
        if len(group) >= 2 and y_true.nunique() > 1
        else np.nan
    )

    battery_results.append({
        "battery_id": battery_id,
        "samples": len(group),
        "actual_soh_mean": y_true.mean(),
        "predicted_soh_mean": y_pred.mean(),
        "actual_soh_min": y_true.min(),
        "actual_soh_max": y_true.max(),
        "aging_min_ah": group["aging_ah"].min(),
        "aging_max_ah": group["aging_ah"].max(),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "mean_error": group["error"].mean(),
        "max_absolute_error": group["absolute_error"].max()
    })


battery_metrics = pd.DataFrame(battery_results)

battery_metrics = battery_metrics.sort_values(
    "MAE",
    ascending=False
)

print("\n" + "=" * 65)
print("BATTERY-WISE ERROR ANALYSIS")
print("=" * 65)

print(
    battery_metrics.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# HIGHEST ERROR MISSIONS
# ============================================================

print("\n" + "=" * 65)
print("TOP 20 HIGHEST ERROR MISSIONS")
print("=" * 65)

worst_missions = df.sort_values(
    "absolute_error",
    ascending=False
).head(20)

print(
    worst_missions[
        [
            "battery_id",
            "mission_id",
            "aging_ah",
            "soh",
            "predicted_soh",
            "error",
            "absolute_error"
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# INVESTIGATE SPECIFIC BATTERIES
# ============================================================

problem_batteries = [
    "battery54",
    "battery13",
    "battery50",
    "battery36"
]

for battery_id in problem_batteries:

    group = df[
        df["battery_id"] == battery_id
    ].copy()

    if group.empty:
        continue

    print("\n" + "=" * 65)
    print(f"DETAILED ANALYSIS: {battery_id}")
    print("=" * 65)

    print("Samples:", len(group))

    print(
        f"Actual SOH range: "
        f"{group['soh'].min():.3f} - "
        f"{group['soh'].max():.3f}"
    )

    print(
        f"Predicted SOH range: "
        f"{group['predicted_soh'].min():.3f} - "
        f"{group['predicted_soh'].max():.3f}"
    )

    print(
        f"Aging range: "
        f"{group['aging_ah'].min():.3f} - "
        f"{group['aging_ah'].max():.3f} Ah"
    )

    print(
        f"Mean error: {group['error'].mean():.4f}"
    )

    print(
        f"MAE: {group['absolute_error'].mean():.4f}"
    )

    print(
        f"Maximum error: "
        f"{group['absolute_error'].max():.4f}"
    )

    print("\nMission-wise predictions:")

    print(
        group[
            [
                "mission_id",
                "aging_ah",
                "soh",
                "predicted_soh",
                "error",
                "absolute_error"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# ERROR VS AGING CORRELATION
# ============================================================

print("\n" + "=" * 65)
print("ERROR VS AGING ANALYSIS")
print("=" * 65)

aging_correlations = []

for battery_id, group in df.groupby("battery_id"):

    if len(group) >= 3:

        correlation = group[
            ["aging_ah", "absolute_error"]
        ].corr().iloc[0, 1]

    else:
        correlation = np.nan

    aging_correlations.append({
        "battery_id": battery_id,
        "samples": len(group),
        "aging_error_correlation": correlation,
        "mean_absolute_error": group["absolute_error"].mean()
    })


aging_correlation_df = pd.DataFrame(
    aging_correlations
)

print(
    aging_correlation_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# ERROR BIAS ANALYSIS
# ============================================================

print("\n" + "=" * 65)
print("PREDICTION BIAS ANALYSIS")
print("=" * 65)

overprediction = (
    df["error"] > 0
).sum()

underprediction = (
    df["error"] < 0
).sum()

correct_direction = (
    df["error"] == 0
).sum()

print("Overpredictions:", overprediction)
print("Underpredictions:", underprediction)
print("Exact predictions:", correct_direction)

print(
    f"Mean prediction bias: {df['error'].mean():.4f}"
)

print(
    "Positive bias = model tends to overpredict SOH."
)

print(
    "Negative bias = model tends to underpredict SOH."
)


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "detailed_oof_errors.csv"
    ),
    index=False
)

battery_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "battery_error_summary.csv"
    ),
    index=False
)

worst_missions.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "worst_prediction_missions.csv"
    ),
    index=False
)

aging_correlation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "aging_error_correlation.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("ERROR ANALYSIS COMPLETED")
print("=" * 65)

print("\nSaved files:")

print("1. results/cv/error_analysis/detailed_oof_errors.csv")
print("2. results/cv/error_analysis/battery_error_summary.csv")
print("3. results/cv/error_analysis/worst_prediction_missions.csv")
print("4. results/cv/error_analysis/aging_error_correlation.csv")

print("\nBatteries with highest MAE:")

print(
    battery_metrics[
        ["battery_id", "samples", "MAE", "RMSE", "R2"]
    ].head(5).to_string(index=False)
)
