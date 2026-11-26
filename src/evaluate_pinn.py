
import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

INPUT_PATH = "results/pinn_test_predictions.csv"

METRICS_PATH = "results/pinn_battery_metrics.csv"
ERRORS_PATH = "results/pinn_prediction_errors.csv"


# --------------------------------------------------
# FIND COLUMNS
# --------------------------------------------------

def find_column(df, candidates):

    normalized = {
        col.lower().strip().replace(" ", "_"): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]

    return None


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = (
        r2_score(actual, predicted)
        if len(actual) >= 2
        else np.nan
    )

    return {
        "samples": len(actual),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("\n" + "=" * 55)
    print("          PINN BATTERY-WISE EVALUATION")
    print("=" * 55)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Prediction file not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("\nLoaded predictions:", len(df))
    print("Available columns:", df.columns.tolist())

    # Identify required columns
    battery_col = find_column(
        df,
        ["battery_id"]
    )

    actual_col = find_column(
        df,
        [
            "actual_soh",
            "soh_actual",
            "true_soh",
            "y_true",
            "actual",
            "soh"
        ]
    )

    predicted_col = find_column(
        df,
        [
            "predicted_soh",
            "soh_predicted",
            "prediction",
            "predicted",
            "y_pred"
        ]
    )

    if battery_col is None:
        raise ValueError("battery_id column not found.")

    if actual_col is None or predicted_col is None:
        raise ValueError(
            "Could not identify actual and predicted SOH columns. "
            "Check the available columns printed above."
        )

    # Convert to numeric
    df[actual_col] = pd.to_numeric(
        df[actual_col],
        errors="coerce"
    )

    df[predicted_col] = pd.to_numeric(
        df[predicted_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            battery_col,
            actual_col,
            predicted_col
        ]
    ).copy()

    if df.empty:
        raise ValueError("No valid prediction records found.")

    actual = df[actual_col].to_numpy()
    predicted = df[predicted_col].to_numpy()

    # Calculate errors
    df["prediction_error"] = predicted - actual
    df["absolute_error"] = np.abs(
        df["prediction_error"]
    )

    # --------------------------------------------------
    # OVERALL PERFORMANCE
    # --------------------------------------------------

    overall = calculate_metrics(
        actual,
        predicted
    )

    print("\nOVERALL TEST PERFORMANCE")
    print("-" * 40)

    print(f"Test samples : {overall['samples']}")
    print(f"MAE          : {overall['MAE']:.4f}")
    print(f"RMSE         : {overall['RMSE']:.4f}")
    print(f"R2           : {overall['R2']:.4f}")

    # --------------------------------------------------
    # BATTERY-WISE PERFORMANCE
    # --------------------------------------------------

    results = []

    for battery_id, group in df.groupby(battery_col):

        battery_actual = group[actual_col].to_numpy()
        battery_predicted = group[predicted_col].to_numpy()

        metrics = calculate_metrics(
            battery_actual,
            battery_predicted
        )

        metrics["battery_id"] = battery_id

        results.append(metrics)

    metrics_df = pd.DataFrame(results)

    metrics_df = metrics_df[
        [
            "battery_id",
            "samples",
            "MAE",
            "RMSE",
            "R2"
        ]
    ]

    metrics_df = metrics_df.sort_values(
        "battery_id"
    )

    print("\nBATTERY-WISE PERFORMANCE")
    print("-" * 65)

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------
    # BEST AND WORST BATTERY
    # --------------------------------------------------

    best = metrics_df.loc[
        metrics_df["MAE"].idxmin()
    ]

    worst = metrics_df.loc[
        metrics_df["MAE"].idxmax()
    ]

    print("\nBEST PREDICTION")
    print("-" * 40)

    print("Battery:", best["battery_id"])
    print(f"MAE    : {best['MAE']:.4f}")

    print("\nLARGEST PREDICTION ERROR")
    print("-" * 40)

    print("Battery:", worst["battery_id"])
    print(f"MAE    : {worst['MAE']:.4f}")

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    os.makedirs("results", exist_ok=True)

    metrics_df.to_csv(
        METRICS_PATH,
        index=False
    )

    df.to_csv(
        ERRORS_PATH,
        index=False
    )

    print("\nSAVED FILES")
    print("-" * 40)

    print(METRICS_PATH)
    print(ERRORS_PATH)

    print("\nEVALUATION COMPLETED")


if __name__ == "__main__":
    main()
