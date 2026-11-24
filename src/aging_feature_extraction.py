
import os
import glob
import numpy as np
import pandas as pd

from scipy.integrate import trapezoid
from preprocessing import extract_discharge_missions


DATASET_PATH = "dataset"
OUTPUT_PATH = "results/aging_features.csv"


def calculate_mission_capacity(mission_data):
    """Calculate discharged Ah using actual elapsed timestamps."""

    time = mission_data["time"].to_numpy(dtype=float)
    current = mission_data["current_load"].to_numpy(dtype=float)

    valid = (
        np.isfinite(time) &
        np.isfinite(current)
    )

    time = time[valid]
    current = current[valid]

    if len(time) < 2:
        return np.nan

    # Sort by elapsed time
    order = np.argsort(time)
    time = time[order]
    current = current[order]

    # Remove duplicate timestamps
    time, unique_indices = np.unique(
        time, return_index=True
    )
    current = current[unique_indices]

    if len(time) < 2:
        return np.nan

    capacity = trapezoid(
        np.abs(current),
        time
    ) / 3600

    return capacity


def process_battery(file_path):

    battery_id = os.path.basename(
        file_path
    ).replace(".csv", "")

    print(f"\nProcessing {battery_id}")

    df = pd.read_csv(file_path, low_memory=False)

    missions = extract_discharge_missions(df)

    mission_records = []

    for mission in missions:

        data = mission["data"].copy()

        capacity = calculate_mission_capacity(data)

        # Validate mission duration
        time_values = pd.to_numeric(
            data["time"], errors="coerce"
        ).dropna()

        if len(time_values) < 2:
            continue

        duration = time_values.max() - time_values.min()

        # Validate voltage
        voltage_values = pd.to_numeric(
            data["voltage_load"], errors="coerce"
        ).dropna()

        if voltage_values.empty:
            continue

        voltage_end = voltage_values.iloc[-1]

        # Apply quality checks to ALL discharge missions
        if (
            len(data) < 100 or
            duration < 60 or
            not np.isfinite(capacity) or
            capacity <= 0 or
            voltage_end <= 0
        ):
            continue

        mission_records.append({
            "battery_id": battery_id,
            "mission_id": mission["mission_id"],
            "mission_type": mission["mission_type"],
            "start_time": time_values.min(),
            "end_time": time_values.max(),
            "capacity_ah": capacity
        })

    if not mission_records:
        return []

    mission_df = pd.DataFrame(mission_records)

    mission_df = mission_df.sort_values(
        ["start_time", "end_time", "mission_id"]
    ).reset_index(drop=True)

    # Cumulative throughput, including both mission types
    mission_df["aging_ah"] = (
        mission_df["capacity_ah"].cumsum()
    )

    # Retain reference missions for matching with SOH labels
    reference_df = mission_df[
        mission_df["mission_type"] == 0
    ].copy()

    return reference_df.to_dict("records")


def main():

    all_records = []

    files = glob.glob(
        os.path.join(
            DATASET_PATH,
            "**",
            "*.csv"
        ),
        recursive=True
    )

    print("=" * 50)
    print("CUMULATIVE DISCHARGE THROUGHPUT EXTRACTION")
    print("=" * 50)

    for file_path in files:

        records = process_battery(file_path)

        all_records.extend(records)

    result = pd.DataFrame(all_records)

    os.makedirs("results", exist_ok=True)

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 50)
    print("AGING FEATURE EXTRACTION COMPLETE")
    print("=" * 50)

    print("Reference missions:", len(result))
    print("Batteries:", result["battery_id"].nunique())

    print("\nPreview:")
    print(
        result[
            [
                "battery_id",
                "mission_id",
                "capacity_ah",
                "aging_ah"
            ]
        ].head(10)
    )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
