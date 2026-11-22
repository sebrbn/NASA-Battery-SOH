
import numpy as np
import pandas as pd

from pathlib import Path
from preprocessing import extract_discharge_missions

import io
from contextlib import redirect_stdout


def calculate_capacity(mission_df):
    """
    Estimate discharge capacity using numerical integration.

    Q (Ah) = integral(abs(current) * time) / 3600
    """

    data = mission_df.copy()

    data["time"] = pd.to_numeric(data["time"], errors="coerce")
    data["current_load"] = pd.to_numeric(
        data["current_load"], errors="coerce"
    )

    data = data.dropna(subset=["time", "current_load"])
    data = data.sort_values("time")

    if len(data) < 2:
        return None

    time = data["time"].to_numpy(dtype=float)
    current = np.abs(
        data["current_load"].to_numpy(dtype=float)
    )

    # Remove invalid or repeated timestamps
    valid = np.concatenate(([True], np.diff(time) > 0))
    time = time[valid]
    current = current[valid]

    if len(time) < 2:
        return None

    capacity_ah = np.trapezoid(current, time) / 3600

    return {
        "capacity_ah": capacity_ah,
        "duration_seconds": time[-1] - time[0],
        "mean_current_a": np.mean(current),
        "min_current_a": np.min(current),
        "max_current_a": np.max(current),
        "voltage_start": data["voltage_load"].iloc[0],
        "voltage_end": data["voltage_load"].iloc[-1],
        "samples": len(time)
    }


def process_battery(file_path):

    df = pd.read_csv(file_path, low_memory=False)

    # Suppress detailed mission logs during batch processing
    with redirect_stdout(io.StringIO()):
        missions = extract_discharge_missions(df)

    results = []

    for mission in missions:

        # Only reference discharge missions
        if mission["mission_type"] != 0:
            continue

        measurements = calculate_capacity(mission["data"])

        if measurements is None:
            continue

        results.append({
            "battery_id": file_path.stem,
            "mission_id": mission["mission_id"],
            "mission_type": mission["mission_type"],
            **measurements
        })

    return results


def main():

    dataset_path = Path("dataset")
    output_path = Path("results")

    output_path.mkdir(exist_ok=True)

    all_results = []

    files = list(dataset_path.rglob("*.csv"))

    print("Total battery files:", len(files))

    for file in files:

        print(f"\nProcessing: {file.name}")

        try:
            results = process_battery(file)

            for result in results:
                result["dataset_group"] = file.parent.name

            all_results.extend(results)

            print("Reference missions processed:", len(results))

        except Exception as error:
            print(f"Error processing {file.name}: {error}")

    if not all_results:
        print("\nNo reference capacities calculated.")
        return

    capacity_df = pd.DataFrame(all_results)

    # Sort each battery's reference measurements chronologically
    capacity_df = capacity_df.sort_values(
        ["battery_id", "mission_id"]
    ).reset_index(drop=True)

    output_file = output_path / "reference_capacities.csv"

    capacity_df.to_csv(output_file, index=False)

    print("\n" + "=" * 55)
    print("CAPACITY ESTIMATION SUMMARY")
    print("=" * 55)

    print("Total reference measurements:", len(capacity_df))
    print("Batteries processed:", capacity_df["battery_id"].nunique())

    print("\nCapacity statistics:")
    print(capacity_df["capacity_ah"].describe())

    print("\nFirst 10 measurements:")
    print(
        capacity_df[
            [
                "battery_id",
                "mission_id",
                "capacity_ah",
                "mean_current_a",
                "duration_seconds"
            ]
        ].head(10).to_string(index=False)
    )

    print("\nSaved to:", output_file)


if __name__ == "__main__":
    main()
