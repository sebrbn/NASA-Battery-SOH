
import pandas as pd
from pathlib import Path


def main():

    input_file = Path("results/reference_capacities.csv")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    df = pd.read_csv(input_file)

    print("Original measurements:", len(df))

    # Quality checks
    valid_mask = (
        (df["samples"] >= 100) &
        (df["duration_seconds"] >= 60) &
        (df["voltage_end"] > 0) &
        (df["capacity_ah"] > 0)
    )

    valid_df = df[valid_mask].copy()
    rejected_df = df[~valid_mask].copy()

    print("Valid measurements:", len(valid_df))
    print("Rejected measurements:", len(rejected_df))

    print("\nRejected measurements:")
    if not rejected_df.empty:
        print(
            rejected_df[
                [
                    "battery_id",
                    "mission_id",
                    "capacity_ah",
                    "duration_seconds",
                    "samples",
                    "voltage_start",
                    "voltage_end"
                ]
            ].to_string(index=False)
        )

    # Chronological order within each battery
    valid_df = valid_df.sort_values(
        ["battery_id", "mission_id"]
    ).reset_index(drop=True)

    # First valid capacity for each battery
    baseline = valid_df.groupby(
        "battery_id"
    )["capacity_ah"].transform("first")

    # Calculate relative SOH
    valid_df["soh_percent"] = (
        valid_df["capacity_ah"] / baseline
    ) * 100

    valid_df["reference_number"] = (
        valid_df.groupby("battery_id").cumcount() + 1
    )

    # Save files
    valid_df.to_csv(
        output_dir / "soh_labels.csv",
        index=False
    )

    rejected_df.to_csv(
        output_dir / "rejected_capacity_measurements.csv",
        index=False
    )

    # Summary
    summary = valid_df.groupby("battery_id").agg(
        measurements=("capacity_ah", "count"),
        initial_capacity=("capacity_ah", "first"),
        final_capacity=("capacity_ah", "last"),
        initial_soh=("soh_percent", "first"),
        final_soh=("soh_percent", "last")
    )

    summary.to_csv(
        output_dir / "soh_summary.csv"
    )

    print("\n" + "=" * 55)
    print("SOH LABEL GENERATION SUMMARY")
    print("=" * 55)

    print("\nSOH statistics:")
    print(valid_df["soh_percent"].describe())

    print("\nPer-battery SOH:")
    print(summary.round(2).to_string())

    print("\nFiles saved:")
    print("results/soh_labels.csv")
    print("results/rejected_capacity_measurements.csv")
    print("results/soh_summary.csv")


if __name__ == "__main__":
    main()
