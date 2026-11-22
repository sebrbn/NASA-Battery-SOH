
import pandas as pd
from pathlib import Path

df = pd.read_csv("results/reference_capacities.csv")

print("\nLOWEST 15 CAPACITY MEASUREMENTS")
print("=" * 75)

print(
    df.nsmallest(15, "capacity_ah")[
        [
            "battery_id",
            "mission_id",
            "capacity_ah",
            "duration_seconds",
            "mean_current_a",
            "samples"
        ]
    ].to_string(index=False)
)

print("\nPER-BATTERY CAPACITY SUMMARY")
print("=" * 75)

summary = df.groupby("battery_id").agg(
    measurements=("capacity_ah", "count"),
    first_capacity=("capacity_ah", "first"),
    last_capacity=("capacity_ah", "last"),
    minimum=("capacity_ah", "min"),
    maximum=("capacity_ah", "max"),
    mean=("capacity_ah", "mean")
)

summary["last_vs_first_percent"] = (
    summary["last_capacity"] /
    summary["first_capacity"] * 100
)

print(summary.round(4).to_string())

# Flag suspiciously low measurements for inspection.
# This is a diagnostic threshold, not an automatic deletion rule.
suspicious = df[df["capacity_ah"] < 0.5]

print("\nSUSPICIOUS MEASUREMENTS (< 0.5 Ah)")
print("=" * 75)

print(suspicious.to_string(index=False))

Path("results").mkdir(exist_ok=True)
summary.to_csv("results/battery_capacity_summary.csv")

print("\nSaved: results/battery_capacity_summary.csv")
