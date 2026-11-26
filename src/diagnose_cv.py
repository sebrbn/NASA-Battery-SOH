
import pandas as pd
import numpy as np

cv = pd.read_csv("results/cv/pinn_group_oof_predictions.csv")
original = pd.read_csv("results/final_feature_dataset.csv")

print("=" * 60)
print("CROSS-VALIDATION PREDICTION DIAGNOSTICS")
print("=" * 60)

for column in ["soh", "predicted_soh", "absolute_error"]:
    print(f"\n{column}")
    print(cv[column].describe())

print("\nTRUE SOH RANGE")
print("Minimum:", original["soh"].min())
print("Maximum:", original["soh"].max())
print("Mean:", original["soh"].mean())

print("\nPREDICTION RANGE")
print("Minimum:", cv["predicted_soh"].min())
print("Maximum:", cv["predicted_soh"].max())
print("Mean:", cv["predicted_soh"].mean())

print("\nFIRST 15 PREDICTIONS")
print(
    cv[
        ["battery_id", "mission_id", "soh",
         "predicted_soh", "absolute_error"]
    ].head(15).to_string(index=False)
)

print("\nFOLD-WISE PREDICTION SUMMARY")

print(
    cv.groupby("fold").agg(
        true_mean=("soh", "mean"),
        predicted_mean=("predicted_soh", "mean"),
        true_std=("soh", "std"),
        predicted_std=("predicted_soh", "std"),
        mae=("absolute_error", "mean")
    ).round(3)
)
