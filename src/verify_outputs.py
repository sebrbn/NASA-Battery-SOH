
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS = os.path.join(BASE_DIR, "results")
MODELS = os.path.join(BASE_DIR, "models")


def check_csv(relative_path, expected_rows=None):

    path = os.path.join(RESULTS, relative_path)

    print("\n" + "=" * 65)
    print(relative_path)
    print("=" * 65)

    if not os.path.exists(path):
        print("MISSING FILE")
        return

    df = pd.read_csv(path)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Column names:", df.columns.tolist())
    print("Missing values:", df.isna().sum().sum())

    if expected_rows is not None:
        print("Expected rows:", expected_rows)
        print("Row count correct:", len(df) == expected_rows)

    if {"battery_id", "mission_id"}.issubset(df.columns):
        duplicates = df.duplicated(
            subset=["battery_id", "mission_id"]
        ).sum()

        print("Duplicate battery/mission records:", duplicates)

    numeric = df.select_dtypes(include=np.number)

    if not numeric.empty:
        print(
            "Infinite numeric values:",
            np.isinf(numeric.to_numpy()).sum()
        )

    return df


# ============================================================
# PREDICTION FILES
# ============================================================

print("\nFINAL PREDICTION FILES")

gpr_predictions = check_csv(
    "pinn_gpr_test_predictions.csv",
    expected_rows=55
)

pinn_predictions = check_csv(
    "pinn_test_predictions.csv",
    expected_rows=55
)

oof_predictions = check_csv(
    "pinn_oof_train_predictions.csv",
    expected_rows=426
)

cv_predictions = check_csv(
    "cv/pinn_group_oof_predictions.csv",
    expected_rows=426
)


# ============================================================
# METRICS
# ============================================================

print("\nMODEL EVALUATION FILES")

comparison = check_csv("pinn_gpr_comparison.csv")

battery_metrics = check_csv(
    "pinn_gpr_battery_metrics.csv"
)

cv_metrics = check_csv(
    "cv/pinn_group_fold_metrics.csv"
)


if comparison is not None:
    print("\nPINN VS PINN + GPR COMPARISON")
    print(comparison.to_string(index=False))


# ============================================================
# MODEL FILES
# ============================================================

print("\nSAVED MODEL FILES")

model_files = [
    "pinn_model.pth",
    "pinn_input_scaler.pkl",
    "pinn_metadata.pkl",
    "gpr_residual_model.pkl",
    "gpr_input_scaler.pkl",
    "ica_pca.pkl",
    "ica_scaler.pkl"
]

for filename in model_files:

    path = os.path.join(MODELS, filename)

    if os.path.exists(path):
        size = os.path.getsize(path)

        print(
            f"[OK] {filename} - {size:,} bytes"
        )

    else:
        print(f"[MISSING] {filename}")


# ============================================================
# FINAL DATASET
# ============================================================

print("\nFINAL DATASET")

check_csv("final_feature_dataset.csv", expected_rows=481)
check_csv("aging_features.csv", expected_rows=481)
check_csv("soh_labels.csv", expected_rows=481)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 65)
print("OUTPUT VERIFICATION COMPLETED")
print("=" * 65)
