
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "results" / "final_feature_dataset.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 55)
print("       FINAL DATASET VALIDATION")
print("=" * 55)

# ==========================================
# 1. BASIC INFORMATION
# ==========================================

print("\n1. DATASET INFORMATION")
print("-" * 40)

print("Dataset shape:", df.shape)
print("Total batteries:", df["battery_id"].nunique())
print("Total missions:", len(df))

print("\nColumns:")
for col in df.columns:
    print(col)

# ==========================================
# 2. DUPLICATE CHECK
# ==========================================

print("\n2. DUPLICATE CHECK")
print("-" * 40)

duplicates = df.duplicated(
    subset=["battery_id", "mission_id"]
).sum()

print("Duplicate mission records:", duplicates)

# ==========================================
# 3. MISSING VALUES
# ==========================================

print("\n3. MISSING VALUE CHECK")
print("-" * 40)

missing = df.isnull().sum()

print(missing[missing > 0] if missing.sum() else "No missing values")

# ==========================================
# 4. COLUMN OVERLAP CHECK
# ==========================================

print("\n4. MERGED COLUMN CHECK")
print("-" * 40)

overlap_columns = [
    col for col in df.columns
    if col.endswith("_x") or col.endswith("_y")
]

print("Overlapping columns:", overlap_columns)

# ==========================================
# 5. SOH DISTRIBUTION
# ==========================================

print("\n5. SOH STATISTICS")
print("-" * 40)

print(df["soh"].describe())

# ==========================================
# 6. BATTERY-WISE RECORD COUNTS
# ==========================================

print("\n6. BATTERY-WISE MEASUREMENTS")
print("-" * 40)

battery_counts = (
    df.groupby("battery_id")
    .agg(
        measurements=("mission_id", "count"),
        initial_soh=("soh", "first"),
        final_soh=("soh", "last")
    )
)

print(battery_counts)

# ==========================================
# 7. POTENTIAL MODEL INPUTS
# ==========================================

print("\n7. POTENTIAL TARGET AND METADATA")
print("-" * 40)

metadata_columns = [
    "battery_id",
    "mission_id",
    "capacity_ah",
    "soh"
]

print("Target: soh")
print("Metadata / excluded from inputs:", metadata_columns)

# ==========================================
# 8. FINAL VALIDATION
# ==========================================

print("\n8. VALIDATION SUMMARY")
print("-" * 40)

checks = {
    "No duplicate mission keys": duplicates == 0,
    "SOH column exists": "soh" in df.columns,
    "No missing SOH values": df["soh"].notna().all(),
    "No invalid SOH values": np.isfinite(df["soh"]).all()
}

for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

print("\nDataset validation inspection completed.")
# Remove duplicate voltage columns created during merging
df = df.drop(columns=[
    "voltage_min_y",
    "voltage_max_y"
])

# Rename remaining columns
df = df.rename(columns={
    "voltage_min_x": "voltage_min",
    "voltage_max_x": "voltage_max"
})

# Save cleaned dataset
output_path = ROOT / "results" / "final_feature_dataset.csv"
df.to_csv(output_path, index=False)

print("\nCleaned dataset shape:", df.shape)
print("Cleaned dataset saved:", output_path)