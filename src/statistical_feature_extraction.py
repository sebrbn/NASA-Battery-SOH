
import os
import numpy as np
import pandas as pd

from preprocessing import extract_discharge_missions


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_DIR = os.path.join(
    PROJECT_ROOT,
    "dataset"
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "results"
)

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "statistical_features_all.csv"
)

REJECTED_FILE = os.path.join(
    RESULTS_DIR,
    "statistical_rejected_missions.csv"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def calculate_stat(values):

    values = np.asarray(values, dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    return values


def safe_mean(values):

    values = calculate_stat(values)

    if isinstance(values, float):
        return values

    return float(np.mean(values))


def safe_std(values):

    values = calculate_stat(values)

    if isinstance(values, float):
        return values

    return float(np.std(values))


def safe_min(values):

    values = calculate_stat(values)

    if isinstance(values, float):
        return values

    return float(np.min(values))


def safe_max(values):

    values = calculate_stat(values)

    if isinstance(values, float):
        return values

    return float(np.max(values))


# ============================================================
# 3. EXTRACT STATISTICAL FEATURES
# ============================================================

def extract_statistical_features(
    mission_df,
    battery_id,
    mission_id
):

    mission_df = mission_df.copy()

    required_columns = [
        "time",
        "voltage_load",
        "current_load"
    ]

    optional_columns = [
        "temperature_battery",
        "temperature_mosfet",
        "temperature_resistor"
    ]

    # Convert columns to numeric
    for column in required_columns + optional_columns:

        if column in mission_df.columns:

            mission_df[column] = pd.to_numeric(
                mission_df[column],
                errors="coerce"
            )

        elif column in required_columns:

            raise ValueError(
                f"Missing required column: {column}"
            )

        else:

            mission_df[column] = np.nan

    # Remove rows without essential measurements
    mission_df = mission_df.dropna(
        subset=required_columns
    )

    mission_df = mission_df.sort_values("time")

    mission_df = mission_df.drop_duplicates(
        subset="time",
        keep="first"
    )

    if len(mission_df) < 100:

        raise ValueError("Insufficient valid samples")

    time = mission_df["time"].to_numpy(dtype=float)

    voltage = mission_df["voltage_load"].to_numpy(dtype=float)

    current = mission_df["current_load"].to_numpy(dtype=float)

    battery_temp = mission_df[
        "temperature_battery"
    ].to_numpy(dtype=float)

    mosfet_temp = mission_df[
        "temperature_mosfet"
    ].to_numpy(dtype=float)

    resistor_temp = mission_df[
        "temperature_resistor"
    ].to_numpy(dtype=float)

    if np.any(np.diff(time) <= 0):

        raise ValueError("Invalid time sequence")

    duration = float(time[-1] - time[0])

    if duration < 60:

        raise ValueError("Duration less than 60 seconds")

    if voltage[-1] <= 0:

        raise ValueError("Invalid ending voltage")

    # Calculate capacity for validation and metadata
    dt = np.diff(time)

    average_current = (
        np.abs(current[:-1]) +
        np.abs(current[1:])
    ) / 2

    capacity_ah = np.sum(
        average_current * dt
    ) / 3600

    if capacity_ah <= 0:

        raise ValueError("Invalid capacity")

    # ========================================================
    # 14 STATISTICAL FEATURES
    # ========================================================

    features = {

        "battery_id": battery_id,

        "mission_id": mission_id,

        # Metadata
        "capacity_ah": capacity_ah,

        "duration_seconds": duration,

        "samples": len(mission_df),

        # Voltage features (4)
        "voltage_mean": safe_mean(voltage),

        "voltage_std": safe_std(voltage),

        "voltage_min": safe_min(voltage),

        "voltage_max": safe_max(voltage),

        # Current features (4)
        "current_mean": safe_mean(current),

        "current_std": safe_std(current),

        "current_min": safe_min(current),

        "current_max": safe_max(current),

        # Battery temperature features (2)
        "battery_temp_mean": safe_mean(battery_temp),

        "battery_temp_std": safe_std(battery_temp),

        # MOSFET temperature features (2)
        "mosfet_temp_mean": safe_mean(mosfet_temp),

        "mosfet_temp_max": safe_max(mosfet_temp),

        # Resistor temperature features (2)
        "resistor_temp_mean": safe_mean(resistor_temp),

        "resistor_temp_max": safe_max(resistor_temp)
    }

    return features


# ============================================================
# 4. FIND ALL BATTERY FILES
# ============================================================

battery_files = []

for root, dirs, files in os.walk(DATASET_DIR):

    for filename in files:

        if filename.lower().endswith(".csv"):

            battery_files.append(
                os.path.join(root, filename)
            )

battery_files.sort()

print("\n========== DATASET INFORMATION ==========")

print("Total battery files:", len(battery_files))

if not battery_files:

    raise FileNotFoundError("No battery CSV files found.")


# ============================================================
# 5. PROCESS ALL BATTERIES
# ============================================================

all_features = []

rejected_missions = []

total_reference_missions = 0

successful_missions = 0

for file_index, file_path in enumerate(
    battery_files,
    start=1
):

    battery_id = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    print("\n--------------------------------------------")

    print(
        f"Processing {file_index}/{len(battery_files)}:"
        f" {battery_id}"
    )

    try:

        df = pd.read_csv(file_path)

        missions = extract_discharge_missions(df)

        reference_missions = [
            mission for mission in missions
            if int(mission["mission_type"]) == 0
        ]

        print("Reference missions:", len(reference_missions))

        for mission in reference_missions:

            total_reference_missions += 1

            mission_id = mission["mission_id"]

            try:

                features = extract_statistical_features(
                    mission["data"],
                    battery_id,
                    mission_id
                )

                all_features.append(features)

                successful_missions += 1

                print(
                    f"Mission {mission_id}: "
                    f"Voltage mean="
                    f"{features['voltage_mean']:.4f} V"
                )

            except Exception as error:

                rejected_missions.append({

                    "battery_id": battery_id,

                    "mission_id": mission_id,

                    "reason": str(error)

                })

                print(
                    f"Rejected mission {mission_id}: {error}"
                )

    except Exception as error:

        print("Battery processing failed:", error)

        rejected_missions.append({

            "battery_id": battery_id,

            "mission_id": None,

            "reason": f"Battery processing failed: {error}"

        })


# ============================================================
# 6. SAVE RESULTS
# ============================================================

features_df = pd.DataFrame(all_features)

rejected_df = pd.DataFrame(rejected_missions)

if not features_df.empty:

    features_df = features_df.sort_values(
        ["battery_id", "mission_id"]
    )

    features_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

if not rejected_df.empty:

    rejected_df.to_csv(
        REJECTED_FILE,
        index=False
    )


# ============================================================
# 7. FINAL SUMMARY
# ============================================================

print("\n============================================")

print("     STATISTICAL FEATURE EXTRACTION")

print("============================================")

print("Total battery files:", len(battery_files))

print("Reference missions:", total_reference_missions)

print("Successfully processed:", successful_missions)

print("Rejected missions:", len(rejected_missions))

if not features_df.empty:

    print("\nFeature dataset shape:", features_df.shape)

    print("\nBatteries processed:",
          features_df["battery_id"].nunique())

    print("\nStatistical feature preview:")

    print(features_df.head())

    statistical_columns = [
        "voltage_mean",
        "voltage_std",
        "voltage_min",
        "voltage_max",
        "current_mean",
        "current_std",
        "current_min",
        "current_max",
        "battery_temp_mean",
        "battery_temp_std",
        "mosfet_temp_mean",
        "mosfet_temp_max",
        "resistor_temp_mean",
        "resistor_temp_max"
    ]

    print(
        "\nNumber of statistical features:",
        len(statistical_columns)
    )

    print("\nMissing values in statistical features:")

    print(
        features_df[statistical_columns]
        .isna()
        .sum()
    )

else:

    print("\nNo statistical features generated.")

print("\nOutput file:", OUTPUT_FILE)

print("Rejected file:", REJECTED_FILE)

print("\nStatistical feature extraction completed.")
