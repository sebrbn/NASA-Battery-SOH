
import os
import numpy as np
import pandas as pd

from scipy.signal import savgol_filter, find_peaks, peak_widths

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
    "ica_features_all.csv"
)

REJECTED_FILE = os.path.join(
    RESULTS_DIR,
    "ica_rejected_missions.csv"
)

VOLTAGE_STEP = 0.005

SAVGOL_WINDOW = 21
SAVGOL_POLYORDER = 3

PEAK_PROMINENCE = 0.02
PEAK_DISTANCE = 5

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. ICA FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_ica_features(mission_df, battery_id, mission_id):

    mission_df = mission_df.copy()

    # Convert required columns to numeric
    required_columns = [
        "time",
        "voltage_load",
        "current_load"
    ]

    for column in required_columns:
        mission_df[column] = pd.to_numeric(
            mission_df[column],
            errors="coerce"
        )

    # Remove invalid measurements
    mission_df = mission_df.dropna(
        subset=required_columns
    )

    mission_df = mission_df.sort_values("time")

    mission_df = mission_df.drop_duplicates(
        subset="time",
        keep="first"
    )

    time = mission_df["time"].to_numpy(dtype=float)

    voltage = mission_df["voltage_load"].to_numpy(dtype=float)

    current = mission_df["current_load"].to_numpy(dtype=float)

    if len(time) < 100:
        raise ValueError("Insufficient valid samples")

    # Ensure time is strictly increasing
    if np.any(np.diff(time) <= 0):
        raise ValueError("Invalid time sequence")

    # --------------------------------------------------------
    # Calculate discharge capacity
    # --------------------------------------------------------

    current_abs = np.abs(current)

    dt = np.diff(time)

    average_current = (
        current_abs[:-1] + current_abs[1:]
    ) / 2

    capacity_increment = (
        average_current * dt / 3600
    )

    capacity_ah = np.concatenate([
        [0],
        np.cumsum(capacity_increment)
    ])

    total_capacity = capacity_ah[-1]

    duration = time[-1] - time[0]

    if total_capacity <= 0:
        raise ValueError("Invalid discharge capacity")

    # --------------------------------------------------------
    # Prepare voltage-capacity relationship
    # --------------------------------------------------------

    valid_mask = (
        np.isfinite(voltage) &
        np.isfinite(capacity_ah) &
        (voltage > 0)
    )

    valid_voltage = voltage[valid_mask]

    valid_capacity = capacity_ah[valid_mask]

    if len(valid_voltage) < 30:
        raise ValueError("Insufficient valid voltage samples")

    voltage_df = pd.DataFrame({
        "voltage": valid_voltage,
        "capacity": valid_capacity
    })

    voltage_df = (
        voltage_df
        .groupby("voltage", as_index=False)["capacity"]
        .mean()
        .sort_values("voltage")
    )

    voltage_unique = voltage_df["voltage"].to_numpy()

    capacity_unique = voltage_df["capacity"].to_numpy()

    if len(voltage_unique) < 30:
        raise ValueError("Insufficient unique voltage values")

    # --------------------------------------------------------
    # Calculate original ICA
    # --------------------------------------------------------

    voltage_grid = np.arange(
        voltage_unique.min(),
        voltage_unique.max(),
        VOLTAGE_STEP
    )

    # Keep the smoothing settings consistent across missions
    if len(voltage_grid) < SAVGOL_WINDOW:
        raise ValueError("Insufficient voltage grid points")

    capacity_grid = np.interp(
        voltage_grid,
        voltage_unique,
        capacity_unique
    )

    capacity_smooth = savgol_filter(
        capacity_grid,
        SAVGOL_WINDOW,
        SAVGOL_POLYORDER
    )

    ica = np.abs(
        np.gradient(
            capacity_smooth,
            voltage_grid
        )
    )

    if not np.all(np.isfinite(ica)):
        raise ValueError("Non-finite ICA values")

    # --------------------------------------------------------
    # ICA statistical features
    # --------------------------------------------------------

    ica_mean = np.mean(ica)

    ica_std = np.std(ica)

    ica_max = np.max(ica)

    ica_min = np.min(ica)

    # --------------------------------------------------------
    # Detect ICA peaks
    # --------------------------------------------------------

    peaks, properties = find_peaks(
        ica,
        prominence=PEAK_PROMINENCE,
        distance=PEAK_DISTANCE
    )

    peak_count = len(peaks)

    # Default values when no peaks are detected
    peak_voltage = np.nan
    peak_height = np.nan
    peak_mean_height = np.nan
    peak_std_height = np.nan
    peak_width = np.nan
    peak_area = np.nan

    if peak_count > 0:

        # Identify strongest peak
        strongest_index = np.argmax(
            ica[peaks]
        )

        strongest_peak = peaks[strongest_index]

        peak_voltage = voltage_grid[strongest_peak]

        peak_height = ica[strongest_peak]

        # Statistics of all detected peaks
        peak_heights = ica[peaks]

        peak_mean_height = np.mean(peak_heights)

        peak_std_height = np.std(peak_heights)

        # Calculate peak widths
        widths, width_heights, left_ips, right_ips = peak_widths(
            ica,
            peaks,
            rel_height=0.5
        )

        voltage_spacing = voltage_grid[1] - voltage_grid[0]

        peak_width = (
            widths[strongest_index] * voltage_spacing
        )

        # Approximate area around strongest peak
        left_index = max(
            0,
            int(np.floor(left_ips[strongest_index]))
        )

        right_index = min(
            len(ica) - 1,
            int(np.ceil(right_ips[strongest_index]))
        )

        peak_area = np.trapezoid(
            ica[left_index:right_index + 1],
            voltage_grid[left_index:right_index + 1]
        )

    # --------------------------------------------------------
    # Create feature dictionary
    # --------------------------------------------------------

    features = {

        "battery_id": battery_id,

        "mission_id": mission_id,

        "capacity_ah": total_capacity,

        "duration_seconds": duration,

        "samples": len(time),

        "mean_current_a": np.mean(current_abs),

        "voltage_min": np.min(voltage),

        "voltage_max": np.max(voltage),

        "ica_mean": ica_mean,

        "ica_std": ica_std,

        "ica_max": ica_max,

        "ica_min": ica_min,

        "ica_peak_count": peak_count,

        "ica_peak_voltage": peak_voltage,

        "ica_peak_height": peak_height,

        "ica_peak_mean_height": peak_mean_height,

        "ica_peak_std": peak_std_height,

        "ica_peak_width": peak_width,

        "ica_peak_area": peak_area
    }

    return features


# ============================================================
# 3. FIND ALL BATTERY CSV FILES
# ============================================================

battery_files = []

for root, dirs, files in os.walk(DATASET_DIR):

    for filename in files:

        if filename.lower().endswith(".csv"):

            file_path = os.path.join(
                root,
                filename
            )

            battery_files.append(file_path)

battery_files.sort()

print("\n========== DATASET INFORMATION ==========")

print("Dataset directory:", DATASET_DIR)

print("Total battery CSV files:", len(battery_files))


if len(battery_files) == 0:
    raise FileNotFoundError("No battery CSV files found.")


# ============================================================
# 4. PROCESS ALL BATTERIES
# ============================================================

all_features = []

rejected_missions = []

total_reference_missions = 0

successful_missions = 0

for file_index, file_path in enumerate(battery_files, start=1):

    battery_id = os.path.splitext(
        os.path.basename(file_path)
    )[0]

    print("\n--------------------------------------------")

    print(
        f"Processing battery {file_index}/{len(battery_files)}:"
        f" {battery_id}"
    )

    try:

        df = pd.read_csv(file_path)

        missions = extract_discharge_missions(df)

        reference_missions = [
            mission for mission in missions
            if int(mission["mission_type"]) == 0
        ]

        print("Total discharge missions:", len(missions))

        print("Reference missions:", len(reference_missions))

        for mission in reference_missions:

            total_reference_missions += 1

            mission_id = mission["mission_id"]

            mission_df = mission["data"].copy()

            # ------------------------------------------------
            # Apply the same basic quality criteria used for
            # SOH labels
            # ------------------------------------------------

            clean_df = mission_df.copy()

            for column in [
                "time",
                "voltage_load",
                "current_load"
            ]:

                clean_df[column] = pd.to_numeric(
                    clean_df[column],
                    errors="coerce"
                )

            clean_df = clean_df.dropna(
                subset=[
                    "time",
                    "voltage_load",
                    "current_load"
                ]
            )

            clean_df = clean_df.sort_values("time")

            clean_df = clean_df.drop_duplicates(
                subset="time",
                keep="first"
            )

            if len(clean_df) < 100:

                rejected_missions.append({
                    "battery_id": battery_id,
                    "mission_id": mission_id,
                    "reason": "Less than 100 valid samples"
                })

                continue

            duration = (
                clean_df["time"].iloc[-1]
                - clean_df["time"].iloc[0]
            )

            if duration < 60:

                rejected_missions.append({
                    "battery_id": battery_id,
                    "mission_id": mission_id,
                    "reason": "Duration less than 60 seconds"
                })

                continue

            if clean_df["voltage_load"].iloc[-1] <= 0:

                rejected_missions.append({
                    "battery_id": battery_id,
                    "mission_id": mission_id,
                    "reason": "Invalid ending voltage"
                })

                continue

            # ------------------------------------------------
            # Extract ICA features
            # ------------------------------------------------

            try:

                features = extract_ica_features(
                    mission_df,
                    battery_id,
                    mission_id
                )

                all_features.append(features)

                successful_missions += 1

                print(
                    f"Mission {mission_id}: "
                    f"Capacity={features['capacity_ah']:.4f} Ah, "
                    f"Peaks={features['ica_peak_count']}"
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
# 5. SAVE RESULTS
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
# 6. FINAL SUMMARY
# ============================================================

print("\n\n============================================")

print("          ICA EXTRACTION SUMMARY")

print("============================================")

print("Total battery files:", len(battery_files))

print("Total reference missions:", total_reference_missions)

print("Successfully processed:", successful_missions)

print("Rejected missions:", len(rejected_missions))

print("\nOutput files:")

print("ICA features:", OUTPUT_FILE)

print("Rejected missions:", REJECTED_FILE)

if not features_df.empty:

    print("\n========== FEATURE DATASET PREVIEW ==========")

    print(features_df.head())

    print("\nFeature dataset shape:", features_df.shape)

    print("\nBatteries processed:")

    print(
        features_df["battery_id"].nunique()
    )

    print("\nTotal ICA feature columns:")

    print(
        len(features_df.columns)
    )

else:

    print("\nNo ICA features were generated.")

print("\nICA batch extraction completed.")
