
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks, peak_widths
from sklearn.isotonic import IsotonicRegression

from preprocessing import extract_discharge_missions


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "recommissioned_batteries",
    "battery02.csv"
)

VOLTAGE_STEP = 0.005

SAVGOL_WINDOW = 21
SAVGOL_POLYORDER = 3

PEAK_PROMINENCE = 0.02
PEAK_DISTANCE = 5


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print("Loaded:", os.path.basename(DATASET_PATH))
print("Total rows:", len(df))

# Extract discharge missions
missions = extract_discharge_missions(df)

print("\nTotal discharge missions:", len(missions))


# ============================================================
# 3. SELECT REFERENCE DISCHARGE MISSION
# ============================================================

reference_missions = [
    mission for mission in missions
    if mission["mission_type"] == 0
]

print("Reference missions:", len(reference_missions))

if not reference_missions:
    raise ValueError("No reference discharge missions found.")

# Select first reference mission
mission = reference_missions[0]

mission_id = mission["mission_id"]

mission_df = mission["data"].copy()

print("Selected mission ID:", mission_id)
print("Mission rows:", len(mission_df))


# ============================================================
# 4. CLEAN DATA
# ============================================================

mission_df["time"] = pd.to_numeric(
    mission_df["time"],
    errors="coerce"
)

mission_df["voltage_load"] = pd.to_numeric(
    mission_df["voltage_load"],
    errors="coerce"
)

mission_df["current_load"] = pd.to_numeric(
    mission_df["current_load"],
    errors="coerce"
)

mission_df = mission_df.dropna(
    subset=["time", "voltage_load", "current_load"]
)

mission_df = mission_df.sort_values("time")

mission_df = mission_df.drop_duplicates(
    subset="time",
    keep="first"
)

time = mission_df["time"].to_numpy(dtype=float)

voltage = mission_df["voltage_load"].to_numpy(dtype=float)

current = mission_df["current_load"].to_numpy(dtype=float)

# Discharge current magnitude
current_abs = np.abs(current)

if len(time) < 3:
    raise ValueError("Not enough valid samples.")

if np.any(np.diff(time) <= 0):
    raise ValueError("Time values must be strictly increasing.")


# ============================================================
# 5. CALCULATE DISCHARGE CAPACITY
# ============================================================

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


# ============================================================
# 6. MISSION INFORMATION
# ============================================================

print("\n========== MISSION INFORMATION ==========")

print("Mission ID:", mission_id)
print("Valid samples:", len(time))

print("Duration:",
      round(time[-1] - time[0], 2), "seconds")

print("Voltage range:",
      round(np.min(voltage), 4),
      "to",
      round(np.max(voltage), 4),
      "V")

print("Current range:",
      round(np.min(current), 4),
      "to",
      round(np.max(current), 4),
      "A")

print("Negative voltage samples:",
      np.sum(voltage < 0))

print("Negative current samples:",
      np.sum(current < 0))

print("Integrated discharge capacity:",
      round(total_capacity, 5), "Ah")


# ============================================================
# 7. VOLTAGE MONOTONICITY VALIDATION
# ============================================================

voltage_diff = np.diff(voltage)

decreases = np.sum(voltage_diff < 0)
increases = np.sum(voltage_diff > 0)
unchanged = np.sum(voltage_diff == 0)

total_changes = len(voltage_diff)

print("\n========== VOLTAGE VALIDATION ==========")

print("Voltage decreases:", decreases)
print("Voltage increases:", increases)
print("Voltage unchanged:", unchanged)

if total_changes > 0:

    print(
        "Percentage decreases:",
        round(decreases / total_changes * 100, 2),
        "%"
    )

    print(
        "Percentage increases:",
        round(increases / total_changes * 100, 2),
        "%"
    )

    print(
        "Percentage unchanged:",
        round(unchanged / total_changes * 100, 2),
        "%"
    )


# ============================================================
# 8. PREPARE VALID VOLTAGE-CAPACITY DATA
# ============================================================

valid_mask = (
    np.isfinite(voltage) &
    np.isfinite(capacity_ah) &
    (voltage > 0)
)

valid_voltage = voltage[valid_mask]

valid_capacity = capacity_ah[valid_mask]

if len(valid_voltage) < 30:
    raise ValueError("Insufficient valid voltage samples.")

# Remove duplicate voltage values by averaging capacity
original_df = pd.DataFrame({
    "voltage": valid_voltage,
    "capacity": valid_capacity
})

original_df = (
    original_df
    .groupby("voltage", as_index=False)["capacity"]
    .mean()
    .sort_values("voltage")
)

voltage_unique = original_df["voltage"].to_numpy()

capacity_unique = original_df["capacity"].to_numpy()

if len(voltage_unique) < 30:
    raise ValueError("Insufficient unique voltage values.")


# ============================================================
# 9. ORIGINAL ICA CALCULATION
# ============================================================

voltage_grid = np.arange(
    voltage_unique.min(),
    voltage_unique.max(),
    VOLTAGE_STEP
)

if len(voltage_grid) < SAVGOL_WINDOW:
    raise ValueError("Voltage grid is too small for smoothing.")

capacity_grid_original = np.interp(
    voltage_grid,
    voltage_unique,
    capacity_unique
)

capacity_smooth_original = savgol_filter(
    capacity_grid_original,
    SAVGOL_WINDOW,
    SAVGOL_POLYORDER
)

ica_original = np.abs(
    np.gradient(
        capacity_smooth_original,
        voltage_grid
    )
)

print("\n========== ORIGINAL ICA INFORMATION ==========")

print("Voltage grid points:", len(voltage_grid))

print("ICA minimum:",
      round(np.min(ica_original), 5))

print("ICA maximum:",
      round(np.max(ica_original), 5))


# ============================================================
# 10. ISOTONIC REGRESSION ICA
# ============================================================

# Sort samples by chronological capacity
sort_indices = np.argsort(capacity_ah)

capacity_sorted = capacity_ah[sort_indices]

voltage_sorted = voltage[sort_indices]

# Remove invalid voltage measurements
iso_mask = (
    np.isfinite(capacity_sorted) &
    np.isfinite(voltage_sorted) &
    (voltage_sorted > 0)
)

capacity_sorted = capacity_sorted[iso_mask]

voltage_sorted = voltage_sorted[iso_mask]

# Fit non-increasing voltage against capacity
isotonic_model = IsotonicRegression(
    increasing=False,
    out_of_bounds="clip"
)

voltage_isotonic = isotonic_model.fit_transform(
    capacity_sorted,
    voltage_sorted
)

# Invert relationship to obtain capacity against voltage
isotonic_df = pd.DataFrame({
    "voltage": voltage_isotonic,
    "capacity": capacity_sorted
})

isotonic_df = (
    isotonic_df
    .groupby("voltage", as_index=False)["capacity"]
    .mean()
    .sort_values("voltage")
)

iso_voltage_unique = isotonic_df["voltage"].to_numpy()

iso_capacity_unique = isotonic_df["capacity"].to_numpy()

voltage_grid_iso = np.arange(
    iso_voltage_unique.min(),
    iso_voltage_unique.max(),
    VOLTAGE_STEP
)

capacity_grid_iso = np.interp(
    voltage_grid_iso,
    iso_voltage_unique,
    iso_capacity_unique
)

capacity_smooth_iso = savgol_filter(
    capacity_grid_iso,
    SAVGOL_WINDOW,
    SAVGOL_POLYORDER
)

ica_isotonic = np.abs(
    np.gradient(
        capacity_smooth_iso,
        voltage_grid_iso
    )
)

print("\n========== ISOTONIC ICA INFORMATION ==========")

print("Voltage grid points:", len(voltage_grid_iso))

print("ICA minimum:",
      round(np.min(ica_isotonic), 5))

print("ICA maximum:",
      round(np.max(ica_isotonic), 5))


# ============================================================
# 11. COMPARE ICA CURVES
# ============================================================

print("\n========== ICA COMPARISON ==========")

print("Original unique voltages:", len(voltage_unique))

print("Isotonic unique voltages:", len(iso_voltage_unique))

# Compare only over their common voltage range
common_min = max(
    voltage_grid.min(),
    voltage_grid_iso.min()
)

common_max = min(
    voltage_grid.max(),
    voltage_grid_iso.max()
)

common_grid = np.arange(
    common_min,
    common_max,
    VOLTAGE_STEP
)

if len(common_grid) > 0:

    original_common = np.interp(
        common_grid,
        voltage_grid,
        ica_original
    )

    isotonic_common = np.interp(
        common_grid,
        voltage_grid_iso,
        ica_isotonic
    )

    curve_rmse = np.sqrt(
        np.mean(
            (original_common - isotonic_common) ** 2
        )
    )

    correlation = np.corrcoef(
        original_common,
        isotonic_common
    )[0, 1]

    print("ICA curve RMSE:", round(curve_rmse, 6))

    print("ICA curve correlation:", round(correlation, 5))


# ============================================================
# 12. ICA PEAK DETECTION
# ============================================================

peaks, properties = find_peaks(
    ica_original,
    prominence=PEAK_PROMINENCE,
    distance=PEAK_DISTANCE
)

print("\n========== ICA PEAK INFORMATION ==========")

print("Number of detected peaks:", len(peaks))

peak_features = {}

if len(peaks) > 0:

    # Strongest peak
    strongest_index = np.argmax(
        ica_original[peaks]
    )

    strongest_peak = peaks[strongest_index]

    strongest_voltage = voltage_grid[strongest_peak]

    strongest_height = ica_original[strongest_peak]

    # Peak width at half prominence
    widths, width_heights, left_ips, right_ips = peak_widths(
        ica_original,
        peaks,
        rel_height=0.5
    )

    voltage_spacing = voltage_grid[1] - voltage_grid[0]

    strongest_width = (
        widths[strongest_index] * voltage_spacing
    )

    # Approximate area around the strongest peak
    left_index = max(
        0,
        int(np.floor(left_ips[strongest_index]))
    )

    right_index = min(
        len(ica_original) - 1,
        int(np.ceil(right_ips[strongest_index]))
    )

    strongest_area = np.trapezoid(
        ica_original[left_index:right_index + 1],
        voltage_grid[left_index:right_index + 1]
    )

    peak_heights = ica_original[peaks]

    peak_features = {
        "ica_peak_count": len(peaks),
        "ica_peak_voltage": strongest_voltage,
        "ica_peak_height": strongest_height,
        "ica_peak_mean_height": np.mean(peak_heights),
        "ica_peak_std": np.std(peak_heights),
        "ica_peak_width": strongest_width,
        "ica_peak_area": strongest_area
    }

    print("Strongest peak voltage:",
          round(strongest_voltage, 5), "V")

    print("Strongest peak height:",
          round(strongest_height, 5), "Ah/V")

    print("Strongest peak width:",
          round(strongest_width, 5), "V")

    print("Strongest peak area:",
          round(strongest_area, 5), "Ah")

    print("\nAll detected peaks:")

    for i, peak in enumerate(peaks):

        print(
            f"Peak {i + 1}: "
            f"Voltage={voltage_grid[peak]:.4f} V, "
            f"Height={ica_original[peak]:.4f} Ah/V"
        )

else:

    print("No peaks detected. Try adjusting prominence.")


# ============================================================
# 13. OVERALL ICA STATISTICS
# ============================================================

print("\n========== ICA STATISTICS ==========")

print("Mean:", round(np.mean(ica_original), 5))

print("Standard deviation:", round(np.std(ica_original), 5))

print("Maximum:", round(np.max(ica_original), 5))

print("Minimum:", round(np.min(ica_original), 5))


# ============================================================
# 14. SAVE FEATURES
# ============================================================

feature_row = {
    "battery_id": "battery02",
    "mission_id": mission_id,
    "capacity_ah": total_capacity,
    "ica_mean": np.mean(ica_original),
    "ica_std": np.std(ica_original),
    "ica_max": np.max(ica_original),
    "ica_min": np.min(ica_original)
}

feature_row.update(peak_features)

feature_df = pd.DataFrame([feature_row])

results_dir = os.path.join(
    PROJECT_ROOT,
    "results"
)

os.makedirs(results_dir, exist_ok=True)

output_path = os.path.join(
    results_dir,
    "battery02_ica_features.csv"
)

feature_df.to_csv(
    output_path,
    index=False
)

print("\nICA features saved to:", output_path)


# ============================================================
# 15. VISUALIZATION 1: VOLTAGE VS CAPACITY
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    capacity_ah,
    voltage,
    label="Original voltage"
)

plt.plot(
    capacity_sorted,
    voltage_isotonic,
    label="Isotonic fitted voltage"
)

plt.xlabel("Discharge Capacity (Ah)")
plt.ylabel("Pack Voltage (V)")
plt.title("Voltage vs Discharge Capacity")
plt.legend()
plt.grid(True)


# ============================================================
# 16. VISUALIZATION 2: ORIGINAL VS ISOTONIC ICA
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    voltage_grid,
    ica_original,
    label="Original ICA"
)

plt.plot(
    voltage_grid_iso,
    ica_isotonic,
    label="Isotonic ICA",
    alpha=0.8
)

plt.xlabel("Pack Voltage (V)")
plt.ylabel("ICA (Ah/V)")
plt.title("Original ICA vs Isotonic ICA")
plt.legend()
plt.grid(True)


# ============================================================
# 17. VISUALIZATION 3: ICA PEAK DETECTION
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    voltage_grid,
    ica_original,
    label="Original ICA"
)

if len(peaks) > 0:

    plt.scatter(
        voltage_grid[peaks],
        ica_original[peaks],
        color="red",
        label="Detected peaks",
        zorder=3
    )

plt.xlabel("Pack Voltage (V)")
plt.ylabel("ICA (Ah/V)")
plt.title("ICA Peak Detection - battery02")
plt.legend()
plt.grid(True)


# ============================================================
# 18. VISUALIZATION 4: CURRENT VS TIME
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    current_abs
)

plt.xlabel("Time (seconds)")
plt.ylabel("Current (A)")
plt.title("Discharge Current vs Time")
plt.grid(True)


# ============================================================
# 19. SHOW ALL GRAPHS
# ============================================================

plt.tight_layout()

plt.show()
