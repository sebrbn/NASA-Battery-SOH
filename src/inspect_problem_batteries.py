
import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(BASE_DIR, "results")
CV_DIR = os.path.join(RESULTS_DIR, "cv")

FINAL_DATASET = os.path.join(
    RESULTS_DIR, "final_feature_dataset.csv"
)

AGING_FILE = os.path.join(
    RESULTS_DIR, "aging_features.csv"
)

SOH_FILE = os.path.join(
    RESULTS_DIR, "soh_labels.csv"
)

CAPACITY_FILE = os.path.join(
    RESULTS_DIR, "reference_capacities.csv"
)

TRAIN_FILE = os.path.join(
    RESULTS_DIR, "pca_train_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    CV_DIR, "problem_battery_diagnostics"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

PROBLEM_BATTERIES = [
    "battery54",
    "battery13",
    "battery50"
]


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

ICA_FEATURES = [
    "ica_mean",
    "ica_std",
    "ica_max",
    "ica_min",
    "ica_peak_count",
    "ica_peak_voltage",
    "ica_peak_height",
    "ica_peak_mean_height",
    "ica_peak_std",
    "ica_peak_width",
    "ica_peak_area",
    "voltage_min",
    "voltage_max"
]

STATISTICAL_FEATURES = [
    "voltage_mean",
    "voltage_std",
    "mean_current_a",
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

FEATURES = ICA_FEATURES + STATISTICAL_FEATURES


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):

    print("\n" + "=" * 75)
    print(title)
    print("=" * 75)


def save_csv(data, filename):

    path = os.path.join(OUTPUT_DIR, filename)

    data.to_csv(path, index=False)

    print("Saved:", path)


def safe_r2_values(values):

    if len(values) < 2:
        return np.nan

    return values.std()


def print_battery_data(df, battery_id):

    group = df[
        df["battery_id"] == battery_id
    ].sort_values("aging_ah")

    if group.empty:
        print(f"No data found for {battery_id}")
        return

    print(f"\n{battery_id}")

    print(
        group[
            [
                "mission_id",
                "aging_ah",
                "capacity_ah",
                "soh"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


# ============================================================
# LOAD DATASETS
# ============================================================

section("LOADING DATASETS")

final_df = pd.read_csv(FINAL_DATASET)

aging_df = pd.read_csv(AGING_FILE)

train_df = pd.read_csv(TRAIN_FILE)

print("Final dataset:", final_df.shape)
print("Aging dataset:", aging_df.shape)
print("Training dataset:", train_df.shape)

training_batteries = sorted(
    train_df["battery_id"].unique()
)

print("\nTraining batteries:", len(training_batteries))


# ============================================================
# MERGE AGING FEATURES
# ============================================================

section("MERGING DATASETS")

df = final_df.merge(
    aging_df[
        [
            "battery_id",
            "mission_id",
            "aging_ah",
            "capacity_ah",
            "mission_type",
            "start_time"
        ]
    ],
    on=["battery_id", "mission_id"],
    how="left",
    suffixes=("", "_aging"),
    validate="one_to_one"
)

# Use capacity from aging data if the final dataset
# does not already contain a capacity column.

if "capacity_ah" not in df.columns:

    df["capacity_ah"] = df["capacity_ah_aging"]

df = df[
    df["battery_id"].isin(training_batteries)
].copy()

print("Merged training dataset:", df.shape)

print(
    "Missing aging values:",
    df["aging_ah"].isna().sum()
)


# ============================================================
# ADD ORIGINAL REFERENCE CAPACITY INFORMATION
# ============================================================

section("REFERENCE CAPACITY DATA")

capacity_source = None

if os.path.exists(SOH_FILE):

    capacity_source = pd.read_csv(SOH_FILE)

    print("Using SOH labels:", capacity_source.shape)

elif os.path.exists(CAPACITY_FILE):

    capacity_source = pd.read_csv(CAPACITY_FILE)

    print("Using reference capacities:", capacity_source.shape)

else:

    print("No reference capacity file found.")


if capacity_source is not None:

    print(
        "Available columns:",
        capacity_source.columns.tolist()
    )

    if (
        "battery_id" in capacity_source.columns
        and "mission_id" in capacity_source.columns
        and "capacity_ah" in capacity_source.columns
    ):

        capacity_source = capacity_source[
            [
                "battery_id",
                "mission_id",
                "capacity_ah"
            ]
        ].drop_duplicates(
            ["battery_id", "mission_id"]
        )

        capacity_source = capacity_source.rename(
            columns={
                "capacity_ah": "reference_capacity_ah"
            }
        )

        df = df.merge(
            capacity_source,
            on=["battery_id", "mission_id"],
            how="left",
            validate="one_to_one"
        )

    else:

        print(
            "Capacity file does not contain the "
            "required columns. Skipping capacity merge."
        )


# ============================================================
# FILTER REFERENCE MISSIONS
# ============================================================

if "mission_type" in df.columns:

    reference_df = df[
        df["mission_type"] == 0
    ].copy()

else:

    reference_df = df.copy()

reference_df = reference_df.sort_values(
    ["battery_id", "aging_ah"]
)

print("\nReference mission records:", len(reference_df))


# ============================================================
# BATTERY-WISE CAPACITY AND SOH SUMMARY
# ============================================================

section("BATTERY CAPACITY AND SOH SUMMARY")

summary_rows = []

for battery_id, group in reference_df.groupby("battery_id"):

    group = group.sort_values("aging_ah")

    soh = group["soh"].dropna()

    capacity = group["capacity_ah"].dropna()

    first = group.iloc[0]

    summary_rows.append({

        "battery_id": battery_id,

        "samples": len(group),

        "initial_capacity_ah": first["capacity_ah"],

        "mean_capacity_ah": capacity.mean(),

        "min_capacity_ah": capacity.min(),

        "max_capacity_ah": capacity.max(),

        "final_capacity_ah": group.iloc[-1]["capacity_ah"],

        "initial_soh": first["soh"],

        "mean_soh": soh.mean(),

        "min_soh": soh.min(),

        "max_soh": soh.max(),

        "final_soh": group.iloc[-1]["soh"],

        "aging_end_ah": group["aging_ah"].max(),

        "soh_above_100_count": (soh > 100).sum(),

        "soh_below_100_count": (soh < 100).sum()

    })


capacity_summary = pd.DataFrame(summary_rows)

print(
    capacity_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

save_csv(
    capacity_summary,
    "battery_capacity_soh_summary.csv"
)


# ============================================================
# DETAILED CAPACITY TRAJECTORIES
# ============================================================

section("PROBLEM BATTERY CAPACITY TRAJECTORIES")

for battery_id in PROBLEM_BATTERIES:

    print_battery_data(
        reference_df,
        battery_id
    )


# ============================================================
# SOH MONOTONICITY CHECK
# ============================================================

section("SOH MONOTONICITY ANALYSIS")

monotonic_rows = []

for battery_id, group in reference_df.groupby("battery_id"):

    group = group.sort_values("aging_ah")

    soh = group["soh"].dropna().to_numpy()

    if len(soh) < 2:
        continue

    differences = np.diff(soh)

    increases = np.sum(differences > 0)

    decreases = np.sum(differences < 0)

    unchanged = np.sum(differences == 0)

    monotonic_rows.append({

        "battery_id": battery_id,

        "samples": len(soh),

        "soh_increases": increases,

        "soh_decreases": decreases,

        "soh_unchanged": unchanged,

        "total_soh_increase": differences[
            differences > 0
        ].sum(),

        "total_soh_decrease": abs(
            differences[differences < 0].sum()
        ),

        "largest_soh_increase": (
            differences.max()
            if len(differences) else 0
        ),

        "largest_soh_decrease": (
            differences.min()
            if len(differences) else 0
        )

    })


monotonic_df = pd.DataFrame(monotonic_rows)

print(
    monotonic_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

save_csv(
    monotonic_df,
    "soh_monotonicity_analysis.csv"
)


# ============================================================
# INITIAL CAPACITY COMPARISON
# ============================================================

section("INITIAL CAPACITY COMPARISON")

initial_rows = []

for battery_id, group in reference_df.groupby("battery_id"):

    group = group.sort_values("aging_ah")

    first = group.iloc[0]

    initial_rows.append({

        "battery_id": battery_id,

        "initial_capacity_ah": first["capacity_ah"],

        "initial_soh": first["soh"],

        "initial_aging_ah": first["aging_ah"]

    })


initial_df = pd.DataFrame(initial_rows)

initial_mean = initial_df[
    "initial_capacity_ah"
].mean()

initial_std = initial_df[
    "initial_capacity_ah"
].std()

initial_df["capacity_z_score"] = (
    initial_df["initial_capacity_ah"] - initial_mean
) / initial_std

initial_df = initial_df.sort_values(
    "capacity_z_score"
)

print(
    initial_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

save_csv(
    initial_df,
    "initial_capacity_comparison.csv"
)


# ============================================================
# FEATURE DISTRIBUTION ANALYSIS
# ============================================================

section("FEATURE DISTRIBUTION ANALYSIS")

available_features = [
    feature
    for feature in FEATURES
    if feature in df.columns
]

print("Available features:", len(available_features))

print("\nMissing features:")

print(
    sorted(set(FEATURES) - set(available_features))
)


# Battery-level mean feature values.
# This avoids treating every mission as an independent battery.

battery_feature_means = (
    df.groupby("battery_id")[available_features]
    .mean()
)

feature_deviation_rows = []

for battery_id in PROBLEM_BATTERIES:

    if battery_id not in battery_feature_means.index:
        continue

    other_batteries = battery_feature_means.drop(
        index=battery_id
    )

    target_values = battery_feature_means.loc[
        battery_id
    ]

    peer_mean = other_batteries.mean()

    peer_std = other_batteries.std(ddof=1)

    for feature in available_features:

        if (
            pd.isna(peer_std[feature])
            or peer_std[feature] == 0
        ):
            z_score = np.nan

        else:

            z_score = (
                target_values[feature] - peer_mean[feature]
            ) / peer_std[feature]

        feature_deviation_rows.append({

            "battery_id": battery_id,

            "feature": feature,

            "battery_mean": target_values[feature],

            "other_battery_mean": peer_mean[feature],

            "other_battery_std": peer_std[feature],

            "z_score": z_score,

            "absolute_z_score": abs(z_score)

            if not pd.isna(z_score)
            else np.nan

        })


feature_deviations = pd.DataFrame(
    feature_deviation_rows
)

feature_deviations = feature_deviations.sort_values(
    ["battery_id", "absolute_z_score"],
    ascending=[True, False]
)

for battery_id in PROBLEM_BATTERIES:

    print("\n" + "-" * 65)

    print(f"FEATURE DEVIATIONS: {battery_id}")

    print("-" * 65)

    group = feature_deviations[
        feature_deviations["battery_id"] == battery_id
    ].head(10)

    print(
        group[
            [
                "feature",
                "battery_mean",
                "other_battery_mean",
                "z_score"
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )


save_csv(
    feature_deviations,
    "problem_battery_feature_deviations.csv"
)


# ============================================================
# BATTERY FEATURE SUMMARY
# ============================================================

section("FEATURE SUMMARY FOR PROBLEM BATTERIES")

feature_summary_rows = []

for battery_id in PROBLEM_BATTERIES:

    group = df[
        df["battery_id"] == battery_id
    ]

    if group.empty:
        continue

    for feature in available_features:

        values = group[feature].dropna()

        if values.empty:
            continue

        feature_summary_rows.append({

            "battery_id": battery_id,

            "feature": feature,

            "mean": values.mean(),

            "std": values.std(),

            "min": values.min(),

            "max": values.max()

        })


feature_summary = pd.DataFrame(
    feature_summary_rows
)

save_csv(
    feature_summary,
    "problem_battery_feature_summary.csv"
)


# ============================================================
# CAPACITY VS SOH RELATIONSHIP
# ============================================================

section("CAPACITY AND SOH RELATIONSHIP")

if "reference_capacity_ah" in reference_df.columns:

    capacity_check = reference_df[
        [
            "battery_id",
            "mission_id",
            "capacity_ah",
            "reference_capacity_ah",
            "soh"
        ]
    ].copy()

    capacity_check["capacity_difference"] = (
        capacity_check["capacity_ah"]
        - capacity_check["reference_capacity_ah"]
    )

    print(
        capacity_check[
            capacity_check["battery_id"].isin(
                PROBLEM_BATTERIES
            )
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.5f}"
        )
    )

    save_csv(
        capacity_check,
        "capacity_source_comparison.csv"
    )

else:

    print(
        "Original reference capacity could not be merged. "
        "Capacity source comparison skipped."
    )


# ============================================================
# FINAL REPORT
# ============================================================

section("DIAGNOSTIC SUMMARY")

print("Training batteries:", len(training_batteries))

print("Problem batteries:", PROBLEM_BATTERIES)

print("\nCapacity summary:")

print(
    capacity_summary[
        capacity_summary["battery_id"].isin(
            PROBLEM_BATTERIES
        )
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nInitial capacity comparison:")

print(
    initial_df[
        initial_df["battery_id"].isin(
            PROBLEM_BATTERIES
        )
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nDiagnostic files saved in:")

print(OUTPUT_DIR)

print("\nDIAGNOSTICS COMPLETED")
