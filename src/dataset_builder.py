
import pandas as pd
from pathlib import Path

# ==========================================
# PATHS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"

ICA_FILE = RESULTS_DIR / "ica_features_all.csv"
STAT_FILE = RESULTS_DIR / "statistical_features_all.csv"
SOH_FILE = RESULTS_DIR / "soh_labels.csv"

OUTPUT_FILE = RESULTS_DIR / "final_feature_dataset.csv"
UNMATCHED_FILE = RESULTS_DIR / "unmatched_missions.csv"

KEYS = ["battery_id", "mission_id"]

# Columns used for identification or validation,
# not as model input features.
METADATA_COLUMNS = [
    "battery_id",
    "mission_id",
    "capacity_ah",
    "duration_seconds",
    "samples",
    "voltage_start",
    "voltage_end",
    "current_mean",
    "current_start",
    "current_end",
]

# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    print("\nLoading feature datasets...")

    ica = pd.read_csv(ICA_FILE)
    stats = pd.read_csv(STAT_FILE)
    soh = pd.read_csv(SOH_FILE)

    print("\nICA dataset shape:", ica.shape)
    print("Statistical dataset shape:", stats.shape)
    print("SOH dataset shape:", soh.shape)

    print("\nSOH columns:")
    print(soh.columns.tolist())

    return ica, stats, soh


# ==========================================
# VALIDATE DATA
# ==========================================

def validate_keys(df, name):

    missing_keys = [
        column for column in KEYS
        if column not in df.columns
    ]

    if missing_keys:
        raise ValueError(
            f"{name} is missing key columns: {missing_keys}"
        )

    if df[KEYS].isnull().any().any():
        raise ValueError(
            f"{name} contains missing mission identifiers."
        )

    duplicates = df[df.duplicated(KEYS, keep=False)]

    if not duplicates.empty:
        print(f"\nDuplicate keys in {name}:")
        print(duplicates[KEYS].head(10))

        raise ValueError(
            f"{name} contains duplicate battery/mission keys."
        )

    print(f"{name}: keys validated successfully.")


# ==========================================
# IDENTIFY SOH TARGET
# ==========================================

def identify_soh_column(soh):

    possible_targets = [
        "soh",
        "soh_percent",
        "soh_percentage",
        "SOH",
        "SOH_percent",
    ]

    for column in possible_targets:
        if column in soh.columns:
            print("\nSelected SOH target:", column)
            return column

    raise ValueError(
        "Could not identify the SOH target column. "
        f"Available columns: {soh.columns.tolist()}"
    )


# ==========================================
# SELECT MODEL FEATURES
# ==========================================

def select_features(df, name):

    excluded = set(KEYS + METADATA_COLUMNS)

    feature_columns = [
        column for column in df.columns
        if column not in excluded
    ]

    if not feature_columns:
        raise ValueError(
            f"No feature columns found in {name}."
        )

    print(f"\n{name} feature count:", len(feature_columns))
    print(feature_columns)

    return df[KEYS + feature_columns]


# ==========================================
# COMPARE MISSION IDENTIFIERS
# ==========================================

def check_matching_missions(ica, stats, soh):

    ica_keys = set(map(tuple, ica[KEYS].to_numpy()))
    stats_keys = set(map(tuple, stats[KEYS].to_numpy()))
    soh_keys = set(map(tuple, soh[KEYS].to_numpy()))

    print("\n====================================")
    print("MISSION KEY VALIDATION")
    print("====================================")

    print("ICA missions:", len(ica_keys))
    print("Statistical missions:", len(stats_keys))
    print("SOH missions:", len(soh_keys))

    print(
        "ICA vs Statistical matches:",
        len(ica_keys & stats_keys)
    )

    print(
        "ICA vs SOH matches:",
        len(ica_keys & soh_keys)
    )

    print(
        "Statistical vs SOH matches:",
        len(stats_keys & soh_keys)
    )

    all_keys = ica_keys | stats_keys | soh_keys
    common_keys = ica_keys & stats_keys & soh_keys

    unmatched = all_keys - common_keys

    if unmatched:

        unmatched_df = pd.DataFrame(
            list(unmatched),
            columns=KEYS
        )

        unmatched_df.to_csv(
            UNMATCHED_FILE,
            index=False
        )

        print("\nWARNING: Some missions do not match.")
        print("Unmatched missions:", len(unmatched))
        print("Saved:", UNMATCHED_FILE)

        raise ValueError(
            "Mission key mismatch. Inspect unmatched_missions.csv "
            "before building the final dataset."
        )

    print("\nAll mission keys match successfully.")


# ==========================================
# BUILD FINAL DATASET
# ==========================================

def build_dataset(ica, stats, soh):

    target_column = identify_soh_column(soh)

    # Validate mission identifiers
    validate_keys(ica, "ICA dataset")
    validate_keys(stats, "Statistical dataset")
    validate_keys(soh, "SOH dataset")

    # Check all three datasets contain the same missions
    check_matching_missions(ica, stats, soh)

    # Select model input features
    ica_features = select_features(ica, "ICA")
    stat_features = select_features(stats, "Statistical")

    # Keep only identifiers and target from SOH labels
    soh_target = soh[KEYS + [target_column]].copy()

    # Rename target to a consistent name
    soh_target = soh_target.rename(
        columns={target_column: "soh"}
    )

    # Merge using mission identifiers
    print("\nMerging datasets...")

    final_df = pd.merge(
        ica_features,
        stat_features,
        on=KEYS,
        how="inner",
        validate="one_to_one"
    )

    final_df = pd.merge(
        final_df,
        soh_target,
        on=KEYS,
        how="inner",
        validate="one_to_one"
    )

    # Sort for readability
    final_df = final_df.sort_values(
        ["battery_id", "mission_id"]
    ).reset_index(drop=True)

    return final_df


# ==========================================
# FINAL VALIDATION
# ==========================================

def validate_final_dataset(df):

    print("\n====================================")
    print("FINAL DATASET VALIDATION")
    print("====================================")

    print("Final shape:", df.shape)

    print("Number of batteries:",
          df["battery_id"].nunique())

    print("Number of missions:", len(df))

    print("Number of input features:",
          len([
              col for col in df.columns
              if col not in KEYS + ["soh"]
          ]))

    print("\nMissing values:")
    print(df.isnull().sum())

    if df.isnull().any().any():
        raise ValueError(
            "Final dataset contains missing values. "
            "Investigate them before modelling."
        )

    if not df["soh"].between(0, 150).all():
        raise ValueError(
            "SOH contains unexpected values. "
            "Inspect the SOH labels."
        )

    if df.duplicated(KEYS).any():
        raise ValueError(
            "Duplicate missions found in final dataset."
        )

    print("\nSOH statistics:")
    print(df["soh"].describe())

    print("\nFinal dataset preview:")
    print(df.head())


# ==========================================
# MAIN
# ==========================================

def main():

    print("====================================")
    print("       DATASET BUILDER")
    print("====================================")

    ica, stats, soh = load_data()

    final_df = build_dataset(
        ica,
        stats,
        soh
    )

    validate_final_dataset(final_df)

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n====================================")
    print("DATASET BUILDING COMPLETED")
    print("====================================")

    print("\nFinal dataset saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
