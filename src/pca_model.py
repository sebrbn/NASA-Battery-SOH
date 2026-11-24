
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupShuffleSplit

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "results" / "final_feature_dataset.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VARIANCE_THRESHOLD = 0.95

# ==========================================
# 1. LOAD DATASET
# ==========================================

print("=" * 55)
print("       PCA FEATURE REDUCTION")
print("=" * 55)

df = pd.read_csv(DATA_PATH)

print("\nDataset shape:", df.shape)

# ==========================================
# 2. DEFINE FEATURE COLUMNS
# ==========================================

ica_features = [
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

statistical_features = [
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

target = "soh"
group = "battery_id"

# Check required columns
required_columns = (
    ica_features +
    statistical_features +
    [target, group]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# ==========================================
# 3. TRAIN-TEST SPLIT BY BATTERY
# ==========================================

X = df[ica_features].copy()
y = df[target].copy()
groups = df[group].copy()

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

train_groups = groups.iloc[train_idx]
test_groups = groups.iloc[test_idx]

print("\nTRAIN-TEST SPLIT")
print("-" * 40)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("Training batteries:", train_groups.nunique())
print("Testing batteries:", test_groups.nunique())

print(
    "Battery overlap:",
    set(train_groups).intersection(set(test_groups))
)

# ==========================================
# 4. STANDARDIZATION
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStandardization completed.")

# ==========================================
# 5. PCA
# ==========================================

pca = PCA()

X_train_pca_all = pca.fit_transform(X_train_scaled)

cumulative_variance = np.cumsum(
    pca.explained_variance_ratio_
)

n_components = (
    np.argmax(cumulative_variance >= VARIANCE_THRESHOLD) + 1
)

print("\nPCA RESULTS")
print("-" * 40)

print("Original ICA features:", len(ica_features))
print("Selected components:", n_components)

print("\nExplained variance:")

for i, variance in enumerate(
    pca.explained_variance_ratio_,
    start=1
):
    print(
        f"PC{i}: {variance:.4f} "
        f"(Cumulative: {cumulative_variance[i-1]:.4f})"
    )

# Refit PCA with selected components
pca = PCA(n_components=n_components)

X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# ==========================================
# 6. CREATE PCA DATAFRAMES
# ==========================================

pca_columns = [
    f"PC{i+1}"
    for i in range(n_components)
]

train_pca_df = pd.DataFrame(
    X_train_pca,
    columns=pca_columns,
    index=X_train.index
)

test_pca_df = pd.DataFrame(
    X_test_pca,
    columns=pca_columns,
    index=X_test.index
)

# ==========================================
# 7. COMBINE PCA + STATISTICAL FEATURES
# ==========================================

train_stat = df.loc[
    X_train.index, statistical_features
]

test_stat = df.loc[
    X_test.index, statistical_features
]

train_final = pd.concat(
    [train_pca_df, train_stat],
    axis=1
)

test_final = pd.concat(
    [test_pca_df, test_stat],
    axis=1
)

train_final["battery_id"] = groups.loc[X_train.index]
train_final["mission_id"] = df.loc[X_train.index, "mission_id"]
train_final["soh"] = y_train

test_final["battery_id"] = groups.loc[X_test.index]
test_final["mission_id"] = df.loc[X_test.index, "mission_id"]
test_final["soh"] = y_test

# ==========================================
# 8. SAVE RESULTS
# ==========================================

train_final.to_csv(
    RESULTS_DIR / "pca_train_dataset.csv",
    index=False
)

test_final.to_csv(
    RESULTS_DIR / "pca_test_dataset.csv",
    index=False
)

joblib.dump(
    scaler,
    MODELS_DIR / "ica_scaler.pkl"
)

joblib.dump(
    pca,
    MODELS_DIR / "ica_pca.pkl"
)

# ==========================================
# 9. FINAL SUMMARY
# ==========================================

print("\nFINAL FEATURE DATASETS")
print("-" * 40)

print("Training shape:", train_final.shape)
print("Testing shape:", test_final.shape)

print(
    "Final model input features:",
    len(pca_columns) + len(statistical_features)
)

print("\nTraining dataset saved.")
print("Testing dataset saved.")
print("Scaler and PCA models saved.")

print("\nPCA FEATURE REDUCTION COMPLETED.")
