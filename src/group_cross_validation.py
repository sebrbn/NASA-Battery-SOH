
import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
N_SPLITS = 5

MAX_EPOCHS = 1000
PATIENCE = 100

LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PHYSICS_WEIGHT = 0.1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs("results/cv", exist_ok=True)
os.makedirs("models/cv", exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 2. FEATURE DEFINITIONS
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

STAT_FEATURES = [
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

TARGET = "soh"
GROUP = "battery_id"
AGE_FEATURE = "aging_ah"

META = [
    "battery_id",
    "mission_id"
]


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 65)
print("FOLD-SPECIFIC PCA + PINN CROSS VALIDATION")
print("=" * 65)

features = pd.read_csv(
    "results/final_feature_dataset.csv"
)

aging = pd.read_csv(
    "results/aging_features.csv"
)

train_reference = pd.read_csv(
    "results/pca_train_dataset.csv"
)

# Merge aging feature.
df = features.merge(
    aging[META + [AGE_FEATURE]],
    on=META,
    how="inner",
    validate="one_to_one"
)

if len(df) != len(features):
    raise ValueError(
        "Some feature records did not match aging records."
    )

# Check required columns.
required_columns = (
    ICA_FEATURES
    + STAT_FEATURES
    + [TARGET, AGE_FEATURE, GROUP, "mission_id"]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

if df[required_columns].isnull().any().any():
    raise ValueError(
        "Missing values found in model dataset."
    )

# Preserve original 20-battery training split.
train_batteries = sorted(
    train_reference[GROUP].unique()
)

cv_df = df[
    df[GROUP].isin(train_batteries)
].copy()

cv_df = cv_df.reset_index(drop=True)

print("\nDataset shape:", cv_df.shape)
print("Training batteries:", len(train_batteries))
print("Records:", len(cv_df))
print("Device:", DEVICE)

# Save the battery split for reproducibility.
pd.DataFrame({
    "training_battery": train_batteries
}).to_csv(
    "results/cv/cv_training_batteries.csv",
    index=False
)


# ============================================================
# 4. PINN ARCHITECTURE
# ============================================================

class PINN(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.Tanh(),

            nn.Linear(64, 32),
            nn.Tanh(),

            nn.Linear(32, 16),
            nn.Tanh(),

            nn.Linear(16, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 5. PHYSICS LOSS
# ============================================================

def physics_loss(model, x, aging_index):

    x_physics = (
        x.detach()
        .clone()
        .requires_grad_(True)
    )

    prediction = model(x_physics)

    gradients = torch.autograd.grad(
        outputs=prediction.sum(),
        inputs=x_physics,
        create_graph=True
    )[0]

    aging_gradient = gradients[:, aging_index]

    # Penalize positive SOH derivative with respect
    # to cumulative discharge throughput.
    penalty = torch.relu(aging_gradient) ** 2

    return penalty.mean()


# ============================================================
# 6. FOLD-SPECIFIC PCA
# ============================================================

def fit_preprocessor(train_df):

    # Fit ICA scaler only on fold training batteries.
    ica_scaler = StandardScaler()

    ica_train = ica_scaler.fit_transform(
        train_df[ICA_FEATURES]
    )

    # Determine PCA components using training data.
    full_pca = PCA()

    full_pca.fit(ica_train)

    cumulative_variance = np.cumsum(
        full_pca.explained_variance_ratio_
    )

    n_components = (
        np.searchsorted(
            cumulative_variance,
            0.95
        ) + 1
    )

    # Fit final PCA.
    pca = PCA(
        n_components=n_components
    )

    pca_train = pca.fit_transform(
        ica_train
    )

    return ica_scaler, pca, pca_train


def transform_features(
    data,
    ica_scaler,
    pca
):

    # ICA features.
    ica_scaled = ica_scaler.transform(
        data[ICA_FEATURES]
    )

    pca_scores = pca.transform(
        ica_scaled
    )

    # Statistical features.
    statistical = data[
        STAT_FEATURES
    ].to_numpy(dtype=np.float32)

    # Aging feature.
    aging = data[
        [AGE_FEATURE]
    ].to_numpy(dtype=np.float32)

    X = np.concatenate(
        [
            pca_scores,
            statistical,
            aging
        ],
        axis=1
    )

    return X.astype(np.float32)


# ============================================================
# 7. PINN TRAINING
# ============================================================

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=None
):

    # --------------------------------------------------------
    # INPUT SCALING
    # --------------------------------------------------------

    input_scaler = StandardScaler()

    X_train_scaled = input_scaler.fit_transform(
        X_train
    )

    # --------------------------------------------------------
    # TARGET SCALING
    # --------------------------------------------------------

    target_scaler = StandardScaler()

    y_train_scaled = target_scaler.fit_transform(
        y_train.reshape(-1, 1)
    )

    # --------------------------------------------------------
    # TENSORS
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    y_train_tensor = torch.tensor(
        y_train_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    if X_val is not None:

        X_val_scaled = input_scaler.transform(
            X_val
        )

        y_val_scaled = target_scaler.transform(
            y_val.reshape(-1, 1)
        )

        X_val_tensor = torch.tensor(
            X_val_scaled,
            dtype=torch.float32,
            device=DEVICE
        )

        y_val_tensor = torch.tensor(
            y_val_scaled,
            dtype=torch.float32,
            device=DEVICE
        )

    else:

        X_val_tensor = None
        y_val_tensor = None

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = PINN(
        X_train.shape[1]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    aging_index = X_train.shape[1] - 1

    best_loss = float("inf")
    best_epoch = MAX_EPOCHS
    best_state = None

    patience_counter = 0

    total_epochs = (
        epochs if epochs is not None
        else MAX_EPOCHS
    )

    last_data_loss = 0.0
    last_physics_loss = 0.0

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(total_epochs):

        model.train()

        optimizer.zero_grad()

        prediction = model(
            X_train_tensor
        )

        data_loss = nn.functional.mse_loss(
            prediction,
            y_train_tensor
        )

        p_loss = physics_loss(
            model,
            X_train_tensor,
            aging_index
        )

        loss = (
            data_loss
            + PHYSICS_WEIGHT * p_loss
        )

        loss.backward()

        optimizer.step()

        last_data_loss = data_loss.item()
        last_physics_loss = p_loss.item()

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if X_val_tensor is not None:

            model.eval()

            with torch.no_grad():

                val_prediction = model(
                    X_val_tensor
                )

                val_loss = nn.functional.mse_loss(
                    val_prediction,
                    y_val_tensor
                ).item()

            if val_loss < best_loss:

                best_loss = val_loss

                best_epoch = epoch + 1

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value
                    in model.state_dict().items()
                }

                patience_counter = 0

            else:

                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

    # Restore best validation model.
    if X_val_tensor is not None:

        if best_state is None:
            raise RuntimeError(
                "No valid model checkpoint was selected."
            )

        model.load_state_dict(
            best_state
        )

    return (
        model,
        input_scaler,
        target_scaler,
        best_epoch,
        best_loss,
        last_data_loss,
        last_physics_loss
    )


# ============================================================
# 8. PREDICTION
# ============================================================

def predict(
    model,
    input_scaler,
    target_scaler,
    X
):

    X_scaled = input_scaler.transform(
        X
    )

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32,
        device=DEVICE
    )

    model.eval()

    with torch.no_grad():

        predictions = model(
            X_tensor
        ).cpu().numpy()

    # Convert predictions back to original SOH scale.
    predictions = target_scaler.inverse_transform(
        predictions
    ).flatten()

    return predictions


# ============================================================
# 9. FIVE-FOLD GROUP CROSS VALIDATION
# ============================================================

group_kfold = GroupKFold(
    n_splits=N_SPLITS
)

all_predictions = []

fold_summaries = []

for fold_number, (
    train_idx,
    val_idx
) in enumerate(
    group_kfold.split(
        cv_df,
        groups=cv_df[GROUP]
    ),
    start=1
):

    print("\n" + "=" * 65)
    print(f"FOLD {fold_number}/{N_SPLITS}")
    print("=" * 65)

    outer_train = (
        cv_df.iloc[train_idx].copy()
    )

    outer_val = (
        cv_df.iloc[val_idx].copy()
    )

    print(
        "Training batteries:",
        sorted(outer_train[GROUP].unique())
    )

    print(
        "Validation batteries:",
        sorted(outer_val[GROUP].unique())
    )

    # --------------------------------------------------------
    # INNER BATTERY-WISE VALIDATION SPLIT
    # --------------------------------------------------------

    inner_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED + fold_number
    )

    inner_train_idx, inner_val_idx = next(
        inner_split.split(
            outer_train,
            groups=outer_train[GROUP]
        )
    )

    inner_train = (
        outer_train.iloc[inner_train_idx].copy()
    )

    inner_val = (
        outer_train.iloc[inner_val_idx].copy()
    )

    # --------------------------------------------------------
    # INNER PREPROCESSING
    # --------------------------------------------------------

    (
        inner_ica_scaler,
        inner_pca,
        _
    ) = fit_preprocessor(
        inner_train
    )

    X_inner_train = transform_features(
        inner_train,
        inner_ica_scaler,
        inner_pca
    )

    X_inner_val = transform_features(
        inner_val,
        inner_ica_scaler,
        inner_pca
    )

    y_inner_train = inner_train[
        TARGET
    ].to_numpy(dtype=np.float32)

    y_inner_val = inner_val[
        TARGET
    ].to_numpy(dtype=np.float32)

    # --------------------------------------------------------
    # SELECT EPOCHS
    # --------------------------------------------------------

    set_seed(SEED + fold_number)

    (
        _,
        _,
        _,
        best_epoch,
        best_val_loss,
        inner_data_loss,
        inner_physics_loss
    ) = train_model(
        X_inner_train,
        y_inner_train,
        X_inner_val,
        y_inner_val
    )

    print(
        "Inner PCA components:",
        inner_pca.n_components_
    )

    print(
        "Selected training epochs:",
        best_epoch
    )

    print(
        "Inner validation MSE:",
        round(best_val_loss, 6)
    )

    print(
        "Last training data loss:",
        round(inner_data_loss, 6)
    )

    print(
        "Last training physics loss:",
        round(inner_physics_loss, 8)
    )

    # --------------------------------------------------------
    # REFIT PCA ON COMPLETE OUTER TRAINING GROUPS
    # --------------------------------------------------------

    (
        fold_ica_scaler,
        fold_pca,
        _
    ) = fit_preprocessor(
        outer_train
    )

    X_outer_train = transform_features(
        outer_train,
        fold_ica_scaler,
        fold_pca
    )

    X_outer_val = transform_features(
        outer_val,
        fold_ica_scaler,
        fold_pca
    )

    y_outer_train = outer_train[
        TARGET
    ].to_numpy(dtype=np.float32)

    # --------------------------------------------------------
    # TRAIN FINAL MODEL FOR THIS FOLD
    # --------------------------------------------------------

    set_seed(SEED + 100 + fold_number)

    (
        fold_model,
        fold_input_scaler,
        fold_target_scaler,
        _,
        _,
        final_data_loss,
        final_physics_loss
    ) = train_model(
        X_outer_train,
        y_outer_train,
        None,
        None,
        epochs=best_epoch
    )

    print(
        "Final fold data loss:",
        round(final_data_loss, 6)
    )

    print(
        "Final fold physics loss:",
        round(final_physics_loss, 8)
    )

    # --------------------------------------------------------
    # VALIDATION PREDICTIONS
    # --------------------------------------------------------

    fold_predictions = predict(
        fold_model,
        fold_input_scaler,
        fold_target_scaler,
        X_outer_val
    )

    fold_result = outer_val[
        META + [TARGET]
    ].copy()

    fold_result["predicted_soh"] = (
        fold_predictions
    )

    fold_result["absolute_error"] = np.abs(
        fold_result[TARGET]
        - fold_result["predicted_soh"]
    )

    fold_result["fold"] = fold_number

    fold_result["pca_components"] = (
        fold_pca.n_components_
    )

    all_predictions.append(
        fold_result
    )

    fold_mae = mean_absolute_error(
        fold_result[TARGET],
        fold_predictions
    )

    fold_rmse = np.sqrt(
        mean_squared_error(
            fold_result[TARGET],
            fold_predictions
        )
    )

    fold_r2 = r2_score(
        fold_result[TARGET],
        fold_predictions
    )

    print("\nFold results:")
    print("MAE:", round(fold_mae, 4))
    print("RMSE:", round(fold_rmse, 4))
    print("R2:", round(fold_r2, 4))

    fold_summaries.append({
        "fold": fold_number,
        "training_batteries": len(
            outer_train[GROUP].unique()
        ),
        "validation_batteries": len(
            outer_val[GROUP].unique()
        ),
        "pca_components": fold_pca.n_components_,
        "best_epoch": best_epoch,
        "inner_validation_mse": best_val_loss,
        "MAE": fold_mae,
        "RMSE": fold_rmse,
        "R2": fold_r2
    })

    # --------------------------------------------------------
    # SAVE FOLD MODEL AND PREPROCESSORS
    # --------------------------------------------------------

    torch.save(
        fold_model.state_dict(),
        f"models/cv/fold_{fold_number}_pinn.pth"
    )

    joblib.dump(
        fold_input_scaler,
        f"models/cv/fold_{fold_number}_input_scaler.pkl"
    )

    joblib.dump(
        fold_target_scaler,
        f"models/cv/fold_{fold_number}_target_scaler.pkl"
    )

    joblib.dump(
        fold_ica_scaler,
        f"models/cv/fold_{fold_number}_ica_scaler.pkl"
    )

    joblib.dump(
        fold_pca,
        f"models/cv/fold_{fold_number}_pca.pkl"
    )


# ============================================================
# 10. OVERALL CROSS VALIDATION RESULTS
# ============================================================

oof = pd.concat(
    all_predictions,
    ignore_index=True
)

y_true = oof[TARGET].to_numpy()

y_pred = oof[
    "predicted_soh"
].to_numpy()

overall_mae = mean_absolute_error(
    y_true,
    y_pred
)

overall_rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

overall_r2 = r2_score(
    y_true,
    y_pred
)

print("\n" + "=" * 65)
print("OVERALL GROUP CROSS-VALIDATION RESULTS")
print("=" * 65)

print("OOF records:", len(oof))
print("Batteries:", oof[GROUP].nunique())

print("MAE:", round(overall_mae, 4))
print("RMSE:", round(overall_rmse, 4))
print("R2:", round(overall_r2, 4))


# ============================================================
# 11. BATTERY-WISE METRICS
# ============================================================

battery_metrics = []

for battery_id, group in oof.groupby(GROUP):

    actual = group[
        TARGET
    ].to_numpy()

    predicted = group[
        "predicted_soh"
    ].to_numpy()

    battery_metrics.append({

        "battery_id": battery_id,

        "samples": len(group),

        "MAE": mean_absolute_error(
            actual,
            predicted
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                actual,
                predicted
            )
        ),

        "R2": (
            r2_score(actual, predicted)
            if len(group) >= 2
            else np.nan
        )
    })

battery_metrics = pd.DataFrame(
    battery_metrics
)


# ============================================================
# 12. SAVE RESULTS
# ============================================================

oof.to_csv(
    "results/cv/pinn_group_oof_predictions.csv",
    index=False
)

battery_metrics.to_csv(
    "results/cv/pinn_group_battery_metrics.csv",
    index=False
)

fold_summary = pd.DataFrame(
    fold_summaries
)

fold_summary.to_csv(
    "results/cv/pinn_group_fold_metrics.csv",
    index=False
)


# ============================================================
# 13. PRINT BATTERY-WISE RESULTS
# ============================================================

print("\nBATTERY-WISE RESULTS")

print(
    battery_metrics.round(4).to_string(
        index=False
    )
)

print("\nFOLD-WISE SUMMARY")

print(
    fold_summary.round(4).to_string(
        index=False
    )
)


# ============================================================
# 14. PREDICTION RANGE CHECK
# ============================================================

print("\nPREDICTION DIAGNOSTICS")

print(
    "Actual SOH range:",
    round(y_true.min(), 3),
    "to",
    round(y_true.max(), 3)
)

print(
    "Predicted SOH range:",
    round(y_pred.min(), 3),
    "to",
    round(y_pred.max(), 3)
)

print(
    "Actual SOH mean:",
    round(y_true.mean(), 3)
)

print(
    "Predicted SOH mean:",
    round(y_pred.mean(), 3)
)


# ============================================================
# 15. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("CROSS VALIDATION COMPLETED")
print("=" * 65)

print("\nSaved files:")

print(
    "results/cv/pinn_group_oof_predictions.csv"
)

print(
    "results/cv/pinn_group_battery_metrics.csv"
)

print(
    "results/cv/pinn_group_fold_metrics.csv"
)

print(
    "models/cv/"
)

print("\nExisting project files were not modified.")
