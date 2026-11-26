
import os
import random
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "results/pca_train_dataset.csv"
TEST_PATH = "results/pca_test_dataset.csv"
AGING_PATH = "results/aging_features.csv"

PINN_MODEL_PATH = "models/pinn_model.pth"
PINN_SCALER_PATH = "models/pinn_input_scaler.pkl"
PINN_METADATA_PATH = "models/pinn_metadata.pkl"

GPR_MODEL_PATH = "models/gpr_residual_model.pkl"
GPR_SCALER_PATH = "models/gpr_input_scaler.pkl"

OOF_PATH = "results/pinn_oof_train_predictions.csv"
PREDICTION_PATH = "results/pinn_gpr_test_predictions.csv"
METRICS_PATH = "results/pinn_gpr_comparison.csv"
BATTERY_METRICS_PATH = "results/pinn_gpr_battery_metrics.csv"

RANDOM_STATE = 42
N_SPLITS = 5

MAX_EPOCHS = 1500
PATIENCE = 100

LEARNING_RATE = 0.001
PHYSICS_WEIGHT = 1.0

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# PINN ARCHITECTURE
# Same architecture as pinn_model.py
# ============================================================

class PINN(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_size, 64),
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
# PHYSICS-INFORMED LOSS
# ============================================================

def calculate_loss(model, X, y, age_index):

    X = X.clone().detach().requires_grad_(True)

    predictions = model(X)

    data_loss = nn.functional.mse_loss(
        predictions,
        y
    )

    gradients = torch.autograd.grad(
        outputs=predictions.sum(),
        inputs=X,
        create_graph=True
    )[0]

    age_derivative = gradients[:, age_index]

    physics_loss = torch.relu(
        age_derivative
    ).pow(2).mean()

    total_loss = (
        data_loss +
        PHYSICS_WEIGHT * physics_loss
    )

    return total_loss


# ============================================================
# EARLY STOPPING FOR FOLD EPOCH SELECTION
# ============================================================

def select_best_epoch(
    X_fit,
    y_fit,
    X_val,
    y_val,
    input_size,
    seed
):

    set_seed(seed)

    model = PINN(input_size).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    best_val_loss = float("inf")
    best_epoch = 1
    patience_counter = 0

    for epoch in range(MAX_EPOCHS):

        model.train()

        optimizer.zero_grad()

        loss = calculate_loss(
            model,
            X_fit,
            y_fit,
            age_index=input_size - 1
        )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():

            val_predictions = model(X_val)

            val_loss = nn.functional.mse_loss(
                val_predictions,
                y_val
            ).item()

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    return best_epoch, best_val_loss


# ============================================================
# TRAIN PINN FOR A FIXED NUMBER OF EPOCHS
# ============================================================

def train_fixed_epochs(
    X_train,
    y_train,
    input_size,
    epochs,
    seed
):

    set_seed(seed)

    model = PINN(input_size).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        loss = calculate_loss(
            model,
            X_train,
            y_train,
            age_index=input_size - 1
        )

        loss.backward()
        optimizer.step()

    return model


# ============================================================
# CREATE OUT-OF-FOLD PINN PREDICTIONS
# ============================================================

def generate_oof_predictions(
    train_df,
    feature_columns
):

    print("\n" + "=" * 60)
    print("       GENERATING OUT-OF-FOLD PINN PREDICTIONS")
    print("=" * 60)

    X = train_df[feature_columns].to_numpy(
        dtype=np.float32
    )

    y = (
        train_df["soh"].to_numpy(
            dtype=np.float32
        ) / 100.0
    )

    groups = train_df["battery_id"].to_numpy()

    input_size = X.shape[1]

    oof_predictions = np.full(
        len(train_df),
        np.nan,
        dtype=np.float64
    )

    group_kfold = GroupKFold(
        n_splits=N_SPLITS
    )

    for fold, (fold_train_idx, holdout_idx) in enumerate(
        group_kfold.split(X, y, groups),
        start=1
    ):

        print("\n" + "-" * 55)
        print(f"FOLD {fold}/{N_SPLITS}")
        print("-" * 55)

        fold_train_df = train_df.iloc[
            fold_train_idx
        ]

        holdout_df = train_df.iloc[
            holdout_idx
        ]

        print(
            "Training batteries:",
            fold_train_df["battery_id"].nunique()
        )

        print(
            "Held-out batteries:",
            sorted(holdout_df["battery_id"].unique())
        )

        # ----------------------------------------------------
        # INNER GROUP VALIDATION SPLIT
        # Used to select the epoch count for this fold.
        # ----------------------------------------------------

        inner_splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.20,
            random_state=RANDOM_STATE + fold
        )

        inner_fit_idx, inner_val_idx = next(
            inner_splitter.split(
                fold_train_df,
                groups=fold_train_df["battery_id"]
            )
        )

        X_fold_train = X[fold_train_idx]
        y_fold_train = y[fold_train_idx]

        X_inner_fit = X_fold_train[inner_fit_idx]
        X_inner_val = X_fold_train[inner_val_idx]

        y_inner_fit = y_fold_train[inner_fit_idx]
        y_inner_val = y_fold_train[inner_val_idx]

        # Fit scaler only on inner fit data
        selection_scaler = StandardScaler()

        X_inner_fit_scaled = selection_scaler.fit_transform(
            X_inner_fit
        )

        X_inner_val_scaled = selection_scaler.transform(
            X_inner_val
        )

        X_inner_fit_tensor = torch.tensor(
            X_inner_fit_scaled,
            dtype=torch.float32,
            device=DEVICE
        )

        X_inner_val_tensor = torch.tensor(
            X_inner_val_scaled,
            dtype=torch.float32,
            device=DEVICE
        )

        y_inner_fit_tensor = torch.tensor(
            y_inner_fit.reshape(-1, 1),
            dtype=torch.float32,
            device=DEVICE
        )

        y_inner_val_tensor = torch.tensor(
            y_inner_val.reshape(-1, 1),
            dtype=torch.float32,
            device=DEVICE
        )

        best_epoch, best_val_loss = select_best_epoch(
            X_inner_fit_tensor,
            y_inner_fit_tensor,
            X_inner_val_tensor,
            y_inner_val_tensor,
            input_size,
            seed=RANDOM_STATE + fold
        )

        print("Selected epochs:", best_epoch)
        print(f"Validation MSE: {best_val_loss:.6f}")

        # ----------------------------------------------------
        # RETRAIN ON ALL FOLD TRAINING BATTERIES
        # ----------------------------------------------------

        fold_scaler = StandardScaler()

        X_fold_train_scaled = fold_scaler.fit_transform(
            X_fold_train
        )

        X_holdout_scaled = fold_scaler.transform(
            X[holdout_idx]
        )

        X_fold_train_tensor = torch.tensor(
            X_fold_train_scaled,
            dtype=torch.float32,
            device=DEVICE
        )

        y_fold_train_tensor = torch.tensor(
            y_fold_train.reshape(-1, 1),
            dtype=torch.float32,
            device=DEVICE
        )

        fold_model = train_fixed_epochs(
            X_fold_train_tensor,
            y_fold_train_tensor,
            input_size,
            best_epoch,
            seed=RANDOM_STATE + fold
        )

        # ----------------------------------------------------
        # PREDICT HELD-OUT BATTERIES
        # ----------------------------------------------------

        X_holdout_tensor = torch.tensor(
            X_holdout_scaled,
            dtype=torch.float32,
            device=DEVICE
        )

        fold_model.eval()

        with torch.no_grad():

            predictions = fold_model(
                X_holdout_tensor
            ).cpu().numpy().flatten()

        # Convert SOH fraction to percentage
        predictions = predictions * 100.0

        oof_predictions[holdout_idx] = predictions

        print(
            "OOF predictions generated:",
            len(holdout_idx)
        )

    if np.isnan(oof_predictions).any():
        raise ValueError(
            "Some training records did not receive OOF predictions."
        )

    return oof_predictions


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_metrics(actual, predicted):

    return {
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
            if len(actual) >= 2
            else np.nan
        )
    }


# ============================================================
# MAIN
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module="sklearn.gaussian_process"
    )

    print("=" * 60)
    print("       PINN + GAUSSIAN PROCESS REGRESSION")
    print("=" * 60)

    print("Device:", DEVICE)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    aging_df = pd.read_csv(AGING_PATH)

    metadata = joblib.load(
        PINN_METADATA_PATH
    )

    feature_columns = metadata["feature_columns"]

    if feature_columns[-1] != "aging_ah":
        raise ValueError(
            "Expected aging_ah to be the final PINN input feature."
        )

    print("\nDATASET INFORMATION")
    print("-" * 40)

    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))
    print("Training batteries:", train_df["battery_id"].nunique())
    print("Testing batteries:", test_df["battery_id"].nunique())
    print("Input features:", len(feature_columns))

    # --------------------------------------------------------
    # 2. MERGE AGING FEATURES
    # --------------------------------------------------------

    aging_columns = aging_df[
        ["battery_id", "mission_id", "aging_ah"]
    ]

    if aging_columns.duplicated(
        ["battery_id", "mission_id"]
    ).any():

        raise ValueError(
            "Duplicate aging mission keys found."
        )

    train_df = train_df.merge(
        aging_columns,
        on=["battery_id", "mission_id"],
        how="left",
        validate="one_to_one"
    )

    test_df = test_df.merge(
        aging_columns,
        on=["battery_id", "mission_id"],
        how="left",
        validate="one_to_one"
    )

    if train_df["aging_ah"].isna().any():
        raise ValueError(
            "Missing training aging values."
        )

    if test_df["aging_ah"].isna().any():
        raise ValueError(
            "Missing testing aging values."
        )

    # --------------------------------------------------------
    # 3. VALIDATE INPUTS
    # --------------------------------------------------------

    X_train_df = train_df[feature_columns].copy()
    X_test_df = test_df[feature_columns].copy()

    y_train = train_df["soh"].to_numpy()
    y_test = test_df["soh"].to_numpy()

    X_train = X_train_df.to_numpy(
        dtype=np.float64
    )

    X_test = X_test_df.to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(X_train).all():
        raise ValueError(
            "Invalid training feature values."
        )

    if not np.isfinite(X_test).all():
        raise ValueError(
            "Invalid testing feature values."
        )

    # --------------------------------------------------------
    # 4. GENERATE OOF PINN PREDICTIONS
    # --------------------------------------------------------

    oof_predictions = generate_oof_predictions(
        train_df,
        feature_columns
    )

    # Calculate residuals
    residuals = y_train - oof_predictions

    oof_df = train_df[
        ["battery_id", "mission_id", "soh"]
    ].copy()

    oof_df["pinn_oof_prediction"] = oof_predictions
    oof_df["oof_residual"] = residuals

    oof_df.to_csv(
        OOF_PATH,
        index=False
    )

    print("\nOOF RESIDUAL SUMMARY")
    print("-" * 40)

    print(
        "Mean residual:",
        f"{np.mean(residuals):.4f}"
    )

    print(
        "Residual standard deviation:",
        f"{np.std(residuals):.4f}"
    )

    print(
        "OOF residual MAE:",
        f"{mean_absolute_error(y_train, oof_predictions):.4f}"
    )

    # --------------------------------------------------------
    # 5. SCALE GPR INPUT FEATURES
    # --------------------------------------------------------

    gpr_scaler = StandardScaler()

    X_train_gpr = gpr_scaler.fit_transform(
        X_train
    )

    X_test_gpr = gpr_scaler.transform(
        X_test
    )

    # --------------------------------------------------------
    # 6. TRAIN GPR ON OOF RESIDUALS
    # --------------------------------------------------------

    print("\nTRAINING GPR")
    print("-" * 40)

    kernel = (
        ConstantKernel(
            1.0,
            constant_value_bounds=(1e-3, 1e3)
        )
        *
        Matern(
            length_scale=np.ones(len(feature_columns)),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5
        )
        +
        WhiteKernel(
            noise_level=0.1,
            noise_level_bounds=(1e-5, 1e2)
        )
    )

    gpr_model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=2,
        random_state=RANDOM_STATE
    )

    gpr_model.fit(
        X_train_gpr,
        residuals
    )

    print("\nOptimized GPR kernel:")
    print(gpr_model.kernel_)

    # --------------------------------------------------------
    # 7. LOAD FINAL TRAINED PINN
    # --------------------------------------------------------

    print("\nLOADING FINAL PINN")
    print("-" * 40)

    pinn_scaler = joblib.load(
        PINN_SCALER_PATH
    )

    final_pinn = PINN(
        input_size=len(feature_columns)
    ).to(DEVICE)

    state_dict = torch.load(
        PINN_MODEL_PATH,
        map_location=DEVICE
    )

    final_pinn.load_state_dict(state_dict)
    final_pinn.eval()

    X_test_pinn = pinn_scaler.transform(
        X_test_df
    )

    X_test_pinn_tensor = torch.tensor(
        X_test_pinn,
        dtype=torch.float32,
        device=DEVICE
    )

    with torch.no_grad():

        pinn_predictions = final_pinn(
            X_test_pinn_tensor
        ).cpu().numpy().flatten()

    pinn_predictions = pinn_predictions * 100.0

    # --------------------------------------------------------
    # 8. GPR RESIDUAL PREDICTION
    # --------------------------------------------------------

    print("\nPREDICTING TEST RESIDUALS")
    print("-" * 40)

    predicted_residuals, residual_std = gpr_model.predict(
        X_test_gpr,
        return_std=True
    )

    corrected_predictions = (
        pinn_predictions + predicted_residuals
    )

    # --------------------------------------------------------
    # 9. COMPARE PINN AND PINN + GPR
    # --------------------------------------------------------

    pinn_metrics = calculate_metrics(
        y_test,
        pinn_predictions
    )

    corrected_metrics = calculate_metrics(
        y_test,
        corrected_predictions
    )

    comparison_df = pd.DataFrame([
        {
            "model": "PINN",
            **pinn_metrics
        },
        {
            "model": "PINN + GPR",
            **corrected_metrics
        }
    ])

    print("\nFINAL TEST PERFORMANCE")
    print("=" * 60)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # 10. BATTERY-WISE COMPARISON
    # --------------------------------------------------------

    prediction_df = test_df[
        ["battery_id", "mission_id", "soh"]
    ].copy()

    prediction_df["pinn_prediction"] = pinn_predictions

    prediction_df["predicted_residual"] = (
        predicted_residuals
    )

    prediction_df["gpr_residual_std"] = residual_std

    prediction_df["corrected_prediction"] = (
        corrected_predictions
    )

    prediction_df["pinn_absolute_error"] = np.abs(
        y_test - pinn_predictions
    )

    prediction_df["corrected_absolute_error"] = np.abs(
        y_test - corrected_predictions
    )

    battery_results = []

    for battery_id, group in prediction_df.groupby(
        "battery_id"
    ):

        actual = group["soh"].to_numpy()

        pinn_pred = group["pinn_prediction"].to_numpy()

        corrected_pred = group[
            "corrected_prediction"
        ].to_numpy()

        pinn_result = calculate_metrics(
            actual,
            pinn_pred
        )

        corrected_result = calculate_metrics(
            actual,
            corrected_pred
        )

        battery_results.append({

            "battery_id": battery_id,
            "samples": len(group),

            "PINN_MAE": pinn_result["MAE"],
            "PINN_RMSE": pinn_result["RMSE"],
            "PINN_R2": pinn_result["R2"],

            "GPR_MAE": corrected_result["MAE"],
            "GPR_RMSE": corrected_result["RMSE"],
            "GPR_R2": corrected_result["R2"]
        })

    battery_metrics_df = pd.DataFrame(
        battery_results
    )

    print("\nBATTERY-WISE COMPARISON")
    print("=" * 85)

    print(
        battery_metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # 11. SAVE RESULTS
    # --------------------------------------------------------

    comparison_df.to_csv(
        METRICS_PATH,
        index=False
    )

    battery_metrics_df.to_csv(
        BATTERY_METRICS_PATH,
        index=False
    )

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False
    )

    joblib.dump(
        gpr_model,
        GPR_MODEL_PATH
    )

    joblib.dump(
        gpr_scaler,
        GPR_SCALER_PATH
    )

    print("\nSAVED FILES")
    print("-" * 50)

    print(OOF_PATH)
    print(PREDICTION_PATH)
    print(METRICS_PATH)
    print(BATTERY_METRICS_PATH)
    print(GPR_MODEL_PATH)
    print(GPR_SCALER_PATH)

    print("\nPINN + GPR PIPELINE COMPLETED")


if __name__ == "__main__":
    main()
