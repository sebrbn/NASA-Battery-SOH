
import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = "results/pca_train_dataset.csv"
TEST_PATH = "results/pca_test_dataset.csv"
AGING_PATH = "results/aging_features.csv"

MODEL_PATH = "models/pinn_model.pth"
SCALER_PATH = "models/pinn_input_scaler.pkl"
METADATA_PATH = "models/pinn_metadata.pkl"

PREDICTION_PATH = "results/pinn_test_predictions.csv"

RANDOM_STATE = 42
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

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# NEURAL NETWORK
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

    # Ordinary supervised prediction loss
    data_loss = nn.functional.mse_loss(
        predictions,
        y
    )

    # Calculate derivative of predicted SOH
    gradients = torch.autograd.grad(
        outputs=predictions.sum(),
        inputs=X,
        create_graph=True
    )[0]

    age_derivative = gradients[:, age_index]

    # SOH should not increase as cumulative throughput increases
    physics_loss = torch.relu(
        age_derivative
    ).pow(2).mean()

    total_loss = (
        data_loss +
        PHYSICS_WEIGHT * physics_loss
    )

    return total_loss, data_loss, physics_loss


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
    epochs,
    X_val=None,
    y_val=None
):

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001
    )

    best_val_loss = float("inf")
    best_epoch = epochs
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()

        total_loss, data_loss, physics_loss = calculate_loss(
            model,
            X_train,
            y_train,
            age_index=X_train.shape[1] - 1
        )

        total_loss.backward()

        optimizer.step()

        # Validation is used only during model selection
        if X_val is not None:

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

                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

                patience_counter = 0

            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch + 1}"
                )
                break

        if (epoch + 1) % 100 == 0:

            print(
                f"Epoch {epoch + 1:4d} | "
                f"Total Loss: {total_loss.item():.6f} | "
                f"Data Loss: {data_loss.item():.6f} | "
                f"Physics Loss: {physics_loss.item():.6f}"
            )

    if X_val is not None and best_state is not None:

        model.load_state_dict(best_state)

    return best_epoch, best_val_loss


# ============================================================
# METRICS
# ============================================================

def evaluate_model(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    print("\nMODEL PERFORMANCE")
    print("-" * 40)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return mae, rmse, r2


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 55)
    print("          PHYSICS-INFORMED NEURAL NETWORK")
    print("=" * 55)

    print("Device:", DEVICE)

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    aging_df = pd.read_csv(AGING_PATH)

    # Verify aging mission keys
    if aging_df.duplicated(
        ["battery_id", "mission_id"]
    ).any():
        raise ValueError("Duplicate aging mission keys found.")

    aging_columns = aging_df[
        ["battery_id", "mission_id", "aging_ah"]
    ]

    # Merge aging values
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
        raise ValueError("Missing training aging values.")

    if test_df["aging_ah"].isna().any():
        raise ValueError("Missing testing aging values.")

    # --------------------------------------------------------
    # 2. SELECT FEATURES
    # --------------------------------------------------------

    pca_features = [
        col for col in train_df.columns
        if col.startswith("PC") and col[2:].isdigit()
    ]

    pca_features = sorted(
        pca_features,
        key=lambda col: int(col[2:])
    )

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

    feature_columns = (
        pca_features +
        statistical_features +
        ["aging_ah"]
    )

    target = "soh"

    missing_columns = [
        col for col in feature_columns + [target]
        if col not in train_df.columns
        or col not in test_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    X_train_df = train_df[feature_columns].copy()
    X_test_df = test_df[feature_columns].copy()

    y_train = train_df[target].to_numpy(
        dtype=np.float32
    ) / 100.0

    y_test = test_df[target].to_numpy(
        dtype=np.float32
    ) / 100.0

    # Validate numeric values
    if not np.isfinite(X_train_df.to_numpy()).all():
        raise ValueError("Invalid training feature values.")

    if not np.isfinite(X_test_df.to_numpy()).all():
        raise ValueError("Invalid testing feature values.")

    print("\nDATASET INFORMATION")
    print("-" * 40)
    print("Training records:", len(train_df))
    print("Testing records:", len(test_df))
    print("Training batteries:", train_df["battery_id"].nunique())
    print("Testing batteries:", test_df["battery_id"].nunique())
    print("PCA components:", len(pca_features))
    print("Statistical features:", len(statistical_features))
    print("Total inputs:", len(feature_columns))

    # --------------------------------------------------------
    # 3. GROUP-BASED VALIDATION SPLIT
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    fit_indices, val_indices = next(
        splitter.split(
            X_train_df,
            y_train,
            groups=train_df["battery_id"]
        )
    )

    X_fit_df = X_train_df.iloc[fit_indices]
    X_val_df = X_train_df.iloc[val_indices]

    y_fit = y_train[fit_indices]
    y_val = y_train[val_indices]

    print("\nVALIDATION SPLIT")
    print("-" * 40)
    print(
        "Fit batteries:",
        sorted(train_df.iloc[fit_indices]["battery_id"].unique())
    )
    print(
        "Validation batteries:",
        sorted(train_df.iloc[val_indices]["battery_id"].unique())
    )

    # --------------------------------------------------------
    # 4. SCALE INPUT FEATURES
    # --------------------------------------------------------

    # Fit only on inner training data for model selection
    validation_scaler = StandardScaler()

    X_fit = validation_scaler.fit_transform(
        X_fit_df
    )

    X_val = validation_scaler.transform(
        X_val_df
    )

    X_fit_tensor = torch.tensor(
        X_fit,
        dtype=torch.float32,
        device=DEVICE
    )

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=DEVICE
    )

    y_fit_tensor = torch.tensor(
        y_fit.reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    y_val_tensor = torch.tensor(
        y_val.reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    # --------------------------------------------------------
    # 5. VALIDATION TRAINING
    # --------------------------------------------------------

    print("\nSTARTING VALIDATION TRAINING")
    print("-" * 40)

    validation_model = PINN(
        input_size=len(feature_columns)
    ).to(DEVICE)

    best_epoch, best_val_loss = train_model(
        validation_model,
        X_fit_tensor,
        y_fit_tensor,
        epochs=MAX_EPOCHS,
        X_val=X_val_tensor,
        y_val=y_val_tensor
    )

    print("\nBest epoch:", best_epoch)
    print("Best validation MSE:", best_val_loss)

    # --------------------------------------------------------
    # 6. FINAL TRAINING ON ALL 20 TRAINING BATTERIES
    # --------------------------------------------------------

    print("\nFINAL TRAINING")
    print("-" * 40)

    final_scaler = StandardScaler()

    X_train = final_scaler.fit_transform(
        X_train_df
    )

    X_test = final_scaler.transform(
        X_test_df
    )

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=DEVICE
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32,
        device=DEVICE
    )

    y_train_tensor = torch.tensor(
        y_train.reshape(-1, 1),
        dtype=torch.float32,
        device=DEVICE
    )

    final_model = PINN(
        input_size=len(feature_columns)
    ).to(DEVICE)

    # Train on all outer training batteries
    train_model(
        final_model,
        X_train_tensor,
        y_train_tensor,
        epochs=best_epoch
    )

    # --------------------------------------------------------
    # 7. TEST EVALUATION
    # --------------------------------------------------------

    final_model.eval()

    with torch.no_grad():

        predictions = final_model(
            X_test_tensor
        ).cpu().numpy().flatten()

    # Convert predictions back to percentage
    predictions = predictions * 100

    actual = y_test * 100

    evaluate_model(actual, predictions)

    # --------------------------------------------------------
    # 8. SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    prediction_df = test_df[
        ["battery_id", "mission_id", "soh"]
    ].copy()

    prediction_df["predicted_soh"] = predictions

    prediction_df["absolute_error"] = np.abs(
        prediction_df["soh"] -
        prediction_df["predicted_soh"]
    )

    prediction_df.to_csv(
        PREDICTION_PATH,
        index=False
    )

    # --------------------------------------------------------
    # 9. SAVE MODEL AND SCALER
    # --------------------------------------------------------

    torch.save(
        final_model.state_dict(),
        MODEL_PATH
    )

    joblib.dump(
        final_scaler,
        SCALER_PATH
    )

    metadata = {
        "feature_columns": feature_columns,
        "pca_features": pca_features,
        "statistical_features": statistical_features,
        "target": target,
        "input_size": len(feature_columns),
        "best_epoch": best_epoch,
        "physics_weight": PHYSICS_WEIGHT,
        "random_state": RANDOM_STATE
    }

    joblib.dump(
        metadata,
        METADATA_PATH
    )

    print("\nSAVED FILES")
    print("-" * 40)
    print(MODEL_PATH)
    print(SCALER_PATH)
    print(METADATA_PATH)
    print(PREDICTION_PATH)

    print("\nPINN TRAINING COMPLETED")


if __name__ == "__main__":
    main()
