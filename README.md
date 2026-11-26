@'
# Battery State of Health (SOH) Prediction

A machine learning pipeline for estimating battery State of Health (SOH) using experimental battery aging data.

The project combines statistical feature extraction, Incremental Capacity Analysis (ICA), Principal Component Analysis (PCA), a Physics-Informed Neural Network (PINN), and Gaussian Process Regression (GPR) for residual correction.

## Dataset

This project uses the 2023 Fricke, Nascimento, and Viana accelerated-life-testing battery dataset.

The dataset contains battery cycling measurements from three groups:

- Regular accelerated-life batteries
- Recommissioned batteries
- Second-life batteries

Raw CSV files are expected inside the `dataset/` directory.

The raw dataset is not included in this repository.

## Pipeline

1. **Data Loading and Preprocessing**
   - Load battery measurement CSV files.
   - Identify discharge missions.
   - Extract valid reference capacity measurements.

2. **SOH Label Generation**
   - Calculate reference capacity for each valid mission.
   - Calculate SOH relative to each battery's first valid reference capacity.

   SOH is calculated as:

   SOH (%) = (Current Capacity / Initial Reference Capacity) × 100

3. **Feature Extraction**
   - Statistical discharge features.
   - Incremental Capacity Analysis (ICA) features.
   - Aging throughput features.

4. **Feature Processing**
   - Merge extracted features.
   - Apply scaling and PCA.

5. **PINN Model**
   - Train a neural network for SOH prediction.
   - Include a physics-based penalty encouraging non-increasing predicted SOH with aging throughput.

6. **GPR Residual Correction**
   - Generate out-of-fold PINN predictions.
   - Train a Gaussian Process Regression model on PINN residuals.
   - Correct PINN predictions using estimated residuals.

7. **Evaluation**
   - Battery-grouped cross-validation.
   - Held-out battery evaluation.
   - Prediction error analysis and battery-level diagnostics.

## Project Structure

```text
NASA-Battery-SOH/
│
├── dataset/                 # Raw battery CSV files (not tracked)
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── capacity_estimation.py
│   ├── generate_soh_labels.py
│   ├── ica_feature_extraction.py
│   ├── batch_ica_extraction.py
│   ├── statistical_feature_extraction.py
│   ├── dataset_builder.py
│   ├── aging_feature_extraction.py
│   ├── pca_model.py
│   ├── pinn_model.py
│   ├── evaluate_pinn.py
│   ├── gpr_model.py
│   ├── group_cross_validation.py
│   ├── analyze_cv_errors.py
│   ├── diagnose_cv.py
│   ├── inspect_problem_batteries.py
│   └── verify_outputs.py
│
├── models/
│   ├── PINN and GPR model artifacts
│   └── cv/                  # Fold-specific model artifacts
│
├── results/
│   ├── Extracted features
│   ├── SOH labels
│   ├── Model predictions
│   ├── Evaluation metrics
│   └── cv/                  # Cross-validation and diagnostics
│
├── requirements.txt
└── README.md