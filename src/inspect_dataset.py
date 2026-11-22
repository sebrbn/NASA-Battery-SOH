
import pandas as pd
from pathlib import Path

DATASET_PATH = Path("dataset")

csv_files = list(DATASET_PATH.rglob("*.csv"))

print("Total CSV files:", len(csv_files))

for file in csv_files:

    print("\n" + "=" * 60)
    print("FILE:", file.name)
    print("=" * 60)

    df = pd.read_csv(file)

    print("\nShape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nMode distribution:")
    if "mode" in df.columns:
        print(df["mode"].value_counts())

    print("\nMission type distribution:")
    if "mission type" in df.columns:
        print(df["mission type"].value_counts())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nBasic statistics:")
    print(df.describe())
