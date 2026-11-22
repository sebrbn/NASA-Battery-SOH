
import pandas as pd
from pathlib import Path


GROUPS = [
    "regular_alt_batteries",
    "recommissioned_batteries",
    "second_life_batteries"
]


def load_all_batteries(dataset_path):

    dataset_path = Path(dataset_path)

    batteries = {}

    for group in GROUPS:

        folder = dataset_path / group

        if not folder.exists():
            print("Missing folder:", folder)
            continue

        for file in folder.glob("*.csv"):

            df = pd.read_csv(file , low_memory=False)

            battery_id = file.stem

            batteries[battery_id] = {
                "group": group,
                "data": df
            }

            print(
                f"Loaded {battery_id}: "
                f"{len(df)} rows | {group}"
            )

    return batteries


if __name__ == "__main__":

    batteries = load_all_batteries("dataset")

    print("\nTotal batteries:", len(batteries))
