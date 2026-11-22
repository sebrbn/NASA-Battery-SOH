
import pandas as pd
from pathlib import Path


def clean_battery_data(df):
    """Clean and validate one battery's complete time-series data."""

    df = df.copy()

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    required = [
        "time",
        "mode",
        "voltage_charger",
        "temperature_battery",
        "voltage_load",
        "current_load",
        "temperature_mosfet",
        "temperature_resistor",
        "mission_type"
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Convert numeric columns safely
    numeric_columns = [
        "time",
        "mode",
        "voltage_charger",
        "temperature_battery",
        "voltage_load",
        "current_load",
        "temperature_mosfet",
        "temperature_resistor",
        "mission_type"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep chronological order
    df = df.dropna(subset=["time", "mode"])
    df = df.sort_values("time").reset_index(drop=True)

    # Do not drop rows just because discharge-only columns are NaN.
    return df


def extract_discharge_missions(df):
    """
    Extract continuous discharge periods.

    mode = -1: discharge
    mode =  0: rest
    mode =  1: charge

    A new discharge mission begins after a non-discharge row.
    """

    df = clean_battery_data(df)

    # Remember original row positions before filtering
    df["_row_position"] = range(len(df))

    discharge = df[df["mode"] == -1].copy()

    print("Total rows:", len(df))
    print("Discharge rows:", len(discharge))

    if discharge.empty:
        print("No discharge records found!")
        return []

    # A gap in original row positions means another mode occurred
    discharge["_segment"] = (
        discharge["_row_position"].diff().ne(1)
    ).cumsum()

    missions = []

    for segment_id, mission in discharge.groupby("_segment"):

        # Remove rows without mission type
        mission = mission.dropna(subset=["mission_type"])

        if len(mission) < 2:
            continue

        mission_types = mission["mission_type"].unique()

        if len(mission_types) != 1:
            print(
                f"Warning: multiple mission types in segment {segment_id}: "
                f"{mission_types}"
            )
            continue

        missions.append({
            "mission_id": int(segment_id),
            "data": mission.drop(
                columns=["_row_position", "_segment"]
            ).copy(),
            "mission_type": int(mission_types[0])
        })

    print("Total discharge missions:", len(missions))

    for mission in missions[:10]:
        print(
            f"Mission {mission['mission_id']} | "
            f"Type: {mission['mission_type']} | "
            f"Rows: {len(mission['data'])} | "
            f"Duration: "
            f"{mission['data']['time'].iloc[-1] - mission['data']['time'].iloc[0]:.2f} seconds"
        )

    return missions


if __name__ == "__main__":

    dataset_path = Path("dataset")

    files = list(dataset_path.rglob("*.csv"))

    print("CSV files found:", len(files))

    if not files:
        print("No CSV files found!")
    else:

        file = files[0]

        print("\nTesting file:", file)

        df = pd.read_csv(file, low_memory=False)

        missions = extract_discharge_missions(df)

        # Separate reference and regular missions
        reference = [
            m for m in missions
            if m["mission_type"] == 0
        ]

        regular = [
            m for m in missions
            if m["mission_type"] == 1
        ]

        print("\nMission summary:")
        print("Reference missions (type 0):", len(reference))
        print("Regular missions (type 1):", len(regular))

        print("\nPreprocessing completed!")
