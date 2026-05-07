"""Preprocess Chapman ECG data into train/val/test CSV files."""

import os
from pathlib import Path

import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_ROOT = Path("your_path/data")
SPLIT_ROOT = Path("your_path/finetune/data_split/chapman")
MIN_LABEL_COUNT = 10
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_STATE = 42


def read_header_file(file_path: Path) -> list[str]:
    with open(file_path, "r") as file:
        return [line.strip() for line in file]


def load_records(data_root: Path) -> pd.DataFrame:
    ref = pd.read_csv(data_root / "chapman/ConditionNames_SNOMED-CT.csv")
    ref["Snomed_CT"] = ref["Snomed_CT"].astype(str)

    rows = []
    mat_files = sorted((data_root / "chapman/WFDBRecords").glob("**/*.mat"))
    for mat_file in tqdm(mat_files):
        hea_file = mat_file.with_suffix(".hea")
        if not hea_file.exists():
            continue

        hea = read_header_file(hea_file)
        try:
            dx_line = next(line for line in hea if "Dx" in line)
            codes = dx_line.split()[1].split(",")
            labels = [ref.loc[ref["Snomed_CT"] == code, "Acronym Name"].values[0] for code in codes]
            diagnose = ",".join(labels)
        except Exception:
            diagnose = "Unknown"

        rows.append({
            "ecg_path": str(mat_file),
            "age": hea[0].split()[1],
            "diagnose": diagnose,
        })

    return pd.DataFrame(rows)


def add_label_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["diagnose"] != "Unknown"].reset_index(drop=True)
    labels = sorted({label for value in df["diagnose"] for label in value.split(",")})

    for label in labels:
        df[label] = df["diagnose"].apply(lambda x: int(label in x.split(",")))

    fixed_columns = df.columns[:3].tolist()
    label_columns = df.columns[3:].tolist()
    keep_labels = [label for label in label_columns if df[label].sum() >= MIN_LABEL_COUNT]
    return df[fixed_columns + keep_labels]


def main() -> None:
    df = add_label_columns(load_records(DATA_ROOT))
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_df, val_df = train_test_split(train_df, test_size=VAL_SIZE, random_state=RANDOM_STATE)

    for frame in (train_df, val_df, test_df):
        frame.reset_index(drop=True, inplace=True)

    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLIT_ROOT / "chapman_train.csv", index=False)
    val_df.to_csv(SPLIT_ROOT / "chapman_val.csv", index=False)
    test_df.to_csv(SPLIT_ROOT / "chapman_test.csv", index=False)

    print(f"train_df shape: {train_df.shape}")
    print(f"val_df shape: {val_df.shape}")
    print(f"test_df shape: {test_df.shape}")


if __name__ == "__main__":
    main()