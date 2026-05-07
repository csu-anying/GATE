import os
from typing import Tuple

import numpy as np
import pandas as pd
import wfdb, ast
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

# Configuration constants
DEFAULT_DATASET_PATH = "data/ptb"
DEFAULT_OUTPUT_PATH = "./processed_data"


def apply_scaler(inputs: np.array, scaler: StandardScaler) -> np.array:

    temp = []
    for x in inputs:
        x_shape = x.shape
        temp.append(scaler.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    temp = np.array(temp)
    return temp


def preprocess(path: str = "data/ptb") -> Tuple[np.array]:

    print("Loading dataset...", end="\n" * 2)

    path = os.path.join(os.getcwd(), Path(path))
    Y = pd.read_csv(os.path.join(path, "ptbxl_database.csv"), index_col="ecg_id")
    data = np.array([wfdb.rdsamp(os.path.join(path, f))[0] for f in Y.filename_lr])
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    agg_df = pd.read_csv(os.path.join(path, "scp_statements.csv"), index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def agg(y_dic):
        temp = []

        for key in y_dic.keys():
            if key in agg_df.index:
                c = agg_df.loc[key].diagnostic_class
                if str(c) != "nan":
                    temp.append(c)
        return list(set(temp))

    Y["diagnostic_superclass"] = Y.scp_codes.apply(agg)
    Y["superdiagnostic_len"] = Y["diagnostic_superclass"].apply(lambda x: len(x))
    counts = pd.Series(np.concatenate(Y.diagnostic_superclass.values)).value_counts()
    Y["diagnostic_superclass"] = Y["diagnostic_superclass"].apply(
        lambda x: list(set(x).intersection(set(counts.index.values)))
    )

    X_data = data[Y["superdiagnostic_len"] >= 1]
    Y_data = Y[Y["superdiagnostic_len"] >= 1]

    print("Preprocessing dataset...", end="\n" * 2)

    mlb = MultiLabelBinarizer()
    mlb.fit(Y_data["diagnostic_superclass"])
    y = mlb.transform(Y_data["diagnostic_superclass"].values)

    # Stratified split
    X_train = X_data[Y_data.strat_fold < 9]
    y_train = y[Y_data.strat_fold < 9]

    X_val = X_data[Y_data.strat_fold == 9]
    y_val = y[Y_data.strat_fold == 9]

    X_test = X_data[Y_data.strat_fold == 10]
    y_test = y[Y_data.strat_fold == 10]

    del X_data, Y_data, y, data

    # Standardization
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))
    X_train_scale = apply_scaler(X_train, scaler)
    X_test_scale = apply_scaler(X_test, scaler)
    X_val_scale = apply_scaler(X_val, scaler)

    del X_train, X_test, X_val

    # Shuffling
    X_train_scale, y_train = shuffle(X_train_scale, y_train, random_state=42)

    return X_train_scale, y_train, X_test_scale, y_test, X_val_scale, y_val


def main(dataset_path: str = DEFAULT_DATASET_PATH, output_path: str = DEFAULT_OUTPUT_PATH) -> None:
    print("PTB-XL Dataset Preprocessing Pipeline")
    
    # Step 1: Preprocess data
    print(f"Loading dataset from: {dataset_path}")
    X_train, y_train, X_test, y_test, X_val, y_val = preprocess(dataset_path)
    
    # Print dataset statistics
    print(f"Train set shape: {X_train.shape}, Labels shape: {y_train.shape}")
    print(f"Test set shape: {X_test.shape}, Labels shape: {y_test.shape}")
    print(f"Val set shape: {X_val.shape}, Labels shape: {y_val.shape}")
    
    # Step 2: Create output directory
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving processed data to: {output_path}")
    
    # Step 3: Save ECG data as npy files
    train_npy_path = os.path.join(output_path, 'train.npy')
    val_npy_path = os.path.join(output_path, 'val.npy')
    test_npy_path = os.path.join(output_path, 'test.npy')
    
    np.save(train_npy_path, X_train)
    print(f"Train ECG data saved: {train_npy_path}")
    
    np.save(val_npy_path, X_val)
    print(f"Val ECG data saved: {val_npy_path}")
    
    np.save(test_npy_path, X_test)
    print(f"Test ECG data saved: {test_npy_path}")
    
    # Step 4: Save labels as npy files
    train_labels_path = os.path.join(output_path, 'train_labels.npy')
    val_labels_path = os.path.join(output_path, 'val_labels.npy')
    test_labels_path = os.path.join(output_path, 'test_labels.npy')
    
    np.save(train_labels_path, y_train)
    print(f"Train labels saved: {train_labels_path}")
    
    np.save(val_labels_path, y_val)
    print(f"Val labels saved: {val_labels_path}")
    
    np.save(test_labels_path, y_test)
    print(f"Test labels saved: {test_labels_path}")


if __name__ == "__main__":
    # Run with default paths
    main()