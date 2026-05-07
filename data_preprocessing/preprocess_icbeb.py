"""ICBEB Dataset Preprocessing Pipeline

This module processes the ICBEB2018 dataset by:
1. Converting .mat files to WFDB format (.hea and .dat)
2. Downsampling to 100Hz and 500Hz
3. Organizing labels and creating train/val/test splits
4. Saving processed data to CSV format

Reference: http://2018.icbeb.org/Challenge.html
"""
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import wfdb
from scipy.ndimage import zoom
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuration constants
MIN_SIGNAL_LENGTH = 1  # placeholder, adjust based on data
TRAIN_TEST_RATIO = 0.2
VAL_TRAIN_RATIO = 0.1
RANDOM_STATE = 42
SAMPLING_RATES = [100, 500]
ECG_CHANNELS = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
DEFAULT_UNITS = 'mV'
DEFAULT_FORMAT = '16'

# Label mapping
LABEL_DICT = {
    1: 'NORM', 2: 'AFIB', 3: '1AVB', 4: 'CLBBB', 5: 'CRBBB',
    6: 'PAC', 7: 'VPC', 8: 'STD', 9: 'STE'
}

ALL_LABELS = ['AFIB', 'VPC', 'NORM', '1AVB', 'CRBBB', 'STE', 'PAC', 'CLBBB', 'STD']


def ensure_directories(*paths: str) -> None:
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            print(f'Directory already exists: {path}')


def store_as_wfdb(signame: str, data: np.ndarray, sigfolder: str, fs: int) -> None:
    wfdb.wrsamp(
        signame,
        fs=fs,
        sig_name=ECG_CHANNELS,
        p_signal=data,
        units=[DEFAULT_UNITS] * len(ECG_CHANNELS),
        fmt=[DEFAULT_FORMAT] * len(ECG_CHANNELS),
        write_dir=sigfolder
    )


def process_mat_files(ori_data_folder: str, output_datafolder_100: str, 
                      output_datafolder_500: str, reference_path: str) -> pd.DataFrame:
    # Load reference labels
    df_reference = pd.read_csv(reference_path)
    
    data = {
        'ecg_id': [], 'filename': [], 'validation': [],
        'age': [], 'sex': [], 'scp_codes': []
    }
    
    ecg_counter = 0
    print("Processing .mat files...")
    
    for folder in ['raw']:
        folder_path = os.path.join(ori_data_folder, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found: {folder_path}")
            continue
            
        filenames = os.listdir(folder_path)
        for filename in tqdm(filenames):
            if filename.split('.')[-1] == 'mat':
                ecg_counter += 1
                name = filename.split('.')[0]
                
                try:
                    # Load .mat file
                    mat_data = loadmat(os.path.join(folder_path, filename))
                    sex, age, sig = mat_data['ECG'][0][0]
                    
                    data['ecg_id'].append(ecg_counter)
                    data['filename'].append(name)
                    data['validation'].append(False)
                    data['age'].append(age[0][0])
                    data['sex'].append(1 if sex[0] == 'Male' else 0)
                    
                    # Get labels from reference
                    labels = df_reference[
                        df_reference.Recording == name
                    ][['First_label', 'Second_label', 'Third_label']].values.flatten()
                    labels = labels[~np.isnan(labels)].astype(int)
                    data['scp_codes'].append({LABEL_DICT[key]: 1 for key in labels})
                    
                    # Store at 500Hz
                    store_as_wfdb(str(ecg_counter), sig.T, output_datafolder_500, 500)
                    
                    # Store at 100Hz (0.2 is downsampling ratio from 500Hz to 100Hz)
                    down_sig = np.array([zoom(channel, 0.2) for channel in sig])
                    store_as_wfdb(str(ecg_counter), down_sig.T, output_datafolder_100, 100)
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
                    continue
    
    return pd.DataFrame(data)


def create_label_columns(df: pd.DataFrame, labels: list) -> pd.DataFrame:
    for label in labels:
        df[label] = df['scp_codes'].apply(lambda x: x.get(label, 0))
    return df


def main(ori_data_folder: str, output_folder: str, reference_path: str, 
         train_csv_path: str, val_csv_path: str, test_csv_path: str) -> None:
    print("ICBEB Dataset Preprocessing Pipeline")
    
    # Create output directories
    output_datafolder_100 = os.path.join(output_folder, 'records100')
    output_datafolder_500 = os.path.join(output_folder, 'records500')
    ensure_directories(output_folder, output_datafolder_100, output_datafolder_500)
    
    # Process .mat files
    df = process_mat_files(ori_data_folder, output_datafolder_100, 
                           output_datafolder_500, reference_path)
    
    print(f"\nProcessed {len(df)} ECG records")
    
    # Add patient_id and prepare dataframe
    df['patient_id'] = df['ecg_id']
    cols = ['patient_id'] + [c for c in df.columns if c != 'patient_id']
    df = df[cols]
    
    # Create binary label columns
    df = create_label_columns(df, ALL_LABELS)
    
    print(f"DataFrame columns: {list(df.columns)}")
    
    # Split data
    print("\nSplitting data into train/val/test...")
    train_df, test_df = train_test_split(
        df, test_size=TRAIN_TEST_RATIO, random_state=RANDOM_STATE
    )
    train_df, val_df = train_test_split(
        train_df, test_size=VAL_TRAIN_RATIO, random_state=RANDOM_STATE
    )
    
    print(f"Train set: {train_df.shape}")
    print(f"Val set: {val_df.shape}")
    print(f"Test set: {test_df.shape}")
    
    # Save CSV files
    print("\nSaving CSV files...")
    os.makedirs(os.path.dirname(train_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_csv_path), exist_ok=True)
    
    train_df.to_csv(train_csv_path, index=False)
    print(f"Train CSV saved: {train_csv_path}")
    
    val_df.to_csv(val_csv_path, index=False)
    print(f"Val CSV saved: {val_csv_path}")
    
    test_df.to_csv(test_csv_path, index=False)
    print(f"Test CSV saved: {test_csv_path}")
    
    print("Processing complete!")


if __name__ == "__main__":
    # Configuration - modify these paths as needed
    ORI_DATA_FOLDER = "your_path/data/ICBEB"
    OUTPUT_FOLDER = os.path.join(ORI_DATA_FOLDER, 'icbeb2018')
    REFERENCE_PATH = "your_path/data/ICBEB/REFERENCE.csv"
    
    TRAIN_CSV_PATH = "your_path/finetune/data_split/icbeb/icbeb_train.csv"
    VAL_CSV_PATH = "your_path/finetune/data_split/icbeb/icbeb_val.csv"
    TEST_CSV_PATH = "your_path/finetune/data_split/icbeb/icbeb_test.csv"
    
    main(ORI_DATA_FOLDER, OUTPUT_FOLDER, REFERENCE_PATH,
         TRAIN_CSV_PATH, VAL_CSV_PATH, TEST_CSV_PATH)
