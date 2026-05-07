"""
MIMIC-IV ECG Data Preprocessing Module

Process raw MIMIC-IV ECG data, clean reports, and prepare dataset for training.
"""
import os
import logging
from typing import Tuple
import numpy as np
import pandas as pd
import wfdb
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
META_PATH = 'your_path/data/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
NUM_LEADS = 12
ECG_LENGTH = 5000
MIN_REPORT_LENGTH = 4
NAN_WINDOW = 6
SCALE_FACTOR = 1000
TEST_SIZE = 0.02
RANDOM_STATE = 42

def process_report(row: pd.Series) -> Tuple[int, str]:
    report_cols = [f'report_{i}' for i in range(18)]
    # Select relevant columns and filter out NaNs
    report = row[report_cols].dropna()
    # Concatenate the reports
    report_text = '. '.join(report)
    
    # Text preprocessing
    report_text = report_text.replace('EKG', 'ECG').replace('ekg', 'ecg')
    report_text = report_text.strip(' ***').strip('*** ').strip('***').strip('=-').strip('=')
    report_text = report_text.lower()
    
    # Normalize whitespace
    report_text = report_text.replace('\n', ' ')
    report_text = report_text.replace('\r', ' ')
    report_text = report_text.replace('\t', ' ')
    
    word_count = len(report_text.split())
    
    if word_count > 0:
        total_report = report_text + '.'
    else:
        total_report = 'empty'
    
    return word_count, total_report

def load_and_filter_data(meta_path: str) -> Tuple[np.ndarray, pd.DataFrame]:
    logger.info("Loading report and record metadata...")
    report_csv = pd.read_csv(f'{meta_path}/machine_measurements.csv', low_memory=False)
    record_csv = pd.read_csv(f'{meta_path}/record_list.csv', low_memory=False)
    
    logger.info("Processing reports...")
    tqdm.pandas()
    report_csv['report_length'], report_csv['total_report'] = zip(
        *report_csv.progress_apply(process_report, axis=1)
    )
    
    # Filter reports with minimum length
    initial_count = len(report_csv)
    report_csv = report_csv[report_csv['report_length'] >= MIN_REPORT_LENGTH]
    report_csv.reset_index(drop=True, inplace=True)
    filtered_count = len(report_csv)
    logger.info(f"Filtered reports: {initial_count} -> {filtered_count}")
    
    # Filter records to match reports
    record_csv = record_csv[record_csv['study_id'].isin(report_csv['study_id'])]
    record_csv.reset_index(drop=True, inplace=True)
    
    # Initialize data array
    temp_npy = np.zeros((len(record_csv), NUM_LEADS, ECG_LENGTH), dtype=np.int16)
    
    return temp_npy, record_csv, report_csv


def fill_nan_values(record: np.ndarray, window: int = NAN_WINDOW) -> np.ndarray:
    for i in range(record.shape[0]):
        nan_idx = np.where(np.isnan(record[i, :]))[0]
        for idx in nan_idx:
            start = max(0, idx - window)
            end = min(idx + window, record.shape[1])
            record[i, idx] = np.mean(record[i, start:end])
    return record


def fill_inf_values(record: np.ndarray, window: int = NAN_WINDOW) -> np.ndarray:
    for i in range(record.shape[0]):
        inf_idx = np.where(np.isinf(record[i, :]))[0]
        for idx in inf_idx:
            start = max(0, idx - window)
            end = min(idx + window, record.shape[1])
            record[i, idx] = np.mean(record[i, start:end])
    return record


def normalize_and_scale_ecg(record: np.ndarray) -> np.ndarray:
    record = (record - record.min()) / (record.max() - record.min())
    record *= SCALE_FACTOR
    return record.astype(np.int16)


def process_ecg_records(temp_npy: np.ndarray, record_csv: pd.DataFrame, 
                       meta_path: str) -> np.ndarray:
    logger.info("Processing ECG records...")
    
    for idx, p in tqdm(enumerate(record_csv['path']), total=len(record_csv)):
        try:
            ecg_path = os.path.join(meta_path, p)
            record = wfdb.rdsamp(ecg_path)[0].T  # Shape: [12, time_steps]
            
            # Clean missing/infinite values
            if np.isnan(record).sum() > 0 or np.isinf(record).sum() > 0:
                if np.isnan(record).sum() > 0:
                    record = fill_nan_values(record)
                if np.isinf(record).sum() > 0:
                    record = fill_inf_values(record)
            
            # Normalize and scale
            record = normalize_and_scale_ecg(record)
            
            # Store (ensure proper length)
            temp_npy[idx] = record[:, :ECG_LENGTH]
            
        except Exception as e:
            logger.warning(f"Error processing {p}: {str(e)}")
            continue
    
    return temp_npy

def main(meta_path: str, output_path: str) -> None:
    # Step 1: Load and filter
    temp_npy, record_csv, report_csv = load_and_filter_data(meta_path)
    
    # Step 2: Process ECG records
    temp_npy = process_ecg_records(temp_npy, record_csv, meta_path)
    
    # Step 3: Split dataset
    logger.info("Splitting dataset into train/val...")
    train_npy, train_csv, val_npy, val_csv = train_test_split(
        temp_npy, report_csv, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    train_csv.reset_index(drop=True, inplace=True)
    val_csv.reset_index(drop=True, inplace=True)
    
    logger.info(f"Train set: {train_npy.shape}, Val set: {val_npy.shape}")
    
    # Step 4: Save files
    logger.info(f"Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    np.save(os.path.join(output_path, 'train.npy'), train_npy)
    np.save(os.path.join(output_path, 'val.npy'), val_npy)
    train_csv.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    val_csv.to_csv(os.path.join(output_path, 'val.csv'), index=False)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    output_dir = "your_path/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/processed/"
    main(META_PATH, output_dir)