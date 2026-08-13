"""
Data Loader for NASA C-MAPS FD001 Dataset

This module loads and preprocesses the NASA Commercial Modular Aero-Propulsion
System Simulation (C-MAPS) dataset for turbofan engine Remaining Useful Life (RUL) prediction.

Dataset: FD001 (single operating condition, single failure mode)
- Training: 100 engines run to failure
- Test: 100 engines with partial run data
- Features: 3 operational settings + 21 sensor measurements
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_path: str = "."):
    """
    Load NASA C-MAPS FD001 dataset with RUL calculation.

    This function performs the complete data loading pipeline:
    1. Load training data and calculate RUL for each cycle
    2. Cap RUL at 125 cycles (piece-wise RUL, standard for C-MAPS)
    3. Load test data with same structure
    4. Load ground truth RUL values for test set

    Args:
        data_path (str): Path to directory containing dataset files
                        (train_FD001.txt, test_FD001.txt, RUL_FD001.txt)

    Returns:
        tuple: (train_df, test_df, test_rul)
            - train_df: Training data with RUL column (pd.DataFrame)
            - test_df: Test data without RUL (pd.DataFrame)
            - test_rul: True RUL values for test engines (pd.DataFrame)

    Dataset Format:
        Column 1: unit_id (engine identifier)
        Column 2: cycle (operational cycle number)
        Columns 3-5: op_setting_1, op_setting_2, op_setting_3
        Columns 6-26: sensor_1 to sensor_21
    """

    data_path = Path(data_path)

    # Define column names (26 columns total)
    # No headers in raw files - must specify during loading
    column_names = (
        ['unit_id', 'cycle'] +  # Index columns
        ['op_setting_1', 'op_setting_2', 'op_setting_3'] +  # Operational settings
        [f'sensor_{i}' for i in range(1, 22)]  # 21 sensor measurements
    )

    # ========================================================================
    # STEP 1: Load Training Data
    # ========================================================================
    train_file = data_path / "train_FD001.txt"

    # Load space-separated file without headers
    train_df = pd.read_csv(
        train_file,
        sep=r'\s+',  # Multiple spaces as delimiter
        header=None,  # No header row in file
        names=column_names  # Apply our column names
    )

    print(f"✓ Loaded training data: {train_df.shape}")
    print(f"  - Records: {len(train_df)}")
    print(f"  - Engines: {train_df['unit_id'].nunique()}")
    print(f"  - Features: {len(column_names) - 2} (excluding unit_id, cycle)")

    # ========================================================================
    # STEP 2: Calculate RUL (Remaining Useful Life)
    # ========================================================================
    # For each engine, RUL = max_cycle_of_engine - current_cycle
    # Example: Engine runs for 150 cycles
    #   - At cycle 1:   RUL = 150 - 1 = 149
    #   - At cycle 50:  RUL = 150 - 50 = 100
    #   - At cycle 150: RUL = 150 - 150 = 0 (failure)

    # Group by engine and find max cycle for each
    max_cycle_per_engine = train_df.groupby('unit_id')['cycle'].max()

    # Map max_cycle to each row based on unit_id
    train_df['max_cycle'] = train_df['unit_id'].map(max_cycle_per_engine)

    # Calculate RUL
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']

    # Drop temporary column (no longer needed)
    train_df = train_df.drop(columns=['max_cycle'])

    print(f"\n✓ Calculated RUL for training data:")
    print(f"  - Min RUL: {train_df['RUL'].min()} cycles (end of life)")
    print(f"  - Max RUL: {train_df['RUL'].max()} cycles (early lifecycle)")
    print(f"  - Mean RUL: {train_df['RUL'].mean():.2f} cycles")

    # ========================================================================
    # STEP 3: Cap RUL at 125 Cycles (Piece-Wise RUL)
    # ========================================================================
    # Standard practice in prognostics: early-life predictions are less critical
    # RUL > 125 cycles → set to 125 (healthy state)
    # RUL ≤ 125 cycles → keep actual value (degradation phase)
    # This creates a piece-wise linear RUL profile

    train_df['RUL'] = train_df['RUL'].clip(upper=125)

    print(f"\n✓ Applied RUL capping at 125 cycles:")
    print(f"  - Capped records: {(train_df['RUL'] == 125).sum()} / {len(train_df)}")
    print(f"  - New max RUL: {train_df['RUL'].max()} cycles")

    # ========================================================================
    # STEP 4: Load Test Data
    # ========================================================================
    test_file = data_path / "test_FD001.txt"

    # Same loading procedure as training data
    test_df = pd.read_csv(
        test_file,
        sep=r'\s+',
        header=None,
        names=column_names
    )

    print(f"\n✓ Loaded test data: {test_df.shape}")
    print(f"  - Records: {len(test_df)}")
    print(f"  - Engines: {test_df['unit_id'].nunique()}")

    # ========================================================================
    # STEP 5: Load Ground Truth RUL for Test Set
    # ========================================================================
    # Test data does NOT include failure cycles (censored data)
    # RUL_FD001.txt contains the true RUL at the last observed cycle
    # One value per test engine (100 values)

    rul_file = data_path / "RUL_FD001.txt"

    test_rul = pd.read_csv(
        rul_file,
        header=None,
        names=['RUL']
    )

    # Apply same RUL capping (consistency with training)
    test_rul['RUL'] = test_rul['RUL'].clip(upper=125)

    print(f"\n✓ Loaded test RUL ground truth: {test_rul.shape}")
    print(f"  - Min RUL: {test_rul['RUL'].min()} cycles")
    print(f"  - Max RUL: {test_rul['RUL'].max()} cycles")
    print(f"  - Mean RUL: {test_rul['RUL'].mean():.2f} cycles")

    # ========================================================================
    # Data Loading Complete
    # ========================================================================
    print("\n" + "="*60)
    print("DATA LOADING SUMMARY")
    print("="*60)
    print(f"Training set:   {train_df.shape[0]:>6} rows × {train_df.shape[1]:>2} columns (includes RUL)")
    print(f"Test set:       {test_df.shape[0]:>6} rows × {test_df.shape[1]:>2} columns")
    print(f"Test RUL:       {test_rul.shape[0]:>6} values")
    print("="*60 + "\n")

    return train_df, test_df, test_rul


def get_feature_columns() -> list:
    """
    Get list of feature column names for modeling.

    Returns:
        list: Column names for features (operational settings + sensors)
              Excludes unit_id, cycle, and RUL
    """
    return (
        ['op_setting_1', 'op_setting_2', 'op_setting_3'] +
        [f'sensor_{i}' for i in range(1, 22)]
    )


def get_column_names() -> list:
    """
    Get complete list of column names in dataset.

    Returns:
        list: All 26 column names in order
    """
    return (
        ['unit_id', 'cycle'] +
        ['op_setting_1', 'op_setting_2', 'op_setting_3'] +
        [f'sensor_{i}' for i in range(1, 22)]
    )


# ============================================================================
# Example Usage & Testing
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("NASA C-MAPS FD001 DATA LOADER")
    print("="*60 + "\n")

    # Load dataset from current directory
    # Assumes train_FD001.txt, test_FD001.txt, RUL_FD001.txt are present
    train_df, test_df, test_rul = load_data(".")

    # Display sample data
    print("\n--- Training Data Sample ---")
    print(train_df.head(3))

    print("\n--- Test Data Sample ---")
    print(test_df.head(3))

    print("\n--- Test RUL Sample ---")
    print(test_rul.head(5))

    # Display available features
    features = get_feature_columns()
    print(f"\n--- Available Features ({len(features)}) ---")
    print(f"Operational Settings: {features[:3]}")
    print(f"Sensors: {features[3:6]} ... {features[-3:]}")

    print("\n Data loading successful! Ready for feature engineering.\n")
