"""
Feature Engineering for NASA C-MAPS FD001 Dataset
Simple function-based approach for preparing features for modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data


def prepare_features(data_path='data/'):
    """
    Complete feature engineering pipeline for C-MAPS dataset.

    Steps:
    1. Load raw data
    2. Remove low-variance sensors
    3. Add rolling statistics (mean and std over 5 and 10 cycles)
    4. Normalize features

    Args:
        data_path (str): Path to data directory containing dataset files

    Returns:
        tuple: (X_train, y_train, X_test, test_rul, scaler, feature_names)
            - X_train: Training features (numpy array)
            - y_train: Training targets (numpy array)
            - X_test: Test features (numpy array)
            - test_rul: True RUL values for test set (numpy array)
            - scaler: Fitted MinMaxScaler
            - feature_names: List of feature column names
    """

    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60 + "\n")

    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("Step 1: Loading data...")
    train_df, test_df, test_rul_df = load_data(data_path)

    # ========================================================================
    # STEP 2: Calculate Variance and Remove Low-Variance Sensors
    # ========================================================================
    print("\nStep 2: Removing low-variance sensors...")

    # Get all sensor and operational setting columns
    sensor_cols = [col for col in train_df.columns
                   if col.startswith('sensor_') or col.startswith('op_setting_')]

    # Calculate variance on training data
    variances = train_df[sensor_cols].var()

    # Remove sensors with variance < 0.01
    variance_threshold = 0.01
    removed_sensors = variances[variances < variance_threshold].index.tolist()
    kept_sensors = variances[variances >= variance_threshold].index.tolist()

    print(f"\nRemoved sensors (variance < {variance_threshold}):")
    for sensor in removed_sensors:
        print(f"  - {sensor} (variance: {variances[sensor]:.6f})")

    print(f"\nKept sensors ({len(kept_sensors)} total):")
    for sensor in kept_sensors:
        print(f"  - {sensor} (variance: {variances[sensor]:.2f})")

    # Filter to kept sensors only
    train_df = train_df[['unit_id', 'cycle'] + kept_sensors + ['RUL']]
    test_df = test_df[['unit_id', 'cycle'] + kept_sensors]

    # ========================================================================
    # STEP 3: Add Rolling Statistics
    # ========================================================================
    print(f"\nStep 3: Adding rolling statistics...")

    # Apply rolling features to training data
    train_df = _add_rolling_features(train_df, kept_sensors)

    # Apply rolling features to test data
    test_df = _add_rolling_features(test_df, kept_sensors)

    print(f"✓ Added rolling features (5 and 10 cycle windows)")
    print(f"  - Rolling mean 5 cycles")
    print(f"  - Rolling mean 10 cycles")
    print(f"  - Rolling std 5 cycles")
    print(f"  - Rolling std 10 cycles")

    # ========================================================================
    # STEP 4: Prepare Feature Matrices
    # ========================================================================
    print(f"\nStep 4: Preparing feature matrices...")

    # Get all feature columns (exclude unit_id, cycle, RUL)
    feature_cols = [col for col in train_df.columns
                    if col not in ['unit_id', 'cycle', 'RUL']]

    # Extract features and target
    X_train = train_df[feature_cols].values
    y_train = train_df['RUL'].values
    X_test = test_df[feature_cols].values
    test_rul = test_rul_df['RUL'].values

    print(f"✓ X_train shape: {X_train.shape}")
    print(f"✓ y_train shape: {y_train.shape}")
    print(f"✓ X_test shape: {X_test.shape}")
    print(f"✓ test_rul shape: {test_rul.shape}")

    # ========================================================================
    # STEP 5: Normalize Features (Fit on Train, Transform Both)
    # ========================================================================
    print(f"\nStep 5: Normalizing features...")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"✓ Fitted MinMaxScaler on training data")
    print(f"✓ Transformed both train and test sets to [0, 1] range")

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"Final feature count: {len(feature_cols)}")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print("="*60 + "\n")

    return X_train, y_train, X_test, test_rul, scaler, feature_cols


def _add_rolling_features(df, sensor_cols):
    """
    Add rolling mean and std features for each sensor.

    Args:
        df (pd.DataFrame): Input dataframe with unit_id and cycle columns
        sensor_cols (list): List of sensor column names

    Returns:
        pd.DataFrame: Dataframe with rolling features added
    """
    df = df.copy()

    # For each kept sensor, calculate rolling statistics
    for col in sensor_cols:
        # Rolling mean 5 cycles
        df[f'{col}_rolling_mean_5'] = df.groupby('unit_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )

        # Rolling mean 10 cycles
        df[f'{col}_rolling_mean_10'] = df.groupby('unit_id')[col].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )

        # Rolling std 5 cycles
        df[f'{col}_rolling_std_5'] = df.groupby('unit_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        df[f'{col}_rolling_std_5'] = df[f'{col}_rolling_std_5'].fillna(0)

        # Rolling std 10 cycles
        df[f'{col}_rolling_std_10'] = df.groupby('unit_id')[col].transform(
            lambda x: x.rolling(window=10, min_periods=1).std()
        )
        df[f'{col}_rolling_std_10'] = df[f'{col}_rolling_std_10'].fillna(0)

    return df


if __name__ == "__main__":
    # Test the feature engineering pipeline
    X_train, y_train, X_test, test_rul, scaler, feature_names = prepare_features('data/')

    print(f"\n--- Feature Engineering Test Complete ---")
    print(f"Feature names: {feature_names[:5]} ... ({len(feature_names)} total)")
