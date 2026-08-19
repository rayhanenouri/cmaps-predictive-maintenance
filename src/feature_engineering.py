"""
Feature engineering for C-MAPS FD001.
Removes constant sensors and adds rolling statistics to capture degradation trends.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from data_loader import load_data


def prepare_features(data_path='data/'):
    """Complete feature engineering pipeline.

    Returns X_train, y_train, X_test, test_rul, scaler, and feature names.
    """

    print("\nfeature engineering pipeline\n")

    # load raw data
    train_df, test_df, test_rul_df = load_data(data_path)

    # some sensors show no variation on FD001 single operating condition
    # these add no information for degradation prediction
    sensor_cols = [col for col in train_df.columns
                   if col.startswith('sensor_') or col.startswith('op_setting_')]

    variances = train_df[sensor_cols].var()
    variance_threshold = 0.01
    removed_sensors = variances[variances < variance_threshold].index.tolist()
    kept_sensors = variances[variances >= variance_threshold].index.tolist()

    print(f"removed {len(removed_sensors)} constant sensors (variance < {variance_threshold}):")
    for sensor in removed_sensors:
        print(f"  {sensor}: {variances[sensor]:.6f}")

    print(f"\nkept {len(kept_sensors)} informative sensors")

    # filter to kept sensors
    train_df = train_df[['unit_id', 'cycle'] + kept_sensors + ['RUL']]
    test_df = test_df[['unit_id', 'cycle'] + kept_sensors]

    # turbofan degradation is a temporal process
    # rolling statistics capture short-term trends better than raw values
    print(f"\nadding rolling statistics (5 and 10 cycle windows)")
    train_df = _add_rolling_features(train_df, kept_sensors)
    test_df = _add_rolling_features(test_df, kept_sensors)

    # prepare feature matrices
    feature_cols = [col for col in train_df.columns
                    if col not in ['unit_id', 'cycle', 'RUL']]

    X_train = train_df[feature_cols].values
    y_train = train_df['RUL'].values
    X_test = test_df[feature_cols].values
    test_rul = test_rul_df['RUL'].values

    print(f"\nfeature matrix shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")

    # normalize to [0,1] range
    # fit on training data only to prevent data leakage
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"\napplied MinMax normalization")
    print(f"final feature count: {len(feature_cols)}")

    return X_train, y_train, X_test, test_rul, scaler, feature_cols


def _add_rolling_features(df, sensor_cols):
    """Add rolling mean and std for each sensor.

    Rolling statistics smooth out noise and capture degradation trends.
    """
    df = df.copy()

    for col in sensor_cols:
        # 5-cycle rolling mean
        df[f'{col}_rolling_mean_5'] = df.groupby('unit_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).mean()
        )

        # 10-cycle rolling mean
        df[f'{col}_rolling_mean_10'] = df.groupby('unit_id')[col].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )

        # 5-cycle rolling std (variability indicator)
        df[f'{col}_rolling_std_5'] = df.groupby('unit_id')[col].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        )
        df[f'{col}_rolling_std_5'] = df[f'{col}_rolling_std_5'].fillna(0)

        # 10-cycle rolling std
        df[f'{col}_rolling_std_10'] = df.groupby('unit_id')[col].transform(
            lambda x: x.rolling(window=10, min_periods=1).std()
        )
        df[f'{col}_rolling_std_10'] = df[f'{col}_rolling_std_10'].fillna(0)

    return df


if __name__ == "__main__":
    X_train, y_train, X_test, test_rul, scaler, feature_names = prepare_features('data/')
    print(f"\nfeature engineering test complete")
    print(f"generated {len(feature_names)} features")
