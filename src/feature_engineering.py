"""
Feature Engineering for Turbofan Engine Degradation Prediction
Removes uninformative sensors, adds temporal features, and normalizes data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Optional


class FeatureEngineer:
    """
    Processes raw sensor data into ML-ready features.

    Steps:
    1. Remove constant/low-variance sensors
    2. Add rolling statistics (temporal degradation patterns)
    3. Normalize features to [0, 1] range
    """

    def __init__(self, variance_threshold: float = 0.01):
        """
        Initialize feature engineer.

        Args:
            variance_threshold (float): Sensors with variance below this are removed
        """
        self.variance_threshold = variance_threshold
        self.scaler = MinMaxScaler()
        self.selected_features = None
        self.feature_columns = None

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit feature engineering pipeline and transform training data.

        Args:
            df (pd.DataFrame): Raw training data with RUL

        Returns:
            pd.DataFrame: Engineered features ready for modeling
        """
        df = df.copy()

        print("\n=== Feature Engineering ===\n")

        # Step 1: Remove uninformative sensors
        df = self._remove_constant_features(df)

        # Step 2: Add rolling window features
        df = self._add_rolling_features(df)

        # Step 3: Normalize features
        df = self._normalize_features(df, fit=True)

        print(f"✓ Final feature set: {len(self.feature_columns)} features")
        print(f"  Total samples: {len(df)}\n")

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test data using fitted pipeline.

        Args:
            df (pd.DataFrame): Raw test data

        Returns:
            pd.DataFrame: Engineered features
        """
        if self.selected_features is None:
            raise RuntimeError("Must call fit_transform() before transform()")

        df = df.copy()

        # Apply same transformations as training
        df = df[['unit_id', 'cycle'] + self.selected_features]
        df = self._add_rolling_features(df)
        df = self._normalize_features(df, fit=False)

        return df

    def _remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove sensors with near-zero variance (provide no information).

        Args:
            df (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with constant features removed
        """
        # Identify sensor and operational setting columns
        feature_cols = [col for col in df.columns if col.startswith('sensor_') or col.startswith('op_setting_')]

        # Calculate variance for each feature
        variances = df[feature_cols].var()

        # Keep only features with variance above threshold
        informative_features = variances[variances > self.variance_threshold].index.tolist()

        # Store selected features
        self.selected_features = informative_features

        print(f"✓ Removed {len(feature_cols) - len(informative_features)} low-variance features")
        print(f"  Kept {len(informative_features)} informative sensors")

        # Keep identification columns + selected features + target
        keep_cols = ['unit_id', 'cycle'] + informative_features
        if 'RUL' in df.columns:
            keep_cols.append('RUL')

        return df[keep_cols]

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling mean and std features to capture degradation trends.

        Rolling statistics help the model learn temporal patterns in sensor drift.

        Args:
            df (pd.DataFrame): Input data

        Returns:
            pd.DataFrame: Data with rolling features added
        """
        feature_cols = [col for col in df.columns if col.startswith('sensor_') or col.startswith('op_setting_')]

        rolling_features = []
        windows = [5, 10]  # 5 and 10 cycle windows

        # Calculate rolling statistics per engine unit
        for window in windows:
            for col in feature_cols:
                # Rolling mean
                col_mean = f'{col}_rolling_mean_{window}'
                df[col_mean] = df.groupby('unit_id')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )
                rolling_features.append(col_mean)

                # Rolling std
                col_std = f'{col}_rolling_std_{window}'
                df[col_std] = df.groupby('unit_id')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std()
                )
                # Fill NaN (occurs when std of single value)
                df[col_std] = df[col_std].fillna(0)
                rolling_features.append(col_std)

        print(f"✓ Added {len(rolling_features)} rolling window features (windows: {windows})")

        return df

    def _normalize_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Normalize features to [0, 1] range using MinMaxScaler.

        Args:
            df (pd.DataFrame): Input data
            fit (bool): If True, fit scaler on data; otherwise use existing scaler

        Returns:
            pd.DataFrame: Normalized data
        """
        # Identify all feature columns (exclude unit_id, cycle, RUL)
        feature_cols = [col for col in df.columns if col not in ['unit_id', 'cycle', 'RUL']]

        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
            self.feature_columns = feature_cols
            print(f"✓ Normalized {len(feature_cols)} features to [0, 1] range")
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])

        return df

    def get_feature_names(self) -> list[str]:
        """
        Get list of all engineered feature names.

        Returns:
            list: Feature column names
        """
        return self.feature_columns if self.feature_columns else []


if __name__ == "__main__":
    # Example usage
    from data_loader import CMAPSDataLoader

    loader = CMAPSDataLoader()
    train_df = loader.load_train_data()

    engineer = FeatureEngineer()
    train_processed = engineer.fit_transform(train_df)

    print(f"Original columns: {train_df.shape[1]}")
    print(f"Processed columns: {train_processed.shape[1]}")
    print(f"\nSample features:\n{train_processed.head()}")
