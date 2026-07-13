"""
Data Loader for NASA C-MAPS FD001 Dataset
Loads and preprocesses turbofan engine degradation data
"""

import pandas as pd
import numpy as np
from pathlib import Path


class CMAPSDataLoader:
    """
    Loader for NASA C-MAPS (Commercial Modular Aero-Propulsion System Simulation) dataset.

    Dataset contains run-to-failure sensor readings from turbofan engines operating
    under various conditions. Each engine starts healthy and degrades over time.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize data loader.

        Args:
            data_dir (str): Path to directory containing C-MAPS dataset files
        """
        self.data_dir = Path(data_dir)

        # Column names as per NASA C-MAPS documentation
        self.index_cols = ['unit_id', 'cycle']
        self.operational_settings = ['op_setting_1', 'op_setting_2', 'op_setting_3']
        self.sensor_cols = [f'sensor_{i}' for i in range(1, 22)]  # 21 sensors
        self.all_cols = self.index_cols + self.operational_settings + self.sensor_cols

    def load_train_data(self) -> pd.DataFrame:
        """
        Load training dataset and calculate Remaining Useful Life (RUL).

        Returns:
            pd.DataFrame: Training data with RUL column added
        """
        train_path = self.data_dir / "train_FD001.txt"

        if not train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {train_path}. "
                "Please download from NASA Prognostics Data Repository."
            )

        # Load data - space-separated, no header
        df = pd.read_csv(train_path, sep=r'\s+', header=None, names=self.all_cols)

        # Calculate RUL: for each engine, RUL = max_cycle - current_cycle
        df = self._calculate_rul(df)

        # Cap RUL at 125 cycles (standard practice in C-MAPS research)
        df['RUL'] = df['RUL'].clip(upper=125)

        print(f"✓ Loaded training data: {len(df)} records from {df['unit_id'].nunique()} engines")
        return df

    def load_test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load test dataset and ground truth RUL values.

        Returns:
            tuple: (test_data, true_rul_values)
                - test_data: DataFrame with sensor readings
                - true_rul_values: Series with actual RUL for each engine
        """
        test_path = self.data_dir / "test_FD001.txt"
        rul_path = self.data_dir / "RUL_FD001.txt"

        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found at {test_path}")
        if not rul_path.exists():
            raise FileNotFoundError(f"RUL ground truth not found at {rul_path}")

        # Load test data
        df = pd.read_csv(test_path, sep=r'\s+', header=None, names=self.all_cols)

        # Load ground truth RUL (one value per engine, representing RUL at last observed cycle)
        true_rul = pd.read_csv(rul_path, header=None, names=['RUL'])

        # Cap RUL at 125 cycles (consistent with training)
        true_rul['RUL'] = true_rul['RUL'].clip(upper=125)

        print(f"✓ Loaded test data: {len(df)} records from {df['unit_id'].nunique()} engines")
        return df, true_rul

    def _calculate_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Remaining Useful Life for each observation in training data.

        For each engine unit, RUL = (max cycle for that unit) - (current cycle)

        Args:
            df (pd.DataFrame): Raw data

        Returns:
            pd.DataFrame: Data with RUL column added
        """
        # Get max cycle for each engine
        max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
        max_cycles.columns = ['unit_id', 'max_cycle']

        # Merge max cycle back to original data
        df = df.merge(max_cycles, on='unit_id', how='left')

        # Calculate RUL
        df['RUL'] = df['max_cycle'] - df['cycle']

        # Drop temporary column
        df = df.drop(columns=['max_cycle'])

        return df

    def get_feature_columns(self) -> list[str]:
        """
        Get list of feature columns (operational settings + sensors).

        Returns:
            list: Column names for model features
        """
        return self.operational_settings + self.sensor_cols


if __name__ == "__main__":
    # Example usage
    loader = CMAPSDataLoader()

    print("\n=== Loading NASA C-MAPS FD001 Dataset ===\n")

    # Load training data
    train_df = loader.load_train_data()
    print(f"Training shape: {train_df.shape}")
    print(f"RUL statistics:\n{train_df['RUL'].describe()}\n")

    # Load test data
    test_df, true_rul = loader.load_test_data()
    print(f"Test shape: {test_df.shape}")
    print(f"Ground truth RUL shape: {true_rul.shape}")

    print("\n✓ Data loading successful!")
