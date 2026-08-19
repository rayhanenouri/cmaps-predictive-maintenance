"""
Data loader for NASA C-MAPS FD001 turbofan degradation dataset.
Loads train/test splits and computes RUL labels.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data(data_path: str = "."):
    """Load C-MAPS FD001 splits and compute RUL labels.

    Returns train_df with RUL column, test_df, and ground truth test_rul.
    """

    data_path = Path(data_path)

    # 26 columns total: unit_id, cycle, 3 operational settings, 21 sensors
    column_names = (
        ['unit_id', 'cycle'] +
        ['op_setting_1', 'op_setting_2', 'op_setting_3'] +
        [f'sensor_{i}' for i in range(1, 22)]
    )

    # load training set
    train_file = data_path / "train_FD001.txt"
    train_df = pd.read_csv(
        train_file,
        sep=r'\s+',
        header=None,
        names=column_names
    )

    print(f"loaded training data: {len(train_df)} records from {train_df['unit_id'].nunique()} engines")

    # training set runs each engine to failure so max cycle = end of life
    max_cycle = train_df.groupby('unit_id')['cycle'].max()
    train_df['RUL'] = train_df['unit_id'].map(max_cycle) - train_df['cycle']

    # degradation only becomes measurable in the last 125 cycles on FD001
    # this is piece-wise RUL normalization from the NASA paper
    train_df['RUL'] = train_df['RUL'].clip(upper=125)

    print(f"calculated RUL: min={train_df['RUL'].min()}, max={train_df['RUL'].max()}, mean={train_df['RUL'].mean():.1f}")

    # load test set
    test_file = data_path / "test_FD001.txt"
    test_df = pd.read_csv(
        test_file,
        sep=r'\s+',
        header=None,
        names=column_names
    )

    print(f"loaded test data: {len(test_df)} records from {test_df['unit_id'].nunique()} engines")

    # test set is censored at unknown points before failure
    # ground truth RUL is provided separately
    rul_file = data_path / "RUL_FD001.txt"
    test_rul = pd.read_csv(rul_file, header=None, names=['RUL'])
    test_rul['RUL'] = test_rul['RUL'].clip(upper=125)

    print(f"loaded ground truth RUL: min={test_rul['RUL'].min()}, max={test_rul['RUL'].max()}, mean={test_rul['RUL'].mean():.1f}")

    return train_df, test_df, test_rul


def get_feature_columns():
    """Return list of feature columns (op settings + sensors)."""
    return (
        ['op_setting_1', 'op_setting_2', 'op_setting_3'] +
        [f'sensor_{i}' for i in range(1, 22)]
    )


def get_column_names():
    """Return all 26 column names."""
    return (
        ['unit_id', 'cycle'] +
        ['op_setting_1', 'op_setting_2', 'op_setting_3'] +
        [f'sensor_{i}' for i in range(1, 22)]
    )


if __name__ == "__main__":
    print("\nNASA C-MAPS FD001 Data Loader\n")

    train_df, test_df, test_rul = load_data(".")

    print("\ntraining sample:")
    print(train_df.head(3))

    print("\ntest sample:")
    print(test_df.head(3))

    print("\ndata loading complete")
