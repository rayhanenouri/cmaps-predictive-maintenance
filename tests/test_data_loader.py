# tests for data loading pipeline
# requires data/ directory with FD001 files

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('src')
from src.data_loader import load_data


@pytest.fixture
def data_path():
    """path to data directory"""
    return 'data/'


@pytest.mark.skipif(
    not Path('data/train_FD001.txt').exists(),
    reason="data files not present"
)
def test_load_data_returns_three_objects(data_path):
    """load_data should return train_df, test_df, test_rul"""
    train_df, test_df, test_rul = load_data(data_path)

    assert train_df is not None
    assert test_df is not None
    assert test_rul is not None
    assert isinstance(train_df, pd.DataFrame)
    assert isinstance(test_df, pd.DataFrame)
    assert isinstance(test_rul, pd.DataFrame)


@pytest.mark.skipif(
    not Path('data/train_FD001.txt').exists(),
    reason="data files not present"
)
def test_train_df_shape(data_path):
    """train_df should have 20631 rows and 27 columns"""
    train_df, _, _ = load_data(data_path)

    assert train_df.shape[0] == 20631
    assert train_df.shape[1] == 27


@pytest.mark.skipif(
    not Path('data/train_FD001.txt').exists(),
    reason="data files not present"
)
def test_train_rul_values(data_path):
    """RUL column should have values between 0 and 125"""
    train_df, _, _ = load_data(data_path)

    assert 'RUL' in train_df.columns
    assert train_df['RUL'].min() >= 0
    assert train_df['RUL'].max() <= 125
    assert train_df['RUL'].notna().all()


@pytest.mark.skipif(
    not Path('data/test_FD001.txt').exists(),
    reason="data files not present"
)
def test_test_df_engines(data_path):
    """test_df should have 100 unique engines"""
    _, test_df, _ = load_data(data_path)

    assert 'unit_id' in test_df.columns
    assert test_df['unit_id'].nunique() == 100


@pytest.mark.skipif(
    not Path('data/RUL_FD001.txt').exists(),
    reason="data files not present"
)
def test_test_rul_length(data_path):
    """test_rul should have 100 values"""
    _, _, test_rul = load_data(data_path)

    assert len(test_rul) == 100
    assert 'RUL' in test_rul.columns
