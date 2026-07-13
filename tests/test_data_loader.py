"""
Unit tests for NASA C-MAPS data loader
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append('src')
from data_loader import CMAPSDataLoader


class TestCMAPSDataLoader(unittest.TestCase):
    """Test cases for CMAPSDataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = CMAPSDataLoader(data_dir='data')

    def test_initialization(self):
        """Test loader initializes with correct column names."""
        self.assertEqual(len(self.loader.index_cols), 2)
        self.assertEqual(len(self.loader.operational_settings), 3)
        self.assertEqual(len(self.loader.sensor_cols), 21)
        self.assertEqual(len(self.loader.all_cols), 26)  # 2 + 3 + 21

    def test_column_names(self):
        """Test column naming convention."""
        self.assertIn('unit_id', self.loader.index_cols)
        self.assertIn('cycle', self.loader.index_cols)
        self.assertEqual(self.loader.sensor_cols[0], 'sensor_1')
        self.assertEqual(self.loader.sensor_cols[-1], 'sensor_21')

    def test_get_feature_columns(self):
        """Test feature column retrieval."""
        features = self.loader.get_feature_columns()
        self.assertEqual(len(features), 24)  # 3 op_settings + 21 sensors
        self.assertIn('op_setting_1', features)
        self.assertIn('sensor_1', features)

    # Integration tests (require actual data files)
    @unittest.skipIf(not Path('data/train_FD001.txt').exists(),
                     "Training data not available")
    def test_load_train_data(self):
        """Test training data loading."""
        df = self.loader.load_train_data()

        # Check structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('RUL', df.columns)
        self.assertIn('unit_id', df.columns)

        # Check RUL calculation
        self.assertTrue((df['RUL'] >= 0).all())
        self.assertTrue((df['RUL'] <= 125).all())  # Capped at 125

        # Check no missing values
        self.assertEqual(df.isnull().sum().sum(), 0)

    @unittest.skipIf(not Path('data/test_FD001.txt').exists(),
                     "Test data not available")
    def test_load_test_data(self):
        """Test test data loading."""
        df, true_rul = self.loader.load_test_data()

        # Check structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(true_rul, pd.DataFrame)

        # Check true RUL
        self.assertIn('RUL', true_rul.columns)
        self.assertTrue((true_rul['RUL'] >= 0).all())
        self.assertTrue((true_rul['RUL'] <= 125).all())


class TestRULCalculation(unittest.TestCase):
    """Test RUL calculation logic."""

    def test_rul_calculation_simple(self):
        """Test RUL calculation on synthetic data."""
        # Create simple test data: 1 engine, 5 cycles
        data = {
            'unit_id': [1, 1, 1, 1, 1],
            'cycle': [1, 2, 3, 4, 5],
            'sensor_1': [100, 100, 100, 100, 100]
        }
        df = pd.DataFrame(data)

        # Add dummy columns to match expected structure
        loader = CMAPSDataLoader()
        for col in loader.operational_settings:
            df[col] = 0
        for col in loader.sensor_cols:
            if col not in df.columns:
                df[col] = 0

        # Reorder columns
        df = df[loader.all_cols]

        # Calculate RUL
        result = loader._calculate_rul(df)

        # Expected RUL: [4, 3, 2, 1, 0]
        expected_rul = [4, 3, 2, 1, 0]
        self.assertEqual(result['RUL'].tolist(), expected_rul)


if __name__ == '__main__':
    unittest.main()
