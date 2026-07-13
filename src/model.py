"""
XGBoost Model for Remaining Useful Life (RUL) Prediction
Trains gradient boosting regressor on turbofan engine sensor data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


class RULPredictor:
    """
    Remaining Useful Life predictor using XGBoost.

    XGBoost is well-suited for this task due to:
    - Robustness to feature interactions
    - Built-in regularization to prevent overfitting
    - Ability to capture non-linear degradation patterns
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42
    ):
        """
        Initialize XGBoost regressor with optimized hyperparameters.

        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Step size shrinkage to prevent overfitting
            subsample (float): Fraction of samples used per tree
            random_state (int): Random seed for reproducibility
        """
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            random_state=random_state,
            objective='reg:squarederror',
            n_jobs=-1,  # Use all CPU cores
            verbosity=0
        )

        self.feature_columns = None

    def train(
        self,
        train_df: pd.DataFrame,
        target_col: str = 'RUL',
        exclude_cols: list = None
    ) -> dict:
        """
        Train the RUL prediction model.

        Args:
            train_df (pd.DataFrame): Training data with features and RUL
            target_col (str): Name of target column
            exclude_cols (list): Columns to exclude from features

        Returns:
            dict: Training metrics (RMSE, R2)
        """
        if exclude_cols is None:
            exclude_cols = ['unit_id', 'cycle', target_col]

        # Separate features and target
        self.feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        X_train = train_df[self.feature_columns]
        y_train = train_df[target_col]

        print("\n=== Training XGBoost Model ===\n")
        print(f"Training samples: {len(X_train)}")
        print(f"Features: {len(self.feature_columns)}")

        # Train model
        self.model.fit(X_train, y_train, verbose=False)

        # Evaluate on training set
        y_pred = self.model.predict(X_train)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        r2 = r2_score(y_train, y_pred)

        print(f"\n✓ Training complete!")
        print(f"  Training RMSE: {rmse:.2f} cycles")
        print(f"  Training R² Score: {r2:.4f}\n")

        return {
            'rmse': rmse,
            'r2': r2
        }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL for new data.

        Args:
            df (pd.DataFrame): Feature data

        Returns:
            np.ndarray: Predicted RUL values
        """
        if self.feature_columns is None:
            raise RuntimeError("Model must be trained before prediction")

        X = df[self.feature_columns]
        predictions = self.model.predict(X)

        # RUL cannot be negative
        predictions = np.maximum(predictions, 0)

        return predictions

    def predict_for_engines(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict RUL at the last observed cycle for each engine.

        For test set evaluation, we need predictions at the final cycle only.

        Args:
            df (pd.DataFrame): Test data with multiple cycles per engine

        Returns:
            pd.DataFrame: Predictions at last cycle for each engine
        """
        # Get last cycle for each engine
        last_cycles = df.groupby('unit_id').tail(1).copy()

        # Predict RUL
        last_cycles['predicted_RUL'] = self.predict(last_cycles)

        return last_cycles[['unit_id', 'predicted_RUL']].reset_index(drop=True)

    def save_model(self, filepath: str = "model.pkl"):
        """
        Save trained model to disk.

        Args:
            filepath (str): Path to save model
        """
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

    def load_model(self, filepath: str = "model.pkl"):
        """
        Load trained model from disk.

        Args:
            filepath (str): Path to model file
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']

        print(f"✓ Model loaded from {filepath}")


if __name__ == "__main__":
    # Example training pipeline
    from data_loader import CMAPSDataLoader
    from feature_engineering import FeatureEngineer

    print("\n" + "="*50)
    print("  NASA C-MAPS RUL Prediction - Training Pipeline")
    print("="*50)

    # Load data
    loader = CMAPSDataLoader()
    train_df = loader.load_train_data()

    # Engineer features
    engineer = FeatureEngineer()
    train_processed = engineer.fit_transform(train_df)

    # Train model
    predictor = RULPredictor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8
    )

    metrics = predictor.train(train_processed)

    # Save model
    predictor.save_model("model.pkl")

    print("="*50)
    print("✓ Training pipeline complete!")
    print("="*50 + "\n")
