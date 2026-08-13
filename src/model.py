"""
Model Training for NASA C-MAPS FD001 Predictive Maintenance
XGBoost-based Remaining Useful Life (RUL) prediction
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from feature_engineering import prepare_features


def train_model(data_path='data/'):
    """
    Train XGBoost model for RUL prediction.

    Steps:
    1. Prepare features using feature engineering pipeline
    2. Train XGBoost regressor with specified hyperparameters
    3. Make predictions on test set
    4. Calculate and display metrics (RMSE, R2)
    5. Save model and scaler to models/ directory

    Args:
        data_path (str): Path to data directory

    Returns:
        tuple: (model, predictions, test_rul, feature_names)
            - model: Trained XGBoost model
            - predictions: Predictions on test set
            - test_rul: True RUL values for test set
            - feature_names: List of feature column names
    """

    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60 + "\n")

    # ========================================================================
    # STEP 1: Prepare Features
    # ========================================================================
    print("Step 1: Preparing features...")
    X_train, y_train, X_test, test_rul, scaler, feature_names = prepare_features(data_path)

    # ========================================================================
    # STEP 2: Train XGBoost Regressor
    # ========================================================================
    print("\nStep 2: Training XGBoost model...")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='reg:squarederror',
        n_jobs=-1
    )

    print(f"\nModel hyperparameters:")
    print(f"  - n_estimators: 300")
    print(f"  - max_depth: 6")
    print(f"  - learning_rate: 0.05")
    print(f"  - subsample: 0.8")
    print(f"  - colsample_bytree: 0.8")
    print(f"  - random_state: 42")

    print(f"\nTraining model on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    model.fit(X_train, y_train, verbose=False)
    print(f"✓ Training complete!")

    # ========================================================================
    # STEP 3: Make Predictions
    # ========================================================================
    print(f"\nStep 3: Making predictions on test set...")

    # Get predictions for all test cycles
    all_predictions = model.predict(X_test)
    all_predictions = np.maximum(all_predictions, 0)  # Ensure non-negative

    # Load test data to get last cycle per engine
    from data_loader import load_data
    _, test_df, _ = load_data(data_path)

    # Get last cycle prediction for each engine
    predictions = []
    current_idx = 0
    for unit_id in sorted(test_df['unit_id'].unique()):
        unit_mask = test_df['unit_id'] == unit_id
        unit_cycle_count = unit_mask.sum()

        # Get the last prediction for this unit
        last_prediction = all_predictions[current_idx + unit_cycle_count - 1]
        predictions.append(last_prediction)

        current_idx += unit_cycle_count

    predictions = np.array(predictions)

    print(f"✓ Generated {len(predictions)} predictions (one per engine)")

    # ========================================================================
    # STEP 4: Calculate Metrics
    # ========================================================================
    print(f"\nStep 4: Calculating metrics...")

    rmse = np.sqrt(mean_squared_error(test_rul, predictions))
    r2 = r2_score(test_rul, predictions)

    # ========================================================================
    # STEP 5: Save Model and Scaler
    # ========================================================================
    print(f"\nStep 5: Saving model and scaler...")

    # Create models directory if it doesn't exist
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    # Save model
    model_path = models_dir / 'xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {model_path}")

    # Save scaler
    scaler_path = models_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved to {scaler_path}")

    # ========================================================================
    # Display Results
    # ========================================================================
    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"RMSE: {rmse:.2f} cycles")
    print(f"R2 Score: {r2:.2f}")
    print("="*60 + "\n")

    return model, predictions, test_rul, feature_names


if __name__ == "__main__":
    # Train the model
    model, predictions, test_rul, feature_names = train_model('data/')

    print("\n--- Model Training Test Complete ---")
    print(f"Model type: {type(model).__name__}")
    print(f"Number of features used: {len(feature_names)}")
