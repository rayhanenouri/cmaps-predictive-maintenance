"""
XGBoost model training for turbofan RUL prediction.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.feature_engineering import prepare_features


def train_model(data_path='data/'):
    """Train XGBoost regressor and save to models directory.
       Returns model, predictions, test_rul, and feature_names.
    """

    print("\nmodel training\n")

    # prepare features
    X_train, y_train, X_test, test_rul, scaler, feature_names = prepare_features(data_path)

    # XGBoost with conservative hyperparameters to avoid overfitting
    # learning rate 0.05 with 300 trees gives good convergence
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

    print(f"training on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    model.fit(X_train, y_train, verbose=False)
    print("training complete")

    # generate predictions for all test cycles
    all_predictions = model.predict(X_test)
    all_predictions = np.maximum(all_predictions, 0)  # RUL cannot be negative

    # get last cycle prediction for each engine
    # this is what we compare against ground truth
    from src.data_loader import load_data
    _, test_df, _ = load_data(data_path)

    predictions = []
    current_idx = 0
    for unit_id in sorted(test_df['unit_id'].unique()):
        unit_mask = test_df['unit_id'] == unit_id
        unit_cycle_count = unit_mask.sum()
        last_prediction = all_predictions[current_idx + unit_cycle_count - 1]
        predictions.append(last_prediction)
        current_idx += unit_cycle_count

    predictions = np.array(predictions)

    # calculate metrics
    rmse = np.sqrt(mean_squared_error(test_rul, predictions))
    r2 = r2_score(test_rul, predictions)

    # save model and scaler
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / 'xgboost_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    scaler_path = models_dir / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\nsaved model to {model_path}")
    print(f"saved scaler to {scaler_path}")
    print(f"\ntest set performance:")
    print(f"  RMSE: {rmse:.2f} cycles")
    print(f"  R2: {r2:.2f}")

    return model, predictions, test_rul, feature_names


if __name__ == "__main__":
    model, predictions, test_rul, feature_names = train_model('data/')
    print("\nmodel training complete")
