"""
Baseline comparison for turbofan RUL prediction.
Compares XGBoost against simple baselines using the SAME features
and the SAME evaluation (last cycle of each of the 100 test engines).
Run from the repo root:  python -m src.baseline
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from src.feature_engineering import prepare_features
from src.data_loader import load_data


def last_cycle_predictions(all_predictions, test_df):
    """Keep only the prediction at the last cycle of each test engine."""
    predictions = []
    current_idx = 0
    for unit_id in sorted(test_df['unit_id'].unique()):
        count = (test_df['unit_id'] == unit_id).sum()
        predictions.append(all_predictions[current_idx + count - 1])
        current_idx += count
    return np.array(predictions)


def evaluate(name, model, X_test, test_df, test_rul):
    all_preds = np.maximum(model.predict(X_test), 0)
    preds = last_cycle_predictions(all_preds, test_df)
    rmse = np.sqrt(mean_squared_error(test_rul, preds))
    r2 = r2_score(test_rul, preds)
    print(f"  {name:<28} RMSE = {rmse:6.2f}   R2 = {r2:5.2f}")
    return rmse


def run_baselines(data_path='data/'):
    X_train, y_train, X_test, test_rul, _, _ = prepare_features(data_path)
    _, test_df, _ = load_data(data_path)
    test_rul = np.ravel(np.asarray(test_rul))

    print("\nbaseline comparison (100 test engines)\n")

    # 1. naive baseline: always predicts the average training RUL
    dummy = DummyRegressor(strategy='mean').fit(X_train, y_train)
    rmse_dummy = evaluate("Mean predictor (naive)", dummy, X_test, test_df, test_rul)

    # 2. simple ML baseline: linear regression on the same 55 features
    linear = LinearRegression().fit(X_train, y_train)
    rmse_linear = evaluate("Linear regression", linear, X_test, test_df, test_rul)

    # 3. your trained XGBoost model (run src/model.py first)
    with open(Path('models') / 'xgboost_model.pkl', 'rb') as f:
        xgb = pickle.load(f)
    rmse_xgb = evaluate("XGBoost (final model)", xgb, X_test, test_df, test_rul)

    print("\nRMSE reduction achieved by XGBoost:")
    print(f"  vs mean predictor:     {(1 - rmse_xgb / rmse_dummy) * 100:.1f}%")
    print(f"  vs linear regression:  {(1 - rmse_xgb / rmse_linear) * 100:.1f}%")


if __name__ == "__main__":
    run_baselines('data/')