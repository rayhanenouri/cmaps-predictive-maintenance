"""
Unsupervised anomaly detection for engine health monitoring to complement supervised RUL prediction with isolation-based approach.
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path


def train_isolation_forest(X_train, contamination=0.1):
    """
    Train Isolation Forest for anomaly detection.
    
    contamination: expected proportion of anomalies (default 10%)
    """
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        n_jobs=-1
    )
    
    iso_forest.fit(X_train)
    
    # Save model
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / 'isolation_forest.pkl', 'wb') as f:
        pickle.dump(iso_forest, f)
    
    print(f"Isolation Forest trained and saved")
    return iso_forest


def evaluate_anomaly_detection(iso_forest, X_test, test_rul, threshold=20):
    """
    Evaluate isolation forest against ground truth.
    
    Engines with RUL < threshold are considered "about to fail"
    """
    # Predict anomalies (-1 = anomaly, 1 = normal)
    predictions = iso_forest.predict(X_test)
    anomaly_scores = iso_forest.decision_function(X_test)
    
    # Get last prediction per engine
    # (same logic as your RUL model)
    from src.data_loader import load_data
    _, test_df, _ = load_data('data/')
    
    last_predictions = []
    last_scores = []
    current_idx = 0
    
    for unit_id in sorted(test_df['unit_id'].unique()):
        unit_mask = test_df['unit_id'] == unit_id
        unit_cycle_count = unit_mask.sum()
        last_pred = predictions[current_idx + unit_cycle_count - 1]
        last_score = anomaly_scores[current_idx + unit_cycle_count - 1]
        last_predictions.append(last_pred)
        last_scores.append(last_score)
        current_idx += unit_cycle_count
    
    last_predictions = np.array(last_predictions)
    last_scores = np.array(last_scores)
    
    # Ground truth: engines with RUL < threshold are "failing soon"
    y_true = (test_rul < threshold).astype(int)
    y_pred = (last_predictions == -1).astype(int)  # -1 = anomaly
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # How many engines flagged before failure?
    engines_flagged = np.sum(y_pred == 1)
    true_failures = np.sum(y_true == 1)
    correctly_flagged = np.sum((y_pred == 1) & (y_true == 1))
    
    detection_rate = correctly_flagged / true_failures * 100 if true_failures > 0 else 0
    
    results = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'detection_rate': detection_rate,
        'engines_flagged': engines_flagged,
        'true_failures': true_failures,
        'correctly_flagged': correctly_flagged,
        'threshold_cycles': threshold
    }
    
    print(f"\nIsolation Forest Evaluation (RUL < {threshold} cycles = failure):")
    print(f"  Precision: {precision:.2f}")
    print(f"  Recall: {recall:.2f}")
    print(f"  F1 Score: {f1:.2f}")
    print(f"  Detection rate: {detection_rate:.1f}%")
    print(f"  Engines flagged: {engines_flagged}/{len(test_rul)}")
    print(f"  Correctly flagged: {correctly_flagged}/{true_failures}")
    
    return results


if __name__ == "__main__":
    from src.feature_engineering import prepare_features
    
    X_train, y_train, X_test, test_rul, scaler, feature_names = prepare_features('data/')
    
    # Train
    iso_forest = train_isolation_forest(X_train, contamination=0.1)
    
    # Evaluate at different thresholds
    for threshold in [10, 20, 30]:
        print(f"\n{'='*60}")
        results = evaluate_anomaly_detection(iso_forest, X_test, test_rul, threshold)
