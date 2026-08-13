"""
Model Evaluation for NASA C-MAPS FD001 Predictive Maintenance
Generate visualizations and comprehensive evaluation metrics
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from feature_engineering import prepare_features


def evaluate_model(data_path='data/'):
    """
    Evaluate trained model and generate visualizations.

    Steps:
    1. Load trained model and scaler
    2. Prepare features
    3. Generate predictions
    4. Create visualizations (scatter, error distribution, degradation curves)
    5. Print comprehensive evaluation report

    Args:
        data_path (str): Path to data directory

    Returns:
        dict: Evaluation metrics
    """

    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60 + "\n")

    # ========================================================================
    # STEP 1: Load Model and Scaler
    # ========================================================================
    print("Step 1: Loading model and scaler...")

    models_dir = Path('models')
    model_path = models_dir / 'xgboost_model.pkl'
    scaler_path = models_dir / 'scaler.pkl'

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Loaded model from {model_path}")

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Loaded scaler from {scaler_path}")

    # ========================================================================
    # STEP 2: Prepare Features
    # ========================================================================
    print("\nStep 2: Preparing features...")
    X_train, y_train, X_test, test_rul, _, feature_names = prepare_features(data_path)

    # ========================================================================
    # STEP 3: Generate Predictions
    # ========================================================================
    print("\nStep 3: Generating predictions...")

    # Load test data to get unit_ids for aggregation
    from data_loader import load_data
    _, test_df, _ = load_data(data_path)

    # Get predictions for all test cycles
    all_predictions = model.predict(X_test)
    all_predictions = np.maximum(all_predictions, 0)  # Ensure non-negative

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

    # ========================================================================
    # STEP 4: Create Results Directory
    # ========================================================================
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    print(f"\n✓ Results directory created: {results_dir}")

    # ========================================================================
    # STEP 5: Generate Visualizations
    # ========================================================================
    print("\nStep 4: Generating visualizations...")

    # 1. Predicted vs Actual Scatter Plot
    _plot_predicted_vs_actual(predictions, test_rul, results_dir)

    # 2. Error Distribution Histogram
    _plot_error_distribution(predictions, test_rul, results_dir)

    # 3. Degradation Curves for 3 Sample Engines
    _plot_degradation_curves(test_df, all_predictions, test_rul, results_dir)

    # ========================================================================
    # STEP 6: Calculate Metrics
    # ========================================================================
    print("\nStep 5: Calculating evaluation metrics...")

    rmse = np.sqrt(mean_squared_error(test_rul, predictions))
    r2 = r2_score(test_rul, predictions)
    errors = predictions - test_rul
    mean_error = np.mean(errors)

    # Calculate percentage within thresholds
    within_10 = np.sum(np.abs(errors) <= 10) / len(errors) * 100
    within_20 = np.sum(np.abs(errors) <= 20) / len(errors) * 100

    # ========================================================================
    # Print Evaluation Report
    # ========================================================================
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    print(f"\nPerformance Metrics:")
    print(f"  RMSE: {rmse:.2f} cycles")
    print(f"  R2 Score: {r2:.2f}")
    print(f"  Mean Error: {mean_error:.2f} cycles")
    print(f"\nPrediction Accuracy:")
    print(f"  Within ±10 cycles: {within_10:.1f}%")
    print(f"  Within ±20 cycles: {within_20:.1f}%")
    print(f"\nVisualizations saved to {results_dir}/:")
    print(f"  - predicted_vs_actual.png")
    print(f"  - error_distribution.png")
    print(f"  - degradation_curves.png")
    print("="*60 + "\n")

    return {
        'rmse': rmse,
        'r2': r2,
        'mean_error': mean_error,
        'within_10': within_10,
        'within_20': within_20
    }


def _plot_predicted_vs_actual(predictions, actual, results_dir):
    """Generate scatter plot of predicted vs actual RUL."""
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(actual, predictions, alpha=0.5, s=50, edgecolors='k', linewidth=0.5)

    # Perfect prediction line (y=x)
    max_val = max(actual.max(), predictions.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    # Labels and title
    plt.xlabel('Actual RUL (cycles)', fontsize=12)
    plt.ylabel('Predicted RUL (cycles)', fontsize=12)
    plt.title('Predicted vs Actual Remaining Useful Life', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add R² annotation
    r2 = r2_score(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.2f} cycles'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = results_dir / 'predicted_vs_actual.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def _plot_error_distribution(predictions, actual, results_dir):
    """Generate histogram of prediction errors."""
    errors = predictions - actual

    plt.figure(figsize=(10, 6))

    # Histogram
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')

    # Add vertical line at zero
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')

    # Labels and title
    plt.xlabel('Prediction Error (cycles)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

    # Add statistics annotation
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    textstr = f'Mean Error: {mean_err:.2f} cycles\nStd Dev: {std_err:.2f} cycles'
    plt.text(0.95, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = results_dir / 'error_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


def _plot_degradation_curves(test_df, all_predictions, actual, results_dir):
    """Plot degradation curves for 3 sample engines."""
    # Get last cycle per unit
    last_cycles = test_df.groupby('unit_id')['cycle'].max()

    # Select 3 engines with different RUL ranges (short, medium, long)
    unit_ids = sorted(last_cycles.index.tolist())

    # Get predictions per unit (take the last prediction for each unit)
    unit_predictions = {}
    unit_actuals = {}

    current_cycle_idx = 0
    for engine_idx, unit_id in enumerate(unit_ids):
        unit_mask = test_df['unit_id'] == unit_id
        unit_cycle_count = unit_mask.sum()

        # Get the last prediction for this unit
        unit_predictions[unit_id] = all_predictions[current_cycle_idx + unit_cycle_count - 1]
        unit_actuals[unit_id] = actual[engine_idx]

        current_cycle_idx += unit_cycle_count

    # Sort units by actual RUL and pick 3 distributed samples
    sorted_units = sorted(unit_actuals.items(), key=lambda x: x[1])
    n_units = len(sorted_units)
    sample_units = [
        sorted_units[0][0],  # Low RUL
        sorted_units[n_units // 2][0],  # Medium RUL
        sorted_units[-1][0]  # High RUL
    ]

    # Create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Also get all predictions for each cycle for plotting
    all_unit_predictions = {}
    current_cycle_idx = 0
    for unit_id in unit_ids:
        unit_mask = test_df['unit_id'] == unit_id
        unit_cycle_count = unit_mask.sum()
        all_unit_predictions[unit_id] = all_predictions[current_cycle_idx:current_cycle_idx + unit_cycle_count]
        current_cycle_idx += unit_cycle_count

    for idx, unit_id in enumerate(sample_units):
        ax = axes[idx]

        # Get data for this unit
        unit_data = test_df[test_df['unit_id'] == unit_id].copy()
        cycles = unit_data['cycle'].values

        # Actual RUL at last cycle
        actual_rul = unit_actuals[unit_id]
        predicted_rul = unit_predictions[unit_id]

        # Get predictions for all cycles of this engine
        unit_cycle_predictions = all_unit_predictions[unit_id]

        # Calculate RUL progression (assuming linear from last point)
        max_cycle = cycles[-1]
        actual_rul_progression = actual_rul + (max_cycle - cycles)
        # For predicted, we have the actual predictions at each cycle
        predicted_rul_progression = unit_cycle_predictions

        # Plot
        ax.plot(cycles, actual_rul_progression, 'b-', linewidth=2, label='Actual RUL', marker='o', markersize=4)
        ax.plot(cycles, predicted_rul_progression, 'r--', linewidth=2, label='Predicted RUL', marker='s', markersize=4)

        # Add horizontal line at 0
        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)

        # Labels
        ax.set_xlabel('Cycle', fontsize=10)
        ax.set_ylabel('RUL (cycles)', fontsize=10)
        ax.set_title(f'Engine {unit_id} - Actual RUL: {actual_rul:.1f}, Predicted: {predicted_rul:.1f}',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Degradation Curves for Sample Engines', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = results_dir / 'degradation_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


if __name__ == "__main__":
    # Run evaluation
    metrics = evaluate_model('data/')

    print("\n--- Evaluation Test Complete ---")
    print(f"Metrics: {metrics}")
