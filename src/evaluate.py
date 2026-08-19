"""
Model evaluation and visualization generation.
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from src.feature_engineering import prepare_features


def evaluate_model(data_path='data/'):
    """Evaluate trained model and generate plots.
       Returns dict of evaluation metrics.
    """

    print("\nmodel evaluation\n")

    # load trained model
    models_dir = Path('models')
    model_path = models_dir / 'xgboost_model.pkl'
    scaler_path = models_dir / 'scaler.pkl'

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"loaded model from {model_path}")

    # prepare features
    X_train, y_train, X_test, test_rul, _, feature_names = prepare_features(data_path)

    # generate predictions
    from src.data_loader import load_data
    _, test_df, _ = load_data(data_path)

    all_predictions = model.predict(X_test)
    all_predictions = np.maximum(all_predictions, 0)

    # get last prediction per engine for metric calculation
    predictions = []
    current_idx = 0
    for unit_id in sorted(test_df['unit_id'].unique()):
        unit_mask = test_df['unit_id'] == unit_id
        unit_cycle_count = unit_mask.sum()
        last_prediction = all_predictions[current_idx + unit_cycle_count - 1]
        predictions.append(last_prediction)
        current_idx += unit_cycle_count

    predictions = np.array(predictions)

    # create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    # generate visualizations
    print("\ngenerating plots")
    _plot_predicted_vs_actual(predictions, test_rul, results_dir)
    _plot_error_distribution(predictions, test_rul, results_dir)
    _plot_degradation_curves(test_df, all_predictions, test_rul, results_dir)

    # calculate metrics
    rmse = np.sqrt(mean_squared_error(test_rul, predictions))
    r2 = r2_score(test_rul, predictions)
    errors = predictions - test_rul
    mean_error = np.mean(errors)

    # percentage of predictions within thresholds
    within_10 = np.sum(np.abs(errors) <= 10) / len(errors) * 100
    within_20 = np.sum(np.abs(errors) <= 20) / len(errors) * 100

    print(f"\nevaluation results:")
    print(f"  RMSE: {rmse:.2f} cycles")
    print(f"  R2: {r2:.2f}")
    print(f"  mean error: {mean_error:.2f} cycles")
    print(f"  within +/- 10 cycles: {within_10:.1f}%")
    print(f"  within +/- 20 cycles: {within_20:.1f}%")
    print(f"\nplots saved to {results_dir}/")

    return {
        'rmse': rmse,
        'r2': r2,
        'mean_error': mean_error,
        'within_10': within_10,
        'within_20': within_20
    }


def _plot_predicted_vs_actual(predictions, actual, results_dir):
    """Scatter plot of predicted vs actual RUL."""
    plt.figure(figsize=(10, 8))

    plt.scatter(actual, predictions, alpha=0.5, s=50, edgecolors='k', linewidth=0.5)

    # perfect prediction line
    max_val = max(actual.max(), predictions.max())
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual RUL (cycles)', fontsize=12)
    plt.ylabel('Predicted RUL (cycles)', fontsize=12)
    plt.title('Predicted vs Actual Remaining Useful Life', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # add metrics to plot
    r2 = r2_score(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    textstr = f'R2 = {r2:.3f}\nRMSE = {rmse:.2f} cycles'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path = results_dir / 'predicted_vs_actual.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  saved {save_path}")


def _plot_error_distribution(predictions, actual, results_dir):
    """Histogram of prediction errors."""
    errors = predictions - actual

    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')

    plt.xlabel('Prediction Error (cycles)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3, axis='y')

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
    print(f"  saved {save_path}")


def _plot_degradation_curves(test_df, all_predictions, actual, results_dir):
    """Plot degradation trajectories for 3 sample engines."""

    # get predictions per engine
    unit_ids = sorted(test_df['unit_id'].unique())
    unit_predictions = {}
    unit_actuals = {}

    current_idx = 0
    for engine_idx, unit_id in enumerate(unit_ids):
        unit_mask = test_df['unit_id'] == unit_id
        unit_cycle_count = unit_mask.sum()
        unit_predictions[unit_id] = all_predictions[current_idx + unit_cycle_count - 1]
        unit_actuals[unit_id] = actual[engine_idx]
        current_idx += unit_cycle_count

    # select engines with low, medium, and high RUL
    sorted_units = sorted(unit_actuals.items(), key=lambda x: x[1])
    n_units = len(sorted_units)
    sample_units = [
        sorted_units[0][0],
        sorted_units[n_units // 2][0],
        sorted_units[-1][0]
    ]

    # get all cycle predictions for plotting
    all_unit_predictions = {}
    current_idx = 0
    for unit_id in unit_ids:
        unit_mask = test_df['unit_id'] == unit_id
        unit_cycle_count = unit_mask.sum()
        all_unit_predictions[unit_id] = all_predictions[current_idx:current_idx + unit_cycle_count]
        current_idx += unit_cycle_count

    # create plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    for idx, unit_id in enumerate(sample_units):
        ax = axes[idx]

        unit_data = test_df[test_df['unit_id'] == unit_id].copy()
        cycles = unit_data['cycle'].values

        actual_rul = unit_actuals[unit_id]
        predicted_rul = unit_predictions[unit_id]

        # reconstruct RUL trajectory from last known point
        max_cycle = cycles[-1]
        actual_rul_progression = actual_rul + (max_cycle - cycles)
        predicted_rul_progression = all_unit_predictions[unit_id]

        ax.plot(cycles, actual_rul_progression, 'b-', linewidth=2,
                label='Actual RUL', marker='o', markersize=4)
        ax.plot(cycles, predicted_rul_progression, 'r--', linewidth=2,
                label='Predicted RUL', marker='s', markersize=4)

        ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)

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
    print(f"  saved {save_path}")


if __name__ == "__main__":
    metrics = evaluate_model('data/')
    print("\nevaluation complete")
