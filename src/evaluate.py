"""
Model Evaluation and Visualization
Tests trained model on unseen engines and generates performance plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from data_loader import CMAPSDataLoader
from feature_engineering import FeatureEngineer
from model import RULPredictor


class ModelEvaluator:
    """
    Evaluates RUL prediction model on test set and generates visualizations.
    """

    def __init__(self):
        self.results = None

    def evaluate(
        self,
        test_df: pd.DataFrame,
        true_rul: pd.Series,
        predictor: RULPredictor
    ) -> dict:
        """
        Evaluate model performance on test set.

        Args:
            test_df (pd.DataFrame): Processed test features
            true_rul (pd.Series): Ground truth RUL values
            predictor (RULPredictor): Trained model

        Returns:
            dict: Evaluation metrics
        """
        print("\n=== Model Evaluation ===\n")

        # Get predictions for each engine at final cycle
        predictions_df = predictor.predict_for_engines(test_df)

        # Align with ground truth (same order)
        predictions_df['actual_RUL'] = true_rul['RUL'].values

        # Calculate metrics
        y_true = predictions_df['actual_RUL']
        y_pred = predictions_df['predicted_RUL']

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Calculate custom metric: percentage within ±20 cycles
        within_20 = np.abs(y_true - y_pred) <= 20
        accuracy_20 = (within_20.sum() / len(y_true)) * 100

        print(f"Test Set Performance:")
        print(f"  RMSE: {rmse:.2f} cycles")
        print(f"  MAE: {mae:.2f} cycles")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Predictions within ±20 cycles: {accuracy_20:.1f}%")
        print(f"  Total engines evaluated: {len(predictions_df)}\n")

        self.results = predictions_df

        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'accuracy_20': accuracy_20,
            'n_engines': len(predictions_df)
        }

    def plot_predictions(self, save_path: str = "results_plot.png", show: bool = False):
        """
        Generate visualization of predicted vs actual RUL.

        Args:
            save_path (str): Path to save plot
            show (bool): Whether to display plot
        """
        if self.results is None:
            raise RuntimeError("Must run evaluate() before plotting")

        # Set style
        sns.set_style("whitegrid")
        plt.figure(figsize=(14, 6))

        # Subplot 1: Predicted vs Actual scatter
        plt.subplot(1, 2, 1)
        plt.scatter(
            self.results['actual_RUL'],
            self.results['predicted_RUL'],
            alpha=0.6,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )

        # Perfect prediction line
        max_val = max(self.results['actual_RUL'].max(), self.results['predicted_RUL'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        plt.xlabel('Actual RUL (cycles)', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted RUL (cycles)', fontsize=12, fontweight='bold')
        plt.title('Predicted vs Actual RUL', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Prediction errors
        plt.subplot(1, 2, 2)
        errors = self.results['predicted_RUL'] - self.results['actual_RUL']

        plt.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        plt.xlabel('Prediction Error (cycles)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

    def plot_engine_degradation(
        self,
        test_df: pd.DataFrame,
        predictor: RULPredictor,
        engine_ids: list = None,
        save_path: str = "degradation_curves.png",
        show: bool = False
    ):
        """
        Plot degradation curves for specific engines.

        Args:
            test_df (pd.DataFrame): Test data with all cycles
            predictor (RULPredictor): Trained model
            engine_ids (list): Specific engines to plot (default: first 4)
            save_path (str): Path to save plot
            show (bool): Whether to display plot
        """
        if engine_ids is None:
            engine_ids = sorted(test_df['unit_id'].unique())[:4]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, engine_id in enumerate(engine_ids):
            if idx >= 4:
                break

            # Get data for this engine
            engine_data = test_df[test_df['unit_id'] == engine_id].copy()

            # Predict RUL for all cycles
            engine_data['predicted_RUL'] = predictor.predict(engine_data)

            # Plot
            ax = axes[idx]
            ax.plot(
                engine_data['cycle'],
                engine_data['predicted_RUL'],
                linewidth=2,
                label='Predicted RUL',
                color='steelblue'
            )

            ax.set_xlabel('Cycle', fontsize=11, fontweight='bold')
            ax.set_ylabel('RUL (cycles)', fontsize=11, fontweight='bold')
            ax.set_title(f'Engine Unit {engine_id}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Degradation curves saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()


def main():
    """
    Complete evaluation pipeline.
    """
    print("\n" + "="*60)
    print("  NASA C-MAPS RUL Prediction - Model Evaluation")
    print("="*60)

    # Load test data
    loader = CMAPSDataLoader()
    test_df, true_rul = loader.load_test_data()

    # Load trained feature engineer and model
    print("\n=== Loading Trained Pipeline ===\n")

    # We need to fit the feature engineer on training data first
    train_df = loader.load_train_data()
    engineer = FeatureEngineer()
    engineer.fit_transform(train_df)
    print("✓ Feature engineer fitted")

    # Transform test data
    test_processed = engineer.transform(test_df)
    print("✓ Test data processed")

    # Load model
    predictor = RULPredictor()
    predictor.load_model("model.pkl")

    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(test_processed, true_rul, predictor)

    # Generate plots
    print("\n=== Generating Visualizations ===\n")
    evaluator.plot_predictions(save_path="results_plot.png")
    evaluator.plot_engine_degradation(
        test_processed,
        predictor,
        save_path="degradation_curves.png"
    )

    print("\n" + "="*60)
    print("✓ Evaluation complete!")
    print("="*60 + "\n")

    return metrics


if __name__ == "__main__":
    main()
