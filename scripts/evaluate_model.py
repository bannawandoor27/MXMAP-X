"""Evaluate trained model performance and generate visualizations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from app.ml.xgboost_model import XGBoostPredictor


class ModelEvaluator:
    """
    Comprehensive model evaluation with visualizations.
    """

    def __init__(
        self,
        model_path: str = "models/cache",
        data_path: str = "data/synthetic_training_data.csv",
    ) -> None:
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            data_path: Path to test data
        """
        self.model_path = model_path
        self.data_path = data_path
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.predictor = XGBoostPredictor.load(model_path)
        print(f"✓ Loaded model: {self.predictor.model_version}")

    def load_test_data(self) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """Load and prepare test data."""
        print(f"\nLoading test data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Use last 20% as test set (same split as training)
        test_size = int(len(df) * 0.2)
        df_test = df.iloc[-test_size:].copy()
        
        # Prepare features
        feature_cols = [
            "mxene_type",
            "terminations",
            "electrolyte",
            "electrolyte_concentration",
            "thickness_um",
            "deposition_method",
            "annealing_temp_c",
            "annealing_time_min",
            "interlayer_spacing_nm",
            "specific_surface_area_m2g",
            "pore_volume_cm3g",
            "optical_transmittance",
            "sheet_resistance_ohm_sq",
        ]
        
        X_test = df_test[feature_cols]
        
        # Prepare targets
        y_test = {
            "capacitance": df_test["areal_capacitance_mf_cm2"].values,
            "esr": df_test["esr_ohm"].values,
            "rate_capability": df_test["rate_capability_percent"].values,
            "cycle_life": df_test["cycle_life_cycles"].values,
        }
        
        print(f"✓ Loaded {len(X_test)} test samples")
        return X_test, y_test

    def evaluate(
        self, X_test: pd.DataFrame, y_test: dict[str, np.ndarray]
    ) -> dict[str, dict[str, float]]:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("Model Evaluation")
        print("=" * 60)
        
        # Transform features
        X_transformed = self.predictor.feature_engineer.transform(X_test)
        
        results = {}
        
        for target_name, y_true in y_test.items():
            print(f"\n{target_name.upper()}:")
            
            # Get predictions
            y_pred_mean = self.predictor.models[target_name]["mean"].predict(X_transformed)
            y_pred_lower = self.predictor.models[target_name]["lower"].predict(X_transformed)
            y_pred_upper = self.predictor.models[target_name]["upper"].predict(X_transformed)
            
            # Calculate metrics
            r2 = r2_score(y_true, y_pred_mean)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_mean))
            mae = mean_absolute_error(y_true, y_pred_mean)
            
            # Calculate coverage (% of true values within prediction intervals)
            coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
            
            # Calculate average interval width
            avg_interval_width = np.mean(y_pred_upper - y_pred_lower)
            relative_width = avg_interval_width / np.mean(y_true)
            
            results[target_name] = {
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "coverage": coverage,
                "avg_interval_width": avg_interval_width,
                "relative_width": relative_width,
                "y_true": y_true,
                "y_pred": y_pred_mean,
                "y_lower": y_pred_lower,
                "y_upper": y_pred_upper,
            }
            
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Coverage (95% CI): {coverage:.2%}")
            print(f"  Avg Interval Width: {avg_interval_width:.4f}")
            print(f"  Relative Width: {relative_width:.2%}")
        
        return results

    def plot_predictions(
        self, results: dict[str, dict[str, any]], output_dir: str = "reports"
    ) -> None:
        """
        Create prediction vs actual plots.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating plots in {output_dir}/...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 10)
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        target_labels = {
            "capacitance": "Areal Capacitance (mF/cm²)",
            "esr": "ESR (Ω)",
            "rate_capability": "Rate Capability (%)",
            "cycle_life": "Cycle Life (cycles)",
        }
        
        for idx, (target_name, metrics) in enumerate(results.items()):
            ax = axes[idx]
            
            y_true = metrics["y_true"]
            y_pred = metrics["y_pred"]
            y_lower = metrics["y_lower"]
            y_upper = metrics["y_upper"]
            
            # Scatter plot with error bars
            ax.errorbar(
                y_true,
                y_pred,
                yerr=[y_pred - y_lower, y_upper - y_pred],
                fmt="o",
                alpha=0.5,
                markersize=4,
                elinewidth=1,
                capsize=2,
            )
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction")
            
            # Labels and title
            ax.set_xlabel(f"True {target_labels[target_name]}", fontsize=11)
            ax.set_ylabel(f"Predicted {target_labels[target_name]}", fontsize=11)
            ax.set_title(
                f"{target_name.replace('_', ' ').title()}\n"
                f"R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.2f}",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/predictions_vs_actual.png", dpi=300, bbox_inches="tight")
        print(f"✓ Saved predictions_vs_actual.png")
        plt.close()

    def plot_feature_importance(self, output_dir: str = "reports") -> None:
        """
        Plot feature importance.
        
        Args:
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        importance = self.predictor.get_feature_importance()
        
        # Sort by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_features[:15])  # Top 15
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importances, color="steelblue")
        plt.yticks(range(len(features)), features)
        plt.xlabel("Importance", fontsize=12)
        plt.title("Top 15 Feature Importances", fontsize=14, fontweight="bold")
        plt.gca().invert_yaxis()
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_importance.png", dpi=300, bbox_inches="tight")
        print(f"✓ Saved feature_importance.png")
        plt.close()

    def plot_residuals(
        self, results: dict[str, dict[str, any]], output_dir: str = "reports"
    ) -> None:
        """
        Plot residual distributions.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (target_name, metrics) in enumerate(results.items()):
            ax = axes[idx]
            
            residuals = metrics["y_true"] - metrics["y_pred"]
            
            # Histogram
            ax.hist(residuals, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
            ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Zero Residual")
            ax.set_xlabel("Residual", fontsize=11)
            ax.set_ylabel("Frequency", fontsize=11)
            ax.set_title(
                f"{target_name.replace('_', ' ').title()} Residuals\n"
                f"Mean = {residuals.mean():.3f}, Std = {residuals.std():.3f}",
                fontsize=12,
                fontweight="bold",
            )
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/residuals.png", dpi=300, bbox_inches="tight")
        print(f"✓ Saved residuals.png")
        plt.close()

    def generate_report(
        self, results: dict[str, dict[str, any]], output_dir: str = "reports"
    ) -> None:
        """
        Generate comprehensive evaluation report.
        
        Args:
            results: Evaluation results
            output_dir: Directory to save report
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = f"{output_dir}/evaluation_report.txt"
        
        with open(report_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write("MXMAP-X Model Evaluation Report\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Model Version: {self.predictor.model_version}\n")
            f.write(f"Model Type: XGBoost with Quantile Regression\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("Performance Metrics\n")
            f.write("=" * 70 + "\n\n")
            
            for target_name, metrics in results.items():
                f.write(f"{target_name.upper()}:\n")
                f.write(f"  R² Score:              {metrics['r2']:.4f}\n")
                f.write(f"  RMSE:                  {metrics['rmse']:.4f}\n")
                f.write(f"  MAE:                   {metrics['mae']:.4f}\n")
                f.write(f"  95% CI Coverage:       {metrics['coverage']:.2%}\n")
                f.write(f"  Avg Interval Width:    {metrics['avg_interval_width']:.4f}\n")
                f.write(f"  Relative Width:        {metrics['relative_width']:.2%}\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write("Feature Importance (Top 10)\n")
            f.write("=" * 70 + "\n\n")
            
            importance = self.predictor.get_feature_importance()
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            for i, (feature, imp) in enumerate(sorted_features[:10], 1):
                f.write(f"{i:2d}. {feature:35s} {imp:.4f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"✓ Saved evaluation_report.txt")


def main() -> None:
    """Main evaluation pipeline."""
    evaluator = ModelEvaluator()
    
    # Load test data
    X_test, y_test = evaluator.load_test_data()
    
    # Evaluate model
    results = evaluator.evaluate(X_test, y_test)
    
    # Generate visualizations
    evaluator.plot_predictions(results)
    evaluator.plot_feature_importance()
    evaluator.plot_residuals(results)
    
    # Generate report
    evaluator.generate_report(results)
    
    print("\n" + "=" * 60)
    print("✓ Evaluation completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - reports/predictions_vs_actual.png")
    print("  - reports/feature_importance.png")
    print("  - reports/residuals.png")
    print("  - reports/evaluation_report.txt")


if __name__ == "__main__":
    main()
