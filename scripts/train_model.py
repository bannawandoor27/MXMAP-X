"""Train XGBoost models on synthetic data with cross-validation."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from app.ml.xgboost_model import XGBoostPredictor
from app.ml.feature_engineering import FeatureEngineer
from app.db.session import async_session_maker
from app.models.database import TrainingMetadata
from sqlalchemy import select


class ModelTrainer:
    """
    Model training pipeline with cross-validation and performance tracking.
    """

    def __init__(
        self,
        data_path: str = "data/synthetic_training_data.csv",
        config_path: str = "config/model_config.yaml",
        model_dir: str = "models/cache",
    ) -> None:
        """
        Initialize trainer.
        
        Args:
            data_path: Path to training data CSV
            config_path: Path to model configuration YAML
            model_dir: Directory to save trained models
        """
        self.data_path = data_path
        self.config_path = config_path
        self.model_dir = model_dir
        
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Training results
        self.results: dict[str, any] = {}

    def load_data(self) -> pd.DataFrame:
        """Load training data from CSV."""
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(df)} samples")
        print(f"  Features: {df.shape[1]} columns")
        print(f"  Missing values: {df.isna().sum().sum()} total")
        return df

    def prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, dict[str, np.ndarray]]:
        """
        Prepare training and test sets.
        
        Args:
            df: Raw dataframe
            
        Returns:
            X_train, y_train, X_test, y_test
        """
        print("\nPreparing data...")
        
        # Define target columns
        target_cols = {
            "capacitance": "areal_capacitance_mf_cm2",
            "esr": "esr_ohm",
            "rate_capability": "rate_capability_percent",
            "cycle_life": "cycle_life_cycles",
        }
        
        # Split features and targets
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
        
        X = df[feature_cols].copy()
        y = {name: df[col].values for name, col in target_cols.items()}
        
        # Train-test split
        test_size = self.config["training"]["test_size"]
        random_state = self.config["training"]["random_state"]
        
        X_train, X_test, y_train_cap, y_test_cap = train_test_split(
            X, y["capacitance"], test_size=test_size, random_state=random_state
        )
        
        # Split other targets with same indices
        train_idx = X_train.index
        test_idx = X_test.index
        
        y_train = {
            "capacitance": y["capacitance"][train_idx],
            "esr": y["esr"][train_idx],
            "rate_capability": y["rate_capability"][train_idx],
            "cycle_life": y["cycle_life"][train_idx],
        }
        
        y_test = {
            "capacitance": y["capacitance"][test_idx],
            "esr": y["esr"][test_idx],
            "rate_capability": y["rate_capability"][test_idx],
            "cycle_life": y["cycle_life"][test_idx],
        }
        
        print(f"✓ Train set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        return X_train, y_train, X_test, y_test

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: dict[str, np.ndarray],
        X_test: pd.DataFrame,
        y_test: dict[str, np.ndarray],
    ) -> XGBoostPredictor:
        """
        Train XGBoost model with quantile regression.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Trained predictor
        """
        print("\n" + "=" * 60)
        print("Training XGBoost Models")
        print("=" * 60)
        
        # Create predictor
        model_version = f"v{self.config['model']['version']}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        predictor = XGBoostPredictor(
            model_version=model_version,
            config=self.config,
        )
        
        # Train models
        training_results = predictor.train(X_train, y_train, X_test, y_test)
        
        print("\n" + "=" * 60)
        print("Training Complete")
        print("=" * 60)
        
        # Store results
        self.results["training_metrics"] = training_results
        self.results["model_version"] = model_version
        
        return predictor

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: dict[str, np.ndarray],
    ) -> dict[str, dict[str, float]]:
        """
        Perform k-fold cross-validation.
        
        Args:
            X: Features
            y: Targets
            
        Returns:
            Cross-validation scores for each target
        """
        print("\n" + "=" * 60)
        print("Cross-Validation")
        print("=" * 60)
        
        n_folds = self.config["training"]["cv_folds"]
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        cv_results = {}
        
        for target_name, target_values in y.items():
            print(f"\nCross-validating {target_name}...")
            
            fold_scores = {"r2": [], "rmse": [], "mae": []}
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                # Split data
                X_fold_train = X.iloc[train_idx]
                X_fold_val = X.iloc[val_idx]
                y_fold_train = {name: vals[train_idx] for name, vals in y.items()}
                y_fold_val = target_values[val_idx]
                
                # Train model
                predictor = XGBoostPredictor(
                    model_version=f"cv_fold_{fold}",
                    config=self.config,
                )
                predictor.train(X_fold_train, y_fold_train)
                
                # Evaluate
                feature_engineer = predictor.feature_engineer
                X_val_transformed = feature_engineer.transform(X_fold_val)
                y_pred = predictor.models[target_name]["mean"].predict(X_val_transformed)
                
                # Calculate metrics
                r2 = r2_score(y_fold_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                mae = mean_absolute_error(y_fold_val, y_pred)
                
                fold_scores["r2"].append(r2)
                fold_scores["rmse"].append(rmse)
                fold_scores["mae"].append(mae)
                
                print(f"  Fold {fold}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
            
            # Calculate mean and std
            cv_results[target_name] = {
                "r2_mean": np.mean(fold_scores["r2"]),
                "r2_std": np.std(fold_scores["r2"]),
                "rmse_mean": np.mean(fold_scores["rmse"]),
                "rmse_std": np.std(fold_scores["rmse"]),
                "mae_mean": np.mean(fold_scores["mae"]),
                "mae_std": np.std(fold_scores["mae"]),
            }
            
            print(f"\n  {target_name} CV Results:")
            print(f"    R²: {cv_results[target_name]['r2_mean']:.4f} ± {cv_results[target_name]['r2_std']:.4f}")
            print(f"    RMSE: {cv_results[target_name]['rmse_mean']:.4f} ± {cv_results[target_name]['rmse_std']:.4f}")
            print(f"    MAE: {cv_results[target_name]['mae_mean']:.4f} ± {cv_results[target_name]['mae_std']:.4f}")
        
        self.results["cv_metrics"] = cv_results
        return cv_results

    def save_model(self, predictor: XGBoostPredictor) -> None:
        """Save trained model to disk."""
        print(f"\nSaving model to {self.model_dir}...")
        predictor.save(self.model_dir)
        print("✓ Model saved")

    async def save_metadata_to_db(self, predictor: XGBoostPredictor) -> None:
        """Save training metadata to database."""
        print("\nSaving metadata to database...")
        
        async with async_session_maker() as session:
            # Deactivate existing models
            result = await session.execute(select(TrainingMetadata))
            existing_models = result.scalars().all()
            for model in existing_models:
                model.is_active = 0
            
            # Create new metadata entry
            metrics = predictor.metrics
            metadata = TrainingMetadata(
                model_version=predictor.model_version,
                model_type="xgboost",
                train_r2_capacitance=metrics.get("capacitance", {}).get("r2"),
                train_r2_esr=metrics.get("esr", {}).get("r2"),
                train_r2_rate_capability=metrics.get("rate_capability", {}).get("r2"),
                train_r2_cycle_life=metrics.get("cycle_life", {}).get("r2"),
                test_r2_capacitance=metrics.get("capacitance", {}).get("r2"),
                test_r2_esr=metrics.get("esr", {}).get("r2"),
                test_r2_rate_capability=metrics.get("rate_capability", {}).get("r2"),
                test_r2_cycle_life=metrics.get("cycle_life", {}).get("r2"),
                train_rmse_capacitance=metrics.get("capacitance", {}).get("rmse"),
                test_rmse_capacitance=metrics.get("capacitance", {}).get("rmse"),
                training_samples=len(self.results.get("X_train", [])),
                test_samples=len(self.results.get("X_test", [])),
                hyperparameters=str(self.config.get("hyperparameters", {})),
                trained_at=datetime.utcnow(),
                is_active=1,
            )
            
            session.add(metadata)
            await session.commit()
        
        print("✓ Metadata saved to database")

    def print_summary(self, predictor: XGBoostPredictor) -> None:
        """Print training summary."""
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        
        print(f"\nModel Version: {predictor.model_version}")
        print(f"Model Type: XGBoost with Quantile Regression")
        
        print("\nTest Set Performance:")
        for target_name, metrics in predictor.metrics.items():
            print(f"\n  {target_name.upper()}:")
            print(f"    R²: {metrics['r2']:.4f}")
            print(f"    RMSE: {metrics['rmse']:.4f}")
            print(f"    MAE: {metrics['mae']:.4f}")
        
        if "cv_metrics" in self.results:
            print("\nCross-Validation Performance:")
            for target_name, cv_metrics in self.results["cv_metrics"].items():
                print(f"\n  {target_name.upper()}:")
                print(f"    R²: {cv_metrics['r2_mean']:.4f} ± {cv_metrics['r2_std']:.4f}")
                print(f"    RMSE: {cv_metrics['rmse_mean']:.4f} ± {cv_metrics['rmse_std']:.4f}")
        
        print("\nTop 10 Most Important Features:")
        importance_sorted = sorted(
            predictor.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for i, (feature, importance) in enumerate(importance_sorted[:10], 1):
            print(f"  {i:2d}. {feature:30s} {importance:.4f}")
        
        print("\n" + "=" * 60)


async def main() -> None:
    """Main training pipeline."""
    import os
    
    # Create directories
    os.makedirs("models/cache", exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Load data
    df = trainer.load_data()
    
    # Prepare data
    X_train, y_train, X_test, y_test = trainer.prepare_data(df)
    trainer.results["X_train"] = X_train
    trainer.results["X_test"] = X_test
    
    # Perform cross-validation
    X_full = pd.concat([X_train, X_test])
    y_full = {
        name: np.concatenate([y_train[name], y_test[name]])
        for name in y_train.keys()
    }
    trainer.cross_validate(X_full, y_full)
    
    # Train final model
    predictor = trainer.train_model(X_train, y_train, X_test, y_test)
    
    # Save model
    trainer.save_model(predictor)
    
    # Save metadata to database
    await trainer.save_metadata_to_db(predictor)
    
    # Print summary
    trainer.print_summary(predictor)
    
    print("\n✓ Training pipeline completed successfully!")
    print(f"\nTo use the trained model, update app/ml/model_loader.py")
    print(f"Model saved to: {trainer.model_dir}")


if __name__ == "__main__":
    asyncio.run(main())
