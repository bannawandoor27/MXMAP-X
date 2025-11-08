"""Feature engineering pipeline for MXene supercapacitor data."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


class FeatureEngineer:
    """
    Feature engineering pipeline for MXene device data.
    
    Handles:
    - Categorical encoding (target encoding for high-cardinality features)
    - Numerical scaling (standardization)
    - Missing value imputation
    - Feature interactions
    """

    def __init__(self) -> None:
        """Initialize feature engineer."""
        self.categorical_features = [
            "mxene_type",
            "terminations",
            "electrolyte",
            "deposition_method",
        ]
        
        self.numerical_features = [
            "thickness_um",
            "electrolyte_concentration",
            "annealing_temp_c",
            "annealing_time_min",
            "interlayer_spacing_nm",
            "specific_surface_area_m2g",
            "pore_volume_cm3g",
            "optical_transmittance",
            "sheet_resistance_ohm_sq",
        ]
        
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler | None = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, target_col: str | None = None) -> "FeatureEngineer":
        """
        Fit feature engineering pipeline.
        
        Args:
            df: Training dataframe
            target_col: Optional target column for target encoding
            
        Returns:
            Self for chaining
        """
        # Fit label encoders for categorical features
        for col in self.categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].fillna("missing"))
                self.label_encoders[col] = le
        
        # Fit scaler on numerical features
        numerical_data = self._prepare_numerical_features(df)
        self.scaler = StandardScaler()
        self.scaler.fit(numerical_data)
        
        # Store feature names
        self.feature_names = self._get_feature_names()
        self.is_fitted = True
        
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform dataframe to feature matrix.
        
        Args:
            df: Input dataframe
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        # Encode categorical features
        categorical_encoded = []
        for col in self.categorical_features:
            if col in df.columns:
                values = df[col].fillna("missing")
                # Handle unseen categories
                encoded = []
                for val in values:
                    if val in self.label_encoders[col].classes_:
                        encoded.append(self.label_encoders[col].transform([val])[0])
                    else:
                        # Assign to "missing" category or last category
                        encoded.append(len(self.label_encoders[col].classes_) - 1)
                categorical_encoded.append(np.array(encoded))
            else:
                # Feature not present, use default
                categorical_encoded.append(np.zeros(len(df)))
        
        categorical_matrix = np.column_stack(categorical_encoded)
        
        # Scale numerical features
        numerical_data = self._prepare_numerical_features(df)
        numerical_scaled = self.scaler.transform(numerical_data)  # type: ignore
        
        # Combine features
        features = np.hstack([categorical_matrix, numerical_scaled])
        
        # Add engineered features
        engineered = self._create_engineered_features(df)
        features = np.hstack([features, engineered])
        
        return features

    def fit_transform(self, df: pd.DataFrame, target_col: str | None = None) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            df: Input dataframe
            target_col: Optional target column
            
        Returns:
            Feature matrix
        """
        self.fit(df, target_col)
        return self.transform(df)

    def _prepare_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare numerical features with imputation."""
        numerical_data = []
        
        for col in self.numerical_features:
            if col in df.columns:
                values = df[col].values
                # Impute missing values with median
                median = np.nanmedian(values)
                values = np.where(np.isnan(values), median, values)
                numerical_data.append(values)
            else:
                # Feature not present, use zeros
                numerical_data.append(np.zeros(len(df)))
        
        return np.column_stack(numerical_data)

    def _create_engineered_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create engineered features based on domain knowledge.
        
        Features:
        - thickness^2 (non-linear thickness effect)
        - surface_area * pore_volume (porosity interaction)
        - has_annealing (binary indicator)
        - thickness / interlayer_spacing (packing density proxy)
        """
        engineered = []
        
        # Thickness squared
        if "thickness_um" in df.columns:
            thickness = df["thickness_um"].fillna(5.0).values
            engineered.append(thickness ** 2)
        else:
            engineered.append(np.zeros(len(df)))
        
        # Surface area * pore volume interaction
        if "specific_surface_area_m2g" in df.columns and "pore_volume_cm3g" in df.columns:
            sa = df["specific_surface_area_m2g"].fillna(80.0).values
            pv = df["pore_volume_cm3g"].fillna(0.12).values
            engineered.append(sa * pv)
        else:
            engineered.append(np.zeros(len(df)))
        
        # Has annealing (binary)
        if "annealing_temp_c" in df.columns:
            has_annealing = (~df["annealing_temp_c"].isna()).astype(float).values
            engineered.append(has_annealing)
        else:
            engineered.append(np.zeros(len(df)))
        
        # Packing density proxy
        if "thickness_um" in df.columns and "interlayer_spacing_nm" in df.columns:
            thickness = df["thickness_um"].fillna(5.0).values
            spacing = df["interlayer_spacing_nm"].fillna(1.2).values
            packing = thickness / (spacing + 0.1)  # Add small constant to avoid division by zero
            engineered.append(packing)
        else:
            engineered.append(np.zeros(len(df)))
        
        return np.column_stack(engineered)

    def _get_feature_names(self) -> list[str]:
        """Get feature names for interpretability."""
        names = []
        
        # Categorical features
        names.extend(self.categorical_features)
        
        # Numerical features
        names.extend(self.numerical_features)
        
        # Engineered features
        names.extend([
            "thickness_squared",
            "surface_area_pore_volume",
            "has_annealing",
            "packing_density",
        ])
        
        return names

    def get_feature_names(self) -> list[str]:
        """Get feature names."""
        return self.feature_names

    def save(self, path: str) -> None:
        """Save feature engineer to disk."""
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FeatureEngineer":
        """Load feature engineer from disk."""
        return joblib.load(path)
