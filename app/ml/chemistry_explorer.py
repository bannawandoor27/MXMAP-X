"""Chemistry space exploration using dimensionality reduction."""

import numpy as np
from typing import Any
from umap import UMAP
from sklearn.preprocessing import StandardScaler
import pandas as pd


class ChemistryExplorer:
    """
    Explore chemistry space using UMAP dimensionality reduction.
    
    Visualizes the design space and identifies similar materials.
    """

    def __init__(self, predictor: Any) -> None:
        """
        Initialize explorer.
        
        Args:
            predictor: Trained ML predictor
        """
        self.predictor = predictor
        self.umap_model: UMAP | None = None
        self.scaler: StandardScaler | None = None
        self.embeddings: np.ndarray | None = None
        self.device_data: list[dict[str, Any]] = []

    async def generate_chemistry_map(
        self,
        devices: list[dict[str, Any]],
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
    ) -> dict[str, Any]:
        """
        Generate 2D chemistry map using UMAP.
        
        Args:
            devices: List of device compositions
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            metric: Distance metric
            
        Returns:
            Dictionary with embeddings and metadata
        """
        # Convert devices to feature matrix
        df = pd.DataFrame(devices)
        X = self.predictor.feature_engineer.transform(df)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit UMAP
        self.umap_model = UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42,
        )
        
        self.embeddings = self.umap_model.fit_transform(X_scaled)
        self.device_data = devices
        
        # Get predictions for all devices
        predictions = []
        for device in devices:
            from app.models.schemas import PredictionRequest
            request = PredictionRequest(**device)
            result = await self.predictor.predict(request)
            predictions.append({
                "capacitance": result.areal_capacitance.value,
                "esr": result.esr.value,
                "rate_capability": result.rate_capability.value,
                "cycle_life": result.cycle_life.value,
            })
        
        # Prepare response
        map_data = {
            "embeddings": [
                {"x": float(emb[0]), "y": float(emb[1])}
                for emb in self.embeddings
            ],
            "devices": [
                {
                    "composition": device,
                    "predictions": pred,
                    "embedding": {"x": float(emb[0]), "y": float(emb[1])},
                }
                for device, pred, emb in zip(devices, predictions, self.embeddings)
            ],
            "parameters": {
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "metric": metric,
            },
        }
        
        return map_data

    def find_similar_devices(
        self,
        target_device: dict[str, Any],
        n_similar: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Find similar devices in chemistry space.
        
        Args:
            target_device: Target device composition
            n_similar: Number of similar devices to return
            
        Returns:
            List of similar devices with distances
        """
        if self.embeddings is None or self.umap_model is None:
            raise ValueError("Chemistry map not generated yet")
        
        # Transform target device
        df = pd.DataFrame([target_device])
        X = self.predictor.feature_engineer.transform(df)
        X_scaled = self.scaler.transform(X)  # type: ignore
        target_embedding = self.umap_model.transform(X_scaled)[0]
        
        # Calculate distances
        distances = np.linalg.norm(self.embeddings - target_embedding, axis=1)
        
        # Get top-k similar (excluding exact match if present)
        similar_indices = np.argsort(distances)[1 : n_similar + 1]
        
        similar_devices = [
            {
                "device": self.device_data[idx],
                "distance": float(distances[idx]),
                "embedding": {
                    "x": float(self.embeddings[idx][0]),
                    "y": float(self.embeddings[idx][1]),
                },
            }
            for idx in similar_indices
        ]
        
        return similar_devices

    def get_cluster_statistics(
        self, embeddings: np.ndarray, predictions: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Calculate statistics for clusters in chemistry space.
        
        Args:
            embeddings: 2D embeddings
            predictions: Prediction results
            
        Returns:
            Cluster statistics
        """
        from sklearn.cluster import DBSCAN
        
        # Perform clustering
        clustering = DBSCAN(eps=0.5, min_samples=5)
        labels = clustering.fit_predict(embeddings)
        
        # Calculate statistics per cluster
        cluster_stats = {}
        for label in set(labels):
            if label == -1:  # Noise points
                continue
            
            mask = labels == label
            cluster_preds = [p for p, m in zip(predictions, mask) if m]
            
            cluster_stats[f"cluster_{label}"] = {
                "size": int(mask.sum()),
                "avg_capacitance": float(
                    np.mean([p["capacitance"] for p in cluster_preds])
                ),
                "avg_esr": float(np.mean([p["esr"] for p in cluster_preds])),
                "avg_rate_capability": float(
                    np.mean([p["rate_capability"] for p in cluster_preds])
                ),
                "avg_cycle_life": float(
                    np.mean([p["cycle_life"] for p in cluster_preds])
                ),
            }
        
        return cluster_stats
