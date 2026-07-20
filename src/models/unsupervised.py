# src/models/unsupervised.py

import pandas as pd
import numpy as np
import joblib
import mlflow
import gc
import time
import yaml
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import hdbscan
import umap
from config.logging import get_logger

logger = get_logger(__name__)

class UnsupervisedFeatures:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initializes the unsupervised engine with project root and config."""
        self.root = Path(__file__).resolve().parents[2]
        with open(self.root / config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pca = None
        self.kmeans = None
        self.umap_reducer = None
        self.hdbscan_model = None

    def fit(self, X_train: pd.DataFrame):
        """Learns structures from Training Set and logs internal metrics to MLflow."""
        logger.info(f"Fitting unsupervised models on {X_train.shape[0]} samples...")
        
        # We use a Nested Run to isolate clustering metrics within the main training run
        with mlflow.start_run(run_name="Unsupervised_Clustering", nested=True):
            
            # --- 1. PCA & K-Means Path ---
            logger.info("Path 1: Running PCA + K-Means...")
            pca_var = self.config["models"]["unsupervised"]["kmeans"]["pca_variance"]
            self.pca = PCA(n_components=pca_var, random_state=42)
            X_pca = self.pca.fit_transform(X_train)
            
            mlflow.log_param("pca_n_components", self.pca.n_components_)
            
            n_clusters = self.config["models"]["unsupervised"]["kmeans"]["n_clusters"]
            if n_clusters == "auto": n_clusters = 8 # Fallback simple or Elbow
            
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.kmeans.fit(X_pca)
            
            mlflow.log_param("kmeans_k", n_clusters)
            mlflow.log_metric("kmeans_inertia", self.kmeans.inertia_)

            # --- 2. UMAP & HDBSCAN Path ---
            logger.info("Path 2: Running UMAP + HDBSCAN...")
            # Sample for UMAP fitting to avoid memory/time bottlenecks
            sample_size = min(50000, len(X_train))
            X_sample = X_train.sample(n=sample_size, random_state=42)
            
            n_comp = self.config["models"]["unsupervised"]["hdbscan"]["umap_components"]
            self.umap_reducer = umap.UMAP(n_components=n_comp, n_neighbors=15, min_dist=0.1, random_state=42, verbose=False)
            self.umap_reducer.fit(X_sample)
            
            # Transform train set via UMAP (Batched for safety)
            X_umap = self.umap_reducer.transform(X_train)
            
            min_size = self.config["models"]["unsupervised"]["hdbscan"]["min_cluster_size"]
            self.hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_size, prediction_data=True)
            self.hdbscan_model.fit(X_umap)
            
            # Log HDBSCAN results
            n_hdb_clusters = len(np.unique(self.hdbscan_model.labels_)) - 1
            noise_ratio = (self.hdbscan_model.labels_ == -1).sum() / len(X_train)
            
            mlflow.log_param("hdbscan_min_size", min_size)
            mlflow.log_metric("hdbscan_clusters_found", n_hdb_clusters)
            mlflow.log_metric("hdbscan_noise_ratio", noise_ratio)

            logger.info(f"Unsupervised fit complete. Found {n_hdb_clusters} density clusters.")
            
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies learned models to generate new predictive features.
        Force float64 at each step to prevent 'Buffer dtype mismatch' errors.
        """
        if self.kmeans is None or self.hdbscan_model is None:
            raise ValueError("Transform failed: Models must be fitted first.")
        
        logger.info(f"Transforming dataset of shape {X.shape}...")
        
        # 1. Force input to float64
        X_input = X.astype(np.float64)
        df_feats = pd.DataFrame(index=X.index)

        # --- A. K-MEANS PATH ---
        # 2. Transform via PCA and IMMEDIATELY force float64 to avoid Buffer dtype mismatch
        if self.kmeans.cluster_centers_.dtype != np.float64:
            self.kmeans.cluster_centers_ = self.kmeans.cluster_centers_.astype(np.float64)
        X_pca = self.pca.transform(X_input).astype(np.float64)
        
        # 3. Calculate distances (Soft Labels)
        # kmeans.transform needs float64
        distances = self.kmeans.transform(X_pca)
        for i in range(distances.shape[1]):
            df_feats[f'KM_DIST_C{i}'] = distances[:, i].astype(np.float32) # Store as float32 to save RAM
        
        # 4. Predict Cluster ID
        # kmeans.predict needs float64
        df_feats['KM_CLUSTER_ID'] = self.kmeans.predict(X_pca).astype(np.int32)

        # --- B. HDBSCAN PATH ---
        # 5. UMAP projection (Batched for RAM protection)
        batch_size = 50000
        umap_chunks = []
        for i in range(0, len(X_input), batch_size):
            chunk = X_input.iloc[i : i + batch_size]
            # reducer.transform returns float32/64 depending on fitting
            umap_chunks.append(self.umap_reducer.transform(chunk).astype(np.float64))
            
        X_umap = np.vstack(umap_chunks)

        # 6. HDBSCAN Prediction
        # approximate_predict is also sensitive to dtype
        hdb_labels, hdb_probs = hdbscan.approximate_predict(self.hdbscan_model, X_umap)
        
        df_feats['HDBSCAN_CLUSTER_ID'] = hdb_labels.astype(np.int32)
        df_feats['HDBSCAN_PROB'] = hdb_probs.astype(np.float32)
        
        # Final Memory Cleanup
        del X_pca, X_umap, umap_chunks, X_input
        gc.collect()
        
        return df_feats

    def save(self):
        """Saves all model artifacts and logs them to the current MLflow run."""
        model_dir = self.root / self.config["data_paths"]["models"]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        path = model_dir / "unsupervised_artifacts.joblib"
        artifacts = {
            "pca": self.pca,
            "kmeans": self.kmeans,
            "umap": self.umap_reducer,
            "hdbscan": self.hdbscan_model
        }
        joblib.dump(artifacts, path)
        
        # Log the file to MLflow
        mlflow.log_artifact(str(path))
        logger.info(f"Unsupervised artifacts archived in MLflow and at {path}")

# ==========================================================
# STANDALONE EXECUTION BLOCK
# ==========================================================
if __name__ == "__main__":
    try:
        logger.info("Starting standalone Unsupervised Feature Engineering test...")
        
        # 1. Initialize the engine
        worker = UnsupervisedFeatures()
        
        # 2. Load the ML-Ready dataset from config
        processed_dir = worker.root / worker.config["data_paths"]["processed"]
        input_file = worker.config["tables"]["ml_ready_data"]
        data_path = processed_dir / input_file
        
        if not data_path.exists():
            logger.error(f"Data file missing at {data_path}. Run preprocessor first.")
            exit(1)
            
        logger.info(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        
        # Isolate features (Dropping ID and TARGET)
        main_key = worker.config["fusion"]["main_key"]
        X = df.drop(columns=[main_key, 'TARGET'], errors='ignore')

        # 3. Setup MLflow for standalone test
        mlflow.set_tracking_uri(worker.config["mlops"]["mlflow_tracking_uri"])
        mlflow.set_experiment("Unsupervised_Features_Test")

        # 4. Execute the cycle within a parent run
        # Note: We use a small sample (10k rows) for the standalone test speed
        with mlflow.start_run(run_name="Standalone_Test"):
            sample_size = 10000
            X_sample = X.head(sample_size)
            
            logger.info(f"Running test on sample of {sample_size} rows...")
            worker.fit(X_sample)
            
            # Test transformation
            new_features = worker.transform(X_sample.head(10))
            logger.info(f"Transformation success! New features shape: {new_features.shape}")
            
            # Save artifacts
            worker.save()

        print("\n" + "="*50)
        print("UN-SUPERVISED TEST COMPLETED SUCCESSFULLY")
        print(f"• Input features  : {X.shape[1]}")
        print(f"• Generated feats : {new_features.shape[1]}")
        print(f"• Artifacts saved in MLflow and models/ directory.")
        print("="*50)

        # Cleanup memory
        del df, X, X_sample, new_features
        gc.collect()

    except Exception as e:
        logger.error(f"❌ Unsupervised standalone execution failed: {e}", exc_info=True)