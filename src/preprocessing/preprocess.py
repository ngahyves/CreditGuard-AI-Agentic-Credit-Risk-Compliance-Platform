# src/features/preprocessor.py

import json
import yaml
import pandas as pd
import numpy as np
import pandera as pa
import time
import joblib
import gc
from pathlib import Path
from typing import Dict, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Import the corporate logger
from config.logging import get_logger

logger = get_logger(__name__)

class IntelliLoanPreprocessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initializes the preprocessor by loading configurations, metadata, 
        and the data contract (Pandera schema).
        """
        try:
            # 1. Setup Root and Load Operational Config
            self.root = Path(__file__).resolve().parents[2]
            full_config_path = self.root / config_path
            
            if not full_config_path.exists():
                logger.critical(f"Setup Error: Config file not found at {full_config_path}")
                raise FileNotFoundError(f"Missing config: {full_config_path}")

            with open(full_config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info("Operational configuration loaded successfully.")

            # 2. Load Feature Metadata (Numeric/Categorical lists generated in FE phase)
            metadata_dir = self.root / self.config["data_paths"]["processed"]
            metadata_path = metadata_dir / "feature_metadata.json"
            
            if not metadata_path.exists():
                logger.error("Metadata Error: feature_metadata.json missing.")
                raise FileNotFoundError("Feature metadata must be generated before preprocessing.")

            with open(metadata_path, 'r') as f:
                self.meta = json.load(f)
            
            self.num_features = self.meta.get("numeric_features", [])
            self.cat_features = self.meta.get("categorical_features", [])
            logger.info(f"Metadata loaded: {len(self.num_features)} numeric and {len(self.cat_features)} categorical features.")
            
            # 3. Load Data Contract (Pandera Schema)
            schema_rel_path = self.config["schema"]["processed_schema"]
            schema_path = self.root / schema_rel_path
            
            with open(schema_path, 'r') as f:
                self.schema = pa.DataFrameSchema.from_json(f.read())
            logger.info("Compliance Data Contract (Pandera) loaded.")

            # 4. Build Sklearn Pipeline structure (Unfitted)
            self.fitted_pipeline = None
            self.column_transformer = self._create_pipeline_structure()

        except Exception as e:
            logger.critical(f"Initialization Failed: {str(e)}", exc_info=True)
            raise

    def _create_pipeline_structure(self) -> ColumnTransformer:
        """Defines the transformation rules (Imputation, Scaling, Encoding)."""
        
        # Pipeline for numerical values
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        # Pipeline for categorical values
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        return ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.num_features),
                ('cat', cat_transformer, self.cat_features)
            ],
            remainder='drop', # Drops TARGET, SK_ID_CURR and others during transformation
            verbose_feature_names_out=False # Very important to keep original feature names for model consistency
        )

    def fit(self, df: pd.DataFrame):
        """
        Validates the data against the contract and fits the internal transformer.
        Learns medians and categories from the provided dataframe.
        """
        logger.info(f"Fitting preprocessor on {len(df)} records...")
        self.schema.validate(df, lazy=True)
        self.fitted_pipeline = self.column_transformer.fit(df)
        logger.info("Preprocessor fitting complete.")
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Applies learned transformations to new data payloads."""
        if self.fitted_pipeline is None:
            raise ValueError("The preprocessor has not been fitted yet.")
        return self.fitted_pipeline.transform(df)

    def save(self, output_path: Optional[str] = None):
        """Serializes the fitted preprocessor to a joblib file for production API use."""
        if self.fitted_pipeline is None:
            raise ValueError("Cannot save an unfitted preprocessor.")

        model_dir = self.root / self.config["data_paths"]["models"]
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if output_path is None:
            output_path = model_dir / "fitted_preprocessor.joblib"
        
        joblib.dump(self.fitted_pipeline, output_path)
        logger.info(f"Fitted preprocessor saved at: {output_path}")

    @classmethod
    def load(cls, path: str):
        """Reloads a saved preprocessor from disk."""
        logger.info(f"Loading preprocessor from {path}")
        return joblib.load(path)

    # ==========================================================
    # ORCHESTRATION METHOD 
    # ==========================================================
    def run_preprocessing(self) -> bool:
        """
        Main engine to execute the preprocessing lifecycle.
        Called by the Orchestrator (Prefect) or standalone.
        """
        try:
            logger.info("Starting Production-ready Preprocessing execution...")
            
            # 1. Load Input: Modeling Matrix (The 86-column table from FE)
            processed_dir = self.root / self.config["data_paths"]["processed"]
            input_file = self.config["tables"]["modeling_matrix"]
            df_modeling = pd.read_parquet(processed_dir / input_file)
            
            # 2. Fit and Save the artifact (The "Fitted Brain" for the API)
            self.fit(df_modeling)
            self.save()
            
            # 3. Transform the data into ML-Ready format (All Numeric)
            X_transformed = self.transform(df_modeling)
            
            # 4. Reconstruct DataFrame with new OHE columns
            # We fetch the exact names of the newly created binary columns
            ohe_cols = self.fitted_pipeline.named_transformers_['cat']\
                             .get_feature_names_out(self.cat_features)
            all_new_cols = self.num_features + list(ohe_cols)
            
            df_ml_ready = pd.DataFrame(X_transformed, columns=all_new_cols)
            
            # 5. RE-ADD the TARGET and SK_ID_CURR (which were dropped by 'remainder=drop')
            main_key = self.config["fusion"]["main_key"]
            df_ml_ready[main_key] = df_modeling[main_key].values
            df_ml_ready["TARGET"] = df_modeling["TARGET"].values

            # 6. Save Output: ML Ready Data (216+ cols ready for XGBoost/LightGBM)
            output_file = self.config["tables"]["ml_ready_data"]
            output_path = processed_dir / output_file
            df_ml_ready.to_parquet(output_path, index=False)
            
            logger.info(f"ML-Ready data saved at: {output_path}")
            print(f"\nPreprocessing Finished.")
            print(f"• Input  : {input_file} -> {df_modeling.shape[1]} columns")
            print(f"• Output : {output_file} -> {df_ml_ready.shape[1]} columns")

            # Final Memory Cleanup
            del df_modeling, df_ml_ready
            gc.collect()
            
            return True

        except Exception as e:
            logger.error(f"Preprocessing execution failed: {e}", exc_info=True)
            raise e

# ==========================================================
# STANDALONE ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    # Allows manual execution: python -m src.features.preprocessor
    engine = IntelliLoanPreprocessor()
    engine.run_preprocessing()