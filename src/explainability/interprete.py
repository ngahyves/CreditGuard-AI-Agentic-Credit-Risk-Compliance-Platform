# src/models/interpret.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import yaml
import time
import gc
from pathlib import Path
from config.logging import get_logger

logger = get_logger(__name__)

class IntelliLoanInterpreter:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initializes the Interpretation Engine with production-ready memory management."""
        start_init = time.time()
        logger.info("Starting Interpretation Engine on FINAL enriched data...")
        
        self.root = Path(__file__).resolve().parents[2]
        with open(self.root / config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 1. Load the Champion Model
        model_path = self.root / self.config["data_paths"]["models"] / "champion_model.joblib"
        if not model_path.exists():
            logger.critical("Champion model not found! Run train.py first.")
            raise FileNotFoundError(f"Missing artifact: {model_path}")
        
        self.model = joblib.load(model_path)
        logger.info(f"Champion model loaded successfully from {model_path.name}")

        # 2. Load the Final Enriched Data
        processed_dir = self.root / self.config["data_paths"]["processed"]
        final_file = self.config["tables"]["final_enriched_data"]
        final_path = processed_dir / final_file

        if not final_path.exists():
            logger.critical(f"Final dataset missing: {final_path}.")
            raise FileNotFoundError(f"Missing dataset: {final_file}")
            
        logger.info(f"Loading final enriched matrix: {final_file}")
        df_final = pd.read_parquet(final_path)

        self.target = "TARGET"
        self.main_key = self.config["fusion"]["main_key"] # client's ID
        
        # We keep a lightweight mapping of unique client IDs to original dataframe positions
        # This allows lookups by real Client ID (e.g., SK_ID_CURR) instead of arbitrary positional rows
        self.client_id_mapping = df_final[self.main_key].tolist() if self.main_key in df_final.columns else []

        # 3. Prepare Feature Matrix (X)
        X_full = df_final.drop(columns=[self.target, self.main_key], errors='ignore')
        
        # 4. SHAP Sampling & Storage Optimization
        sample_size = 1000
        logger.info(f"Sampling {sample_size} records for SHAP value computation...")
        self.X_sample = X_full.sample(n=min(sample_size, len(X_full)), random_state=42)

        # Garbage collect X_full immediately to free up gigabytes of RAM in production
        del df_final, X_full
        gc.collect()

        # 5. Initialize SHAP Explainer
        logger.info("Initializing SHAP TreeExplainer for LightGBM...")
        self.explainer = shap.TreeExplainer(self.model)
        
        # 6. Pre-calculate global SHAP values
        calc_start = time.time()
        raw_shap = self.explainer.shap_values(self.X_sample)
        # Handle LightGBM output consistency (Extract class 1 / Risk)
        self.shap_values = raw_shap[1] if isinstance(raw_shap, list) else raw_shap
        
        logger.info(f"SHAP computation finished in {time.time() - calc_start:.2f}s.")
        logger.info(f"Engine Ready. Total Setup Time: {time.time() - start_init:.2f}s")

    def plot_global_importance(self):
        """Generates and saves the Global Feature Importance plot for model audit."""
        logger.info("Generating SHAP Global Summary Plot...")
        try:
            plt.figure(figsize=(14, 10))
            shap.summary_plot(self.shap_values, self.X_sample, show=False)
            plt.title(f"IntelliLoan {self.config['project']['version']} - Global Risk Drivers", fontsize=16)
            
            report_dir = self.root / self.config["data_paths"]["reports"]
            report_dir.mkdir(parents=True, exist_ok=True)
            output_path = report_dir / "shap_global_importance.png"
            
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Global Importance plot saved at: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate global SHAP plot: {e}")

    def get_local_explanation(self, client_data: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the top reason codes for a specific client.
        FIX: Accepts a single-row DataFrame (the actual client being evaluated by the API/RAG).
        """
        logger.info("Extracting local risk drivers for real-time client inference...")
        try:
            # Ensure incoming data doesn't contain tracking keys
            instance = client_data.drop(columns=[self.target, self.main_key], errors='ignore')
            
            # Real-time SHAP calculation for the specific individual query
            local_shap = self.explainer.shap_values(instance)
            if isinstance(local_shap, list):
                local_shap = local_shap[1]

            reasons = pd.DataFrame({
                'feature': instance.columns,
                'shap_impact': local_shap[0]
            })

            # High positive impact = Driving the model towards "Decline / High Risk"
            top_reasons = reasons.sort_values(by='shap_impact', ascending=False).head(5)
            return top_reasons
        except Exception as e:
            logger.error(f"Local interpretation failed: {e}")
            return pd.DataFrame()

# ==========================================================
# EXECUTION
# ==========================================================
if __name__ == "__main__":
    try:
        interpreter = IntelliLoanInterpreter()
        interpreter.plot_global_importance()
        
        print("\n" + "="*60)
        print("INTELLILOAN ADVISOR: SIMULATING API/RAG LOCAL CALL")
        print("="*60)
        
        # Simulate a real single client record payload arriving at our API
        mock_client_payload = interpreter.X_sample.iloc[[0]] 
        
        reasons = interpreter.get_local_explanation(mock_client_payload)
        print(reasons)
        print("="*60)
        logger.info("Interpretation pipeline finished successfully.")
        
    except Exception as e:
        logger.critical(f"Critical Failure in Interpretation: {e}", exc_info=True)


def get_local_explanation(self, client_data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting local risk drivers...")
        try:
            # 1. Nettoyage des colonnes (Comme dans le Predictor)
            import re
            instance = client_data.drop(columns=[self.target, self.main_key], errors='ignore').copy()
            instance.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in instance.columns]
            
            # 2. Reindex pour garantir l'ordre exact du modèle
            # self.model.feature_name_ contient l'ordre appris pendant le fit
            instance = instance.reindex(columns=self.model.feature_name_, fill_value=0.0)

            # 3. Calcul SHAP
            local_shap = self.explainer.shap_values(instance)
            if isinstance(local_shap, list):
                local_shap = local_shap[1]

            reasons = pd.DataFrame({
                'feature': instance.columns, # Utilise les noms sanitizés
                'shap_impact': local_shap[0]
            })

            return reasons.sort_values(by='shap_impact', ascending=False).head(5)
            
        except Exception as e:
            logger.error(f"❌ SHAP Local Error: {e}")
            # Retourne un DataFrame avec les colonnes attendues pour éviter le crash du prochain nœud
            return pd.DataFrame(columns=['feature', 'shap_impact'])