# src/monitoring/drift.py

import pandas as pd
import numpy as np
import yaml
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset, ClassificationPreset

from config.logging import get_logger

logger = get_logger(__name__)

class IntelliLoanDriftAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initializes the analyzer and resolves project paths."""
        self.root = Path(__file__).resolve().parents[2]
        with open(self.root / config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def _load_and_split(self) -> tuple:
        """Reproduces the training split to compare Reference vs Current distributions."""
        data_path = self.root / self.config["data_paths"]["processed"] / self.config["tables"]["final_enriched_data"]
        logger.info(f"Loading modeling matrix from: {data_path.name}")
        df = pd.read_parquet(data_path)
        
        return train_test_split(
            df, 
            test_size=self.config["training"]["test_size"],
            stratify=df["TARGET"],
            random_state=self.config["project"]["seed"]
        )

    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes special characters from column names for HTML/JS compatibility."""
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
        return df

    def run_pre_deployment_audit(self, inject_drift: bool = True) -> str:
        """
        Executes the drift audit. 
        Compares Training data (Reference) vs Testing data (Current).
        Saves the interactive report HTML and returns the output file path.
        """
        try:
            logger.info("🎬 Initializing Pre-deployment Drift Audit...")
            train_df, test_df = self._load_and_split()
            
            # 1. Feature Selection (Exclude ID, keep TARGET for context drift)
            main_key = self.config["fusion"]["main_key"]
            features = [c for c in train_df.columns if c != main_key]
            
            reference = train_df[features].copy()
            current = test_df[features].copy()

            # 2. ARTIFICIAL DRIFT INJECTION (Vectorisé & Sûr pour la RAM)
            if inject_drift:
                logger.warning("⚠️ SIMULATION: Injecting vectorised drift across all numerical features...")
                
                # Sélection instantanée en bloc de toutes les colonnes numériques
                num_cols = current.select_dtypes(include=['number']).columns
                cols_to_drift = [c for c in num_cols if c not in [main_key, 'TARGET']]
                
                # Modification "In-place" : 0% de RAM supplémentaire
                if cols_to_drift:
                    current[cols_to_drift] *= 1.5
                    logger.info(f"⚡ Drift injected instantly into {len(cols_to_drift)} columns.")

            # 3. Format Data for Evidently
            reference = self._sanitize_columns(reference)
            current = self._sanitize_columns(current)

            # 4. Generate the Statistical Report
            logger.info("Computing Drift metrics via Kolmogorov-Smirnov tests...")
            drift_report = Report(metrics=[
                DataSummaryPreset(), 
                DataDriftPreset()
            ])
            
            drift_report_result = drift_report.run(reference_data=reference, current_data=current)

            # 5. Persist HTML Artifact for audit trail
            report_dir = self.root / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            output_path = report_dir / "pre_deployment_drift_report.html"
            
            drift_report_result.save_html(str(output_path))
            logger.info(f"Visual report successfully archived at: {output_path}")
            
            # Return the file path
            return str(output_path)

        except Exception as e:
            logger.error(f"Drift Audit Pipeline failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    analyzer = IntelliLoanDriftAnalyzer()
    # Manual test run
    drift_score = analyzer.run_pre_deployment_audit(inject_drift=True)
    print(f"\nAudit Signal (Drift Share): {drift_score}")