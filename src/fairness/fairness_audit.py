# src/fairness/mitigation.py

import pandas as pd
import numpy as np
import joblib
import yaml
import re
from pathlib import Path
from fairlearn.metrics import (
    MetricFrame, 
    selection_rate, 
    demographic_parity_difference, 
    equalized_odds_difference,
    false_negative_rate
)
from sklearn.metrics import accuracy_score, recall_score
from config.logging import get_logger

logger = get_logger(__name__)

class IntelliLoanFairness:
    """
    Executes algorithmic fairness audits on production models 
    to detect and quantify bias across protected demographic attributes.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        # Path resolution from src/fairness/ to project root (2 levels up)
        self.root = Path(__file__).resolve().parents[2]
        with open(self.root / config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # 1. Load trained core ML model
        model_path = self.root / self.config["data_paths"]["models"] / "champion_model.joblib"
        self.model = joblib.load(model_path)
        
        # 2. Load the reference evaluation matrix containing target and sensitive attributes
        data_path = self.root / self.config["data_paths"]["processed"] / self.config["tables"]["final_enriched_data"]
        self.df = pd.read_parquet(data_path)

        # 3. Create reports output directory if it does not exist
        self.reports_dir = self.root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # 4. Reconstruct sensitive attributes from One-Hot encoded representations
        self._reconstruct_sensitive_features()

    def _reconstruct_sensitive_features(self):
        """Helper to safely reconstruct original categorical columns from One-Hot encoded representations."""
        # A. Gender
        if 'CODE_GENDER_F' in self.df.columns:
            self.df['GENDER_GROUP'] = np.where(self.df['CODE_GENDER_F'] == 1, 'Female', 'Male')
        
        # B. Family Status
        family_cols = [c for c in self.df.columns if c.startswith('NAME_FAMILY_STATUS_')]
        if family_cols:
            self.df['FAMILY_STATUS_GROUP'] = self.df[family_cols].idxmax(axis=1).str.replace('NAME_FAMILY_STATUS_', '')

        # C. Housing Type
        housing_cols = [c for c in self.df.columns if c.startswith('NAME_HOUSING_TYPE_')]
        if housing_cols:
            self.df['HOUSING_TYPE_GROUP'] = self.df[housing_cols].idxmax(axis=1).str.replace('NAME_HOUSING_TYPE_', '')

        # D. Income Type
        income_cols = [c for c in self.df.columns if c.startswith('NAME_INCOME_TYPE_')]
        if income_cols:
            self.df['INCOME_TYPE_GROUP'] = self.df[income_cols].idxmax(axis=1).str.replace('NAME_INCOME_TYPE_', '')

        # E. Organization Type
        org_cols = [c for c in self.df.columns if c.startswith('ORGANIZATION_TYPE_')]
        if org_cols:
            self.df['ORGANIZATION_TYPE_GROUP'] = self.df[org_cols].idxmax(axis=1).str.replace('ORGANIZATION_TYPE_', '')

    def run_comprehensive_audit(self, sensitive_feature: str) -> MetricFrame:
        """Runs a comprehensive fairness evaluation utilizing Fairlearn metrics framework."""
        logger.info(f"⚖️ Starting Fairlearn Audit for sensitive attribute: {sensitive_feature}")

        # 5. Dynamic Feature Alignment with training columns
        X = self.df.copy()
        X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]
        X_inference = X.reindex(columns=self.model.feature_name_, fill_value=0.0)
        
        # 6. Generate Targets and Predictions
        y_true = self.df['TARGET']
        y_pred = (self.model.predict_proba(X_inference)[:, 1] > 0.5).astype(int)
        
        # 7. Define Fairness Metrics Portfolio
        metrics = {
            'accuracy': accuracy_score,
            'recall': recall_score,
            'selection_rate': selection_rate,
            'false_negative_rate': false_negative_rate
        }

        # 8. Compute Disaggregated Metrics via Fairlearn MetricFrame
        mf = MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=self.df[sensitive_feature]
        )

        # 9. Compute Macro-level Parity Differences
        dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=self.df[sensitive_feature])
        eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=self.df[sensitive_feature])

        # 10. Console Output Report
        logger.info(f"Audit Complete for {sensitive_feature}")
        print(f"\n" + "="*60)
        print(f"FAIRNESS AUDIT REPORT: {sensitive_feature.upper()}")
        print( "="*60)
        print(f"Demographic Parity Difference : {dp_diff:.4f}")
        print(f"Equalized Odds Difference     : {eo_diff:.4f}")
        print("\n--- Disaggregated Metrics By Sub-Group ---")
        print(mf.by_group)
        print("="*60 + "\n")
        
        # 11. Artifact Persistence: Export metrics by group to the /reports directory
        # Replaces raw group dictionary into a clean pandas DataFrame structure for easy analytical reads
        report_df = mf.by_group.copy()
        # Inject macro-level scores as new columns for compliance aggregation
        report_df['macro_demographic_parity_diff'] = dp_diff
        report_df['macro_equalized_odds_diff'] = eo_diff
        
        report_path = self.reports_dir / f"fairness_audit_{sensitive_feature.lower()}.csv"
        report_df.to_csv(report_path)
        logger.info(f"💾 Fairness report saved successfully at: {report_path}")
        
        return mf

if __name__ == "__main__":
    logger.info("Starting Fairness Audit execution block...")
    auditor = IntelliLoanFairness()
    
    # Audit 1: Gender
    if 'GENDER_GROUP' in auditor.df.columns:
        mf_gender = auditor.run_comprehensive_audit('GENDER_GROUP')

    # Audit 2: Family Status
    if 'FAMILY_STATUS_GROUP' in auditor.df.columns:
        mf_family = auditor.run_comprehensive_audit('FAMILY_STATUS_GROUP')

    # Audit 3: Housing Type
    if 'HOUSING_TYPE_GROUP' in auditor.df.columns:
        mf_housing = auditor.run_comprehensive_audit('HOUSING_TYPE_GROUP')

    # Audit 4: Income Type
    if 'INCOME_TYPE_GROUP' in auditor.df.columns:
        mf_income = auditor.run_comprehensive_audit('INCOME_TYPE_GROUP')

    # Audit 5: Organization Type
    if 'ORGANIZATION_TYPE_GROUP' in auditor.df.columns:
        mf_org = auditor.run_comprehensive_audit('ORGANIZATION_TYPE_GROUP')
