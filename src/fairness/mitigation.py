# src/fairness/mitigation.py

import pandas as pd
import numpy as np
import joblib
import yaml
import re
from pathlib import Path
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference, false_negative_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score
from config.logging import get_logger

logger = get_logger(__name__)

class IntelliLoanFairnessMitigation:
    """Applies ThresholdOptimizer post-processing to mitigate detected algorithmic biases."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.root = Path(__file__).resolve().parents[2]
        with open(self.root / config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.model = joblib.load(self.root / self.config["data_paths"]["models"] / "champion_model.joblib")
        self.df = pd.read_parquet(self.root / self.config["data_paths"]["processed"] / self.config["tables"]["final_enriched_data"])
        self.reports_dir = self.root / "reports"
        self._reconstruct_features()

    def _reconstruct_features(self):
        """Reconstructs GENDER_GROUP from One-Hot Encoded features for fairness grouping."""
        if 'CODE_GENDER_F' in self.df.columns:
            self.df['GENDER_GROUP'] = np.where(self.df['CODE_GENDER_F'] == 1, 'Female', 'Male')

    def apply_mitigation(self, sensitive_feature: str, constraint: str = "equalized_odds"):
        """
        Applies mitigation using the chosen constraint.
        Use 'equalized_odds' for banking (precision focus) 
        instead of 'demographic_parity' (volume focus).
        """
        logger.info(f"⚖️ Starting {constraint} mitigation for: {sensitive_feature}")
        
        # 1. Align features with model expectations
        X = self.df.copy()
        X.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in X.columns]
        X_inference = X.reindex(columns=self.model.feature_name_, fill_value=0.0)
        
        y_true = self.df['TARGET']
        sensitive_data = self.df[sensitive_feature]
        
        # 2. Train post-processing thresholds
        # We use 'equalized_odds' to avoid the 0.5% approval rate trap of 'demographic_parity'
        optimizer = ThresholdOptimizer(
            estimator=self.model, 
            constraints=constraint, 
            predict_method="predict_proba",
            objective="balanced_accuracy_score", # Better for imbalanced credit data
            prefit=True
        )
        
        logger.info("Fitting ThresholdOptimizer (this might take a minute)...")
        optimizer.fit(X_inference, y_true, sensitive_features=sensitive_data)
        
        # 3. Predict & Evaluate
        y_pred_mitigated = optimizer.predict(X_inference, sensitive_features=sensitive_data)
        
        mf_mitigated = MetricFrame(
            metrics={
                'accuracy': accuracy_score, 
                'recall': recall_score, 
                'selection_rate': selection_rate, # Approval Rate
                'false_negative_rate': false_negative_rate
            },
            y_true=y_true, 
            y_pred=y_pred_mitigated, 
            sensitive_features=sensitive_data
        )
        
        new_dp_diff = demographic_parity_difference(y_true, y_pred_mitigated, sensitive_features=sensitive_data)
        new_eo_diff = equalized_odds_difference(y_true, y_pred_mitigated, sensitive_features=sensitive_data)
        
        # 4. Reporting
        print(f"\nMITIGATION COMPLETE: {constraint.upper()} on {sensitive_feature.upper()}")
        print(f"-> Selection Rate (Avg Approval): {mf_mitigated.overall['selection_rate']:.2%}")
        print(f"-> Demographic Parity Diff: {new_dp_diff:.4f}")
        print(f"-> Equalized Odds Diff: {new_eo_diff:.4f}\n")
        print(mf_mitigated.by_group)
        
        # 5. Persistence
        report_name = f"mitigated_{constraint}_{sensitive_feature.lower()}.csv"
        mf_mitigated.by_group.to_csv(self.reports_dir / report_name)
        
        model_name = f"mitigated_model_{constraint}_{sensitive_feature.lower()}.joblib"
        joblib.dump(optimizer, self.root / self.config["data_paths"]["models"] / model_name)
        logger.info(f"Mitigated model saved as {model_name}")

if __name__ == "__main__":
    mitigator = IntelliLoanFairnessMitigation()
    if 'GENDER_GROUP' in mitigator.df.columns:
        # TEST 1: Equalized Odds
        mitigator.apply_mitigation('GENDER_GROUP', constraint="equalized_odds")