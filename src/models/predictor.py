# src/models/predictor.py

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.unsupervised import UnsupervisedFeatures # Import indispensable
from config.logging import get_logger

logger = get_logger(__name__)

class IntelliLoanPredictor:
    """
    Production-ready real-time inference pipeline for credit risk assessment.
    Supports Strategy A (Fixed) and Strategy B (Risk-Based Terciles).
    """
    def __init__(self, config: dict):
        self.root = Path(__file__).resolve().parents[2]
        self.config = config
        model_dir = self.root / config["data_paths"]["models"]
        
        # 1. Load core ML artifacts
        logger.info("Loading inference artifacts...")
        self.preprocessor = joblib.load(model_dir / "fitted_preprocessor.joblib")
        self.model = joblib.load(model_dir / "champion_model.joblib")
        
        # 2. Reconstruct the Unsupervised Engine
        # We need the class to apply the same PCA/UMAP transformation logic
        self.unsup_engine = UnsupervisedFeatures()
        artifacts = joblib.load(model_dir / "unsupervised_artifacts.joblib")
        self.unsup_engine.pca = artifacts['pca']
        self.unsup_engine.kmeans = artifacts['kmeans']
        self.unsup_engine.umap_reducer = artifacts['umap']
        self.unsup_engine.hdbscan_model = artifacts['hdbscan']

        # 3. Native Pandas Output Configuration
        self.preprocessor.set_output(transform="pandas")

    def _get_risk_segment(self, probability: float) -> str:
        """Assigns a risk segment based on pre-calculated tercile boundaries."""
        bounds = self.config["ab_testing"]["tercile_boundaries"]
        if probability <= bounds["low_max"]:
            return "low_risk"
        elif probability <= bounds["medium_max"]:
            return "medium_risk"
        return "high_risk"

    def get_credit_decision(self, raw_data: pd.DataFrame, strategy: str = "A") -> dict:
        """
        Main decision gate. 
        strategy: "A" (Standard) or "B" (Segmented)
        """
        # A. Preprocessing
        X_clean = self.preprocessor.transform(raw_data)
        
        # B. Unsupervised Enrichment (Using our reconstructed engine)
        X_unsup = self.unsup_engine.transform(X_clean)
        
        # C. Feature Fusion (Aligning indexes to avoid NaNs)
        X_enriched = pd.concat([X_clean.reset_index(drop=True), 
                                X_unsup.reset_index(drop=True)], axis=1)
        
        # D. Production Alignment
        X_inference = X_enriched.reindex(columns=self.model.feature_name_, fill_value=0.0)

        # E. Model Inference
        prob = float(self.model.predict_proba(X_inference)[0, 1])
        credit_score = int((1 - prob) * 1000)
        
        # F. A/B Testing Strategy Selection
        if strategy == "A":
            conf = self.config["ab_testing"]["strategies"]["A"]["thresholds"]
            risk_segment = "N/A"
        else:
            risk_segment = self._get_risk_segment(prob)
            conf = self.config["ab_testing"]["strategies"]["B"][risk_segment]
        
        approve_th = conf["approve"]
        decline_th = conf["decline"]

        # G. Business Decision Logic
        if credit_score > approve_th:
            verdict, risk_label = "APPROVE", "LOW"
        elif credit_score >= decline_th:
            verdict, risk_label = "MANUAL REVIEW", "MEDIUM"
        else:
            verdict, risk_label = "DECLINE", "HIGH"

        return {
            "probability_of_default": round(prob, 4),
            "credit_score": credit_score,
            "verdict": verdict,
            "risk_level": risk_label,
            "strategy_used": strategy,
            "risk_segment": risk_segment,
            "shap_reasons": [] # Placeholder for the Agentic RAG step
        }