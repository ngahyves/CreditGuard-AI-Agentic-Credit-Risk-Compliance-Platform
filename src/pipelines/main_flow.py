# src/pipelines/main_flow.py

import os
from pathlib import Path
from prefect import flow, task

# Pipeline Module Imports
from src.ingestion.ingest import Ingestion
from src.features.feature_engineering import FeatureEngineering
from src.preprocessing.preprocess import IntelliLoanPreprocessor
from src.models.train import IntelliLoanTrainer
from src.monitoring.drift import IntelliLoanDriftAnalyzer
from config.logging import get_logger

# 1. Define the UNIQUE configuration path relative to the project root
# Since this script is located in src/pipelines/, we go up two levels to reach the root
ROOT_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = "config/config.yaml" # Relative path from root

logger = get_logger(__name__)

# ==========================================================
# TASK DEFINITIONS
# (Each class is designed to accept CONFIG_PATH in its constructor)
# ==========================================================

@task(name="Data_Ingestion", retries=1)
def run_ingestion_task():
    """Triggers the raw data ingestion and initial validation."""
    logger.info("[Task] Starting Data Ingestion and Structural Validation...")
    ingestor = Ingestion(config_path=CONFIG_PATH)
    return ingestor.run_pipeline()

@task(name="Feature_Engineering")
def run_feature_eng_task():
    """Triggers advanced feature engineering and statistical pruning."""
    logger.info("[Task] Starting Advanced Feature Engineering...")
    fe = FeatureEngineering(config_path=CONFIG_PATH)
    return fe.run_pipeline()

@task(name="Production_Preprocessing")
def run_preprocessing_task():
    """Fits the production transformers and generates the ML-ready matrix."""
    logger.info("[Task] Running Production Preprocessing (Fit & Transform)...")
    # 1. Instantiate the preprocessor class
    prep = IntelliLoanPreprocessor(config_path=CONFIG_PATH)
    # 2. Execute the full internal logic (Fit + Transform + Persistence)
    prep.run_preprocessing()
    return True

@task(name="Model_Training")
def run_training_task():
    """Handles model benchmarking, Optuna optimization, and MLflow registration."""
    logger.info("[Task] Starting Model Benchmarking and Hyperparameter Tuning...")
    trainer = IntelliLoanTrainer(config_path=CONFIG_PATH)
    trainer.run_full_pipeline()

@task(name="Drift_Audit")
def run_drift_task():
    """Generates a statistical drift report for pre-deployment audit."""
    logger.info("[Task] Generating Statistical Data Drift Report...")
    analyzer = IntelliLoanDriftAnalyzer(config_path=CONFIG_PATH)
    # inject_drift=True simulates a shift to test the monitoring engine
    analyzer.run_pre_deployment_audit(inject_drift=True)

# ==========================================================
# MASTER ORCHESTRATION FLOW
# ==========================================================

@flow(name="IntelliLoan_E2E_Pipeline", log_prints=False) #to avoid conflicts with UMAP
def intelliloan_pipeline():
    """
    Main flow orchestrating the Home Credit project's logical execution order:
    Ingest -> Engineer -> Preprocess -> Train -> Audit.
    """
    logger.info(f"Launching IntelliLoan Lifecycle from Root: {ROOT_DIR}")

    # Step-by-step pipeline execution
    run_ingestion_task()
    run_feature_eng_task()
    run_preprocessing_task()
    run_training_task()
    run_drift_task()

    logger.info("Full Pipeline executed successfully. Model is ready for inference.")

if __name__ == "__main__":
    # To execute: python -m src.pipelines.main_flow
    intelliloan_pipeline()