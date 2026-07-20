# src/models/train.py

import os
import yaml
import pandas as pd
import numpy as np
import gc
import time
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.sklearn
import optuna
import joblib
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
from scipy.stats import ks_2samp

# Internal imports
from config.logging import get_logger
from src.models.unsupervised import UnsupervisedFeatures

logger = get_logger(__name__)

class IntelliLoanTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initializes the training engine and MLflow tracking."""
        self.root = Path(__file__).resolve().parents[2]
        with open(self.root / config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # MLflow setup from config
        try:
            mlflow.set_tracking_uri(self.config["mlops"]["mlflow_tracking_uri"])
            mlflow.set_experiment(self.config["mlops"]["experiment_name"])
            logger.info(f"MLflow initialized at {self.config['mlops']['mlflow_tracking_uri']}")
        except Exception as e:
            logger.error(f"Failed to connect to MLflow: {e}")
            raise

        self.main_key = self.config["fusion"]["main_key"]
        self.target = "TARGET"

    def load_and_split(self):
        """Loads ML-ready data and performs stratified split."""
        path = self.root / self.config["data_paths"]["processed"] / self.config["tables"]["ml_ready_data"]
        
        if not path.exists():
            logger.critical(f"ML-ready data missing at {path}")
            raise FileNotFoundError(f"File {path} not found.")

        df = pd.read_parquet(path)
        
        X = df.drop(columns=[self.target, self.main_key])
        y = df[self.target]
        
        logger.info(f"Data loaded: {X.shape}. Splitting into Train/Test (Stratified)...")
        return train_test_split(
            X, y, 
            test_size=self.config["training"]["test_size"], 
            stratify=y, 
            random_state=self.config["project"]["seed"]
        )
    
    def _sanitize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes special characters from column names to prevent LightGBM/XGBoost crashes.
        Replaces spaces, brackets, and special JSON characters with underscores.
        """
        import re
        new_cols = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
        df.columns = new_cols
        return df

    def get_scale_weight(self, y):
        """Calculates the ratio for class imbalance handling (Ratio 0/1)."""
        counts = y.value_counts()
        return float(counts[0] / counts[1])

    def _log_metrics(self, y_true, y_proba, prefix="test"):
        """
        Calculates and logs banking-specific metrics to MLflow.
        Used for both Benchmarking and Final Training.
        """
        from sklearn.metrics import precision_recall_curve, auc, brier_score_loss
        from scipy.stats import ks_2samp

        # 1. Standard AUC
        auc_score = roc_auc_score(y_true, y_proba)
        
        # 2. Gini Coefficient
        gini = 2 * auc_score - 1
        
        # 3. Kolmogorov-Smirnov (KS) Statistic
        # Compares the distribution of Good vs Bad
        data = pd.DataFrame({'target': y_true, 'proba': y_proba})
        dist_good = data[data['target'] == 0]['proba']
        dist_bad = data[data['target'] == 1]['proba']
        ks_stat = ks_2samp(dist_good, dist_bad).statistic
        
        # 4. Precision-Recall AUC (Better for 8% imbalanced target)
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        # 5. Brier Score (Calibration check)
        brier = brier_score_loss(y_true, y_proba)

        # Log all to MLflow
        metrics = {
            f"{prefix}_auc": auc_score,
            f"{prefix}_gini": gini,
            f"{prefix}_ks": ks_stat,
            f"{prefix}_pr_auc": pr_auc,
            f"{prefix}_brier": brier
        }
        mlflow.log_metrics(metrics)
        return auc_score # We return AUC to keep track of the champion

    def benchmark_models(self, X_train, X_test, y_train, y_test):
        """Phase 1: Compares default models using multiple banking metrics."""
        logger.info(" Starting Phase 1: Model Benchmarking with Banking Metrics...")
        
        weight = self.get_scale_weight(y_train)
        models = {
            "XGBoost": xgb.XGBClassifier(scale_pos_weight=weight, random_state=42, tree_method='hist'),
            "LightGBM": lgb.LGBMClassifier(scale_pos_weight=weight, random_state=42, verbosity=-1),
            "CatBoost": CatBoostClassifier(auto_class_weights='Balanced', random_state=42, verbose=0)
        }

        best_model_name = None
        best_auc = 0

        for name, model in models.items():
            # Use 'nested=True' because this is part of the larger pipeline
            with mlflow.start_run(run_name=f"Benchmark_{name}", nested=True):
                logger.info(f"Evaluating {name}...")
                
                start_time = time.time()
                model.fit(X_train, y_train)
                duration = time.time() - start_time
                
                y_proba = model.predict_proba(X_test)[:, 1]
                
                # Use our central logging function
                current_auc = self._log_metrics(y_test, y_proba, prefix="bench")
                
                mlflow.log_param("model_type", name)
                mlflow.log_metric("train_duration_sec", duration)
                
                if current_auc > best_auc:
                    best_auc = current_auc
                    best_model_name = name
                    
                logger.info(f"-> {name} AUC: {current_auc:.4f} | Gini: {2*current_auc-1:.4f}")

        logger.info(f"Champion selected: {best_model_name}")
        return best_model_name

    def objective(self, trial, X, y, model_name):
        """Optuna objective with RAM optimization."""
        import gc # local import
        
        weight = self.get_scale_weight(y)
        
        # --- OPTIMISATION 1: downcast in float32 to gather 50% RAM ---
        X = X.astype(np.float32)

        if model_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'scale_pos_weight': weight,
                'tree_method': 'hist',
                'n_jobs': -1 
            }
            model_class = xgb.XGBClassifier
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000), # trees
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),       # leaves
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0), # columns subsampling
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0), # rows subsampling
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'scale_pos_weight': weight,
                'random_state': 42,
                'verbosity': -1,
                'n_jobs': -1
            }
            model_class = lgb.LGBMClassifier

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_t, X_v = X.iloc[train_idx], X.iloc[val_idx]
            y_t, y_v = y.iloc[train_idx], y.iloc[val_idx]
            
            clf = model_class(**params)
            clf.fit(X_t, y_t)
            
            auc = roc_auc_score(y_v, clf.predict_proba(X_v)[:, 1])
            cv_scores.append(auc)
            
            # --- OPTIMISATION 2: Nettoyage manuel après chaque Fold ---
            del X_t, X_v, y_t, y_v, clf
            gc.collect()

        # --- OPTIMISATION 3: Nettoyage final avant de rendre le score ---
        final_score = np.mean(cv_scores)
        gc.collect()
        
        return final_score

    def run_full_pipeline(self):
        """
        Main training workflow: 
        1. Split -> 2. Enrich -> 3. Benchmark -> 4. Optimize -> 5. Final Training & Save.
        """
        import re
        # ==========================================================
        # 1. LOADING & SPLITTING
        # ==========================================================
        X_train_raw, X_test_raw, y_train, y_test = self.load_and_split()
        
        # Memory optimization
        X_train_raw = X_train_raw.astype(np.float32)
        X_test_raw = X_test_raw.astype(np.float32)

        # ==========================================================
        # 2. FEATURE ENRICHMENT (UNSUPERVISED)
        # ==========================================================
        logger.info("--- Phase 1: Unsupervised Feature Enrichment ---")
        unsup = UnsupervisedFeatures()
        
        # Fit clustering only on Training data
        unsup.fit(X_train_raw)
        
        # Transform both sets (Addition of the 11 cluster features)
        X_train_enriched = pd.concat([X_train_raw, unsup.transform(X_train_raw)], axis=1).astype(np.float32)
        X_test_enriched = pd.concat([X_test_raw, unsup.transform(X_test_raw)], axis=1).astype(np.float32)
        
        # Sanitization of column names for LightGBM compatibility
        X_train_enriched = self._sanitize_column_names(X_train_enriched)
        X_test_enriched = self._sanitize_column_names(X_test_enriched)
        
        logger.info(f"Enrichment complete. Modeling features: {X_train_enriched.shape[1]}")
        
        # Free up memory
        del X_train_raw, X_test_raw
        gc.collect()

        # ==========================================================
        # 3. BENCHMARKING (ON ENRICHED DATA)
        # ==========================================================
        logger.info("--- Phase 2: Benchmarking Architectures ---")
        champion_name = self.benchmark_models(X_train_enriched, X_test_enriched, y_train, y_test)
        gc.collect()

        # ==========================================================
        # 4. HYPERPARAMETER TUNING (OPTUNA)
        # ==========================================================
        # Tuning on 50k samples to prevent memory allocation errors
        tuning_size = min(50000, len(X_train_enriched))
        logger.info(f"--- Phase 3: Optimizing {champion_name} on {tuning_size} samples ---")
        
        X_tuning = X_train_enriched.sample(n=tuning_size, random_state=42)
        y_tuning = y_train.loc[X_tuning.index]

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, X_tuning, y_tuning, champion_name), 
                       n_trials=self.config["mlops"]["optuna"]["n_trials"])

        del X_tuning, y_tuning
        gc.collect()

        # ==========================================================
        # 5. FINAL CHAMPION TRAINING & PERSISTENCE
        # ==========================================================
        with mlflow.start_run(run_name="Champion_Final_Training"):
            logger.info(f"Training final {champion_name} on full enriched dataset...")
            
            # Prepare final parameters
            weight = self.get_scale_weight(y_train)
            final_params = {**study.best_params, "scale_pos_weight": weight, "random_state": 42}
            
            if champion_name == "XGBoost":
                final_model = xgb.XGBClassifier(**final_params, tree_method='hist')
            else:
                final_model = lgb.LGBMClassifier(**final_params)
            
            # Final Fit
            final_model.fit(X_train_enriched, y_train)

            # --- EVALUATION ---
            y_proba = final_model.predict_proba(X_test_enriched)[:, 1]
            final_auc = self._log_metrics(y_test, y_proba, prefix="final")
            mlflow.log_params(study.best_params)
            
            # --- MODEL REGISTRY ---
            reg_name = self.config["mlops"]["model_registry_name"]
            if champion_name == "XGBoost":
                mlflow.xgboost.log_model(final_model, "model", registered_model_name=reg_name)
            else:
                mlflow.lightgbm.log_model(final_model, "model", registered_model_name=reg_name)

            # --- LOCAL SAVING (FOR API & INTERPRETATION) ---
            model_dir = self.root / self.config["data_paths"]["models"]
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Save the model object
            model_path = model_dir / "champion_model.joblib"
            joblib.dump(final_model, model_path)
            
            # 2. Save the Unsupervised artifacts
            unsup.save() 
            
            # 3. Save the final enriched dataset (X + y + ID)
            logger.info("Persisting the final enriched modeling matrix...")
            df_final_data = X_train_enriched.copy()
            df_final_data['TARGET'] = y_train.values
            df_final_data[self.main_key] = y_train.index 

            # Use name from config: final_enriched_modeling_matrix.parquet
            output_name = self.config["tables"]["final_enriched_data"]
            processed_dir = self.root / self.config["data_paths"]["processed"]
            output_path = processed_dir / output_name
            
            df_final_data.to_parquet(output_path, index=False)
            
            # --- MLFLOW ARTIFACTS ---
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(output_path))
            
            # Link the preprocessor created earlier
            prep_path = model_dir / "fitted_preprocessor.joblib"
            if prep_path.exists():
                mlflow.log_artifact(str(prep_path))
            
            logger.info(f"PIPELINE SUCCESSFUL. Final AUC: {final_auc:.4f}")
            print(f"\nFinal Test AUC: {final_auc:.4f}")
            print(f"Data saved at: {output_path}")

if __name__ == "__main__":
    try:
        trainer = IntelliLoanTrainer()
        trainer.run_full_pipeline()
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)