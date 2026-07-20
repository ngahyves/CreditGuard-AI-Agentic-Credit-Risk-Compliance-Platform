# src/features/feature_engineering.py

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import gc
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis
from scipy.stats import pearsonr, f_oneway, chi2_contingency, ttest_ind, spearmanr
import pandera as pa
import black # for pandera contract exporting
import json
 
# Import custom logger
from config.logging import get_logger
logger = get_logger(__name__)

class FeatureEngineering:
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initializes paths and loads configuration."""
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_path = self.project_root / config_path
        self.config = self._load_config()

        #File path
        processed_dir = self.config["data_paths"]["processed"]
        file_name = self.config["tables"]["ingested_table"]
        self.file_path = self.project_root / processed_dir / file_name
        logger.info(f"Feature Engineering initialized for file: {self.file_path}")

    def _load_config(self) -> Dict:
        """Loads YAML configuration."""
        if not self.config_path.exists():
            logger.critical(f"Config file missing at {self.config_path}")
            raise FileNotFoundError(f"Missing config: {self.config_path}")
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def data_integrity(self) -> pd.DataFrame:
        """Checks data integrity and removes useless columns."""
        if not self.file_path.exists():
            logger.critical(f"Ingested file missing at {self.file_path}")
            raise FileNotFoundError(f"Missing ingested file: {self.file_path}")
            
        logger.info(f'Reading ingested file at {self.file_path}')
        df = pd.read_parquet(self.file_path)
        total_rows = len(df)

        # 1. Unique ID Check (Essential for Credit Risk)
        main_key = self.config["fusion"]["main_key"]
        if not df[main_key].is_unique:
            logger.critical("Data Corruption: SK_ID_CURR is not unique!")
            raise ValueError("Duplicate IDs found in Master Table.")

        # 2. Sparsity & Constant Variance Audit
        stats = pd.DataFrame({
            'missing_pct': (df.isnull().sum() / total_rows) * 100,
            'nunique': df.nunique()
        })
        too_sparse_cols = stats[stats['missing_pct'] > 90].index.tolist()
        constant_cols = stats[stats['nunique'] <= 1].index.tolist()
        cols_to_drop = list(set(too_sparse_cols + constant_cols))
        
        # We ensure TARGET and ID are NEVER dropped
        cols_to_drop = [c for c in cols_to_drop if c not in [main_key, 'TARGET']]
        df_cleaned = df.drop(columns=cols_to_drop)
        logger.info(f"Integrity check passed. Dropped {len(cols_to_drop)} columns. New shape: {df_cleaned.shape}")
        return df_cleaned

    def transform_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts days to years and fixes the 365243 anomaly."""
        df = df.copy()
        # Handling DAYS_BIRTH
        if 'DAYS_BIRTH' in df.columns:
            df['YEARS_BIRTH'] = df['DAYS_BIRTH'] / -365.25
        
        # Handling DAYS_EMPLOYED with its specific anomaly
        if 'DAYS_EMPLOYED' in df.columns:
            # We fix the anomaly before dividing to avoid precision issues
            df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
            df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / -365.25

        logger.info("Temporal features converted. DAYS_EMPLOYED anomaly addressed.")
        return df

    #Categorizes each feature into one of the 6 strategic risk pillars
    @staticmethod
    def categorize_pillar(col: str) -> str:
        """
        Categorizes each feature into one of the 6 strategic risk pillars 
        based on naming conventions and prefixes.
        """
        # 1. Extern Scores: Third-party risk scores
        if 'EXT_SOURCE' in col: 
            pillar = 'Extern Scores'
        # 2. Profile Client: Personal and demographic data
        elif any(x in col for x in ['YEARS_BIRTH', 'YEARS_EMPLOYED', 'EDUCATION', 'GENDER', 'FAMILY_STATUS']):
            pillar = 'Client Profile'
        # 3. Bureau: External credit history
        elif 'DAYS_CREDIT' in col or col.startswith('BU_') or 'BUREAU' in col:
            pillar = 'Bureau'
        # 4. Previous Applications: Static data from past requests
        elif col.startswith('PRV_') and not any(x in col for x in ['INST_', 'POS_', 'CC_']):
            pillar = 'Previous Applications'
        # 5. Intern aggregations: Behavioral data (Payments, CC, POS)
        elif any(x in col for x in ['INST_', 'POS_', 'CC_', 'PAYMENT', 'BALANCE']):
            pillar = 'Intern aggregations'
        # 6. Financial: Basic amounts and count metrics
        elif any(col.startswith(prefix) for prefix in ['AMT_', 'CNT_', 'DAYS_']):
            pillar = 'Financial'
        else:
            pillar = 'Others'
            
        return pillar

    # Compute information value
    def compute_iv(self, df_input: pd.DataFrame, feature: str, target: str, bins: int = 10) -> float:
        """Computes Information Value (IV) for a numerical feature."""
        # Use a minimal slice of data to save memory
        df_local = df_input[[feature, target]].dropna().copy()
        if df_local.empty or df_local[feature].nunique() <= 1:
            return np.nan
        try:
            # Automatic binning
            df_local['bin'] = pd.qcut(df_local[feature], q=bins, duplicates='drop')
            # Contingency table
            grouped = df_local.groupby('bin', observed=False)[target].agg(['count', 'sum'])
            grouped.columns = ['total', 'bad']
            grouped['good'] = grouped['total'] - grouped['bad']

            total_good = grouped['good'].sum()
            total_bad = grouped['bad'].sum()

            # Safety check: avoid division by zero if a bin has only one class
            if total_good == 0 or total_bad == 0:
                return np.nan

            grouped['dist_good'] = grouped['good'] / total_good
            grouped['dist_bad'] = grouped['bad'] / total_bad
            
            # Numerical stability: replace 0 with a very small value for log
            grouped = grouped.replace(0, 1e-6)
            grouped['woe'] = np.log(grouped['dist_good'] / grouped['dist_bad'])
            grouped['iv'] = (grouped['dist_good'] - grouped['dist_bad']) * grouped['woe']

            return float(grouped['iv'].sum())
            
        except Exception as e:
            logger.debug(f"IV calculation failed for {feature}: {str(e)}")
            return np.nan

    # ==============================================================================
    # UNIVARIATE AUDIT
    #Interpret iv
    def _interpret_iv(self, iv_val: float) -> str:
        """Standard banking interpretation of Information Value."""
        if pd.isna(iv_val): return 'Unknown'
        if iv_val < 0.02: return 'Useless'
        if iv_val < 0.1: return 'Weak'
        if iv_val < 0.3: return 'Medium'
        if iv_val < 0.5: return 'Strong'
        return 'Suspicious (Too high)'

    #Calculation of skewness, kurtosis, AUC, iv
    def compute_univariate_metrics(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """Calculates Skewness, Kurtosis, AUC, and IV for all numerical features."""
        results = []
        numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != target]
        total_features = len(numeric_cols)
        logger.info(f"Starting univariate audit for {total_features} features.")
        
        for i, col in enumerate(numeric_cols):
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{total_features} features processed...")
                
            df_temp = df[[col, target]].dropna() #removing missings for calculation
            if df_temp[col].nunique() <= 1:
                continue
                
            # Statistical calculations
            results.append({
                "variable": col,
                "skewness": skew(df_temp[col]),
                "kurtosis": kurtosis(df_temp[col]),
                "auc": roc_auc_score(df_temp[target], df_temp[col]),
                "iv": self.compute_iv(df, col, target)
            })

        # Build final audit dataframe
        results_df = pd.DataFrame(results)
        results_df['pillar'] = results_df['variable'].apply(self.categorize_pillar)
        results_df['predictive_power'] = (results_df['auc'] - 0.5).abs()
        results_df['iv_interpretation'] = results_df['iv'].apply(self._interpret_iv)
        
        logger.info("Univariate metrics audit finalized.")
        return results_df.sort_values("predictive_power", ascending=False)

    # ==============================================================================
    # AUTOMATED SELECTION & REFINEMENT
    # Outlier detection using isolation forest
    def _run_outlier_detection(self, df: pd.DataFrame, features: list, target: str) -> pd.DataFrame:
        """Identifies financial outliers using Isolation Forest and reports risk."""
        from sklearn.ensemble import IsolationForest
        from sklearn.impute import SimpleImputer

        logger.info("Running Isolation Forest for global outlier detection...")
        imputer = SimpleImputer(strategy='median')
        X_outliers = imputer.fit_transform(df[features])

        iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
        df['is_outlier'] = iso_forest.fit_predict(X_outliers)
        df['is_outlier'] = df['is_outlier'].map({1: 0, -1: 1})

        # Summary statistics for outliers
        if df['is_outlier'].nunique() > 1:
            outlier_risk = df.groupby('is_outlier', observed=False)[target].mean()
            print(f"\nDefault rate for Normal Clients: {outlier_risk.get(0, 0):.2%}")
            print(f"Default rate for Outlier Clients: {outlier_risk.get(1, 0):.2%}")
        
        return df

    # Winsorization (capping) and Log-transforms
    def _apply_shape_corrections(self, df: pd.DataFrame, df_metrics: pd.DataFrame) -> pd.DataFrame:
        """Applies Winsorization (capping) and Log-transforms based on audit results."""
        current_cols = df.columns.tolist()
        
        # 1. Target variables for Winsorization (Kurtosis > 100)
        extreme_vars = df_metrics[(df_metrics['kurtosis'] > 100) & 
                                  (df_metrics['variable'].isin(current_cols))]['variable'].tolist()
        for var in extreme_vars:
            df[var] = df[var].clip(upper=df[var].quantile(0.99))

        # 2. Target variables for Log-transform (Skewness > 2)
        possible_skewed = df_metrics[(df_metrics['skewness'].abs() > 2) & 
                                     (df_metrics['variable'].isin(current_cols)) & 
                                     (~df_metrics['variable'].isin(extreme_vars))]['variable'].tolist()
        
        logged_count = 0
        for var in possible_skewed:
            if np.nanmin(df[var].values) >= 0:
                df[var] = np.log1p(df[var])
                logged_count += 1
        
        logger.info(f"Corrections: {len(extreme_vars)} variables capped, {logged_count} variables log-transformed.")
        return df

    # Features selection function based on predictive power
    def automated_feature_selection(self, df_cleaned: pd.DataFrame, df_metrics: pd.DataFrame, target: str = 'TARGET') -> pd.DataFrame:
        """Main orchestrator for feature selection, outlier detection, and shape correction."""
        
        logger.info("Initializing automated feature selection...")
        
        # 1. Filtering by Predictive Power
        threshold = self.config["fusion"].get("PREDICTIVE_POWER_THRESHOLD", 0.01)
        kept_features = df_metrics[df_metrics['predictive_power'] >= threshold]['variable'].tolist()
        
        essential_cols = [self.config["fusion"]["main_key"], target]
        final_cols = [c for c in list(set(essential_cols + kept_features)) if c in df_cleaned.columns]
        
        df_filtered = df_cleaned[final_cols].copy()

        # Print Selection Report
        print("\n" + "="*50)
        print(f"FEATURE SELECTION: {len(kept_features)} features retained (Threshold: {threshold})")
        print("="*50)

        # 3. Recombine with categorical variables
        all_cat_cols = df_cleaned.select_dtypes(include=['object', 'category']).columns.tolist()
        cat_to_add = [c for c in all_cat_cols if c not in df_filtered.columns]
        df_bivariate = pd.concat([df_filtered, df_cleaned[cat_to_add]], axis=1)

        # 4. Statistical Shape Corrections
        df_final = self._apply_shape_corrections(df_bivariate, df_metrics)

        logger.info(f"Feature selection and refinement completed. Final Shape: {df_final.shape}")
        return df_final

    # ==============================================================================
    # BIVARIATE STATISTICAL AUDIT

    def _analyze_numeric_predictors(self, df: pd.DataFrame, num_feats: list, target: str) -> list:
        """
        Compares each NUMERIC variable (e.g., Age, Income) against the TARGET (0/1).
        Goal: Is the average value different between 'Good' and 'Bad' payers?
        """
        import scipy.stats as stats
        results = []
        
        for var in num_feats:
            clean_df = df[[var, target]].dropna()
            # Split the variable into two groups based on the Target
            group_repaid = clean_df[clean_df[target] == 0][var].values
            group_default = clean_df[clean_df[target] == 1][var].values
            
            if len(group_repaid) > 0 and len(group_default) > 0:
                # Welch's t-test: Do 'Good' and 'Bad' payers have statistically different means?
                # We use Welch because our groups are imbalanced (92% vs 8%).
                t_stat, p_val = stats.ttest_ind(group_repaid, group_default, equal_var=False)
                
                results.append({
                    'variable': var, 
                    'test_type': "Welch's t-test", 
                    'metric_value': t_stat, 
                    'p_value': p_val, 
                    'importance': 'High' if p_val < 0.05 else 'Low'
                })
        return results

    def _analyze_categorical_predictors(self, df: pd.DataFrame, cat_feats: list, target: str) -> list:
        """
        Compares each CATEGORICAL variable (e.g., Education) against the TARGET (0/1).
        Goal: Does the category influence the probability of default?
        """
        import scipy.stats as stats
        results = []
        
        for var in cat_feats:
            # Create a contingency table (Cross-tabulation)
            # Row: Education level, Column: Target (0 or 1)
            contingency = pd.crosstab(df[var], df[target])
            
            if not contingency.empty:
                # Chi-Square test: Is the distribution of default 'dependent' on this category?
                chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                
                results.append({
                    'variable': var, 
                    'test_type': 'Chi-Square', 
                    'metric_value': chi2, 
                    'p_value': p_val, 
                    'importance': 'High' if p_val < 0.05 else 'Low'
                })
        return results

    def run_bivariate_analysis(self, df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main function to run the bivariate audit against our binary TARGET (0/1).
        """
        logger.info(f"Starting bivariate analysis against binary target: {target_col}")
        
        # 1. Separate our features
        main_key = self.config["fusion"]["main_key"]
        features = [col for col in df.columns if col not in [target_col, main_key, 'is_outlier']]
        num_feats = df[features].select_dtypes(include=[np.number]).columns.tolist()
        cat_feats = df[features].select_dtypes(include=['object', 'category']).columns.tolist()

        # 2. Run the two specialized audits (Both compare vs Target 0/1)
        num_results = self._analyze_numeric_predictors(df, num_feats, target_col)
        cat_results = self._analyze_categorical_predictors(df, cat_feats, target_col)

        # 3. Format as DataFrames for easy reading
        df_num = pd.DataFrame(num_results).sort_values(by='p_value')
        df_cat = pd.DataFrame(cat_results).sort_values(by='p_value')

        logger.info(f"Bivariate audit complete. {len(df_num)} numeric and {len(df_cat)} categorical features tested.")
        return df_num, df_cat

    # ==============================================================================
    # FINAL SELECTION (SIGNIFICANCE & MULTICOLLINEARITY)

    def perform_final_selection(self, df_bivariate: pd.DataFrame, df_res_num: pd.DataFrame, 
                               df_res_cat: pd.DataFrame, df_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Final pruning of the dataset:
        1. Drops variables with p-value > 0.05 (Non-significant).
        2. Resolves multicollinearity (> 0.80) using an AUC-based 'Duel'.
        """
        logger.info("Starting final feature selection pipeline...")

        # --- STEP 1: STATISTICAL PRUNING (Significance Gate) ---
        # Identify variables with 'Low' importance from your run_bivariate_analysis results
        useless_num = df_res_num[df_res_num['importance'] == 'Low']['variable'].tolist()
        useless_cat = df_res_cat[df_res_cat['importance'] == 'Low']['variable'].tolist()
        useless_vars = list(set(useless_num + useless_cat))
        
        df_significant = df_bivariate.drop(columns=useless_vars, errors='ignore')
        logger.info(f"Dropped {len(useless_vars)} non-significant features. Remaining: {df_significant.shape[1]}")

        # --- STEP 2: MULTICOLLINEARITY CLEANUP (Redundancy Gate) ---
        logger.info("Starting multicollinearity analysis using Spearman correlation and AUC duel...")

        # 1. Identify numerical and categorical features
        cat_features = df_significant.select_dtypes(exclude=[np.number]).columns.tolist()
        num_features = df_significant.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude keys and target from correlation analysis
        main_key = self.config["fusion"]["main_key"]
        for col in [main_key, 'TARGET', 'is_outlier']:
            if col in num_features: num_features.remove(col)

        # 2. Compute Spearman Correlation Matrix (Robust to non-linear relationships)
        corr_matrix = df_significant[num_features].corr(method='spearman').abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # 3. Identify pairs to drop based on AUC Power
        threshold = 0.80
        to_drop_corr = set()
        high_corr_pairs = []

        for col in upper.columns:
            # Find peers correlated > threshold
            peers = upper.index[upper[col] > threshold].tolist()
            
            for peer in peers:
                if col in to_drop_corr or peer in to_drop_corr:
                    continue
                
                # Retrieve AUC power from df_metrics
                auc_col_series = df_metrics[df_metrics['variable'] == col]['auc']
                auc_peer_series = df_metrics[df_metrics['variable'] == peer]['auc']
                
                if auc_col_series.empty or auc_peer_series.empty:
                    continue
                
                # Calculate predictive strength: distance from random chance (0.5)
                power_col = abs(auc_col_series.values[0] - 0.5)
                power_peer = abs(auc_peer_series.values[0] - 0.5)
                
                # The 'Duel': Drop the one with lower predictive power
                if power_col < power_peer:
                    drop_candidate, keep_candidate = col, peer
                else:
                    drop_candidate, keep_candidate = peer, col
                
                to_drop_corr.add(drop_candidate)
                high_corr_pairs.append((keep_candidate, drop_candidate, corr_matrix.loc[peer, col]))

        # 4. Final Dataset Reconstruction
        num_features_to_keep = [f for f in num_features if f not in to_drop_corr]
        structural_keys = [c for c in [main_key, 'TARGET'] if c in df_significant.columns]
        
        final_columns = structural_keys + num_features_to_keep + cat_features
        df_final_modeling = df_significant[final_columns].copy()

        # --- STEP 3: FINAL REPORTING ---
        print("\n" + "="*50)
        print("FINAL SELECTION REPORT")
        print("="*50)
        print(f"Features analyzed (Numeric) : {len(num_features)}")
        print(f"Non-significant dropped     : {len(useless_vars)}")
        print(f"Redundant features dropped  : {len(to_drop_corr)}")
        print(f"Final total feature count   : {df_final_modeling.shape[1]}")
        print("="*50)

        if high_corr_pairs:
            print("\nTop 5 Statistical Duels (Kept vs Dropped):")
            for p in high_corr_pairs[:5]:
                print(f"- Kept: {p[0]} vs Dropped: {p[1]} (Corr: {p[2]:.2f})")

        return df_final_modeling
    
    # ARTIFACTS GENERATION (SAVE COLUMN NAMES & SCHEMA)
    # ==========================================================
    def _save_final_artifacts(self, df_final: pd.DataFrame, target: str, main_key: str):
        """
        Generates and saves the data contract (Pandera) and feature metadata.
        This allows the Preprocessor to know exactly what to do later.
        """
        import json
        import pandera as pa

        logger.info("Generating final artifacts (Metadata & Schema)...")

        # 1. Define and Create Processed Directory
        processed_dir = self.project_root / self.config["data_paths"]["processed"]
        processed_dir.mkdir(parents=True, exist_ok=True)

        # 2. Infer and Save Pandera Schema (The Data Contract)
        # Using the path from config.yaml
        schema_rel_path = self.config["schema"]["processed_schema"]
        schema_path = self.project_root / schema_rel_path
        
        final_schema = pa.infer_schema(df_final)
        final_schema.to_json(schema_path)
        logger.info(f"Modeling Data Contract saved at: {schema_path}")

        # 3. Identify feature types for the Preprocessor
        # We drop the Target and ID from the numerical list
        num_features = df_final.select_dtypes(include=[np.number]).columns.tolist()
        for col in [target, main_key, 'is_outlier']:
            if col in num_features:
                num_features.remove(col)
                
        cat_features = df_final.select_dtypes(include=['object', 'category']).columns.tolist()

        metadata = {
            "numeric_features": num_features,
            "categorical_features": cat_features,
            "target": target,
            "id": main_key
        }

        # 4. Save Metadata as JSON
        metadata_path = processed_dir / "feature_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Feature metadata saved at: {metadata_path}")

    # MAIN PIPELINE ORCHESTRATION
    # ==========================================================
    def run_pipeline(self) -> pd.DataFrame:
        """
        Main execution flow:
        Integrity -> Temporal -> Univariate Audit -> Selection -> 
        Bivariate Audit -> Final Pruning -> Artifacts -> Persistence.
        """
        logger.info("Starting Feature Engineering Pipeline...")
        target = "TARGET"
        main_key = self.config["fusion"]["main_key"]
        
        # 1. Data Cleaning & Basic Transformation
        df = self.data_integrity()
        df = self.transform_temporal_features(df)
        
        # 2. Statistical Audits
        df_metrics = self.compute_univariate_metrics(df, target)
        
        # 3. Filtering & Refinement
        df = self.automated_feature_selection(df, df_metrics, target)
        
        # 4. Significance & Multicollinearity
        df_res_num, df_res_cat = self.run_bivariate_analysis(df, target)
        df_final = self.perform_final_selection(df, df_res_num, df_res_cat, df_metrics)
        
        # 5. Final "DAYS_BIRTH" and "DAYS_EMPLOYED" conversion in years
        logger.info("Converting temporal features to Years for final output.")
        if 'DAYS_BIRTH' in df_final.columns:
            df_final['YEARS_BIRTH'] = df_final['DAYS_BIRTH'] / -365.25
            df_final = df_final.drop(columns=['DAYS_BIRTH'], errors='ignore')
        
        if 'DAYS_EMPLOYED' in df_final.columns:
            df_final['YEARS_EMPLOYED'] = df_final['DAYS_EMPLOYED'] / -365.25
            df_final = df_final.drop(columns=['DAYS_EMPLOYED'], errors='ignore')

        # --- save artifacts---
        self._save_final_artifacts(df_final, target, main_key)

        # 6. Final Persistence to Parquet
        processed_dir = self.project_root / self.config["data_paths"]["processed"]
        output_name = self.config["tables"]["modeling_matrix"]
        output_path = processed_dir / output_name
        
        df_final.to_parquet(output_path, index=False)
        
        logger.info(f"FEATURE ENGINEERING COMPLETE. Final Shape: {df_final.shape}")
        print(f"\nPipeline Finished Successfully!")
        print(f" Final rows: {df_final.shape[0]}")
        print(f" Final features: {df_final.shape[1]}")
        print(f" Artifacts & Data saved in: {processed_dir}")
        
        return df_final

# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    try:
        fe = FeatureEngineering()
        # Run the pipeline
        df_final = fe.run_pipeline()
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)