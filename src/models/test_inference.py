# src/models/test_inference.py

import pandas as pd
import yaml
import numpy as np
from pathlib import Path
from src.models.predictor import IntelliLoanPredictor
from config.logging import get_logger

logger = get_logger(__name__)

def test_inference_pipeline():
    # 1. Setup Root and Load Config
    root = Path(__file__).resolve().parents[2]
    with open(root / "config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    predictor = IntelliLoanPredictor(config)

    # 2. Load "Modeling Matrix" (Data before preprocessing)
    data_path = root / config["data_paths"]["processed"] / config["tables"]["modeling_matrix"]
    logger.info(f"Loading sample data from {data_path}")
    df_all = pd.read_parquet(data_path)

    # 3. Select samples (Good vs Bad borrowers)
    # We drop the ID and Target to simulate a real incoming API payload
    main_key = config["fusion"]["main_key"]
    
    client_good = df_all[df_all['TARGET'] == 0].head(1).drop(columns=['TARGET'])
    client_bad = df_all[df_all['TARGET'] == 1].head(1).drop(columns=['TARGET'])
    
    client_good_id = int(client_good[main_key].values[0])
    client_bad_id = int(client_bad[main_key].values[0])

    # 4. RUN INFERENCE (Testing Strategy A and Strategy B)
    logger.info("Launching multi-strategy inference tests...")
    
    # Testing Good Client with Strategy A
    res_good_a = predictor.get_credit_decision(client_good, strategy="A")
    
    # Testing Bad Client with Strategy B (The risk-based one)
    res_bad_b = predictor.get_credit_decision(client_bad, strategy="B")

    # 5. DISPLAY RESULTS
    print("\n" + "═"*60)
    print(f"TEST CASE: GOOD BORROWER (ID: {client_good_id})")
    print(f"   Policy: STRATEGY A (Fixed Thresholds)")
    print("═"*60)
    print(f"Result: {res_good_a['verdict']} (Score: {res_good_a['credit_score']}/1000)")
    print(f"Prob. of Default: {res_good_a['probability_of_default']:.2%}")
    print(f"Risk Tier: {res_good_a['risk_level']}")

    print("\n" + "═"*60)
    print(f"TEST CASE: BAD BORROWER (ID: {client_bad_id})")
    print(f"   Policy: STRATEGY B (Risk-Based Segments)")
    print("═"*60)
    print(f"Result: {res_bad_b['verdict']} (Score: {res_bad_b['credit_score']}/1000)")
    print(f"Prob. of Default: {res_bad_b['probability_of_default']:.2%}")
    print(f"Risk Segment: {res_bad_b['risk_segment']} (Tier: {res_bad_b['risk_level']})")
    print("═"*60 + "\n")

if __name__ == "__main__":
    test_inference_pipeline()