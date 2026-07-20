# src/rag/test_agent.py
# We will test our agent wit the B strategy we implement for credit decision

import pandas as pd
import yaml
from pathlib import Path
from src.rag.agents import IntelliLoanAgent
from config.logging import get_logger

logger = get_logger(__name__)

def run_agent_demo():
    root = Path(__file__).resolve().parents[2]
    with open(root / "config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    logger.info("Initializing IntelliLoan Agentic System...")
    agent_system = IntelliLoanAgent(config)

    # 2. Extract specific client (100002 is a known high-risk case for demo)
    data_path = root / config["data_paths"]["processed"] / config["tables"]["modeling_matrix"]
    df_samples = pd.read_parquet(data_path)
    
    # Test with a client that is likely to be declined to see the RAG in action
    client_id = 100002
    client_to_test = df_samples[df_samples[config["fusion"]["main_key"]] == client_id]
    
    if client_to_test.empty:
        client_to_test = df_samples.head(1)
        client_id = int(client_to_test[config["fusion"]["main_key"]].values[0])

    # 3. RUN INFERENCE WITH STRATEGY B (The risk-based one)
    logger.info(f"Launching Agent with STRATEGY B for Client: {client_id}")
    final_output = agent_system.run_inference(client_id, client_to_test, strategy="B")

    # 4. Results
    print("\n" + "═"*60)
    print(f"ML RESULTS (Strategy {final_output['ml_results']['strategy_used']})")
    print(f"Score: {final_output['ml_results']['credit_score']}/1000 | Verdict: {final_output['ml_results']['verdict']}")
    print(f"Risk Segment: {final_output['ml_results']['risk_segment']}")
    print("═"*60)
    print(f"FINAL AGENTIC MEMO")
    print("-" * 60)
    print(final_output["final_memo"])
    print("═"*60 + "\n")

if __name__ == "__main__":
    run_agent_demo()