# src/rag/evaluate.py

import os
import pandas as pd
from pathlib import Path
from langchain_groq import ChatGroq 
from config.logging import get_logger
from dotenv import load_dotenv

logger = get_logger(__name__)
load_dotenv()

class IntelliLoanEvaluatorAutonome:
    """
    Custom deterministic evaluation engine utilizing an LLM-as-a-Judge architecture 
    to score pipeline Faithfulness, Context Precision, and Answer Relevancy via Groq.
    """
    def __init__(self):
        self.root = Path(__file__).resolve().parents[2] # Fixed path resolution to project root
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("Missing GROQ_API_KEY in your .env configuration file.")
        
        # Deploying llama-3.3-70b-versatile via Groq as the core auditing judge engine
        self.judge_llm = ChatGroq(
            model="llama-3.3-70b-versatile", 
            groq_api_key=groq_key,
            temperature=0.0 # Deterministic scoring
        )

    def evaluate_memo(self, context: str, answer: str):
        """Evaluates a generated memo artifact directly against its source context using standard prompts."""
        logger.info("Executing isolated compliance audit on generated credit memo via Groq...")
        
        prompt = f"""
        YOU ARE A PRINCIPAL DATA QUALITY AUDITOR. Evaluate the following Generated Memo against the Reference Policy Context.
        
        REFERENCE POLICY CONTEXT:
        {context}
        
        GENERATED CREDIT MEMO:
        {answer}
        
        SCORING MANDATE:
        Compute three distinct metrics from 0.00 (fail) to 1.00 (perfect score):
        1. FAITHFULNESS: Is every fact/number in the memo fully supported by the reference text? (1.00 = no hallucinations)
        2. CONTEXT_PRECISION: Did the text extract only the rules relevant to the credit file?
        3. ANSWER_RELEVANCY: Is the final memo helpful, objective, and directly answering the loan application?
        
        OUTPUT FORMAT: You must strictly respond with a single valid JSON block containing no other text:
        {{
            "faithfulness": 1.00,
            "context_precision": 1.00,
            "answer_relevancy": 1.00,
            "audit_justification": "Provide a brief structural synthesis here."
        }}
        """
        response = self.judge_llm.invoke(prompt)
        
        # Display Results
        print("\n" + "═"*60)
        print("STANDALONE METRICS EVALUATION PERFORMANCE REPORT (GROQ JUDGE)")
        print("═"*60)
        print(response.content)
        print("═"*60 + "\n")
        
        # Save evaluation result to reports folder
        reports_dir = self.root / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = reports_dir / "ragas_pipeline_benchmarks.csv"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str(response.content))
        logger.info(f"💾 Custom compliance benchmarks archived successfully at: {output_file}")

if __name__ == "__main__":
    # Test record mapping client 100002 actual outputs
    mock_context = "SECTION 3: FINANCIAL RATIOS. Down Payment: Minimum down payment of 10% is required for consumer loans. Lack of down payment must be compensated by high EXT_SOURCE scores (>0.65)."
    mock_answer = "The application is recommended for DECLINE. The risk score is 243/1000. The applicant triggers a financial ratio policy violation: the minimum down payment requirement of 10% is not met and EXT_SOURCE scores are below 0.65."
    
    evaluator = IntelliLoanEvaluatorAutonome()
    evaluator.evaluate_memo(mock_context, mock_answer)
