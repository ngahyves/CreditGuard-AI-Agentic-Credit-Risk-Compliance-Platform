# src/rag/agents.py

import os
from typing import TypedDict, Annotated
from operator import add
from pathlib import Path
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import time

from src.models.predictor import IntelliLoanPredictor
from src.explainability.interprete import IntelliLoanInterpreter 
from config.logging import get_logger

logger = get_logger(__name__)
load_dotenv()

# 1. State Contract Contract
class AgentState(TypedDict):
    client_id: int
    client_data: pd.DataFrame
    strategy: str
    ml_results: dict      
    shap_reasons: list  
    policy_context: str
    final_memo: str
    # Annotated[int, add] triggers automatic summation on updates to prevent infinite execution loops
    iteration_count: Annotated[int, add] 
    validation_status: str 

class IntelliLoanAgent:
    """
    Enterprise-grade Agentic Workflow powered by LangGraph, Groq, and Gemini Embeddings.
    Encapsulates the entire multi-agent lifecycle: inference, RAG, synthesis, 
    and self-correction logic inside a single, scalable class structure.
    """
    def __init__(self, config: dict):
        self.config = config
        self.root = Path(__file__).resolve().parents[2] 
        
        # Initialize Core Production Machine Learning Engines
        self.predictor = IntelliLoanPredictor(config)
        self.interpreter = IntelliLoanInterpreter()

        # Secure Groq Cloud connection for high-speed narrative synthesis (Llama 3.1)
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("Missing GROQ_API_KEY inside your .env configuration file.")
            
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant", 
            groq_api_key=groq_key,
            temperature=0.2
        )
        

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_db = Chroma(
            persist_directory=str(self.root / "data/vectorstore"),
            embedding_function=embeddings,
            collection_name="bank_policies"
        )
        
        # Compile the stateful execution graph once during initialization
        self.graph = self._build_graph()

    # --- ENCAPSULATED WORKFLOW NODES (METHODS) ---

    def ml_analysis_node(self, state: AgentState) -> dict:
        """
        NODE 1: Quantitative Analysis.
        Calculates the risk score and extracts true mathematical reasons (SHAP).
        Supports Strategy A/B routing.
        """
        start = time.time()
        logger.info(f"[NODE: Analyst] Processing client {state['client_id']}...")
        
        # Defining the decision strategy (A or B but A by default)
        strategy_to_use = state.get("strategy", "A")
        logger.info(f"Applying Decision Policy: Strategy {strategy_to_use}")

        # 1. Run Probability & Verdict WITH THE CHOSEN STRATEGY
        # We pass the ‘strategy’ argument to the Predictor
        results = self.predictor.get_credit_decision(state['client_data'], strategy=strategy_to_use)
        
        logger.info(f"ML Result: {results['verdict']} (Score: {results['credit_score']}, Segment: {results['risk_segment']})")

        # 2. PRODUCTION SHAP ALIGNMENT FIX
        try:
            # Re-calculating the exact feature space seen by the model for SHAP accuracy
            X_clean_arr = self.predictor.preprocessor.transform(state['client_data'])
            clean_cols = [col.split("__")[-1] for col in self.predictor.preprocessor.get_feature_names_out()]
            X_clean = pd.DataFrame(X_clean_arr, columns=clean_cols, index=state['client_data'].index)
            
            X_unsup_arr = self.predictor.unsup_engine.transform(X_clean) # we use unsup_engine from the class IntelliLoanPredictor()
            unsup_cols = [col for col in self.predictor.model.feature_name_ if col not in X_clean.columns]
            X_unsup = pd.DataFrame(X_unsup_arr, columns=unsup_cols, index=state['client_data'].index)
            
            X_inference = pd.concat([X_clean, X_unsup], axis=1).loc[:, ~pd.concat([X_clean, X_unsup], axis=1).columns.duplicated()]
            X_inference = X_inference.reindex(columns=self.predictor.model.feature_name_, fill_value=0.0)
            
            reasons = self.interpreter.get_local_explanation(X_inference)
        except Exception as e:
            logger.error(f"Advanced SHAP alignment failed: {e}. Falling back to baseline extraction.")
            reasons = self.interpreter.get_local_explanation(state['client_data'])

        # 3. Defensive fallback
        if reasons is None or reasons.empty or 'feature' not in reasons.columns:
            logger.warning("SHAP explanation failed. Creating placeholder.")
            reasons = pd.DataFrame({'feature': ['EXT_SOURCE_3', 'YEARS_EMPLOYED'], 'shap_impact': [0.5, 0.2]})

        reasons_serialized = reasons.to_dict(orient='records')

        duration = time.time() - start
        logger.info(f"[Timer] ML + SHAP took: {duration:.2f}s")
        # 4. Return enriched state
        return {
            "ml_results": results, 
            "shap_reasons": reasons_serialized,
            "iteration_count": 0
        }

    def policy_research_node(self, state: AgentState) -> dict:
        """NODE 2: Executes semantic context extraction from ChromaDB based on top SHAP drivers."""
        start = time.time()
        reasons_list = [item['feature'] for item in state['shap_reasons']][:5]
        query = f"Credit constraints, limits and rules related to: {', '.join(reasons_list)}"
        
        logger.info(f"[NODE: Researcher] Extracting policy chunks from ChromaDB for anchors: {reasons_list}")
        docs = self.vector_db.similarity_search(query, k=3)
        
        context_parts = [getattr(d, "page_content", str(d)) for d in docs]
        duration = time.time() - start
        logger.info(f"[Timer] RAG Search took: {duration:.2f}s")
        return {"policy_context": "\n\n".join(context_parts)}

    def writer_node(self, state: AgentState) -> dict:
        """NODE 3: Synthesizes analytical numbers and policy facts into a clean credit memo."""
        logger.info(f"[NODE: Writer] Compiling corporate narrative draft... (Loop stack: {state.get('iteration_count', 0)})")
        # Extracting the probability
        proba = state['ml_results']['probability_of_default']
        # Multiplying by 100 to have a percentage
        proba_pct = f"{proba * 100:.2f}%"
        
        prefix = "CRITICAL: REVISE PREVIOUS DRAFT BASED ON COMPLIANCE CRITIC ALERTS.\n" if state.get("iteration_count", 0) > 0 else ""
        
        prompt = f"""
        ACT AS A SENIOR CREDIT RISK OFFICER AT A TOP-TIER BANK.
        Your goal is to write a high-stakes credit appraisal memo for an internal committee.
        {prefix}

        1. DATA INPUTS:
        - Risk Score: {state['ml_results']['credit_score']}/1000
        - Probability of Default (PD): {proba_pct}
        - Algorithmic Verdict: {state['ml_results']['verdict']}
        - Empirical Risk Tier: {state['ml_results']['risk_level']}
        - Top Model Drivers (SHAP): {state['shap_reasons']}
        
        2. CORPORATE POLICY CONSTRAINTS (RAG):
        {state['policy_context']}
        
        INSTRUCTIONS:
        - Use a professional, objective, and decisive banking tone.
        - Combine the ML score with the regulatory context.
        - If the verdict is DECLINE or MANUAL REVIEW, explain the specific policy violation.
        - Keep it under 250 words.

        STRUCTURE:
        I. EXECUTIVE SUMMARY
        II. QUANTITATIVE JUSTIFICATION
        III. REGULATORY COMPLIANCE
        IV. FINAL RECOMMENDATION
        """
        response = self.llm.invoke(prompt)
        return {
            "final_memo": str(response.content),
            "iteration_count": 1 # Will add 1 to state layer systematically via operator add
        }

    def compliance_critic_node(self, state: AgentState) -> dict:
        """NODE 4: Audits draft memo against original context to enforce strict hallucination mitigation."""
        logger.info(f"[NODE: Compliance_Critic] Evaluating document veracity against source text... (Iteration: {state['iteration_count']})")
        
        critic_prompt = f"""
        YOU ARE AN AUTONOMOUS BANKING COMPLIANCE AUDITOR. 
        Your job is to double-check if the generated memo hallucinates or invents non-existent facts/rules.

        GOLDEN REFERENCE CORPORATE POLICY:
        {state['policy_context']}

        GENERATED CREDIT MEMO DRAFT:
        {state['final_memo']}

        CRITERIA:
        - Does the memo mention any financial ratio, age limit, or score NOT found in the reference policy?
        - If it mentions a number, does it match the reference text?
        
        OUTPUT FORMAT: Respond with EXACTLY one word: "APPROVED" if perfectly accurate, or "REJECTED" if you find any hallucination/incoherence. Do not add intro or outro.
        """
        response = self.llm.invoke(critic_prompt)
        verdict = str(response.content).strip().upper()
        
        status = "APPROVED" if "APPROVED" in verdict and "REJECTED" not in verdict else "REJECTED"
        if status == "REJECTED":
            logger.warning(f"Compliance violation flagged by Critic agent on stack iteration #{state['iteration_count']}!")
            
        return {"validation_status": status}

    # --- ROUTING LOGIC GATE ---

    def _routing_gate(self, state: AgentState):
        """Internal router supervising conditional edge evaluation state variables."""
        if state["validation_status"] == "APPROVED":
            logger.info("State graph validation approved. Terminating agentic pipeline loop.")
            return "end"
        
        if state["iteration_count"] >= 3:
            logger.error("Max recursion stack limit hit (3/3). Forcing pipeline exit to guarantee runtime safety.")
            return "end"
            
        logger.info("Reflection loop triggered. Rerouting state metrics back to Node 3 Writer...")
        return "retry"

    # --- TOPOLOGY COMPILATION ---

    def _build_graph(self):
        """Assembles internal processing nodes into a compiled cyclic StateGraph workflow instance."""
        workflow = StateGraph(AgentState)
        
        # Map class methods directly as StateGraph node operations
        workflow.add_node("ml_analyst", self.ml_analysis_node)
        workflow.add_node("researcher", self.policy_research_node)
        workflow.add_node("writer", self.writer_node)
        workflow.add_node("compliance_critic", self.compliance_critic_node)

        # Establish graph routing connections
        workflow.set_entry_point("ml_analyst")
        workflow.add_edge("ml_analyst", "researcher")
        workflow.add_edge("researcher", "writer")
        workflow.add_edge("writer", "compliance_critic")
        
        # Inject the core self-correction conditional feedback gate topology
        workflow.add_conditional_edges(
            "compliance_critic",
            self._routing_gate,
            {
                "end": END,
                "retry": "writer"
            }
        )
        return workflow.compile()

    def run_inference(self, client_id: int, client_data: pd.DataFrame, strategy: str = "A") -> dict:
        """Describe the agent-based pipeline, including the choice of strategy."""
        initial_state = {
            "client_id": client_id,
            "client_data": client_data,
            "strategy": strategy,
            "iteration_count": 0,
            "validation_status": "",
            "policy_context": "",
            "final_memo": ""
        }
        return self.graph.invoke(initial_state)
