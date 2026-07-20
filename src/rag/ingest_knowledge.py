# src/rag/ingest_knowledge.py

import os
import yaml
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma
from config.logging import get_logger
from dotenv import load_dotenv

# Initialize logger
logger = get_logger(__name__)
# Explicitly load environment variables from the .env file at root
load_dotenv()

class PolicyIngestor:
    """
    Handles corporate regulatory text document generation, chunking strategies,
    and vector persistence using ChromaDB and Google Gemini Free Embeddings.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        # Resolve the project root directory
        self.root = Path(__file__).resolve().parents[2]
        with open(self.root / config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # 2. Define Vector Store Path
        self.db_path = self.root / "data/vectorstore"
        
        # 3. Secure hugging face Validation
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def create_simulated_policy(self) -> Path:
        """Creates a realistic banking policy document for RAG demonstration."""
        policy_content = """
        INTELLILOAN CORPORATE CREDIT POLICY - v0.3.0
        
        SECTION 1: DEMOGRAPHIC RISK FACTORS
        - Gender Bias Mitigation: Historically, gender-based disparities exist. Decisions must be justified 
          by financial stability (DTI) rather than gender identifiers.
        - Age Thresholds: Borrowers under 30 years old must demonstrate at least 2 years of steady employment 
          seniority to mitigate generational volatility.
        
        SECTION 2: OCCUPATIONAL STABILITY
        - Seniority: Job seniority below 1.5 years is considered a high-risk factor (Red Flag).
        - Education: Applicants with 'Higher Education' or 'Academic Degree' status benefit from 
          preferential risk tiers due to historically lower default rates in the portfolio.
        
        SECTION 3: FINANCIAL RATIOS (THRESHOLD REQUIREMENTS)
        - Debt-to-Income (DTI): Any applicant with a monthly annuity exceeding 40% of their total income 
          requires manual intervention by a Senior Risk Officer.
        - Down Payment: Minimum down payment of 10% is required for consumer loans. Lack of down payment 
          must be compensated by high EXT_SOURCE scores (>0.65).
        
        SECTION 4: EXTERNAL SIGNALS
        - EXT_SOURCE scores are the primary anchors for risk assessment. A combined average below 0.35 
          triggers an automatic 'DECLINE' recommendation regardless of other factors.
        """
        raw_dir = self.root / "data/raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        policy_file = raw_dir / "credit_policy_manual.txt"
        
        with open(policy_file, "w", encoding="utf-8") as f:
            f.write(policy_content.strip())
        
        return policy_file

    def run_ingestion(self):
        """Loads, splits, and stores the document into ChromaDB."""
        try:
            logger.info("Starting Knowledge Ingestion into ChromaDB...")
            
            # 1. Get or Create Document Path
            policy_file = self.create_simulated_policy()
            
            # 2. Native Modern Load: No TextLoader needed here anymore
            with open(policy_file, "r", encoding="utf-8") as f:
                raw_text = f.read()
            
            # 3. Split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400, 
                chunk_overlap=50,
                separators=["SECTION", "\n\n", "\n", " "]
            )
            chunks = text_splitter.split_text(raw_text)
            logger.info(f"Policy split into {len(chunks)} contextual text chunks.")

            # 4. Store in ChromaDB database using Gemini embeddings
            logger.info(f"Persisting vectors to {self.db_path}...")
            
            Chroma.from_texts(
                texts=chunks,
                embedding=self.embeddings,
                persist_directory=str(self.db_path),
                collection_name="bank_policies"
            )
            
            logger.info("Vector Store successfully created and persisted.")
            print("\n" + "="*50)
            print("KNOWLEDGE BASE INGESTED INTO CHROMADB VIA HUGGING FACE!")
            print("="*50 + "\n")
            
        except Exception as e:
            logger.error(f"Ingestion failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    ingestor = PolicyIngestor()
    ingestor.run_ingestion()
