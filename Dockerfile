# 1. Base image
FROM python:3.11-slim

# 2. System dependencies (crucial for LightGBM and UMAP)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Workdir
WORKDIR /app

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the project structure
# We only copy what's strictly necessary for inference
COPY src/ ./src/
COPY config/ ./config/
RUN mkdir -p models
COPY models/champion_model.joblib ./models/
COPY models/fitted_preprocessor.joblib ./models/
COPY models/unsupervised_artifacts.joblib ./models/
COPY models/mitigated_model_equalized_odds_gender_group.joblib ./models/

# 6. copy of the demo database and RAG memory
RUN mkdir -p data/processed data/vectorstore
COPY data/processed/feature_metadata.json ./data/processed/
COPY data/processed/final_modeling_matrix.parquet ./data/processed/
COPY data/processed/final_enriched_modeling_matrix.parquet ./data/processed/
COPY data/vectorstore/ ./data/vectorstore/

# 7. Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 8. Expose port needed  by GCP
EXPOSE 8080

# 9. Start the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]