# src/data/ingest.py

import os
import yaml
import json
import hashlib
import time
import gc
from pathlib import Path
from typing import Dict
import pandas as pd
import pandera as pa
import pyarrow.parquet as pq
import pyarrow as pa_arrow

# Import the custom logger
from config.logging import get_logger

logger = get_logger(__name__)

class Ingestion:
    """
    Orchestrates industrial ingestion using PyArrow Row Groups.
    Validates chunks with Pandera, optimizes memory, and persists versioned snapshots.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        # 1. Setup paths relative to project root
        self.project_root = Path(__file__).resolve().parents[2]
        self.config_path = self.project_root / config_path
        
        # 2. Load Configuration
        self.config = self._load_config()
        
        # 3. Extract metadata
        self.main_key = self.config["fusion"]["main_key"]
        self.target_col = "TARGET"
        self.version = self.config["project"]["version"]
        
        # 4. Load Pandera Schema from JSON (handling the 'raw_shema' typo in config)
        self.schema_json_path = self.project_root / self.config["schema"]["raw_schema"]
        self.schema = self._load_validation_schema()

    #Load config.yaml
    def _load_config(self) -> Dict:
        """Loads operational configurations from YAML file."""
        if not self.config_path.exists():
            logger.critical(f"Config file missing at {self.config_path}")
            raise FileNotFoundError(f"Missing config: {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)
        
    #Load raw_schema
    def _load_validation_schema(self) -> pa.DataFrameSchema:
        """Loads the Pandera validation contract from JSON file."""
        try:
            with open(self.schema_json_path, "r") as f:
                schema_dict = json.load(f)
            schema = pa.DataFrameSchema.from_json(json.dumps(schema_dict))
            logger.info(f"Compliance schema loaded from {self.schema_json_path}")
            return schema
        except Exception as e:
            logger.error(f"Failed to load JSON schema: {e}. Fallback to basic validation.")
            return pa.DataFrameSchema(
                columns={self.main_key: pa.Column(pa.Int, unique=True, nullable=False)},
                strict=False
            )
        
    # Data integrity check
    def _verify_md5(self, filepath: Path) -> str:
        """Ensures data lineage traceability via MD5 hash."""
        file_hash = hashlib.md5(filepath.read_bytes()).hexdigest()
        logger.info(f"Lineage Verification - Input MD5: {file_hash}")
        return file_hash

    #Downcasting
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downcasts numeric types to reduce RAM footprint."""
        start_mem = df.memory_usage().sum() / 1024**2
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')
        
        for col in df.select_dtypes(include=['int64']).columns:
            if col not in [self.main_key, self.target_col]:
                df[col] = df[col].astype('int32')
                
        end_mem = df.memory_usage().sum() / 1024**2
        logger.debug(f"Chunk RAM Optimization: {start_mem:.1f}MB -> {end_mem:.1f}MB")
        return df

    # Run pipeline function
    def run_pipeline(self, filename: str = "df_master_uncleaned.parquet") -> pd.DataFrame:
        """
        Executes the Ingestion Pipeline using Row Group Chunks.
        Returns a unified optimized DataFrame for further feature engineering.
        """
        start_time = time.time()
        source_file = self.project_root / self.config["data_paths"]["interim"] / filename
        
        if not source_file.exists():
            logger.error(f"Source file not found at {source_file}")
            raise FileNotFoundError(f"Source missing: {source_file}")

        # Integrity Check
        self._verify_md5(source_file)

        # Setup Persistence
        processed_dir = self.project_root / self.config["data_paths"]["processed"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        export_path = processed_dir / f"master_ingested_v{self.version}.parquet"

        # PyArrow Chunked Reading
        parquet_file = pq.ParquetFile(source_file)
        num_row_groups = parquet_file.num_row_groups
        logger.info(f"Opening {filename}: splitting into {num_row_groups} row groups.")

        parquet_writer = None
        chunk_list = []

        for i in range(num_row_groups):
            logger.info(f"Processing row group {i+1}/{num_row_groups}...")
            # Convert row group to pandas
            df_chunk = parquet_file.read_row_group(i).to_pandas()

            # 1. Validation Gate
            try:
                self.schema.validate(df_chunk, lazy=True)
            except (pa.errors.SchemaErrors, pa.errors.SchemaError) as err:
                logger.critical(f"Compliance Violation in Chunk {i+1}: {err}")
                raise err

            # 2. Optimization
            df_chunk = self._optimize_memory(df_chunk)
            
            # 3. Accumulate for return
            chunk_list.append(df_chunk)

            # 4. Stream-write to final processed file
            table = pa_arrow.Table.from_pandas(df_chunk, preserve_index=False)
            if parquet_writer is None:
                parquet_writer = pq.ParquetWriter(export_path, table.schema, compression='snappy')
            parquet_writer.write_table(table)

        if parquet_writer:
            parquet_writer.close()

        # Unify all optimized chunks
        logger.info("Unifying all chunks in memory...")
        df_final = pd.concat(chunk_list, ignore_index=True)
        
        # Final cleanup of chunk list to free RAM
        del chunk_list
        gc.collect()

        duration = time.time() - start_time
        logger.info(f"Ingestion finalized in {duration:.2f}s. Final Shape: {df_final.shape}")
        
        return df_final

if __name__ == "__main__":
    try:
        ingestor = Ingestion()
        df_ingested = ingestor.run_pipeline()
        print(f"Ingestion successful. Shape: {df_ingested.shape}")
    except Exception as e:
        logger.exception("An error occurred during ingestion:") 
        print(f"Ingestion failed: {e}")