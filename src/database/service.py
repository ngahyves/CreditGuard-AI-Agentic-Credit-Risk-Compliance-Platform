# src/database/service.py

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.models import Base
from config.logging import get_logger
from dotenv import load_dotenv

logger = get_logger(__name__)
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./fallback.db"

# Neon data base(Postgres Cloud)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Creates the tables on the Neon servers."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Remote Neon PostgreSQL initialized successfully.")
    except Exception as e:
        logger.error(f"Cloud DB Connection Failed: {e}")
        raise

def save_inference(data: dict):
    from src.database.models import InferenceLog
    db = SessionLocal()
    try:
        new_log = InferenceLog(**data)
        db.add(new_log)
        db.commit()
        logger.info(f"Decision for client {data['client_id']} archived in Neon Cloud.")
    except Exception as e:
        db.rollback()
        logger.error(f"Cloud DB Save Error: {e}")
    finally:
        db.close()