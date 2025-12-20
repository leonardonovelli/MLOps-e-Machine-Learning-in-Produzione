"""
FastAPI inference service.
Usa modello Hugging Face per sentiment analysis.
"""

import os
import time
from datetime import datetime

import pyodbc
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

HF_MODEL = os.environ.get("HF_REPO_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")

SQL_SERVER = os.environ.get("SQL_SERVER")
SQL_PORT = os.environ.get("SQL_PORT")
SQL_USER = os.environ.get("SQL_USER")
SQL_PASSWORD = os.environ.get("SQL_PASSWORD")
SQL_DB = os.environ.get("SQL_DB")

if not HF_MODEL or not HF_TOKEN:
    raise RuntimeError("HF_REPO_ID o HF_TOKEN mancanti")

app = FastAPI(title="Sentiment API")

# -------------------------------------------------
# Model preso da HF
# -------------------------------------------------
classifier = pipeline(
    "sentiment-analysis",
    model=HF_MODEL,
    token=HF_TOKEN
)

# -------------------------------------------------
# DB CONNECTION
# -------------------------------------------------
conn = pyodbc.connect(
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={SQL_SERVER},{SQL_PORT};"
    f"DATABASE={SQL_DB};"
    f"UID={SQL_USER};"
    f"PWD={SQL_PASSWORD};"
    "Encrypt=no;"
)
cursor = conn.cursor()

cursor.execute("""
IF NOT EXISTS (
    SELECT * FROM sysobjects WHERE name='metrics' AND xtype='U'
)
CREATE TABLE metrics (
    id INT IDENTITY(1,1) PRIMARY KEY,
    timestamp DATETIME2,
    text NVARCHAR(100),
    label NVARCHAR(32),
    confidence FLOAT,
    latency_ms FLOAT
)
""")
conn.commit()

# -------------------------------------------------
# REQUEST / RESPONSE API
# -------------------------------------------------
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float
    latency_ms: float


# -------------------------------------------------
# ENDPOINT
# -------------------------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    start = time.time()

    result = classifier(payload.text)[0]
    label = result["label"]
    score = float(result["score"])

    latency_ms = (time.time() - start) * 1000
    timestamp = datetime.utcnow()

    cursor.execute(
        """
        INSERT INTO metrics (timestamp, text, label, confidence, latency_ms)
        VALUES (?, ?, ?, ?, ?)
        """,
        timestamp,
        payload.text,
        label,
        score,
        latency_ms
    )
    conn.commit()

    return PredictResponse(
        label=label,
        score=score,
        latency_ms=round(latency_ms, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
