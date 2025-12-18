"""
FastAPI inference service.
Usa il modello caricato su HF e calcola le metriche scrivendole in un CSV che importo in grafana.
"""

import csv
import os
import time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

HF_MODEL = os.environ.get("HF_REPO_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")
CSV_PATH = "/data/metrics.csv"

app = FastAPI(title="Sentiment API")

classifier = pipeline(
    "sentiment-analysis",
    model=HF_MODEL,
    token=HF_TOKEN
)

file_exists = os.path.exists(CSV_PATH)

# Metric counters (in-memory, coerenti col servizio)
requests_total = 0
positive_count = 0
confidence_sum = 0.0

if not file_exists:
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "requests_total",
            "positive_ratio",
            "avg_confidence",
            "latency_ms"
        ])


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    global requests_total, positive_count, confidence_sum

    start = time.time()

    result = classifier(payload.text)[0]
    label = result["label"]
    score = float(result["score"])

    requests_total += 1
    confidence_sum += score
    if label == "positive":
        positive_count += 1

    latency_ms = (time.time() - start) * 1000
    positive_ratio = positive_count / requests_total
    avg_confidence = confidence_sum / requests_total

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            requests_total,
            round(positive_ratio, 4),
            round(avg_confidence, 4),
            round(latency_ms, 2)
        ])

    return PredictResponse(label=label, score=score)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
