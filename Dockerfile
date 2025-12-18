FROM python:3.10-slim

WORKDIR /app

RUN pip install \
    fastapi \
    uvicorn \
    transformers \
    torch \
    huggingface_hub \
    python-dotenv

COPY app.py .
COPY .env .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]