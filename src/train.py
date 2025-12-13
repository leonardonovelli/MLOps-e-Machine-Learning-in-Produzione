# train.py

"""
Training pipeline per CI/CD.
Esegue un training rapido su dataset HuggingFace e valuta metriche.
Eseguo solo su un range di 200 osservazioni per non rendere lungo il processo
"""

import json
import os
import random
from datetime import datetime

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = "model_data"
BEST_METRICS_FILE = "best_metrics.json"

# parametri dataset
TRAIN_SIZE = 200      # numero di esempi di train
EVAL_SIZE = 50        # numero di esempi di eval
SEED = 42             # seed per riproducibilità

# ------------------------------
# Dataset rapido
# ------------------------------
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
dataset = dataset.rename_column("label", "labels")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# selezione random dei sample
def sample_dataset(dataset_split, n, seed=42):
    indices = list(range(len(dataset_split)))
    random.Random(seed).shuffle(indices)
    selected_indices = indices[:n]
    return dataset_split.select(selected_indices)

# tokenizzazione
def preprocess(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

# train ed eval dataset random
train_dataset = sample_dataset(dataset["train"], TRAIN_SIZE, SEED).map(preprocess, batched=True)
eval_dataset = sample_dataset(dataset["validation"], EVAL_SIZE, SEED).map(preprocess, batched=True)

# ------------------------------
# Modello
# ------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ------------------------------
# Metriche
# ------------------------------
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    """Calcola accuracy sul dataset di validazione."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"]}

# ------------------------------
# TrainingArguments rapidi
# ------------------------------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    save_strategy="no",
    logging_strategy="steps",
    logging_steps=10,
    push_to_hub=False
)

# ------------------------------
# Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ------------------------------
# Train
# ------------------------------
trainer.train()

# ------------------------------
# Valutazione e salvataggio metriche
# ------------------------------

# Salvo il modello per eventuale push su HF
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics = trainer.evaluate(eval_dataset)
metrics["timestamp"] = datetime.utcnow().isoformat()  # UTC in formato ISO
print("Validation metrics:", metrics)

with open(OUTPUT_DIR+"/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# --- controllo best accuracy ---
val_accuracy = metrics["eval_accuracy"]

if os.path.exists(OUTPUT_DIR+"/"+BEST_METRICS_FILE):
    with open(OUTPUT_DIR+"/"+BEST_METRICS_FILE, encoding="utf-8") as f:
        best_metrics = json.load(f)
    best_accuracy = best_metrics.get("accuracy", 0)
else:
    best_accuracy = 0

if val_accuracy > best_accuracy:
    print(f"Accuracy migliorata: {val_accuracy:.4f} > {best_accuracy:.4f}.")
    best_metrics = {
        "accuracy": val_accuracy,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(OUTPUT_DIR+"/"+BEST_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
else:
    print(f"Accuracy {val_accuracy:.4f} non supera il best {best_accuracy:.4f}.")
