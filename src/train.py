# train.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate
import json
import os
from datetime import datetime

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = "model_data"
BEST_METRICS_FILE = "best_metrics.json"

# ------------------------------
# Dataset rapido
# ------------------------------
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")
dataset = dataset.rename_column("label", "labels")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

def preprocess(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

train_dataset = dataset["train"].select(range(200)).map(preprocess, batched=True)
eval_dataset = dataset["validation"].select(range(50)).map(preprocess, batched=True)

# ------------------------------
# Modello
# ------------------------------
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# ------------------------------
# Metriche
# ------------------------------
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
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
metrics = trainer.evaluate(eval_dataset)
metrics["timestamp"] = datetime.utcnow().isoformat()  # UTC in formato ISO
print("Validation metrics:", metrics)

with open(OUTPUT_DIR+"/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# --- controllo best accuracy ---
val_accuracy = metrics["eval_accuracy"]

if os.path.exists(OUTPUT_DIR+"/"+BEST_METRICS_FILE):
    with open(OUTPUT_DIR+"/"+BEST_METRICS_FILE) as f:
        best_metrics = json.load(f)
    best_acc = best_metrics.get("accuracy", 0)
else:
    best_acc = 0

if val_accuracy > best_acc:
    print(f"Accuracy migliorata: {val_accuracy:.4f} > {best_acc:.4f}.")
    best_metrics = {
        "accuracy": val_accuracy,
        "timestamp": datetime.utcnow().isoformat()
    }
    with open(OUTPUT_DIR+"/"+BEST_METRICS_FILE, "w") as f:
        json.dump(best_metrics, f, indent=2)
else:
    print(f"Accuracy {val_accuracy:.4f} non supera il best {best_acc:.4f}.")