# tests.py

"""
Test di base sul modello di sentiment.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

def test_model_load():
    """Controlla che modello e tokenizer si possano caricare"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    assert tokenizer is not None
    assert model is not None

def test_model_output_shape():
    """Controlla che l’output del modello corrisponda al numero di label"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sample_text = "I love this product!"
    inputs = tokenizer(sample_text, return_tensors="pt")
    outputs = model(**inputs)
    assert outputs.logits.shape[-1] == model.config.num_labels
