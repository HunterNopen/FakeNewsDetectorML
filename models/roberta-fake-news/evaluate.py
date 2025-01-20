from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import classification_report
import pandas as pd
import torch

MODEL_PATH = "models/outputs/roberta-fine-tuned"
model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

test_data = pd.read_csv("data/processed/test.csv")

inputs = tokenizer(list(test_data['text']), truncation=True, padding="max_length", max_length=512, return_tensors="pt")
labels = torch.tensor(test_data['label'].values)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

print(classification_report(labels, predictions))
