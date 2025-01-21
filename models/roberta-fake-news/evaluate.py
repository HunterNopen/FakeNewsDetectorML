from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import classification_report
import pandas as pd
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def convert_to_dataset(df):
    return Dataset.from_pandas(df[['text', 'label']])

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

print("Getting model...")
model = RobertaForSequenceClassification.from_pretrained("./outputs/roberta-fine-tuned/model")
model.to(device)
tokenizer = RobertaTokenizer.from_pretrained("./outputs/roberta-fine-tuned/tokenizer")

test_data = pd.read_csv("../../data/processed/test.csv")
test_data = test_data.dropna(subset=["text"])
test_data = test_data[test_data["text"].str.strip() != ""]

test_dataset = convert_to_dataset(test_data)
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataloader = DataLoader(test_dataset, batch_size=16)

all_preds = []
all_labels = []

print('Evaluating...')
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        inputs = {key: value.to(device) for key, value in batch.items() if key in ["input_ids", "attention_mask"]}
        labels = batch["label"].to(device)
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(all_labels)
print(all_preds)
print(classification_report(all_labels, all_preds))
