from transformers import AlbertTokenizer, AlbertModel, AlbertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd

MODEL_NAME = "XSY/albert-base-v2-fakenews-discriminator"

tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)
model = AlbertForSequenceClassification.from_pretrained(MODEL_NAME)

print("Getting data...")
train_data = pd.read_csv("../../data/processed/train.csv")
val_data = pd.read_csv("../../data/processed/validation.csv")
test_data = pd.read_csv("../../data/processed/test.csv")

train_data = train_data.dropna(subset=["text"])
train_data = train_data[train_data["text"].str.strip() != ""]

val_data = val_data.dropna(subset=["text"])
val_data = val_data[val_data["text"].str.strip() != ""]

test_data = test_data.dropna(subset=["text"])
test_data = test_data[test_data["text"].str.strip() != ""]

def convert_to_dataset(df):
    return Dataset.from_pandas(df[['text', 'label']])

train_dataset = convert_to_dataset(train_data)
val_dataset = convert_to_dataset(val_data)
test_dataset = convert_to_dataset(test_data)

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

print("Tokenizing...")

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.with_format("torch")
val_dataset = val_dataset.with_format("torch")
test_dataset = test_dataset.with_format("torch")

training_args = TrainingArguments(
    output_dir="models/outputs",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="models/outputs/logs",
    save_strategy="epoch",
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

print("Starting training...")

trainer.train()

model.save_pretrained("../outputs/roberta-fine-tuned")

tokenizer.save_pretrained("../outputs/roberta-fine-tuned-tokenizer")

