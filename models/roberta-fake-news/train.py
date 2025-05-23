from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from datasets import Dataset
import pandas as pd
import torch
from torch.optim import AdamW

def tokenize_function(data):
    return tokenizer(data["text"], truncation=True, padding="max_length", max_length=512)

def convert_to_dataset(df):
    return Dataset.from_pandas(df[['text', 'label']])

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

print("Getting model...")

MODEL_NAME = "hamzab/roberta-fake-news-classification"

tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
model = RobertaForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

print("Getting data...")
train_data = pd.read_csv("../../data/processed/train.csv")
val_data = pd.read_csv("../../data/processed/validation.csv")
test_data = pd.read_csv("../../data/processed/test.csv")

print(f"Data shape: {train_data.shape}")

train_data = train_data.dropna(subset=["text"])
train_data = train_data[train_data["text"].str.strip() != ""]

val_data = val_data.dropna(subset=["text"])
val_data = val_data[val_data["text"].str.strip() != ""]

test_data = test_data.dropna(subset=["text"])
test_data = test_data[test_data["text"].str.strip() != ""]


train_dataset = convert_to_dataset(train_data)
val_dataset = convert_to_dataset(val_data)
test_dataset = convert_to_dataset(test_data)

print("Tokenizing...")

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset = train_dataset.with_format("torch")
val_dataset = val_dataset.with_format("torch")
test_dataset = test_dataset.with_format("torch")

training_args = TrainingArguments(
    output_dir="./outputs",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./outputs/logs",
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    gradient_accumulation_steps=4
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

print("Starting training...")

training_loss = trainer.train().training_loss
evaluation_loss = trainer.evaluate()['eval_loss']
preds = trainer.predict(test_dataset)
report = classification_report(test_dataset["label"], preds.predictions.argmax(-1))

with open("./outputs/roberta-fine-tuned/losses_and_classification.txt", 'w') as f:
    f.write(f"Train Loss: {training_loss}\n")
    f.write(f"Eval Loss: {evaluation_loss}\n\n")
    f.write("Classification Report:\n")
    f.write(report)

model.save_pretrained("./outputs/roberta-fine-tuned-3-epoch/model")

tokenizer.save_pretrained("./outputs/roberta-fine-tuned-3-epoch/tokenizer")
