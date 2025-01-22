import torch
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AlbertTokenizer, AlbertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification
)

bert_model = BertForSequenceClassification.from_pretrained("./models/fake-news-bert-detect/outputs/bert-fine-tuned/model")
bert_tokenizer = BertTokenizer.from_pretrained("./models/fake-news-bert-detect/outputs/bert-fine-tuned/tokenizer")

albert_model = AlbertForSequenceClassification.from_pretrained("./models/albert-fake-news/outputs/albert-fine-tuned/model")
albert_tokenizer = AlbertTokenizer.from_pretrained("./models/albert-fake-news/outputs/albert-fine-tuned/tokenizer")

roberta_model = RobertaForSequenceClassification.from_pretrained("./models/roberta-fake-news/outputs/roberta-fine-tuned/model")
roberta_tokenizer = RobertaTokenizer.from_pretrained("./models/roberta-fake-news/outputs/roberta-fine-tuned/tokenizer")

def predict_with_model(model, tokenizer, text):
    inputs = tokenizer(text, truncation=True, padding="max_length",
                       max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)[0].tolist()
    return probabilities

def get_all_predictions(text):
    bert_probs = predict_with_model(bert_model, bert_tokenizer, text)
    albert_probs = predict_with_model(albert_model, albert_tokenizer, text)
    roberta_probs = predict_with_model(roberta_model, roberta_tokenizer, text)
    return {
        "bert": bert_probs,
        "albert": albert_probs,
        "roberta": roberta_probs
    }

# Example usage:
text = "Breaking news: Scientists discovered life on Mars!"
all_model_outputs = get_all_predictions(text)
print(all_model_outputs)