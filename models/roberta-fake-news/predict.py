from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

model = RobertaForSequenceClassification.from_pretrained("./outputs/roberta-fine-tuned/model")
tokenizer = RobertaTokenizer.from_pretrained("./outputs/roberta-fine-tuned/tokenizer")

def predict(text):
    inputs = tokenizer(text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1)
    return "Fake News" if prediction.item() == 1 else "True News"

text = "Breaking news: Scientists discovered life on Mars!"
result = predict(text)
print(f"The news is classified as: {result}")
