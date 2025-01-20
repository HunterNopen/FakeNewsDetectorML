import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
fake_news = pd.read_csv("../raw/Fake.csv")
true_news = pd.read_csv("../raw/True.csv")

fake_news['label'] = 1
true_news['label'] = 0

data = pd.concat([fake_news, true_news], ignore_index=True)

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    stop_words = set(stopwords.words("english"))
    text = [word for word in text if word not in stop_words]
    return " ".join(text)

print("Cleaning text...")
data['text'] = data['text'].apply(clean_text)

train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print("Saving processed datasets...")
train_data.to_csv("./train.csv", index=False)
val_data.to_csv("./validation.csv", index=False)
test_data.to_csv("./test.csv", index=False)

print("Preprocessing complete!")
