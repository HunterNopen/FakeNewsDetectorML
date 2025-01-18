import pandas as pd
import matplotlib.pyplot as plt

fake_news = pd.read_csv("../raw/Fake.csv")
true_news = pd.read_csv("../raw/True.csv")

print("Fake News Shape:", fake_news.shape)
print("True News Shape:", true_news.shape)
print("Fake News Sample:\n", fake_news.head())
print("True News Sample:\n", true_news.head())

fake_news['text_length'] = fake_news['text'].apply(len)
true_news['text_length'] = true_news['text'].apply(len)

plt.figure(figsize=(10, 5))
plt.hist(fake_news['text_length'], bins=50, alpha=0.5, label='Fake News', color='red')
plt.hist(true_news['text_length'], bins=50, alpha=0.5, label='True News', color='blue')
plt.legend()
plt.title("Distribution of Text Lengths")
plt.xlabel("Text Length")
plt.ylabel("Frequency")
plt.show()

print("Fake News Missing Values:\n", fake_news.isnull().sum())
print("True News Missing Values:\n", true_news.isnull().sum())
