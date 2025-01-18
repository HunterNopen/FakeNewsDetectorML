import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../processed/train.csv")

plt.figure(figsize=(10, 5))
sns.boxplot(x='label', y='word_count', data=data)
plt.title("Word Count by Label")
plt.xlabel("Label (1 = Fake, 0 = True)")
plt.ylabel("Word Count")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x='label', y='sentiment', data=data)
plt.title("Sentiment by Label")
plt.xlabel("Label (1 = Fake, 0 = True)")
plt.ylabel("Sentiment Polarity")
plt.show()

fake_keywords = " ".join(data[data['label'] == 1]['text'])
true_keywords = " ".join(data[data['label'] == 0]['text'])

print("Common Fake News Keywords:", fake_keywords[:500])
print("Common True News Keywords:", true_keywords[:500])
