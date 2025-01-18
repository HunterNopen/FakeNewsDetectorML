import pandas as pd
from textblob import TextBlob

train_data = pd.read_csv("data/processed/train.csv")

train_data['word_count'] = train_data['text'].apply(lambda x: len(x.split()))

train_data['sentiment'] = train_data['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

print("Saving feature-engineered data...")
train_data.to_csv("data/processed/feature_engineered.csv", index=False)

print("Feature engineering complete!")
