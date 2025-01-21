from datasets import Dataset


def tokenize_function(example, tokenizer):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

def convert_to_dataset(df):
    return Dataset.from_pandas(df[['text', 'label']])