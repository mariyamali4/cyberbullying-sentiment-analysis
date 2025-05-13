# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
# from datasets import Dataset
# import torch
# import os
#
#
# def eda(df):
#     print(df.head())
#
#     # Rename 'CyberHate' to 'label' for consistency
#     if 'CyberHate' in df.columns:
#         df = df.rename(columns={'CyberHate': 'label'})
#     elif 'Label' in df.columns:
#         df = df.rename(columns={'Label': 'label'})
#
#     # Plot class distribution
#     plt.figure(figsize=(5, 4))
#     sns.countplot(data=df, x='label')
#     plt.title("Cyberhate vs Non-Cyberhate Distribution")
#     plt.xlabel("Label")
#     plt.ylabel("Number of Posts")
#     plt.show()
#
#     # Word count distribution
#     df['word_count'] = df['Title'].astype(str).apply(lambda x: len(x.split()))
#     plt.figure(figsize=(6, 4))
#     sns.histplot(data=df, x='word_count', hue='label', bins=30, kde=True)
#     plt.title("Post Word Count Distribution")
#     plt.xlabel("Word Count")
#     plt.show()
#
#     # Top keywords in cyberhate posts
#     if isinstance(df['label'].iloc[0], str):  # if not mapped yet
#         df['label'] = df['label'].map({'no bullying': 0, 'bullying': 1})
#
#     bullying_texts = df[df['label'] == 1]['Title'].dropna().astype(str)
#     vectorizer = CountVectorizer(stop_words='english', max_features=20)
#     X = vectorizer.fit_transform(bullying_texts)
#     top_words = pd.DataFrame(X.sum(axis=0).tolist()[0], index=vectorizer.get_feature_names_out(), columns=['Count'])
#     top_words.sort_values('Count').plot(kind='barh', figsize=(6, 5), title="Top Words in Cyberhate Posts")
#     plt.show()
#
#
# def prepare_data(csv_path):
#     df = pd.read_csv(csv_path)
#
#     # Normalize column names
#     if 'CyberHate' in df.columns:
#         df = df[['Title', 'CyberHate']].rename(columns={'Title': 'text', 'CyberHate': 'label'})
#     elif 'Label' in df.columns:
#         df = df[['Title', 'Label']].rename(columns={'Title': 'text', 'Label': 'label'})
#
#     df['label'] = df['label'].astype(int)
#
#     # Balance the dataset
#     bully_df = df[df['label'] == 1]
#     no_bully_df = df[df['label'] == 0]
#     oversampled_bully_df = bully_df.sample(len(no_bully_df), replace=True, random_state=42)
#     balanced_df = pd.concat([no_bully_df, oversampled_bully_df]).sample(frac=1, random_state=42)
#
#     return train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=42)
#
#
# def tokenize_data(train_df, test_df, tokenizer):
#     train_dataset = Dataset.from_pandas(train_df)
#     test_dataset = Dataset.from_pandas(test_df)
#
#     def tokenize_function(example):
#         return tokenizer(example["text"], truncation=True)
#
#     train_dataset = train_dataset.map(tokenize_function, batched=True)
#     test_dataset = test_dataset.map(tokenize_function, batched=True)
#
#     return train_dataset, test_dataset
#
#
# def train_model(train_dataset, test_dataset, tokenizer):
#     model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
#     data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
#
#     training_args = TrainingArguments(
#         output_dir="./roberta-bullying",
#         eval_strategy="epoch",
#         save_strategy="epoch",
#         learning_rate=2e-5,
#         per_device_train_batch_size=8,
#         per_device_eval_batch_size=8,
#         num_train_epochs=3,
#         weight_decay=0.01,
#         load_best_model_at_end=True,
#         logging_dir="./logs",
#         report_to="none"
#     )
#
#     def compute_metrics(eval_pred):
#         predictions, labels = eval_pred
#         preds = predictions.argmax(axis=-1)
#         report = classification_report(labels, preds, output_dict=True)
#         return {
#             "accuracy": report["accuracy"],
#             "precision_bullying": report["1"]["precision"],
#             "recall_bullying": report["1"]["recall"],
#             "f1_bullying": report["1"]["f1-score"]
#         }
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=test_dataset,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics
#     )
#
#     trainer.train()
#     model.save_pretrained("roberta-bullying")
#     tokenizer.save_pretrained("roberta-bullying")
#     return model
#
#
# def predict_text(text, model, tokenizer):
#     model.eval()
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         prediction = torch.argmax(outputs.logits, dim=1).item()
#     return "bullying" if prediction == 1 else "non-bullying"
#
#
# def main():
#     os.environ["WANDB_DISABLED"] = "true"
#
#     csv_path = "labeled_reddit.csv"
#     df = pd.read_csv(csv_path)
#
#     eda(df)
#     train_df, test_df = prepare_data(csv_path)
#     tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
#     train_dataset, test_dataset = tokenize_data(train_df, test_df, tokenizer)
#     model = train_model(train_dataset, test_dataset, tokenizer)
#
#     # Test prediction
#     test_text = "everyone hates you."
#     result = predict_text(test_text, model, tokenizer)
#     print("🔎 Comment:", test_text)
#     print("📢 Prediction:", result)
#
#
# if __name__ == "__main__":
#     main()


# -*- coding: utf-8 -*-
"""DE PROJECT.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19q56BfGTxrcsMIhq-tRUnkjf7eT8MVaQ

**EDA**
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (if not loaded yet)
df = pd.read_csv("/content/reddit_posts_labeled.csv")

# ✅ Preview
print(df.head())

# ✅ Class balance
plt.figure(figsize=(5, 4))
sns.countplot(data=df, x='Label')
plt.title("Bullying vs Non-Bullying Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Posts")
plt.show()

# ✅ Word count distribution
df['word_count'] = df['Title'].astype(str).apply(lambda x: len(x.split()))
plt.figure(figsize=(6, 4))
sns.histplot(data=df, x='word_count', hue='Label', bins=30, kde=True)
plt.title("Post Word Count Distribution")
plt.xlabel("Word Count")
plt.show()

# ✅ Top keywords (bullying only)
from sklearn.feature_extraction.text import CountVectorizer

bullying_texts = df[df['Label'] == 'bullying']['Title'].dropna().astype(str)
vectorizer = CountVectorizer(stop_words='english', max_features=20)
X = vectorizer.fit_transform(bullying_texts)
top_words = pd.DataFrame(X.sum(axis=0).tolist()[0], index=vectorizer.get_feature_names_out(), columns=['Count'])
top_words.sort_values('Count').plot(kind='barh', figsize=(6,5), title="Top Words in Bullying Posts")
plt.show()

!pip install transformers datasets scikit-learn

!pip install -U transformers

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
import torch
import os

# Load your dataset (uploaded CSV)
df = pd.read_csv("reddit_posts_labeled.csv")

# Preprocess: rename columns and convert labels
df = df[['Title', 'Label']].rename(columns={'Title': 'text', 'Label': 'label'})
df['label'] = df['label'].map({'no bullying': 0, 'bullying': 1})

# ✅ Balance the dataset by oversampling minority class
bully_df = df[df['label'] == 1]
no_bully_df = df[df['label'] == 0]

oversampled_bully_df = bully_df.sample(len(no_bully_df), replace=True, random_state=42)
balanced_df = pd.concat([no_bully_df, oversampled_bully_df]).sample(frac=1, random_state=42)  # shuffle

# Split into train/test
train_df, test_df = train_test_split(balanced_df, test_size=0.2, stratify=balanced_df['label'], random_state=42)

# Tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True)

# Convert to HuggingFace Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Data loader helper
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training configuration
training_args = TrainingArguments(
    output_dir="./roberta-bullying",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_dir="./logs",
    report_to="none"  # ✅ disables wandb
)

# Metric reporting
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)
    report = classification_report(labels, preds, output_dict=True)
    return {
        "accuracy": report["accuracy"],
        "precision_bullying": report["1"]["precision"],
        "recall_bullying": report["1"]["recall"],
        "f1_bullying": report["1"]["f1-score"]
    }

# ✅ Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

# Save trained model and tokenizer
model.save_pretrained("roberta-bullying")
tokenizer.save_pretrained("roberta-bullying")

from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Load fine-tuned model
model_path = "./roberta-bullying"  # or path where you saved the model
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# Switch to evaluation mode
model.eval()

# 🔍 Test input
text = "everyone hates you."

# Tokenize
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()

# Output result
print("🔎 Comment:", text)
print("📢 Prediction:", "bullying" if prediction == 1 else "non-bullying")