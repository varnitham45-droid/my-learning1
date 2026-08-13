mport os
import re
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AdamW
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

DATA_PATH = "spam.csv"

df = pd.read_csv(DATA_PATH)

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

df.columns = ["text", "label"]

df = df.dropna()

print("\nDuplicate rows:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

print("\nOriginal labels:")
print(df["label"].value_counts())

def normalize_label(label):
    label = str(label).strip().lower()

    if label in ["spam", "1", "true"]:
        return 1
    elif label in ["ham", "0", "false", "not spam"]:
        return 0
    else:
        return np.nan

df["label"] = df["label"].apply(normalize_label)

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

print("\nNormalized labels:")
print(df["label"].value_counts())

def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_text"] = df["text"].apply(clean_text)

print("\nExample preprocessing:")
print("Original:")
print(df["text"].iloc[0])

print("\nCleaned:")
print(df["clean_text"].iloc[0])

plt.figure(figsize=(7, 5))

sns.countplot(
    x=df["label"]
)

plt.xticks(
    [0, 1],
    ["Ham", "Spam"]
)

plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Emails")

plt.tight_layout()

plt.savefig(
    "class_distribution.png",
    dpi=300
)

plt.show()

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("\nTraining TF-IDF shape:")
print(X_train_tfidf.shape)

print("Testing TF-IDF shape:")
print(X_test_tfidf.shape)

def evaluate_model(model_name, y_true, y_pred):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("\n================================")
    print(model_name)
    print("================================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Ham", "Spam"],
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Confusion Matrix": cm
    }

start_time = time.time()

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=SEED
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_predictions = lr_model.predict(
    X_test_tfidf
)

lr_time = time.time() - start_time

lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions
)

print(
    "Training time:",
    round(lr_time, 2),
    "seconds"
)

start_time = time.time()

nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_predictions = nb_model.predict(
    X_test_tfidf
)

nb_time = time.time() - start_time

nb_results = evaluate_model(
    "Naive Bayes",
    y_test,
    nb_predictions
)

print(
    "Training time:",
    round(nb_time, 2),
    "seconds"
)

MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizer.from_pretrained(
    MODEL_NAME
)

class SpamDataset(Dataset):

    def _init_(
        self,
        texts,
        labels,
        tokenizer,
        max_length=128
    ):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, index):

        text = str(self.texts[index])

        label = int(self.labels[index])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids":
                encoding["input_ids"].squeeze(0),

            "attention_mask":
                encoding["attention_mask"].squeeze(0),

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }

train_dataset = SpamDataset(
    X_train,
    y_train,
    tokenizer
)

test_dataset = SpamDataset(
    X_test,
    y_test,
    tokenizer
)

BATCH_SIZE = 16

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model = model.to(device)

for param in model.distilbert.parameters():
    param.requires_grad = True

LEARNING_RATE = 2e-5
EPOCHS = 3
WEIGHT_DECAY = 0.01

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

criterion = nn.CrossEntropyLoss()

train_losses = []
validation_losses = []

transformer_start = time.time()

for epoch in range(EPOCHS):

    print(
        f"\n========== EPOCH {epoch + 1}/{EPOCHS} =========="
    )

    model.train()

    total_train_loss = 0

    for batch in train_loader:

        input_ids = batch["input_ids"].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(
            outputs.logits,
            labels
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss /
        len(train_loader)
    )

    train_losses.append(
        average_train_loss
    )

    model.eval()

    total_validation_loss = 0

    with torch.no_grad():

        for batch in test_loader:

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch[
                "labels"
            ].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = criterion(
                outputs.logits,
                labels
            )

            total_validation_loss += loss.item()

    average_validation_loss = (
        total_validation_loss /
        len(test_loader)
    )

    validation_losses.append(
        average_validation_loss
    )

    print(
        "Training Loss:",
        round(average_train_loss, 4)
    )

    print(
        "Validation Loss:",
        round(average_validation_loss, 4)
    )

transformer_time = (
    time.time() -
    transformer_start
)

print(
    "\nDistilBERT training time:",
    round(transformer_time, 2),
    "seconds"
)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch[
            "input_ids"
        ].to(device)

        attention_mask = batch[
            "attention_mask"
        ].to(device)

        labels = batch[
            "labels"
        ].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

distilbert_predictions = np.array(
    all_predictions
)

distilbert_labels = np.array(
    all_labels
)

distilbert_results = evaluate_model(
    "DistilBERT",
    distilbert_labels,
    distilbert_predictions
)

results = pd.DataFrame([

    {
        "Model": "Logistic Regression",
        "Accuracy": lr_results["Accuracy"],
        "Precision": lr_results["Precision"],
        "Recall": lr_results["Recall"],
        "F1-score": lr_results["F1-score"]
    },

    {
        "Model": "Naive Bayes",
        "Accuracy": nb_results["Accuracy"],
        "Precision": nb_results["Precision"],
        "Recall": nb_results["Recall"],
        "F1-score": nb_results["F1-score"]
    },

    {
        "Model": "DistilBERT",
        "Accuracy":
            distilbert_results["Accuracy"],
        "Precision":
            distilbert_results["Precision"],
        "Recall":
            distilbert_results["Recall"],
        "F1-score":
            distilbert_results["F1-score"]
    }

])

print("\nFINAL MODEL COMPARISON")
print(results)

results_plot = results.set_index(
    "Model"
)

results_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title(
    "Performance Comparison of Spam Classification Models"
)

plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)

plt.xticks(
    rotation=0
)

plt.legend(
    title="Metrics"
)

plt.tight_layout()

plt.savefig(
    "metric_comparison.png",
    dpi=300
)

plt.show()

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

sns.heatmap(
    lr_results["Confusion Matrix"],
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
    ax=axes[0]
)

axes[0].set_title(
    "Logistic Regression"
)

axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

sns.heatmap(
    nb_results["Confusion Matrix"],
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
    ax=axes[1]
)

axes[1].set_title(
    "Naive Bayes"
)

axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

sns.heatmap(
    distilbert_results["Confusion Matrix"],
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=["Ham", "Spam"],
    yticklabels=["Ham", "Spam"],
    ax=axes[2]
)

axes[2].set_title(
    "DistilBERT"
)

axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "confusion_matrices.png",
    dpi=300
)

plt.show()

plt.figure(figsize=(9, 5))

plt.plot(
    range(1, EPOCHS + 1),
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    range(1, EPOCHS + 1),
    validation_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "DistilBERT Training Behavior"
)

plt.xticks(
    range(1, EPOCHS + 1)
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "training_behavior.png",
    dpi=300
)

plt.show()

training_times = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Naive Bayes",
        "DistilBERT"
    ],

    "Training Time": [
        lr_time,
        nb_time,
        transformer_time
    ]

})

print("\nTraining Time:")
print(training_times)

plt.figure(figsize=(9, 5))

sns.barplot(
    data=training_times,
    x="Model",
    y="Training Time"
)

plt.title(
    "Computational Cost Comparison"
)

plt.xlabel("Model")

plt.ylabel(
    "Training Time (seconds)"
)

plt.xticks(
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "computational_cost.png",
    dpi=300
)

plt.show()

results.to_csv(
    "model_comparison_results.csv",
    index=False
)

training_times.to_csv(
    "training_times.csv",
    index=False
)

print("\nPROJECT COMPLETED SUCCESSFULLY")
