# train_medical_model.py

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

def train_medical_classifier():
    """
    Trains a text classification model to predict a disease based on symptoms.
    """
    # --- 1. Load and Prepare Data ---
    print("Loading Symptom to Disease dataset...")
    df = pd.read_csv("Symptom2Disease.csv")
    
    # The model needs numerical labels, not text labels. We create a mapping.
    labels = df['label'].unique().tolist()
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    # Apply the mapping to our dataframe
    df['label_id'] = df['label'].map(label2id)
    
    # Split data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # --- 2. Tokenize Data ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], padding="max_length", truncation=True)
        tokenized["label"] = examples["label_id"]
        return tokenized

    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # --- 3. Load Model ---
    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # --- 4. Define Training Arguments ---
    training_args = TrainingArguments(
        output_dir="./sahara-medical-results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3, # Increased to 3 for better performance on this dataset
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # --- 5. Define Evaluation Metrics ---
    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
        f1 = f1_score(p.label_ids, preds, average="weighted")
        acc = accuracy_score(p.label_ids, preds)
        return {"accuracy": acc, "f1": f1}

    # --- 6. Train the Model ---
    print("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # --- 7. Save the Final Model ---
    print("Training complete. Saving final model...")
    trainer.save_model("./sahara-medical-model")
    print("Model saved to './sahara-medical-model'. You're ready for the final step!")

if __name__ == "__main__":
    train_medical_classifier()