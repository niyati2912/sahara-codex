# train_medical_model.py

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def train_medical_classifier():
    # --- 1. Load and Prepare the Data ---
    print("Loading dataset...")
    try:
        df = pd.read_csv("Symptom2Disease.csv")
    except FileNotFoundError:
        print("Error: Symptom2Disease.csv not found.")
        print("Please make sure the dataset file is in the same folder as this script.")
        return

    # Create the numerical label column
    le = LabelEncoder()
    df['label_id'] = le.fit_transform(df['label'])
    
    # FIX #1: The Trainer expects the numerical labels to be in a column named "label".
    # We drop the old text column and rename our new numerical column.
    df = df.drop(columns=['label'])
    df = df.rename(columns={'label_id': 'label'})
    
    # Create mappings to convert between number IDs and disease names
    id2label = {id: label for id, label in enumerate(le.classes_)}
    label2id = {label: id for id, label in id2label.items()}
    print("Dataset loaded and prepared.")

    # Convert pandas DataFrame to Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1)
    
    # --- 2. Tokenize the Data ---
    print("Tokenizing data...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)
    print("Data tokenized.")

    # --- 3. Configure and Train the Model ---
    print("Loading pre-trained model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id
    )
    print("Model loaded.")

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy="epoch",
        # FIX #2: This avoids all errors related to evaluation_strategy
        # by simply saving the final model instead of the "best" one.
        load_best_model_at_end=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    print("Training arguments set. Starting training...")

    # Start the training process.
    trainer.train()
    
    print("Training complete.")

    # --- 4. Save the Final Model ---
    print("Saving final model...")
    trainer.save_model("./sahara-medical-model")
    print("Model saved to ./sahara-medical-model")
    print("AI model training is complete!")


if __name__ == "__main__":
    train_medical_classifier()