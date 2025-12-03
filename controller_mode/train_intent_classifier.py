"""
Intent Classification using Fine-tuned DistilBERT

This script fine-tunes DistilBERT for binary intent classification on the 
synthetic dataset to distinguish between harmful/bio-related prompts (label=1)
and benign/general prompts (label=0).

Usage:
    python train_intent_classifier.py [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
"""

import json
import argparse
import os
from pathlib import Path

# Enable MPS fallback for operations not supported on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "model_name": "distilbert-base-cased",
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "output_dir": "./intent_classifier_model",
    "seed": 42,
}

LABEL_NAMES = {0: "benign", 1: "harmful"}


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_dataset_from_json(json_path: str) -> Dataset:
    """Load the JSON dataset and convert to HuggingFace Dataset format."""
    from datasets import ClassLabel
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    dataset = Dataset.from_dict({
        "text": [item["prompt"] for item in data],
        "label": [item["label"] for item in data]
    })
    
    # Cast label column to ClassLabel for stratified splitting
    dataset = dataset.cast_column(
        "label", 
        ClassLabel(names=["benign", "harmful"])
    )
    
    print(f"Loaded {len(dataset)} samples from {json_path}")
    print(f"Label distribution: {dict(zip(*np.unique(dataset['label'], return_counts=True)))}")
    
    return dataset


def prepare_splits(dataset: Dataset, seed: int = 42) -> DatasetDict:
    """Split dataset into train (80%), validation (10%), and test (10%) sets."""
    # First split: 80% train, 20% temp
    train_test = dataset.train_test_split(
        test_size=0.2, 
        seed=seed,
        stratify_by_column="label"
    )
    
    # Second split: 50% of temp for val, 50% for test (each 10% of total)
    val_test = train_test["test"].train_test_split(
        test_size=0.5, 
        seed=seed,
        stratify_by_column="label"
    )
    
    splits = DatasetDict({
        "train": train_test["train"],
        "validation": val_test["train"],
        "test": val_test["test"]
    })
    
    print(f"\nDataset splits:")
    print(f"  Train:      {len(splits['train'])} samples")
    print(f"  Validation: {len(splits['validation'])} samples")
    print(f"  Test:       {len(splits['test'])} samples")
    
    return splits


def tokenize_dataset(dataset: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    """Tokenize the dataset using the provided tokenizer."""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing"
    )
    
    return tokenized


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(eval_pred):
    """Compute evaluation metrics for the Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    accuracy = accuracy_score(labels, predictions)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def evaluate_on_test_set(trainer: Trainer, test_dataset, label_names: dict):
    """Perform detailed evaluation on the test set."""
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    
    # Get predictions
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    
    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        labels, preds, 
        target_names=[label_names[0], label_names[1]]
    ))
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    print("Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  {label_names[0]:>8} {label_names[1]:>8}")
    print(f"Actual {label_names[0]:>8}  {cm[0][0]:>8} {cm[0][1]:>8}")
    print(f"       {label_names[1]:>8}  {cm[1][0]:>8} {cm[1][1]:>8}")
    
    return {
        "accuracy": accuracy,
        "predictions": preds,
        "labels": labels,
        "confusion_matrix": cm
    }


# ============================================================================
# Training
# ============================================================================

def train_model(
    tokenized_datasets: DatasetDict,
    model_name: str,
    output_dir: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    warmup_ratio: float,
    weight_decay: float,
    seed: int,
):
    """Fine-tune the DistilBERT model."""
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load model
    print(f"\nLoading model: {model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id={v: k for k, v in LABEL_NAMES.items()}
    )
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Training arguments
    # Note: dataloader_pin_memory=False is required for MPS (Apple Silicon) compatibility
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        seed=seed,
        report_to="none",  # Disable wandb/tensorboard by default
        push_to_hub=False,
        dataloader_pin_memory=False,  # Required for MPS compatibility
        use_cpu=device == "cpu",  # Only use CPU if no GPU/MPS available
    )
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    
    # Train
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    trainer.train()
    
    # Save the best model
    trainer.save_model(output_dir)
    print(f"\nModel saved to: {output_dir}")
    
    return trainer


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for intent classification"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="controller_mode/final_unique_dataset.json",
        help="Path to the JSON dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_CONFIG["output_dir"],
        help="Directory to save the model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_CONFIG["num_epochs"],
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Batch size for training and evaluation"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_CONFIG["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_CONFIG["max_length"],
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_CONFIG["seed"],
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Resolve data path
    script_dir = Path(__file__).parent
    if not os.path.isabs(args.data_path):
        # Try relative to script directory first
        data_path = script_dir / Path(args.data_path).name
        if not data_path.exists():
            # Try as-is (relative to cwd)
            data_path = Path(args.data_path)
    else:
        data_path = Path(args.data_path)
    
    print("=" * 60)
    print("DISTILBERT INTENT CLASSIFICATION")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Data path:    {data_path}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max length:   {args.max_length}")
    print(f"  Seed:         {args.seed}")
    
    # Load and prepare data
    print("\n" + "=" * 60)
    print("DATA PREPARATION")
    print("=" * 60)
    
    dataset = load_dataset_from_json(str(data_path))
    splits = prepare_splits(dataset, seed=args.seed)
    
    # # Load tokenizer
    print(f"\nLoading tokenizer: {DEFAULT_CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CONFIG["model_name"])
    
    # # Tokenize datasets
    tokenized_datasets = tokenize_dataset(splits, tokenizer, args.max_length)
    
    # Train model
    trainer = train_model(
        tokenized_datasets=tokenized_datasets,
        model_name=DEFAULT_CONFIG["model_name"],
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=DEFAULT_CONFIG["warmup_ratio"],
        weight_decay=DEFAULT_CONFIG["weight_decay"],
        seed=args.seed,
    )
    
    # Evaluate on test set
    test_results = evaluate_on_test_set(
        trainer, 
        tokenized_datasets["test"], 
        LABEL_NAMES
    )
    
    # Save tokenizer alongside model
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nTokenizer saved to: {args.output_dir}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Model and tokenizer saved to: {args.output_dir}")
    print("\nTo use the model for inference:")
    print(f"  from transformers import pipeline")
    print(f"  classifier = pipeline('text-classification', model='{args.output_dir}')")
    print(f"  result = classifier('Your text here')")


if __name__ == "__main__":
    main()
