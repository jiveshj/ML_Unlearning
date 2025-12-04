"""
Evaluate Fine-tuned DistilBERT Intent Classifier on MMLU Dataset

This script loads the fine-tuned model and evaluates it on the MMLU dataset,
computing accuracy, precision, recall, F1-score, and other metrics.

The MMLU dataset contains benign/general knowledge questions (label=0), so this
evaluation tests the model's ability to correctly identify safe content.

Usage:
    python evaluate_mmlu.py [--model_path MODEL_PATH] [--data_path DATA_PATH] [--batch_size BATCH_SIZE]
"""

import json
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "model_path": "./intent_classifier_model",
    "data_path": "./mmlu_dataset.json",
    "max_length": 128,
    "batch_size": 32,
}

LABEL_NAMES = {0: "benign", 1: "harmful"}


# ============================================================================
# Dataset
# ============================================================================

class MMLUDataset(Dataset):
    """PyTorch Dataset for MMLU evaluation."""
    
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long)
        }


# ============================================================================
# Evaluation
# ============================================================================

def load_data(data_path: str) -> list:
    """Load the JSON dataset."""
    with open(data_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {data_path}")
    
    # Count label distribution
    labels = [item["label"] for item in data]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique, counts))}")
    
    return data


def evaluate_model(model, dataloader, device):
    """Run evaluation and return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities and predictions
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of harmful class
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def compute_and_print_metrics(predictions, labels, probs, label_names):
    """Compute and display all evaluation metrics."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # For MMLU (mostly benign samples), we calculate metrics appropriately
    # Using macro average for balanced view across classes
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    
    # Also get per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 37)
    print(f"{'Accuracy':<25} {accuracy:>10.4f}")
    print(f"{'Macro Precision':<25} {precision_macro:>10.4f}")
    print(f"{'Macro Recall':<25} {recall_macro:>10.4f}")
    print(f"{'Macro F1-Score':<25} {f1_macro:>10.4f}")
    
    # ROC-AUC (only if both classes are present)
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        roc_auc = roc_auc_score(labels, probs)
        print(f"{'ROC-AUC':<25} {roc_auc:>10.4f}")
    else:
        roc_auc = None
        print(f"{'ROC-AUC':<25} {'N/A (single class)':>15}")
    
    # Classification report
    print("\n" + "-" * 70)
    print("CLASSIFICATION REPORT")
    print("-" * 70)
    
    # Handle case where not all classes are present in predictions/labels
    present_labels = sorted(set(labels) | set(predictions))
    target_names = [label_names[i] for i in present_labels]
    
    print(classification_report(
        labels, predictions,
        labels=present_labels,
        target_names=target_names,
        digits=4,
        zero_division=0
    ))
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    print("-" * 70)
    print("CONFUSION MATRIX")
    print("-" * 70)
    print(f"\n{'':>15} {'Predicted':^20}")
    print(f"{'':>15} {label_names[0]:>10} {label_names[1]:>10}")
    print(f"{'Actual':>15}")
    
    for i, label in enumerate([label_names[0], label_names[1]]):
        row = cm[i] if i < len(cm) else [0, 0]
        print(f"{label:>15} {row[0]:>10} {row[1]:>10}")
    
    # Additional statistics for benign class detection (important for MMLU)
    print("\n" + "-" * 70)
    print("CLASS DETECTION STATISTICS")
    print("-" * 70)
    
    benign_mask = labels == 0
    benign_correct = (predictions[benign_mask] == 0).sum() if benign_mask.sum() > 0 else 0
    benign_total = benign_mask.sum()
    
    harmful_mask = labels == 1
    harmful_correct = (predictions[harmful_mask] == 1).sum() if harmful_mask.sum() > 0 else 0
    harmful_total = harmful_mask.sum()
    
    if benign_total > 0:
        print(f"\nBenign samples correctly identified: {benign_correct}/{benign_total} ({100*benign_correct/benign_total:.2f}%)")
    else:
        print("\nNo benign samples in dataset")
        
    if harmful_total > 0:
        print(f"Harmful samples correctly identified: {harmful_correct}/{harmful_total} ({100*harmful_correct/harmful_total:.2f}%)")
    else:
        print("No harmful samples in dataset")
    
    # False positive rate (benign classified as harmful) - important metric for MMLU
    if benign_total > 0:
        false_positive_rate = (predictions[benign_mask] == 1).sum() / benign_total
        print(f"\nFalse Positive Rate (benign → harmful): {false_positive_rate:.4f} ({100*false_positive_rate:.2f}%)")
    
    # Probability distribution
    print("\n" + "-" * 70)
    print("PREDICTION PROBABILITY STATISTICS")
    print("-" * 70)
    print(f"\nProbability of 'harmful' class (label=1):")
    print(f"  Mean:   {probs.mean():.4f}")
    print(f"  Std:    {probs.std():.4f}")
    print(f"  Min:    {probs.min():.4f}")
    print(f"  Max:    {probs.max():.4f}")
    print(f"  Median: {np.median(probs):.4f}")
    
    # Probability distribution by actual class
    if benign_total > 0:
        print(f"\nFor actual BENIGN samples:")
        print(f"  Mean harmful prob:   {probs[benign_mask].mean():.4f}")
        print(f"  Std:                 {probs[benign_mask].std():.4f}")
    
    if harmful_total > 0:
        print(f"\nFor actual HARMFUL samples:")
        print(f"  Mean harmful prob:   {probs[harmful_mask].mean():.4f}")
        print(f"  Std:                 {probs[harmful_mask].std():.4f}")
    
    # Return metrics as dict
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "benign_correct": int(benign_correct),
        "benign_total": int(benign_total),
        "harmful_correct": int(harmful_correct),
        "harmful_total": int(harmful_total),
        "false_positive_rate": float(false_positive_rate) if benign_total > 0 else None,
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std()),
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned DistilBERT on MMLU dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_CONFIG["model_path"],
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_CONFIG["data_path"],
        help="Path to the MMLU dataset JSON"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_CONFIG["max_length"],
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional: Save results to JSON file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional: Limit evaluation to first N samples (for quick testing)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model_path if not os.path.isabs(args.model_path) else Path(args.model_path)
    data_path = script_dir / args.data_path if not os.path.isabs(args.data_path) else Path(args.data_path)
    
    print("=" * 70)
    print("MMLU DATASET EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model path:   {model_path}")
    print(f"  Data path:    {data_path}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Max length:   {args.max_length}")
    if args.max_samples:
        print(f"  Max samples:  {args.max_samples}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device:       {device}")
    
    # Load model and tokenizer
    print("\n" + "-" * 70)
    print("LOADING MODEL")
    print("-" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    print(f"Model loaded: {model.config.architectures}")
    print(f"Labels: {model.config.id2label}")
    
    # Load data
    print("\n" + "-" * 70)
    print("LOADING DATA")
    print("-" * 70)
    
    data = load_data(str(data_path))
    
    # Optionally limit samples
    if args.max_samples and args.max_samples < len(data):
        print(f"Limiting to first {args.max_samples} samples")
        data = data[:args.max_samples]
    
    # Create dataset and dataloader
    dataset = MMLUDataset(data, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Run evaluation
    print("\n" + "-" * 70)
    print("RUNNING EVALUATION")
    print("-" * 70)
    
    predictions, labels, probs = evaluate_model(model, dataloader, device)
    
    # Compute and print metrics
    metrics = compute_and_print_metrics(predictions, labels, probs, LABEL_NAMES)
    
    # Save results if output file specified
    if args.output_file:
        output_path = script_dir / args.output_file if not os.path.isabs(args.output_file) else Path(args.output_file)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
