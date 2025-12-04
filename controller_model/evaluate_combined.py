"""
Combined Evaluation: MMLU (benign) + WMDP (harmful) Classification

This script evaluates the fine-tuned DistilBERT intent classifier on a combined
dataset of MMLU (label=0, benign) and WMDP (label=1, harmful) samples.

It provides comprehensive classification metrics including:
- Overall accuracy, precision, recall, F1-score
- Per-dataset performance breakdown
- Confusion matrix
- ROC-AUC and probability statistics

Usage:
    python evaluate_combined.py [--model_path MODEL_PATH] [--mmlu_path MMLU_PATH] [--wmdp_path WMDP_PATH]
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
    roc_curve,
)
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "model_path": "./intent_classifier_model",
    "mmlu_path": "./mmlu_dataset.json",
    "wmdp_path": "./wmdp_bio_dataset.json",
    "max_length": 128,
    "batch_size": 32,
}

LABEL_NAMES = {0: "benign (MMLU)", 1: "harmful (WMDP)"}
SHORT_LABEL_NAMES = {0: "benign", 1: "harmful"}


# ============================================================================
# Dataset
# ============================================================================

class CombinedDataset(Dataset):
    """PyTorch Dataset for combined MMLU + WMDP evaluation."""
    
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
            "label": torch.tensor(item["label"], dtype=torch.long),
            "source": item["source"]  # Track which dataset sample came from
        }


# ============================================================================
# Data Loading
# ============================================================================

def load_mmlu_data(data_path: str) -> list:
    """Load MMLU dataset and ensure all samples have label=0 (benign)."""
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Force label=0 for MMLU (benign) and add source tag
    for item in data:
        item["label"] = 0
        item["source"] = "mmlu"
    
    print(f"Loaded {len(data)} MMLU samples (label=0, benign)")
    return data


def load_wmdp_data(data_path: str) -> list:
    """Load WMDP dataset and ensure all samples have label=1 (harmful)."""
    with open(data_path, "r") as f:
        data = json.load(f)
    
    # Force label=1 for WMDP (harmful) and add source tag
    for item in data:
        item["label"] = 1
        item["source"] = "wmdp"
    
    print(f"Loaded {len(data)} WMDP samples (label=1, harmful)")
    return data


def load_combined_data(mmlu_path: str, wmdp_path: str, max_samples_per_class: int = None) -> list:
    """Load and combine MMLU and WMDP datasets."""
    mmlu_data = load_mmlu_data(mmlu_path)
    wmdp_data = load_wmdp_data(wmdp_path)
    
    # Optionally limit samples per class for balanced evaluation
    if max_samples_per_class:
        mmlu_data = mmlu_data[:max_samples_per_class]
        wmdp_data = wmdp_data[:max_samples_per_class]
        print(f"Limited to {max_samples_per_class} samples per class")
    
    # Combine datasets
    combined_data = mmlu_data + wmdp_data
    
    # Shuffle for fair evaluation
    np.random.seed(42)
    indices = np.random.permutation(len(combined_data))
    combined_data = [combined_data[i] for i in indices]
    
    print(f"\nCombined dataset: {len(combined_data)} total samples")
    print(f"  - MMLU (benign):  {len(mmlu_data)} samples")
    print(f"  - WMDP (harmful): {len(wmdp_data)} samples")
    
    return combined_data


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(model, dataloader, device):
    """Run evaluation and return predictions, labels, probs, and sources."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_sources = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            sources = batch["source"]
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get probabilities and predictions
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of harmful class
            all_sources.extend(sources)
    
    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_sources)
    )


def compute_and_print_metrics(predictions, labels, probs, sources, label_names):
    """Compute and display all evaluation metrics."""
    
    print("\n" + "=" * 80)
    print("COMBINED CLASSIFICATION RESULTS (MMLU + WMDP)")
    print("=" * 80)
    
    # =========================================================================
    # Overall Metrics
    # =========================================================================
    print("\n" + "-" * 80)
    print("OVERALL METRICS")
    print("-" * 80)
    
    accuracy = accuracy_score(labels, predictions)
    
    # Binary classification metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1, zero_division=0
    )
    
    # Macro metrics (for balanced view)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    
    # Weighted metrics
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    print(f"{'Total Samples':<30} {len(labels):>10}")
    print(f"{'Accuracy':<30} {accuracy:>10.4f}")
    print()
    print(f"{'Precision (harmful)':<30} {precision:>10.4f}")
    print(f"{'Recall (harmful)':<30} {recall:>10.4f}")
    print(f"{'F1-Score (harmful)':<30} {f1:>10.4f}")
    print()
    print(f"{'Macro Precision':<30} {precision_macro:>10.4f}")
    print(f"{'Macro Recall':<30} {recall_macro:>10.4f}")
    print(f"{'Macro F1-Score':<30} {f1_macro:>10.4f}")
    print()
    print(f"{'Weighted Precision':<30} {precision_weighted:>10.4f}")
    print(f"{'Weighted Recall':<30} {recall_weighted:>10.4f}")
    print(f"{'Weighted F1-Score':<30} {f1_weighted:>10.4f}")
    
    # ROC-AUC
    roc_auc = roc_auc_score(labels, probs)
    print(f"\n{'ROC-AUC':<30} {roc_auc:>10.4f}")
    
    # =========================================================================
    # Classification Report
    # =========================================================================
    print("\n" + "-" * 80)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 80)
    print(classification_report(
        labels, predictions,
        target_names=[SHORT_LABEL_NAMES[0], SHORT_LABEL_NAMES[1]],
        digits=4
    ))
    
    # =========================================================================
    # Confusion Matrix
    # =========================================================================
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    
    print("-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    print(f"\n{'':>20} {'Predicted':^25}")
    print(f"{'':>20} {SHORT_LABEL_NAMES[0]:>12} {SHORT_LABEL_NAMES[1]:>12}")
    print(f"{'Actual':<20}")
    
    tn, fp, fn, tp = cm.ravel()
    print(f"{SHORT_LABEL_NAMES[0]:<20} {tn:>12} {fp:>12}")
    print(f"{SHORT_LABEL_NAMES[1]:<20} {fn:>12} {tp:>12}")
    
    print(f"\n  True Negatives (TN):  {tn:>6} (benign correctly classified)")
    print(f"  False Positives (FP): {fp:>6} (benign misclassified as harmful)")
    print(f"  False Negatives (FN): {fn:>6} (harmful misclassified as benign)")
    print(f"  True Positives (TP):  {tp:>6} (harmful correctly classified)")
    
    # =========================================================================
    # Per-Dataset Performance
    # =========================================================================
    print("\n" + "-" * 80)
    print("PER-DATASET PERFORMANCE BREAKDOWN")
    print("-" * 80)
    
    # MMLU (benign) performance
    mmlu_mask = sources == "mmlu"
    mmlu_preds = predictions[mmlu_mask]
    mmlu_labels = labels[mmlu_mask]
    mmlu_probs = probs[mmlu_mask]
    
    mmlu_correct = (mmlu_preds == mmlu_labels).sum()
    mmlu_total = len(mmlu_labels)
    mmlu_accuracy = mmlu_correct / mmlu_total if mmlu_total > 0 else 0
    
    # True negative rate (specificity) for MMLU
    mmlu_tnr = (mmlu_preds == 0).sum() / mmlu_total if mmlu_total > 0 else 0
    mmlu_fpr = (mmlu_preds == 1).sum() / mmlu_total if mmlu_total > 0 else 0
    
    print(f"\n📊 MMLU Dataset (Benign, Label=0)")
    print(f"  Total samples:           {mmlu_total}")
    print(f"  Correctly classified:    {mmlu_correct} ({100*mmlu_accuracy:.2f}%)")
    print(f"  True Negative Rate:      {mmlu_tnr:.4f} ({100*mmlu_tnr:.2f}%)")
    print(f"  False Positive Rate:     {mmlu_fpr:.4f} ({100*mmlu_fpr:.2f}%)")
    print(f"  Mean harmful prob:       {mmlu_probs.mean():.4f}")
    
    # WMDP (harmful) performance
    wmdp_mask = sources == "wmdp"
    wmdp_preds = predictions[wmdp_mask]
    wmdp_labels = labels[wmdp_mask]
    wmdp_probs = probs[wmdp_mask]
    
    wmdp_correct = (wmdp_preds == wmdp_labels).sum()
    wmdp_total = len(wmdp_labels)
    wmdp_accuracy = wmdp_correct / wmdp_total if wmdp_total > 0 else 0
    
    # True positive rate (sensitivity/recall) for WMDP
    wmdp_tpr = (wmdp_preds == 1).sum() / wmdp_total if wmdp_total > 0 else 0
    wmdp_fnr = (wmdp_preds == 0).sum() / wmdp_total if wmdp_total > 0 else 0
    
    print(f"\n📊 WMDP Dataset (Harmful, Label=1)")
    print(f"  Total samples:           {wmdp_total}")
    print(f"  Correctly classified:    {wmdp_correct} ({100*wmdp_accuracy:.2f}%)")
    print(f"  True Positive Rate:      {wmdp_tpr:.4f} ({100*wmdp_tpr:.2f}%)")
    print(f"  False Negative Rate:     {wmdp_fnr:.4f} ({100*wmdp_fnr:.2f}%)")
    print(f"  Mean harmful prob:       {wmdp_probs.mean():.4f}")
    
    # =========================================================================
    # Probability Statistics
    # =========================================================================
    print("\n" + "-" * 80)
    print("PREDICTION PROBABILITY STATISTICS")
    print("-" * 80)
    
    print(f"\nOverall probability of 'harmful' class (label=1):")
    print(f"  Mean:   {probs.mean():.4f}")
    print(f"  Std:    {probs.std():.4f}")
    print(f"  Min:    {probs.min():.4f}")
    print(f"  Max:    {probs.max():.4f}")
    print(f"  Median: {np.median(probs):.4f}")
    
    print(f"\nBy actual class:")
    print(f"  Benign (MMLU) - Mean harmful prob:  {mmlu_probs.mean():.4f} (std: {mmlu_probs.std():.4f})")
    print(f"  Harmful (WMDP) - Mean harmful prob: {wmdp_probs.mean():.4f} (std: {wmdp_probs.std():.4f})")
    
    # Separation between classes
    prob_separation = wmdp_probs.mean() - mmlu_probs.mean()
    print(f"\n  Class probability separation: {prob_separation:.4f}")
    
    # =========================================================================
    # Additional Rates
    # =========================================================================
    print("\n" + "-" * 80)
    print("SUMMARY RATES")
    print("-" * 80)
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    balanced_accuracy = (specificity + sensitivity) / 2
    
    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 47)
    print(f"{'Specificity (TNR)':<35} {specificity:>10.4f}")
    print(f"{'Sensitivity (TPR/Recall)':<35} {sensitivity:>10.4f}")
    print(f"{'Balanced Accuracy':<35} {balanced_accuracy:>10.4f}")
    print(f"{'False Positive Rate (FPR)':<35} {fp/(tn+fp) if (tn+fp) > 0 else 0:>10.4f}")
    print(f"{'False Negative Rate (FNR)':<35} {fn/(tp+fn) if (tp+fn) > 0 else 0:>10.4f}")
    
    # Return all metrics as dict
    metrics = {
        "total_samples": len(labels),
        "accuracy": float(accuracy),
        "precision_harmful": float(precision),
        "recall_harmful": float(recall),
        "f1_harmful": float(f1),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "mmlu_dataset": {
            "total": int(mmlu_total),
            "correct": int(mmlu_correct),
            "accuracy": float(mmlu_accuracy),
            "true_negative_rate": float(mmlu_tnr),
            "false_positive_rate": float(mmlu_fpr),
            "mean_harmful_prob": float(mmlu_probs.mean()),
        },
        "wmdp_dataset": {
            "total": int(wmdp_total),
            "correct": int(wmdp_correct),
            "accuracy": float(wmdp_accuracy),
            "true_positive_rate": float(wmdp_tpr),
            "false_negative_rate": float(wmdp_fnr),
            "mean_harmful_prob": float(wmdp_probs.mean()),
        },
        "specificity": float(specificity),
        "sensitivity": float(sensitivity),
        "balanced_accuracy": float(balanced_accuracy),
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std()),
        "class_probability_separation": float(prob_separation),
    }
    
    return metrics


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned DistilBERT on combined MMLU + WMDP dataset"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_CONFIG["model_path"],
        help="Path to the fine-tuned model"
    )
    parser.add_argument(
        "--mmlu_path",
        type=str,
        default=DEFAULT_CONFIG["mmlu_path"],
        help="Path to the MMLU dataset JSON"
    )
    parser.add_argument(
        "--wmdp_path",
        type=str,
        default=DEFAULT_CONFIG["wmdp_path"],
        help="Path to the WMDP bio dataset JSON"
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
        "--max_samples_per_class",
        type=int,
        default=None,
        help="Optional: Limit samples per class for balanced evaluation"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    model_path = script_dir / args.model_path if not os.path.isabs(args.model_path) else Path(args.model_path)
    mmlu_path = script_dir / args.mmlu_path if not os.path.isabs(args.mmlu_path) else Path(args.mmlu_path)
    wmdp_path = script_dir / args.wmdp_path if not os.path.isabs(args.wmdp_path) else Path(args.wmdp_path)
    
    print("=" * 80)
    print("COMBINED EVALUATION: MMLU (benign) + WMDP (harmful)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Model path:   {model_path}")
    print(f"  MMLU path:    {mmlu_path}")
    print(f"  WMDP path:    {wmdp_path}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Max length:   {args.max_length}")
    if args.max_samples_per_class:
        print(f"  Max samples per class: {args.max_samples_per_class}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device:       {device}")
    
    # Load model and tokenizer
    print("\n" + "-" * 80)
    print("LOADING MODEL")
    print("-" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    
    print(f"Model loaded: {model.config.architectures}")
    print(f"Labels: {model.config.id2label}")
    
    # Load combined data
    print("\n" + "-" * 80)
    print("LOADING DATA")
    print("-" * 80)
    
    combined_data = load_combined_data(
        str(mmlu_path),
        str(wmdp_path),
        max_samples_per_class=args.max_samples_per_class
    )
    
    # Create dataset and dataloader
    # Custom collate function to handle string 'source' field
    def collate_fn(batch):
        return {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "label": torch.stack([item["label"] for item in batch]),
            "source": [item["source"] for item in batch]
        }
    
    dataset = CombinedDataset(combined_data, tokenizer, max_length=args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Run evaluation
    print("\n" + "-" * 80)
    print("RUNNING EVALUATION")
    print("-" * 80)
    
    predictions, labels, probs, sources = evaluate_model(model, dataloader, device)
    
    # Compute and print metrics
    metrics = compute_and_print_metrics(predictions, labels, probs, sources, LABEL_NAMES)
    
    # Save results if output file specified
    if args.output_file:
        output_path = script_dir / args.output_file if not os.path.isabs(args.output_file) else Path(args.output_file)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    
    return metrics


if __name__ == "__main__":
    main()
