#!/usr/bin/env python3
"""
Baseline Inference Script for Gemma 2 2B on WMDP and MMLU

This script evaluates the baseline (no SAE intervention) performance of 
google/gemma-2-2b on:
- WMDP-Bio (harmful biosecurity knowledge)
- Select MMLU subjects (benign general knowledge)

The MMLU subjects used match the paper:
- High School US History (204 questions)
- High School Geography (198 questions)
- Human Aging (223 questions)  
- College Computer Science (100 questions)

Usage:
    python run_baseline_inference.py
    python run_baseline_inference.py --max_samples 100
    python run_baseline_inference.py --batch_size 8 --output results.json

Based on: "Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning"
https://arxiv.org/pdf/2503.11127
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    "model_name": "google/gemma-2-2b",
    "batch_size": 4,
    "max_new_tokens": 1,
    "max_length": 512,
}

MMLU_SUBJECTS = [
    "high_school_us_history",
    "high_school_geography", 
    "human_aging",
    "college_computer_science",
]

CHOICE_LETTERS = ["A", "B", "C", "D"]


# ============================================================================
# Dataset Classes
# ============================================================================

class MultipleChoiceDataset(Dataset):
    """Dataset for multiple-choice question answering."""
    
    def __init__(
        self,
        data: List[dict],
        tokenizer,
        max_length: int = 512,
        dataset_name: str = "unknown"
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        prompt = self._format_prompt(item)
        
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "answer": item["answer"],
            "prompt": prompt,
        }
    
    def _format_prompt(self, item: dict) -> str:
        """Format a multiple-choice question into a prompt."""
        question = item["question"]
        choices = item["choices"]
        
        prompt = f"Question: {question}\n\n"
        for i, choice in enumerate(choices):
            prompt += f"{CHOICE_LETTERS[i]}. {choice}\n"
        prompt += "\nAnswer:"
        
        return prompt


def collate_fn(batch: List[dict]) -> dict:
    """Custom collate function for variable-length sequences."""
    # Find max length in batch
    max_len = max(item["input_ids"].size(0) for item in batch)
    
    # Pad sequences
    input_ids = []
    attention_masks = []
    answers = []
    prompts = []
    
    for item in batch:
        seq_len = item["input_ids"].size(0)
        padding_len = max_len - seq_len
        
        # Pad on the left for causal LM
        padded_ids = torch.cat([
            torch.zeros(padding_len, dtype=torch.long),
            item["input_ids"]
        ])
        padded_mask = torch.cat([
            torch.zeros(padding_len, dtype=torch.long),
            item["attention_mask"]
        ])
        
        input_ids.append(padded_ids)
        attention_masks.append(padded_mask)
        answers.append(item["answer"])
        prompts.append(item["prompt"])
    
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "answers": answers,
        "prompts": prompts,
    }


# ============================================================================
# Data Loading
# ============================================================================

def load_wmdp_bio(max_samples: Optional[int] = None) -> List[dict]:
    """Load WMDP-Bio dataset from HuggingFace."""
    print("\nLoading WMDP-Bio dataset...")
    
    try:
        dataset = load_dataset("cais/wmdp", "wmdp-bio", split="test")
        data = list(dataset)
        print(f"  Loaded {len(data)} samples")
        
        if max_samples and max_samples < len(data):
            data = data[:max_samples]
            print(f"  Limited to {max_samples} samples")
        
        return data
    except Exception as e:
        print(f"  Error loading WMDP-Bio: {e}")
        return []


def load_mmlu_subjects(
    subjects: List[str] = None,
    max_samples_per_subject: Optional[int] = None
) -> Tuple[List[dict], Dict[str, int]]:
    """Load MMLU subjects from HuggingFace."""
    subjects = subjects or MMLU_SUBJECTS
    print("\nLoading MMLU subjects...")
    
    all_data = []
    subject_counts = {}
    
    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            data = list(dataset)
            
            if max_samples_per_subject and max_samples_per_subject < len(data):
                data = data[:max_samples_per_subject]
            
            # Add subject info to each item
            for item in data:
                item["subject"] = subject
            
            all_data.extend(data)
            subject_counts[subject] = len(data)
            print(f"  {subject}: {len(data)} samples")
            
        except Exception as e:
            print(f"  Error loading {subject}: {e}")
            subject_counts[subject] = 0
    
    print(f"  Total MMLU samples: {len(all_data)}")
    return all_data, subject_counts


# ============================================================================
# Inference
# ============================================================================

def get_answer_token_ids(tokenizer) -> Dict[str, int]:
    """Get token IDs for answer letters A, B, C, D."""
    answer_ids = {}
    
    for letter in CHOICE_LETTERS:
        # Try different tokenization formats
        tokens = tokenizer.encode(letter, add_special_tokens=False)
        if tokens:
            answer_ids[letter] = tokens[0]
        else:
            # Fallback: try with space prefix
            tokens = tokenizer.encode(f" {letter}", add_special_tokens=False)
            if tokens:
                answer_ids[letter] = tokens[-1]
    
    return answer_ids


def evaluate_batch(
    model,
    tokenizer,
    batch: dict,
    answer_token_ids: Dict[str, int],
    device: str
) -> Tuple[List[int], List[int]]:
    """Evaluate a batch of questions and return predictions and ground truth."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    answers = batch["answers"]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Get logits for the last token (where we predict the answer)
    last_token_logits = logits[:, -1, :]
    
    predictions = []
    for i in range(last_token_logits.size(0)):
        # Get probabilities for each answer choice
        choice_probs = {}
        for letter, token_id in answer_token_ids.items():
            choice_probs[letter] = last_token_logits[i, token_id].item()
        
        # Select the answer with highest probability
        predicted_letter = max(choice_probs, key=choice_probs.get)
        predicted_idx = CHOICE_LETTERS.index(predicted_letter)
        predictions.append(predicted_idx)
    
    return predictions, answers


def run_evaluation(
    model,
    tokenizer,
    dataloader: DataLoader,
    answer_token_ids: Dict[str, int],
    device: str,
    dataset_name: str = "Dataset"
) -> Dict:
    """Run evaluation on a dataset."""
    model.eval()
    
    all_predictions = []
    all_answers = []
    
    print(f"\nEvaluating {dataset_name}...")
    
    for batch in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        predictions, answers = evaluate_batch(
            model, tokenizer, batch, answer_token_ids, device
        )
        all_predictions.extend(predictions)
        all_answers.extend(answers)
    
    # Calculate metrics
    correct = sum(p == a for p, a in zip(all_predictions, all_answers))
    total = len(all_predictions)
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": all_predictions,
        "answers": all_answers,
    }


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline inference on WMDP and MMLU with Gemma 2 2B"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_CONFIG["model_name"],
        help="HuggingFace model name or path"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_CONFIG["batch_size"],
        help="Batch size for inference"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for quick testing)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=DEFAULT_CONFIG["max_length"],
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--wmdp_only",
        action="store_true",
        help="Only evaluate on WMDP-Bio"
    )
    parser.add_argument(
        "--mmlu_only",
        action="store_true",
        help="Only evaluate on MMLU"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model precision"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("BASELINE INFERENCE: Gemma 2 2B on WMDP & MMLU")
    print("=" * 70)
    
    # ========================================================================
    # Setup
    # ========================================================================
    
    script_dir = Path(__file__).parent
    
    # Determine device
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Set dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    print(f"\nConfiguration:")
    print(f"  Model:       {args.model_name}")
    print(f"  Device:      {device}")
    print(f"  Dtype:       {args.dtype}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Max length:  {args.max_length}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    
    # ========================================================================
    # Load Model
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("LOADING MODEL")
    print("-" * 70)
    
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading model from {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    
    if device != "cuda":
        model = model.to(device)
    
    model.eval()
    print(f"✓ Model loaded successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get answer token IDs
    answer_token_ids = get_answer_token_ids(tokenizer)
    print(f"  Answer token IDs: {answer_token_ids}")
    
    # ========================================================================
    # Load Datasets
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("LOADING DATASETS")
    print("-" * 70)
    
    results = {
        "model": args.model_name,
        "device": device,
        "dtype": args.dtype,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Load WMDP-Bio
    if not args.mmlu_only:
        wmdp_data = load_wmdp_bio(max_samples=args.max_samples)
        if wmdp_data:
            wmdp_dataset = MultipleChoiceDataset(
                wmdp_data, tokenizer, args.max_length, "WMDP-Bio"
            )
            wmdp_loader = DataLoader(
                wmdp_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
    
    # Load MMLU
    if not args.wmdp_only:
        mmlu_data, subject_counts = load_mmlu_subjects(
            max_samples_per_subject=args.max_samples
        )
        if mmlu_data:
            mmlu_dataset = MultipleChoiceDataset(
                mmlu_data, tokenizer, args.max_length, "MMLU"
            )
            mmlu_loader = DataLoader(
                mmlu_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
    
    # ========================================================================
    # Run Evaluation
    # ========================================================================
    
    print("\n" + "-" * 70)
    print("RUNNING EVALUATION")
    print("-" * 70)
    
    # Evaluate WMDP-Bio
    if not args.mmlu_only and wmdp_data:
        wmdp_results = run_evaluation(
            model, tokenizer, wmdp_loader, answer_token_ids, device, "WMDP-Bio"
        )
        results["wmdp_bio"] = {
            "accuracy": wmdp_results["accuracy"],
            "correct": wmdp_results["correct"],
            "total": wmdp_results["total"],
        }
        print(f"\n  WMDP-Bio Accuracy: {wmdp_results['accuracy']:.4f} ({wmdp_results['correct']}/{wmdp_results['total']})")
    
    # Evaluate MMLU
    if not args.wmdp_only and mmlu_data:
        mmlu_results = run_evaluation(
            model, tokenizer, mmlu_loader, answer_token_ids, device, "MMLU"
        )
        results["mmlu"] = {
            "accuracy": mmlu_results["accuracy"],
            "correct": mmlu_results["correct"],
            "total": mmlu_results["total"],
            "subjects": subject_counts,
        }
        print(f"\n  MMLU Accuracy: {mmlu_results['accuracy']:.4f} ({mmlu_results['correct']}/{mmlu_results['total']})")
        
        # Per-subject breakdown
        print("\n  Per-subject results:")
        for subject, count in subject_counts.items():
            print(f"    {subject}: {count} questions")
    
    # ========================================================================
    # Results Summary
    # ========================================================================
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Dataset':<30} {'Accuracy':<15} {'Correct/Total':<20}")
    print("-" * 65)
    
    if "wmdp_bio" in results:
        wmdp = results["wmdp_bio"]
        print(f"{'WMDP-Bio':<30} {wmdp['accuracy']:<15.4f} {wmdp['correct']}/{wmdp['total']}")
    
    if "mmlu" in results:
        mmlu = results["mmlu"]
        print(f"{'MMLU (weighted)':<30} {mmlu['accuracy']:<15.4f} {mmlu['correct']}/{mmlu['total']}")
    
    print("-" * 65)
    
    # Save results
    if args.output:
        output_path = script_dir / args.output if not os.path.isabs(args.output) else Path(args.output)
    else:
        output_path = script_dir / "baseline_results.json"
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
