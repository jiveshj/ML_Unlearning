"""
Evaluator module for multiple choice question answering evaluation.

Provides functions for evaluating model performance on WMDP and MMLU
benchmarks using log-likelihood scoring.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def format_multiple_choice_prompt(question: str, choices: List[str]) -> str:
    """
    Format a multiple choice question for the model.
    
    Args:
        question: The question text
        choices: List of answer choices
        
    Returns:
        Formatted prompt string
    """
    prompt = f"Question: {question}\n\n"
    for i, choice in enumerate(choices):
        prompt += f"{chr(65+i)}) {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def evaluate_multiple_choice(
    model,
    tokenizer,
    dataset: Dataset,
    batch_size: int = 1,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[float, List[bool], Dict]:
    """
    Evaluate model on multiple choice dataset using log-likelihood scoring.
    
    For each question, scores each answer choice by computing the log
    probability of the answer token given the prompt.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        dataset: Dataset with 'question', 'choices', and 'answer' fields
        batch_size: Not used currently (processes one at a time)
        max_samples: Optional limit on number of samples
        device: Device to run on (auto-detected if None)
        
    Returns:
        Tuple of (accuracy, list of correct/incorrect, details dict)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    results = []
    all_confidences = []
    
    samples = list(dataset)
    if max_samples:
        samples = samples[:max_samples]
    
    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating"):
            question = sample['question']
            choices = sample['choices']
            answer = sample['answer']
            
            # Score each choice
            choice_scores = []
            
            for choice_idx, choice in enumerate(choices):
                # Format prompt
                prompt = format_multiple_choice_prompt(question, choices)
                answer_text = chr(65 + choice_idx)  # A, B, C, D
                
                full_text = prompt + " " + answer_text
                
                # Tokenize
                inputs = tokenizer(full_text, return_tensors='pt').to(device)
                
                # Get logits
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Get the token ID for the answer letter
                answer_token_id = tokenizer.encode(answer_text, add_special_tokens=False)[0]
                
                # Get probability at the answer position
                answer_pos = inputs['input_ids'].shape[1] - 1
                log_probs = torch.log_softmax(logits[0, answer_pos-1, :], dim=0)
                score = log_probs[answer_token_id].item()
                
                choice_scores.append(score)
            
            # Pick choice with highest score
            predicted = np.argmax(choice_scores)
            correct = (predicted == answer)
            results.append(correct)
            
            # Confidence: difference between top and second choice
            sorted_scores = sorted(choice_scores, reverse=True)
            confidence = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 0
            all_confidences.append(confidence)
    
    accuracy = sum(results) / len(results) if results else 0.0
    
    details = {
        'accuracy': accuracy,
        'num_correct': sum(results),
        'num_total': len(results),
        'mean_confidence': float(np.mean(all_confidences)),
        'std_confidence': float(np.std(all_confidences))
    }
    
    return accuracy, results, details


def evaluate_with_interventions(
    model,
    tokenizer,
    pipeline,
    wmdp_dataset: Dataset,
    mmlu_dataset: Dataset,
    use_refusal: bool = True,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Evaluate model with SAE interventions applied.
    
    Applies the clamping hooks, runs evaluation, then removes hooks.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        pipeline: UnlearningPipeline with interventions setup
        wmdp_dataset: WMDP-Bio dataset
        mmlu_dataset: MMLU dataset
        use_refusal: Whether to use refusal clamping
        max_samples: Optional sample limit
        
    Returns:
        Dictionary with evaluation results
    """
    device = next(model.parameters()).device
    
    # Apply interventions
    pipeline.apply_hooks(use_refusal=use_refusal)
    
    print("Evaluating on WMDP-Bio (with intervention)...")
    wmdp_acc, wmdp_results, wmdp_details = evaluate_multiple_choice(
        model, tokenizer, wmdp_dataset,
        max_samples=max_samples, device=device
    )
    
    print("Evaluating on MMLU (with intervention)...")
    mmlu_acc, mmlu_results, mmlu_details = evaluate_multiple_choice(
        model, tokenizer, mmlu_dataset,
        max_samples=max_samples, device=device
    )
    
    # Remove interventions
    pipeline.remove_hooks()
    
    return {
        'wmdp_accuracy': wmdp_acc,
        'wmdp_details': wmdp_details,
        'mmlu_accuracy': mmlu_acc,
        'mmlu_details': mmlu_details
    }


def run_baseline_evaluation(
    model,
    tokenizer,
    wmdp_dataset: Dataset,
    mmlu_dataset: Dataset,
    max_samples: Optional[int] = None
) -> Dict:
    """
    Evaluate model without any interventions (baseline).
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        wmdp_dataset: WMDP-Bio dataset
        mmlu_dataset: MMLU dataset
        max_samples: Optional sample limit
        
    Returns:
        Dictionary with baseline evaluation results
    """
    device = next(model.parameters()).device
    
    print("Evaluating baseline on WMDP-Bio...")
    wmdp_acc, wmdp_results, wmdp_details = evaluate_multiple_choice(
        model, tokenizer, wmdp_dataset,
        max_samples=max_samples, device=device
    )
    
    print("Evaluating baseline on MMLU...")
    mmlu_acc, mmlu_results, mmlu_details = evaluate_multiple_choice(
        model, tokenizer, mmlu_dataset,
        max_samples=max_samples, device=device
    )
    
    return {
        'wmdp_accuracy': wmdp_acc,
        'wmdp_details': wmdp_details,
        'mmlu_accuracy': mmlu_acc,
        'mmlu_details': mmlu_details
    }


def hyperparameter_sweep(
    model,
    tokenizer,
    pipeline,
    wmdp_dataset,
    mmlu_dataset,
    param_name: str,
    param_values: List,
    baseline_wmdp: float,
    baseline_mmlu: float,
    max_samples: Optional[int] = 50
) -> List[Dict]:
    """
    Sweep over a hyperparameter and evaluate.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        pipeline: UnlearningPipeline
        wmdp_dataset: WMDP-Bio dataset
        mmlu_dataset: MMLU dataset
        param_name: Parameter to sweep ('top_k_features', 'clamp_coefficient', etc.)
        param_values: List of values to try
        baseline_wmdp: Baseline WMDP accuracy
        baseline_mmlu: Baseline MMLU accuracy
        max_samples: Sample limit for faster sweeps
        
    Returns:
        List of result dictionaries
    """
    from .metrics import alignment_metric
    
    results = []
    
    for value in tqdm(param_values, desc=f"Sweeping {param_name}"):
        print(f"\nTrying {param_name}={value}")
        
        # Update config based on parameter
        if param_name == 'top_k_features':
            pipeline.config.top_k_features = value
            # Re-identify features with new top_k
            layer_idx = pipeline.layer_indices[0]
            harmful_features, refusal_feature = pipeline.identify_features(
                layer_idx,
                pipeline.forget_acts[layer_idx],
                pipeline.retain_acts[layer_idx],
                refusal_data=None
            )
            pipeline.setup_interventions(layer_idx, harmful_features, refusal_feature)
        
        elif param_name == 'clamp_coefficient':
            pipeline.config.clamp_coefficient = value
            for interventor in pipeline.interventors.values():
                interventor.config.clamp_coefficient = value
        
        elif param_name == 'refusal_coefficient':
            pipeline.config.refusal_coefficient = value
            for interventor in pipeline.interventors.values():
                interventor.config.refusal_coefficient = value
        
        # Evaluate
        eval_results = evaluate_with_interventions(
            model, tokenizer, pipeline,
            wmdp_dataset, mmlu_dataset,
            use_refusal=True,
            max_samples=max_samples
        )
        
        # Compute alignment
        alignment, R_good, R_bad = alignment_metric(
            eval_results['mmlu_accuracy'],
            baseline_mmlu,
            eval_results['wmdp_accuracy'],
            baseline_wmdp
        )
        
        results.append({
            'param_value': value,
            'wmdp_acc': eval_results['wmdp_accuracy'],
            'mmlu_acc': eval_results['mmlu_accuracy'],
            'alignment': alignment,
            'R_good': R_good,
            'R_bad': R_bad
        })
    
    return results

