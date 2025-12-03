"""
LM Evaluation Harness wrapper module.

Provides integration with EleutherAI's lm-evaluation-harness for
standardized evaluation on WMDP and MMLU benchmarks.
"""

import os
import json
from typing import Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .metrics import alignment_metric

# Try to import lm_eval - it's optional
try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
    LM_EVAL_AVAILABLE = True
except ImportError:
    LM_EVAL_AVAILABLE = False
    HFLM = object  # Placeholder for type hints


class InterventionWrapper(HFLM if LM_EVAL_AVAILABLE else object):
    """
    Wrapper around HuggingFace model that applies SAE interventions during evaluation.
    
    Integrates with lm-eval harness by extending HFLM and applying
    clamping hooks during model calls.
    
    Attributes:
        pipeline: UnlearningPipeline with interventions setup
        use_refusal: Whether to use refusal intervention
        hooks_applied: Whether hooks are currently active
    """
    
    def __init__(
        self,
        pretrained: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        pipeline=None,
        use_refusal: bool = False,
        **kwargs
    ):
        """
        Initialize the wrapper.
        
        Args:
            pretrained: The base HuggingFace model
            tokenizer: Tokenizer
            pipeline: UnlearningPipeline with interventions setup
            use_refusal: Whether to use refusal intervention
            **kwargs: Additional arguments for HFLM
        """
        if not LM_EVAL_AVAILABLE:
            raise ImportError(
                "lm_eval is not installed. Install with: pip install lm-eval"
            )
        
        super().__init__(
            pretrained=pretrained,
            tokenizer=tokenizer,
            **kwargs
        )
        
        self.pipeline = pipeline
        self.use_refusal = use_refusal
        self.hooks_applied = False
    
    def _model_call(self, inps, attn_mask=None, labels=None):
        """Override model call to apply interventions."""
        # Apply hooks if pipeline is provided and not already applied
        if self.pipeline is not None and not self.hooks_applied:
            self.pipeline.apply_hooks(use_refusal=self.use_refusal)
            self.hooks_applied = True
        
        # Call the original model
        return super()._model_call(inps, attn_mask=attn_mask, labels=labels)
    
    def cleanup(self):
        """Remove hooks after evaluation."""
        if self.pipeline is not None and self.hooks_applied:
            self.pipeline.remove_hooks()
            self.hooks_applied = False


def evaluate_with_lm_eval(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pipeline=None,
    use_refusal: bool = False,
    batch_size: int = 8,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate model using lm-eval harness on WMDP-Bio and MMLU subsets.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        pipeline: UnlearningPipeline (None for baseline)
        use_refusal: Whether to use refusal intervention
        batch_size: Evaluation batch size
        device: Device to use
        
    Returns:
        Dictionary with all evaluation results
    """
    if not LM_EVAL_AVAILABLE:
        raise ImportError(
            "lm_eval is not installed. Install with: pip install lm-eval"
        )
    
    # Define tasks exactly as in paper
    mmlu_tasks = [
        "mmlu_high_school_us_history",
        "mmlu_high_school_geography",
        "mmlu_human_aging",
        "mmlu_college_computer_science"
    ]
    
    wmdp_task = "wmdp_bio"
    
    all_tasks = mmlu_tasks + [wmdp_task]
    
    # Create wrapper model
    lm_obj = InterventionWrapper(
        pretrained=model,
        tokenizer=tokenizer,
        pipeline=pipeline,
        use_refusal=use_refusal,
        batch_size=batch_size,
        device=device
    )
    
    method_name = 'Baseline' if pipeline is None else ('Refusal Boost' if use_refusal else 'Clamp Prime')
    print(f"\n{'='*70}")
    print(f"Running LM Evaluation Harness")
    print(f"Intervention: {method_name}")
    print(f"{'='*70}\n")
    
    # Run evaluation
    results = evaluator.simple_evaluate(
        model=lm_obj,
        tasks=all_tasks,
        batch_size=batch_size,
        log_samples=False,
        device=device
    )
    
    # Cleanup hooks
    lm_obj.cleanup()
    
    return results


def compute_weighted_mmlu(results: Dict) -> tuple:
    """
    Compute weighted average MMLU accuracy as in paper.
    
    Weights (number of questions):
        - High School History: 204
        - High School Geography: 198
        - Human Aging: 223
        - College Computer Science: 100
    Total: 725 questions
    
    Args:
        results: Output from lm-eval harness
        
    Returns:
        Tuple of (weighted_accuracy, individual_accuracies dict)
    """
    weights = {
        "mmlu_high_school_us_history": 204,
        "mmlu_high_school_geography": 198,
        "mmlu_human_aging": 223,
        "mmlu_college_computer_science": 100
    }
    
    total_weight = sum(weights.values())
    
    individual_accs = {}
    weighted_sum = 0.0
    
    for task_name, weight in weights.items():
        if task_name in results.get("results", {}):
            acc = results["results"][task_name].get("acc,none", 0.0)
            individual_accs[task_name] = acc
            weighted_sum += acc * weight
            print(f"  {task_name}: {acc:.4f} (weight={weight})")
        else:
            print(f"  Warning: {task_name} not found in results")
    
    weighted_acc = weighted_sum / total_weight
    print(f"\n  Weighted MMLU Accuracy: {weighted_acc:.4f}")
    
    return weighted_acc, individual_accs


def extract_wmdp_accuracy(results: Dict) -> float:
    """
    Extract WMDP-Bio accuracy from results.
    
    Args:
        results: Output from lm-eval harness
        
    Returns:
        WMDP-Bio accuracy
    """
    if "wmdp_bio" in results.get("results", {}):
        acc = results["results"]["wmdp_bio"].get("acc,none", 0.0)
        print(f"  WMDP-Bio Accuracy: {acc:.4f} (1,273 questions)")
        return acc
    else:
        print("  Warning: wmdp_bio not found in results")
        return 0.0


def run_full_evaluation_suite(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    pipeline=None,
    batch_size: int = 8,
    device: str = "cuda",
    output_dir: str = "eval_results"
) -> Dict:
    """
    Run complete evaluation suite: baseline, clamp_prime, and refusal methods.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        pipeline: UnlearningPipeline (optional)
        batch_size: Batch size for evaluation
        device: Device to use
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all results and alignment metrics
    """
    if not LM_EVAL_AVAILABLE:
        raise ImportError(
            "lm_eval is not installed. Install with: pip install lm-eval"
        )
    
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # 1. Baseline Evaluation
    print("\n" + "="*70)
    print("BASELINE EVALUATION")
    print("="*70)
    
    baseline_results = evaluate_with_lm_eval(
        model=model,
        tokenizer=tokenizer,
        pipeline=None,
        batch_size=batch_size,
        device=device
    )
    
    baseline_mmlu, baseline_mmlu_breakdown = compute_weighted_mmlu(baseline_results)
    baseline_wmdp = extract_wmdp_accuracy(baseline_results)
    
    all_results["baseline"] = {
        "mmlu_weighted": baseline_mmlu,
        "mmlu_breakdown": baseline_mmlu_breakdown,
        "wmdp_bio": baseline_wmdp,
    }
    
    # 2. Clamp Prime Evaluation
    if pipeline is not None:
        print("\n" + "="*70)
        print("CLAMP PRIME EVALUATION")
        print("="*70)
        
        clamp_results = evaluate_with_lm_eval(
            model=model,
            tokenizer=tokenizer,
            pipeline=pipeline,
            use_refusal=False,
            batch_size=batch_size,
            device=device
        )
        
        clamp_mmlu, clamp_mmlu_breakdown = compute_weighted_mmlu(clamp_results)
        clamp_wmdp = extract_wmdp_accuracy(clamp_results)
        
        all_results["clamp_prime"] = {
            "mmlu_weighted": clamp_mmlu,
            "mmlu_breakdown": clamp_mmlu_breakdown,
            "wmdp_bio": clamp_wmdp,
        }
        
        # 3. Refusal Boost Evaluation
        print("\n" + "="*70)
        print("REFUSAL BOOST EVALUATION")
        print("="*70)
        
        refusal_results = evaluate_with_lm_eval(
            model=model,
            tokenizer=tokenizer,
            pipeline=pipeline,
            use_refusal=True,
            batch_size=batch_size,
            device=device
        )
        
        refusal_mmlu, refusal_mmlu_breakdown = compute_weighted_mmlu(refusal_results)
        refusal_wmdp = extract_wmdp_accuracy(refusal_results)
        
        all_results["refusal_boost"] = {
            "mmlu_weighted": refusal_mmlu,
            "mmlu_breakdown": refusal_mmlu_breakdown,
            "wmdp_bio": refusal_wmdp,
        }
        
        # 4. Compute Alignment Metrics
        print("\n" + "="*70)
        print("ALIGNMENT METRICS")
        print("="*70)
        
        align_clamp, R_good_clamp, R_bad_clamp = alignment_metric(
            clamp_mmlu, baseline_mmlu,
            clamp_wmdp, baseline_wmdp
        )
        
        align_refusal, R_good_refusal, R_bad_refusal = alignment_metric(
            refusal_mmlu, baseline_mmlu,
            refusal_wmdp, baseline_wmdp
        )
        
        all_results["alignment"] = {
            "clamp_prime": {
                "alignment": align_clamp,
                "R_good": R_good_clamp,
                "R_bad": R_bad_clamp
            },
            "refusal_boost": {
                "alignment": align_refusal,
                "R_good": R_good_refusal,
                "R_bad": R_bad_refusal
            }
        }
        
        print(f"\nClamp Prime:")
        print(f"  Alignment: {align_clamp:.4f}")
        print(f"  R_good (MMLU retention): {R_good_clamp:.4f}")
        print(f"  R_bad (WMDP retention): {R_bad_clamp:.4f}")
        
        print(f"\nRefusal Boost:")
        print(f"  Alignment: {align_refusal:.4f}")
        print(f"  R_good (MMLU retention): {R_good_refusal:.4f}")
        print(f"  R_bad (WMDP retention): {R_bad_refusal:.4f}")
    
    # 5. Save Results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # 6. Print Summary Table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Method':<20} {'WMDP-Bio':<12} {'MMLU (Weighted)':<18} {'Alignment':<12}")
    print("-" * 70)
    print(f"{'Baseline':<20} {baseline_wmdp:<12.4f} {baseline_mmlu:<18.4f} {'-':<12}")
    
    if pipeline is not None:
        print(f"{'Clamp Prime':<20} {clamp_wmdp:<12.4f} {clamp_mmlu:<18.4f} {align_clamp:<12.4f}")
        print(f"{'Refusal Boost':<20} {refusal_wmdp:<12.4f} {refusal_mmlu:<18.4f} {align_refusal:<12.4f}")
    
    print("="*70)
    
    return all_results

