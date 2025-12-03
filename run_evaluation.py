#!/usr/bin/env python3
"""
Evaluation Script for SAE-based Unlearning

Run standalone evaluation on WMDP-Bio and MMLU benchmarks with or without
SAE interventions.

Usage:
    python run_evaluation.py --model google/gemma-2-2b --max-samples 100
    python run_evaluation.py --use-lm-eval  # Use EleutherAI harness

Based on: "Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning"
https://arxiv.org/pdf/2503.11127
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_unlearning.config import UnlearningConfig
from sae_unlearning.pipeline import UnlearningPipeline
from sae_unlearning.data import WMDPDataset, MMLUDataset
from sae_unlearning.evaluation import (
    evaluate_multiple_choice,
    run_baseline_evaluation,
    evaluate_with_interventions,
    alignment_metric,
    format_metrics_report,
)
from sae_unlearning.utils import setup_environment, print_device_info, save_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model on WMDP-Bio and MMLU benchmarks"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b",
        help="Model name or path"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=7,
        help="Layer index to intervene on"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (None for all)"
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=None,
        help="JSON file with harmful features"
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only run baseline evaluation"
    )
    parser.add_argument(
        "--use-lm-eval",
        action="store_true",
        help="Use EleutherAI lm-evaluation-harness"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for lm-eval"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for results"
    )
    return parser.parse_args()


def load_features(features_file):
    """Load features from file."""
    if features_file and os.path.exists(features_file):
        with open(features_file, 'r') as f:
            data = json.load(f)
        return data['harmful_features'], data.get('refusal_feature', 15864)
    return None, None


def run_simple_evaluation(model, tokenizer, pipeline, wmdp_dataset, mmlu_dataset, 
                          harmful_features, refusal_feature, max_samples, device, output_dir):
    """Run evaluation using built-in evaluator."""
    results = {}
    
    # Baseline evaluation
    print("\n" + "=" * 70)
    print("BASELINE EVALUATION")
    print("=" * 70)
    
    baseline = run_baseline_evaluation(
        model, tokenizer, wmdp_dataset, mmlu_dataset,
        max_samples=max_samples
    )
    results['baseline'] = baseline
    
    print(f"\nBaseline Results:")
    print(f"  WMDP-Bio: {baseline['wmdp_accuracy']:.4f}")
    print(f"  MMLU: {baseline['mmlu_accuracy']:.4f}")
    
    # If no features provided, we're done
    if harmful_features is None:
        print("\nNo harmful features provided. Skipping intervention evaluation.")
        return results
    
    # Setup interventions
    layer_idx = pipeline.layer_indices[0]
    pipeline.setup_interventions(layer_idx, harmful_features, refusal_feature)
    
    # Clamp Prime evaluation
    print("\n" + "=" * 70)
    print("CLAMP PRIME EVALUATION")
    print("=" * 70)
    
    clamp_results = evaluate_with_interventions(
        model, tokenizer, pipeline,
        wmdp_dataset, mmlu_dataset,
        use_refusal=False,
        max_samples=max_samples
    )
    results['clamp_prime'] = clamp_results
    
    print(f"\nClamp Prime Results:")
    print(f"  WMDP-Bio: {clamp_results['wmdp_accuracy']:.4f}")
    print(f"  MMLU: {clamp_results['mmlu_accuracy']:.4f}")
    
    # Refusal Clamp evaluation
    print("\n" + "=" * 70)
    print("REFUSAL CLAMP EVALUATION")
    print("=" * 70)
    
    refusal_results = evaluate_with_interventions(
        model, tokenizer, pipeline,
        wmdp_dataset, mmlu_dataset,
        use_refusal=True,
        max_samples=max_samples
    )
    results['refusal_clamp'] = refusal_results
    
    print(f"\nRefusal Clamp Results:")
    print(f"  WMDP-Bio: {refusal_results['wmdp_accuracy']:.4f}")
    print(f"  MMLU: {refusal_results['mmlu_accuracy']:.4f}")
    
    # Compute alignment metrics
    align_cp, R_good_cp, R_bad_cp = alignment_metric(
        clamp_results['mmlu_accuracy'], baseline['mmlu_accuracy'],
        clamp_results['wmdp_accuracy'], baseline['wmdp_accuracy']
    )
    
    align_rc, R_good_rc, R_bad_rc = alignment_metric(
        refusal_results['mmlu_accuracy'], baseline['mmlu_accuracy'],
        refusal_results['wmdp_accuracy'], baseline['wmdp_accuracy']
    )
    
    results['alignment'] = {
        'clamp_prime': {'alignment': align_cp, 'R_good': R_good_cp, 'R_bad': R_bad_cp},
        'refusal_clamp': {'alignment': align_rc, 'R_good': R_good_rc, 'R_bad': R_bad_rc}
    }
    
    # Print summary
    print("\n" + "=" * 70)
    print("ALIGNMENT METRICS")
    print("=" * 70)
    print(f"\nClamp Prime:")
    print(f"  Alignment: {align_cp:.4f}")
    print(f"  R_good (MMLU retention): {R_good_cp:.4f}")
    print(f"  R_bad (WMDP retention): {R_bad_cp:.4f}")
    
    print(f"\nRefusal Clamp:")
    print(f"  Alignment: {align_rc:.4f}")
    print(f"  R_good (MMLU retention): {R_good_rc:.4f}")
    print(f"  R_bad (WMDP retention): {R_bad_rc:.4f}")
    
    # Print formatted report
    print("\n" + format_metrics_report(baseline, clamp_results, "Clamp Prime"))
    
    return results


def run_lm_eval_evaluation(model, tokenizer, pipeline, harmful_features, refusal_feature,
                           batch_size, device, output_dir):
    """Run evaluation using EleutherAI lm-evaluation-harness."""
    try:
        from sae_unlearning.evaluation.lm_eval_wrapper import run_full_evaluation_suite
    except ImportError:
        print("Error: lm-eval not installed. Install with: pip install lm-eval")
        return None
    
    # Setup interventions if features provided
    if harmful_features is not None:
        layer_idx = pipeline.layer_indices[0]
        pipeline.setup_interventions(layer_idx, harmful_features, refusal_feature)
    
    results = run_full_evaluation_suite(
        model=model,
        tokenizer=tokenizer,
        pipeline=pipeline if harmful_features is not None else None,
        batch_size=batch_size,
        device=str(device),
        output_dir=output_dir
    )
    
    return results


def main():
    args = parse_args()
    
    # Setup
    output_dir = setup_environment(output_dir=args.output_dir)
    print_device_info()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========================================================================
    # Load Model
    # ========================================================================
    print(f"\nLoading model: {args.model}...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    if not torch.cuda.is_available():
        model = model.to("cpu")
    
    print("✓ Model loaded")
    
    # ========================================================================
    # Load Features (if provided)
    # ========================================================================
    harmful_features, refusal_feature = load_features(args.features_file)
    
    if harmful_features:
        print(f"✓ Loaded {len(harmful_features)} harmful features")
    elif not args.baseline_only:
        print("⚠ No features file provided. Will only run baseline evaluation.")
        print("  Run run_pipeline.py first to identify harmful features.")
    
    # ========================================================================
    # Setup Pipeline
    # ========================================================================
    cfg = UnlearningConfig(
        clamp_coefficient=-300.0,
        refusal_coefficient=-500.0,
        device=device,
    )
    
    pipeline = None
    if harmful_features and not args.baseline_only:
        print("\nSetting up SAE pipeline...")
        pipeline = UnlearningPipeline(
            model=model,
            layer_indices=[args.layer],
            config=cfg
        )
        print("✓ Pipeline ready")
    
    # ========================================================================
    # Run Evaluation
    # ========================================================================
    
    if args.use_lm_eval:
        print("\nRunning evaluation with lm-evaluation-harness...")
        results = run_lm_eval_evaluation(
            model, tokenizer, pipeline,
            harmful_features, refusal_feature,
            args.batch_size, device, output_dir
        )
    else:
        print("\nLoading datasets...")
        wmdp_dataset = WMDPDataset(split='test')
        mmlu_dataset = MMLUDataset(split='test')
        
        print(f"  WMDP-Bio: {len(wmdp_dataset)} samples")
        print(f"  MMLU: {len(mmlu_dataset)} samples")
        
        if args.baseline_only:
            harmful_features = None
        
        results = run_simple_evaluation(
            model, tokenizer, pipeline,
            wmdp_dataset, mmlu_dataset,
            harmful_features, refusal_feature,
            args.max_samples, device, output_dir
        )
    
    # Save results
    if results:
        save_results(results, output_dir, 'evaluation_results.json')
    
    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Method':<20} {'WMDP-Bio':<15} {'MMLU':<15} {'Alignment':<12}")
    print("-" * 70)
    
    if 'baseline' in results:
        baseline = results['baseline']
        wmdp = baseline.get('wmdp_accuracy', baseline.get('wmdp_bio', 0))
        mmlu = baseline.get('mmlu_accuracy', baseline.get('mmlu_weighted', 0))
        print(f"{'Baseline':<20} {wmdp:<15.4f} {mmlu:<15.4f} {'-':<12}")
    
    if 'clamp_prime' in results:
        cp = results['clamp_prime']
        wmdp = cp.get('wmdp_accuracy', cp.get('wmdp_bio', 0))
        mmlu = cp.get('mmlu_accuracy', cp.get('mmlu_weighted', 0))
        align = results.get('alignment', {}).get('clamp_prime', {}).get('alignment', 0)
        print(f"{'Clamp Prime':<20} {wmdp:<15.4f} {mmlu:<15.4f} {align:<12.4f}")
    
    if 'refusal_clamp' in results:
        rc = results['refusal_clamp']
        wmdp = rc.get('wmdp_accuracy', rc.get('wmdp_bio', 0))
        mmlu = rc.get('mmlu_accuracy', rc.get('mmlu_weighted', 0))
        align = results.get('alignment', {}).get('refusal_clamp', {}).get('alignment', 0)
        print(f"{'Refusal Clamp':<20} {wmdp:<15.4f} {mmlu:<15.4f} {align:<12.4f}")
    
    print("=" * 70)
    print(f"\n✓ Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()

