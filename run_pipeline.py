#!/usr/bin/env python3
"""
Main Pipeline Script for SAE-based Unlearning

Runs the complete unlearning pipeline:
1. Load model and datasets
2. Collect activations
3. Identify harmful features
4. Setup interventions
5. Evaluate and visualize results

Usage:
    python run_pipeline.py --model google/gemma-2-2b --max-samples 500

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
    alignment_metric,
)
from sae_unlearning.visualization import (
    plot_accuracy_comparison,
    plot_pareto_frontier,
    plot_sae_reconstruction_quality,
)
from sae_unlearning.utils import (
    setup_environment,
    collect_activations_for_texts,
    print_device_info,
    save_results,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SAE-based unlearning pipeline"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b",
        help="Model name or path (default: google/gemma-2-2b)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=7,
        help="Layer index to intervene on (default: 7)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples to use (default: 500)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top harmful features to identify (default: 50)"
    )
    parser.add_argument(
        "--clamp-coef",
        type=float,
        default=-300.0,
        help="Clamping coefficient (default: -300.0)"
    )
    parser.add_argument(
        "--refusal-coef",
        type=float,
        default=-500.0,
        help="Refusal coefficient (default: -500.0)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not specified)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for activation collection (default: 4)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save checkpoints and results"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup
    output_dir = setup_environment(output_dir=args.output_dir)
    print_device_info()
    
    print("=" * 70)
    print("SAE CONDITIONAL CLAMPING UNLEARNING - FULL PIPELINE")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = UnlearningConfig(
        activation_threshold=0.05,
        clamp_coefficient=args.clamp_coef,
        refusal_coefficient=args.refusal_coef,
        top_k_features=args.top_k,
        retain_frequency_threshold=1e-4,
        device=device,
    )
    
    layer_indices = [args.layer]
    
    # ========================================================================
    # STEP 1: Load Model
    # ========================================================================
    print(f"\n[1/7] Loading model: {args.model}...")
    
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
    
    d_model = model.config.hidden_size
    print(f"✓ Model loaded: d_model={d_model}, intervening on layers {layer_indices}")
    
    # ========================================================================
    # STEP 2: Load Datasets
    # ========================================================================
    print("\n[2/7] Loading datasets...")
    
    wmdp_dataset = WMDPDataset(split='test')
    mmlu_dataset = MMLUDataset(split='test')
    
    forget_texts = wmdp_dataset.get_questions()[:args.max_samples]
    retain_texts = mmlu_dataset.get_questions()[:args.max_samples]
    
    print(f"✓ Forget set: {len(forget_texts)} samples")
    print(f"✓ Retain set: {len(retain_texts)} samples")
    
    # ========================================================================
    # STEP 3: Collect Activations
    # ========================================================================
    print("\n[3/7] Collecting activations...")
    
    forget_acts = collect_activations_for_texts(
        model, tokenizer, forget_texts, layer_indices,
        batch_size=args.batch_size, device=device
    )
    
    retain_acts = collect_activations_for_texts(
        model, tokenizer, retain_texts, layer_indices,
        batch_size=args.batch_size, device=device
    )
    
    layer_idx = layer_indices[0]
    print(f"✓ Forget activations: {forget_acts[layer_idx].shape}")
    print(f"✓ Retain activations: {retain_acts[layer_idx].shape}")
    
    if not args.no_save:
        torch.save({
            'forget_acts': forget_acts,
            'retain_acts': retain_acts
        }, os.path.join(output_dir, 'activations.pt'))
        print(f"✓ Saved activations to {output_dir}/activations.pt")
    
    # ========================================================================
    # STEP 4: Initialize Pipeline and Load SAEs
    # ========================================================================
    print("\n[4/7] Loading SAEs...")
    
    pipeline = UnlearningPipeline(
        model=model,
        layer_indices=layer_indices,
        config=cfg
    )
    pipeline.forget_acts = forget_acts
    pipeline.retain_acts = retain_acts
    
    # Visualize SAE quality
    combined_acts = torch.cat([forget_acts[layer_idx], retain_acts[layer_idx]], dim=0)
    plot_sae_reconstruction_quality(
        pipeline.saes[str(layer_idx)],
        combined_acts.to(device),
        save_path=os.path.join(output_dir, "sae_reconstruction.png")
    )
    
    # ========================================================================
    # STEP 5: Identify Features
    # ========================================================================
    print("\n[5/7] Identifying harmful features...")
    
    harmful_features, refusal_feature = pipeline.identify_features(
        layer_idx,
        forget_acts[layer_idx],
        retain_acts[layer_idx],
        refusal_data=None
    )
    
    print(f"✓ Found {len(harmful_features)} harmful features")
    print(f"  Top 10: {harmful_features[:10]}")
    print(f"✓ Refusal feature: {refusal_feature}")
    
    if not args.no_save:
        with open(os.path.join(output_dir, 'features.json'), 'w') as f:
            json.dump({
                'harmful_features': harmful_features,
                'refusal_feature': refusal_feature,
                'num_harmful': len(harmful_features)
            }, f, indent=2)
    
    # ========================================================================
    # STEP 6: Setup Interventions and Evaluate
    # ========================================================================
    print("\n[6/7] Setting up interventions and evaluating...")
    
    pipeline.setup_interventions(layer_idx, harmful_features, refusal_feature)
    
    all_results = {}
    
    # Baseline
    print("  Evaluating baseline...")
    baseline_wmdp, _, _ = evaluate_multiple_choice(
        model, tokenizer, wmdp_dataset,
        max_samples=args.max_samples, device=device
    )
    baseline_mmlu, _, _ = evaluate_multiple_choice(
        model, tokenizer, mmlu_dataset,
        max_samples=args.max_samples, device=device
    )
    all_results['Baseline'] = {'WMDP-Bio': baseline_wmdp, 'MMLU': baseline_mmlu}
    
    # Clamp Prime
    print("  Evaluating Clamp Prime...")
    pipeline.apply_hooks(use_refusal=False)
    clamp_prime_wmdp, _, _ = evaluate_multiple_choice(
        model, tokenizer, wmdp_dataset,
        max_samples=args.max_samples, device=device
    )
    clamp_prime_mmlu, _, _ = evaluate_multiple_choice(
        model, tokenizer, mmlu_dataset,
        max_samples=args.max_samples, device=device
    )
    pipeline.remove_hooks()
    all_results['Clamp Prime'] = {'WMDP-Bio': clamp_prime_wmdp, 'MMLU': clamp_prime_mmlu}
    
    # Refusal Clamp
    print("  Evaluating Refusal Clamp...")
    pipeline.apply_hooks(use_refusal=True)
    refusal_wmdp, _, _ = evaluate_multiple_choice(
        model, tokenizer, wmdp_dataset,
        max_samples=args.max_samples, device=device
    )
    refusal_mmlu, _, _ = evaluate_multiple_choice(
        model, tokenizer, mmlu_dataset,
        max_samples=args.max_samples, device=device
    )
    pipeline.remove_hooks()
    all_results['Refusal Clamp'] = {'WMDP-Bio': refusal_wmdp, 'MMLU': refusal_mmlu}
    
    # Compute alignment
    alignment_cp, R_good_cp, R_bad_cp = alignment_metric(
        clamp_prime_mmlu, baseline_mmlu, clamp_prime_wmdp, baseline_wmdp
    )
    alignment_rc, R_good_rc, R_bad_rc = alignment_metric(
        refusal_mmlu, baseline_mmlu, refusal_wmdp, baseline_wmdp
    )
    
    print("✓ Evaluation complete")
    
    # ========================================================================
    # STEP 7: Generate Visualizations
    # ========================================================================
    print("\n[7/7] Generating visualizations...")
    
    plot_accuracy_comparison(
        all_results,
        save_path=os.path.join(output_dir, "accuracy_comparison.png")
    )
    
    pareto_points = [
        {'method': 'Baseline', 'wmdp': baseline_wmdp, 'mmlu': baseline_mmlu},
        {'method': 'Clamp Prime', 'wmdp': clamp_prime_wmdp, 'mmlu': clamp_prime_mmlu},
        {'method': 'Refusal Clamp', 'wmdp': refusal_wmdp, 'mmlu': refusal_mmlu}
    ]
    plot_pareto_frontier(
        pareto_points,
        save_path=os.path.join(output_dir, "pareto_frontier.png")
    )
    
    print("✓ All visualizations saved")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for method, scores in all_results.items():
        print(f"\n{method}:")
        for dataset, acc in scores.items():
            print(f"  {dataset}: {acc:.1%}")
    
    print(f"\nAlignment Metrics:")
    print(f"  Clamp Prime:  {alignment_cp:.4f} (R_good={R_good_cp:.3f}, R_bad={R_bad_cp:.3f})")
    print(f"  Refusal Clamp: {alignment_rc:.4f} (R_good={R_good_rc:.3f}, R_bad={R_bad_rc:.3f})")
    
    # Save final summary
    if not args.no_save:
        summary = {
            'config': cfg.to_dict(),
            'results': all_results,
            'alignment': {
                'clamp_prime': {'alignment': alignment_cp, 'R_good': R_good_cp, 'R_bad': R_bad_cp},
                'refusal_clamp': {'alignment': alignment_rc, 'R_good': R_good_rc, 'R_bad': R_bad_rc}
            },
            'harmful_features': harmful_features[:20]
        }
        save_results(summary, output_dir, 'summary.json')
    
    print(f"\n✓ All results saved to: {output_dir}/")
    print("=" * 70)
    
    return pipeline, all_results


if __name__ == "__main__":
    main()

