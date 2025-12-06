#!/usr/bin/env python3
"""
Run lm-evaluation-harness with SAE clamping hook applied.

Uses EleutherAI's standardized evaluation on WMDP and MMLU.

Usage:
    python run_lm_eval_with_sae.py --tasks mmlu --limit 50
    python run_lm_eval_with_sae.py --tasks wmdp_bio --limit 100
    python run_lm_eval_with_sae.py --tasks mmlu,wmdp_bio --no_clamping
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sae_lens import SAE
from sae_unlearning.models.sae_wrapper import GemmaScopeWrapper

import lm_eval
from lm_eval.models.huggingface import HFLM


# ============================================================================
# SAE Clamping Hook
# ============================================================================

class ClampPrimeHook:
    """Forward hook that applies Clamp Prime intervention."""
    
    def __init__(
        self,
        sae_wrapper: GemmaScopeWrapper,
        harmful_features: list,
        clamp_coefficient: float = 0.0,
        device: torch.device = None
    ):
        self.sae_wrapper = sae_wrapper
        self.harmful_features = harmful_features
        self.clamp_coefficient = clamp_coefficient
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __call__(self, module, input, output):
        """Apply clamping intervention to layer output."""
        if isinstance(output, tuple):
            activations = output[0]
        else:
            activations = output
        
        original_dtype = activations.dtype
        original_device = activations.device
        
        # Encode to SAE latent space
        latents = self.sae_wrapper.encode(activations)
        
        # Compute reconstruction error BEFORE clamping
        reconstruction = self.sae_wrapper.decode(latents)
        reconstruction = reconstruction.to(dtype=original_dtype, device=original_device)
        reconstruction_error = activations - reconstruction
        
        # Apply clamping to harmful features (unconditionally)
        for feat_idx in self.harmful_features:
            latents[..., feat_idx] = self.clamp_coefficient
        
        # Decode back to activation space and ADD error back
        modified_activations = self.sae_wrapper.decode(latents)
        modified_activations = modified_activations.to(dtype=original_dtype, device=original_device)
        modified_activations = modified_activations + reconstruction_error
        
        # Modify in-place
        activations.copy_(modified_activations)
        return None


def load_features_to_clamp(filepath: str) -> list:
    """Load feature indices to clamp from a text file."""
    features = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                features.append(int(line))
    return features


def main():
    parser = argparse.ArgumentParser(description="Run lm-eval with SAE clamping")
    parser.add_argument("--model_name", type=str, default="google/gemma-2-2b")
    parser.add_argument("--tasks", type=str, default="mmlu", 
                        help="Comma-separated list of tasks (e.g., mmlu,wmdp_bio)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples per task")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sae_layer", type=int, default=7)
    parser.add_argument("--clamp_coefficient", type=float, default=-300.0)
    parser.add_argument("--features_file", type=str, default=None)
    parser.add_argument("--no_clamping", action="store_true",
                        help="Run without SAE clamping (baseline)")
    parser.add_argument("--output_path", type=str, default="lm_eval_results")
    args = parser.parse_args()
    
    # Setup
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.features_file:
        features_file = Path(args.features_file)
    else:
        features_file = project_root / "frequencies_second_time" / "features_to_clamp_layer7.txt"
    
    print("=" * 70)
    print("LM-EVAL WITH SAE CLAMPING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model:            {args.model_name}")
    print(f"  Tasks:            {args.tasks}")
    print(f"  Limit:            {args.limit}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  SAE Layer:        {args.sae_layer}")
    print(f"  Clamp coeff:      {args.clamp_coefficient}")
    print(f"  No clamping:      {args.no_clamping}")
    print(f"  Features file:    {features_file}")
    
    # Load model
    print("\n" + "-" * 70)
    print("LOADING MODEL")
    print("-" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    hook_handle = None
    
    if not args.no_clamping:
        # Load SAE and setup hook
        print("\n" + "-" * 70)
        print("LOADING SAE AND SETTING UP HOOK")
        print("-" * 70)
        
        harmful_features = load_features_to_clamp(str(features_file))
        print(f"  Loaded {len(harmful_features)} features to clamp")
        
        sae_id = f"layer_{args.sae_layer}/width_16k/canonical"
        print(f"  Loading SAE: {sae_id}...")
        
        sae_result = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id=sae_id,
        )
        if isinstance(sae_result, tuple):
            sae_model = sae_result[0]
        else:
            sae_model = sae_result
        
        sae_wrapper = GemmaScopeWrapper(sae_model, device=torch.device(device))
        print(f"  SAE loaded: d_model={sae_wrapper.d_model}, d_sae={sae_wrapper.d_sae}")
        
        clamp_hook = ClampPrimeHook(
            sae_wrapper=sae_wrapper,
            harmful_features=harmful_features,
            clamp_coefficient=args.clamp_coefficient,
            device=torch.device(device)
        )
        
        target_layer = model.model.layers[args.sae_layer]
        hook_handle = target_layer.register_forward_hook(clamp_hook)
        print(f"  Hook registered on layer {args.sae_layer}")
    
    # Wrap model for lm-eval
    print("\n" + "-" * 70)
    print("RUNNING LM-EVAL")
    print("-" * 70)
    
    # Create HFLM wrapper
    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
    )
    
    # Parse tasks
    task_list = [t.strip() for t in args.tasks.split(",")]
    
    # Run evaluation
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=task_list,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    
    # Remove hook if applied
    if hook_handle:
        hook_handle.remove()
        print("  Hook removed")
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    for task_name, task_results in results["results"].items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    # Save results
    import json
    output_file = Path(args.output_path) / "results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()

