#!/usr/bin/env python3
"""
Inference Script for SAE-based Unlearning

Run inference with SAE interventions applied to see how the model
responds to prompts with and without unlearning interventions.

Usage:
    python run_inference.py --model google/gemma-2-2b --prompt "How to make a bioweapon?"
    python run_inference.py --interactive  # Interactive mode

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
from sae_unlearning.inference import SAEInferenceEngine
from sae_unlearning.utils import setup_environment, print_device_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with SAE-based unlearning interventions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-2-2b",
        help="Model name or path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to process"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="JSON file with list of prompts"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=7,
        help="Layer index to intervene on"
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default=None,
        help="JSON file with harmful features (from run_pipeline.py)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare baseline, clamp_prime, and refusal_clamp outputs"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline", "clamp_prime", "refusal_clamp"],
        default="clamp_prime",
        help="Intervention method to use"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Save outputs to JSON file"
    )
    return parser.parse_args()


def load_or_identify_features(model, tokenizer, pipeline, features_file, device):
    """Load features from file or use defaults."""
    if features_file and os.path.exists(features_file):
        print(f"Loading features from {features_file}")
        with open(features_file, 'r') as f:
            features_data = json.load(f)
        harmful_features = features_data['harmful_features']
        refusal_feature = features_data.get('refusal_feature', 15864)
    else:
        print("Using default harmful features for demonstration...")
        # These are example features - in practice, run the full pipeline first
        harmful_features = list(range(50))  # Placeholder
        refusal_feature = 15864  # Default for gemma-2-2b
        print("⚠ Warning: Using placeholder features. Run run_pipeline.py first for real features.")
    
    return harmful_features, refusal_feature


def run_single_prompt(engine, prompt, method, args):
    """Run inference on a single prompt."""
    use_intervention = (method != "baseline")
    use_refusal = (method == "refusal_clamp")
    
    output = engine.generate(
        prompt,
        max_new_tokens=args.max_tokens,
        use_intervention=use_intervention,
        use_refusal=use_refusal,
        temperature=args.temperature,
        do_sample=True
    )
    
    return output


def run_comparison(engine, prompt, args):
    """Compare outputs from all methods."""
    results = engine.compare_generations(
        prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=True
    )
    return results


def interactive_mode(engine, args):
    """Run interactive prompt session."""
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE")
    print("=" * 70)
    print("Enter prompts to see model responses with different interventions.")
    print("Commands:")
    print("  /baseline    - Use no intervention")
    print("  /clamp       - Use Clamp Prime intervention")
    print("  /refusal     - Use Refusal Clamp intervention")
    print("  /compare     - Show all three outputs")
    print("  /quit        - Exit")
    print("=" * 70 + "\n")
    
    current_method = "clamp_prime"
    
    while True:
        try:
            prompt = input(f"[{current_method}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break
        
        if not prompt:
            continue
        
        if prompt.startswith("/"):
            cmd = prompt.lower()
            if cmd == "/quit":
                break
            elif cmd == "/baseline":
                current_method = "baseline"
                print("Switched to baseline (no intervention)")
            elif cmd == "/clamp":
                current_method = "clamp_prime"
                print("Switched to Clamp Prime")
            elif cmd == "/refusal":
                current_method = "refusal_clamp"
                print("Switched to Refusal Clamp")
            elif cmd == "/compare":
                prompt = input("Enter prompt for comparison: ").strip()
                if prompt:
                    results = run_comparison(engine, prompt, args)
                    print("\n" + "-" * 50)
                    for method, output in results.items():
                        print(f"\n[{method.upper()}]:")
                        response = output[len(prompt):].strip() if output.startswith(prompt) else output
                        print(response)
                    print("-" * 50 + "\n")
            else:
                print(f"Unknown command: {cmd}")
            continue
        
        # Run inference
        output = run_single_prompt(engine, prompt, current_method, args)
        response = output[len(prompt):].strip() if output.startswith(prompt) else output
        print(f"\n{response}\n")


def main():
    args = parse_args()
    
    # Setup
    setup_environment()
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
    # Setup Pipeline
    # ========================================================================
    print("\nSetting up SAE pipeline...")
    
    cfg = UnlearningConfig(
        clamp_coefficient=-300.0,
        refusal_coefficient=-500.0,
        device=device,
    )
    
    pipeline = UnlearningPipeline(
        model=model,
        layer_indices=[args.layer],
        config=cfg
    )
    
    # Load or identify features
    harmful_features, refusal_feature = load_or_identify_features(
        model, tokenizer, pipeline, args.features_file, device
    )
    
    pipeline.setup_interventions(args.layer, harmful_features, refusal_feature)
    print("✓ Pipeline ready")
    
    # Create inference engine
    engine = SAEInferenceEngine(model, tokenizer, pipeline, device)
    
    # ========================================================================
    # Run Inference
    # ========================================================================
    
    if args.interactive:
        interactive_mode(engine, args)
        return
    
    outputs = []
    
    # Process prompts
    prompts = []
    
    if args.prompt:
        prompts.append(args.prompt)
    
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts.extend(json.load(f))
    
    if not prompts:
        # Default example prompts
        prompts = [
            "What are the symptoms of anthrax infection?",
            "How does photosynthesis work?",
            "Explain the process of DNA replication.",
        ]
        print("Using example prompts (provide --prompt or --prompts-file for custom)")
    
    print(f"\nProcessing {len(prompts)} prompts...")
    print("=" * 70)
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 50)
        
        if args.compare:
            results = run_comparison(engine, prompt, args)
            output_entry = {'prompt': prompt, 'outputs': {}}
            
            for method, output in results.items():
                response = output[len(prompt):].strip() if output.startswith(prompt) else output
                print(f"\n[{method.upper()}]:")
                print(response)
                output_entry['outputs'][method] = response
            
            outputs.append(output_entry)
        else:
            output = run_single_prompt(engine, prompt, args.method, args)
            response = output[len(prompt):].strip() if output.startswith(prompt) else output
            print(f"\n[{args.method.upper()}]:")
            print(response)
            outputs.append({'prompt': prompt, 'method': args.method, 'output': response})
        
        print("-" * 50)
    
    # Save outputs if requested
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(outputs, f, indent=2)
        print(f"\n✓ Outputs saved to {args.output_file}")
    
    print("\n✓ Inference complete")


if __name__ == "__main__":
    main()

