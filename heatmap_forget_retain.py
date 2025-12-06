#!/usr/bin/env python3
"""
Create heatmap of SAE feature activations for layer 7 on WMDP Bio Forget vs Bio Retain datasets.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sae_lens import SAE
from typing import List, Optional

# =========================================================================
# Configuration
# =========================================================================
LAYER = 7
MODEL_NAME = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
FEATURES_FILE = "/home/ubuntu/ML_Unlearning/frequencies_second_time_7/features_to_clamp_layer7.txt"
OUTPUT_DIR = "/home/ubuntu/ML_Unlearning/sae_activations"
MAX_LENGTH = 128
BATCH_SIZE = 4
MAX_SAMPLES = 2000  # Match sae_feature_analysis.py


# =========================================================================
# SAE Activation Collector
# =========================================================================
class SAEActivationCollector:
    """Collects SAE-encoded activations for all tokens in a sequence."""
    
    def __init__(self, model, sae, layer_idx: int, device: torch.device):
        self.model = model
        self.sae = sae
        self.layer_idx = layer_idx
        self.device = device
        self.hook_handle = None
        self.captured_activations = None
        
    def _get_layer(self):
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[self.layer_idx]
        raise RuntimeError("Could not find transformer layers")
    
    def _hook_fn(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        with torch.no_grad():
            batch_size, seq_len, hidden_dim = hidden_states.shape
            flat_hidden = hidden_states.reshape(-1, hidden_dim)
            sae_latents = self.sae.encode(flat_hidden)
            d_sae = sae_latents.shape[-1]
            sae_latents = sae_latents.reshape(batch_size, seq_len, d_sae)
            self.captured_activations = sae_latents.cpu()
    
    def register_hook(self):
        layer = self._get_layer()
        self.hook_handle = layer.register_forward_hook(self._hook_fn)
    
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def collect_activations(self, texts: list, tokenizer, max_length: int = 128, batch_size: int = 4) -> torch.Tensor:
        self.model.eval()
        all_activations = []
        
        self.register_hook()
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
                batch_texts = texts[i:i + batch_size]
                
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                batch_activations = self.captured_activations
                all_activations.append(batch_activations)
                
                if i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        finally:
            self.remove_hook()
        
        return torch.cat(all_activations, dim=0)


# =========================================================================
# Dataset Loaders (from sae_feature_analysis.py)
# =========================================================================
def load_wmdp_bio_forget(max_samples: Optional[int] = None) -> List[str]:
    """Load WMDP-Bio forget set (harmful biosecurity knowledge to unlearn)."""
    print("\nLoading WMDP-Bio Forget set...")
    
    ds = load_dataset("cais/wmdp-bio-forget-corpus", split='train')
    
    # Format questions as prompts
    prompts = []
    for item in ds:
        abstract = item["abstract"]
        text = item["text"]
        prompt = f"Abstract: {abstract}\n\nText: {text}\n\n"
        prompts.append(prompt)
    
    print(f"  Loaded {len(prompts)} forget prompts")
    
    if max_samples and max_samples < len(prompts):
        prompts = prompts[:max_samples]
        print(f"  Limited to {max_samples} samples")
    
    return prompts


def load_wmdp_bio_retain(max_samples: Optional[int] = None) -> List[str]:
    """Load WMDP bio-retain-corpus (benign bio knowledge to retain)."""
    print("\nLoading WMDP Bio-Retain-Corpus...")
    
    # Load the bio-retain-corpus from WMDP
    dataset = load_dataset("cais/wmdp-corpora", "bio-retain-corpus", split='train')
    
    prompts = []
    for item in dataset:
        # The retain corpus contains text passages
        text = item['text']
        if text:
            prompts.append(text)
    
    print(f"  Loaded {len(prompts)} retain prompts")
    
    if max_samples and max_samples < len(prompts):
        prompts = prompts[:max_samples]
        print(f"  Limited to {max_samples} samples")
    
    return prompts


# =========================================================================
# Heatmap Creation
# =========================================================================
def compute_activation_stats(activations: torch.Tensor, feature_indices: List[int]) -> dict:
    """
    Compute activation statistics for specific features.
    
    Args:
        activations: Tensor of shape (n_samples, seq_len, d_sae)
        feature_indices: List of feature indices to analyze
        
    Returns:
        Dict with 'mean', 'frequency', 'max' for each feature
    """
    # Reshape to (n_samples * seq_len, d_sae)
    n_samples, seq_len, d_sae = activations.shape
    flat = activations.reshape(-1, d_sae)
    total_positions = flat.shape[0]
    
    feature_indices_tensor = torch.tensor(feature_indices)
    selected = flat[:, feature_indices_tensor]  # (total_positions, n_features)
    
    # Compute statistics
    mean_activation = selected.mean(dim=0).numpy()
    frequency = ((selected > 0).sum(dim=0).float() / total_positions).numpy()
    max_activation = selected.max(dim=0).values.numpy()
    
    return {
        'mean': mean_activation,
        'frequency': frequency,
        'max': max_activation
    }


def create_activation_heatmap(
    forget_activations: torch.Tensor,
    retain_activations: torch.Tensor,
    feature_indices: List[int],
    save_path: str = "feature_activation_heatmap_forget_retain.png"
):
    """
    Create a heatmap comparing feature activations on Forget vs Retain sets.
    """
    # Compute stats for both datasets
    forget_stats = compute_activation_stats(forget_activations, feature_indices)
    retain_stats = compute_activation_stats(retain_activations, feature_indices)
    
    # Create figure with multiple heatmaps
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    
    # Prepare data matrices (2 rows: Forget, Retain)
    n_features = len(feature_indices)
    
    # 1. Activation Frequency Heatmap
    freq_matrix = np.array([forget_stats['frequency'], retain_stats['frequency']])
    im1 = axes[0].imshow(freq_matrix, aspect='auto', cmap='YlOrRd', vmin=0)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(['Bio Forget (Harmful)', 'Bio Retain (Benign)'], fontsize=12)
    axes[0].set_title('Activation Frequency (proportion of positions with activation > 0)', 
                       fontsize=14, fontweight='bold')
    if n_features <= 50:
        axes[0].set_xticks(range(n_features))
        axes[0].set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.6)
    cbar1.set_label('Frequency', fontsize=10)
    
    # 2. Mean Activation Heatmap
    mean_matrix = np.array([forget_stats['mean'], retain_stats['mean']])
    im2 = axes[1].imshow(mean_matrix, aspect='auto', cmap='viridis')
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(['Bio Forget (Harmful)', 'Bio Retain (Benign)'], fontsize=12)
    axes[1].set_title('Mean Activation Value', fontsize=14, fontweight='bold')
    if n_features <= 50:
        axes[1].set_xticks(range(n_features))
        axes[1].set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.6)
    cbar2.set_label('Mean Activation', fontsize=10)
    
    # 3. Frequency Difference Heatmap (Forget - Retain)
    diff_matrix = forget_stats['frequency'] - retain_stats['frequency']
    diff_matrix = diff_matrix.reshape(1, -1)
    im3 = axes[2].imshow(diff_matrix, aspect='auto', cmap='RdBu_r', 
                          vmin=-np.max(np.abs(diff_matrix)), 
                          vmax=np.max(np.abs(diff_matrix)))
    axes[2].set_yticks([0])
    axes[2].set_yticklabels(['Forget - Retain'], fontsize=12)
    axes[2].set_title('Frequency Difference (positive = more active on Forget/harmful)', 
                       fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Feature Index', fontsize=12)
    if n_features <= 50:
        axes[2].set_xticks(range(n_features))
        axes[2].set_xticklabels(feature_indices, rotation=45, ha='right', fontsize=8)
    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.6)
    cbar3.set_label('Difference', fontsize=10)
    
    plt.suptitle(f'Layer {LAYER} - Activation Patterns for {n_features} Identified Features\n'
                 f'WMDP Bio Forget (harmful) vs Bio Retain (benign)', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved heatmap to: {save_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(f"{'Feature':<10} {'Forget Freq':>12} {'Retain Freq':>12} {'Difference':>12}")
    print("-" * 46)
    for i, feat in enumerate(feature_indices[:10]):  # Show first 10
        print(f"{feat:<10} {forget_stats['frequency'][i]:>12.4f} {retain_stats['frequency'][i]:>12.4f} "
              f"{forget_stats['frequency'][i] - retain_stats['frequency'][i]:>12.4f}")
    if len(feature_indices) > 10:
        print(f"... and {len(feature_indices) - 10} more features")


# =========================================================================
# Main
# =========================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the 50 identified features
    print(f"\nLoading features from: {FEATURES_FILE}")
    with open(FEATURES_FILE, 'r') as f:
        feature_indices = [int(line.strip()) for line in f if line.strip()]
    print(f"Loaded {len(feature_indices)} features: {feature_indices[:5]}...")
    
    # Check if activation files exist
    forget_path = os.path.join(OUTPUT_DIR, f"forget_sae_activations_layer{LAYER}.pt")
    retain_path = os.path.join(OUTPUT_DIR, f"retain_sae_activations_layer{LAYER}.pt")
    
    if os.path.exists(forget_path) and os.path.exists(retain_path):
        print(f"\nLoading cached activations from {OUTPUT_DIR}...")
        forget_activations = torch.load(forget_path)
        retain_activations = torch.load(retain_path)
    else:
        print("\nActivation files not found. Collecting new activations...")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Load model and SAE
        print(f"\nLoading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        sae_id = f"layer_{LAYER}/width_16k/canonical"
        print(f"Loading SAE: {SAE_RELEASE} / {sae_id}")
        sae_result = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id)
        sae = sae_result[0] if isinstance(sae_result, tuple) else sae_result
        sae = sae.to(device)
        
        # Create collector
        collector = SAEActivationCollector(model=model, sae=sae, layer_idx=LAYER, device=device)
        
        # Load datasets
        forget_texts = load_wmdp_bio_forget(max_samples=MAX_SAMPLES)
        retain_texts = load_wmdp_bio_retain(max_samples=MAX_SAMPLES)
        
        # Collect activations
        print("\n" + "=" * 60)
        print("Collecting FORGET set activations...")
        print("=" * 60)
        forget_activations = collector.collect_activations(
            texts=forget_texts, tokenizer=tokenizer, 
            max_length=MAX_LENGTH, batch_size=BATCH_SIZE
        )
        print(f"Forget activations shape: {forget_activations.shape}")
        torch.save(forget_activations, forget_path)
        
        torch.cuda.empty_cache()
        
        print("\n" + "=" * 60)
        print("Collecting RETAIN set activations...")
        print("=" * 60)
        retain_activations = collector.collect_activations(
            texts=retain_texts, tokenizer=tokenizer,
            max_length=MAX_LENGTH, batch_size=BATCH_SIZE
        )
        print(f"Retain activations shape: {retain_activations.shape}")
        torch.save(retain_activations, retain_path)
    
    print(f"\nForget activations: {forget_activations.shape}")
    print(f"Retain activations: {retain_activations.shape}")
    
    # Create heatmap
    print("\n" + "=" * 60)
    print("Creating heatmap visualization...")
    print("=" * 60)
    
    create_activation_heatmap(
        forget_activations=forget_activations,
        retain_activations=retain_activations,
        feature_indices=feature_indices,
        save_path=os.path.join(OUTPUT_DIR, f"feature_activation_heatmap_forget_retain_layer{LAYER}.png")
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

