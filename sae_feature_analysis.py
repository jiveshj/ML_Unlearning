#!/usr/bin/env python3
"""
SAE Feature Analysis - Capture full sequence SAE activations.

This script hooks the SAE between Gemma layers to capture activation matrices
for ALL tokens (not just the last token). Each prompt produces a tensor of
shape (max_length, d_sae) and these are concatenated to create a 3D tensor
of shape (n_prompts, max_len, d_sae).

Output:
    - forget_sae_activations_layer{N}.pt: (n_forget, max_len, 16384)
    - retain_sae_activations_layer{N}.pt: (n_retain, max_len, 16384)
    - sae_activations_metadata.json: configuration and shape info
"""

import os
import json
import argparse
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Optional
from datasets import load_dataset
from sae_lens import SAE


class SAEActivationCollector:
    """
    Collects SAE-encoded activations for all tokens in a sequence.
    
    Hooks into a transformer layer and passes the hidden states through
    the SAE encoder to get sparse latent activations.
    """
    
    def __init__(self, model, sae, layer_idx: int, device: torch.device):
        """
        Initialize the collector.
        
        Args:
            model: HuggingFace transformer model
            sae: SAE lens model
            layer_idx: Which layer to hook
            device: Device for computation
        """
        self.model = model
        self.sae = sae
        self.layer_idx = layer_idx
        self.device = device
        self.hook_handle = None
        self.captured_activations = None
        
    def _get_layer(self):
        """Get the transformer layer to hook."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Gemma, Llama architecture
            return self.model.model.layers[self.layer_idx]
        else:
            raise RuntimeError("Could not find transformer layers for this model architecture")
    
    def _hook_fn(self, module, input, output):
        """
        Forward hook that captures hidden states and encodes through SAE.
        
        Args:
            module: The hooked module
            input: Input to the module
            output: Output from the module (hidden_states, ...)
        """
        # Extract hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # hidden_states shape: (batch_size, seq_len, hidden_dim)
        # Encode through SAE to get sparse latents
        with torch.no_grad():
            # SAE expects (batch, seq, hidden_dim) or (batch*seq, hidden_dim)
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Flatten for SAE encoding
            flat_hidden = hidden_states.reshape(-1, hidden_dim)
            
            # Encode through SAE
            sae_latents = self.sae.encode(flat_hidden)
            
            # Reshape back to (batch, seq, d_sae)
            d_sae = sae_latents.shape[-1]
            sae_latents = sae_latents.reshape(batch_size, seq_len, d_sae)
            
            # Store on CPU to save GPU memory
            self.captured_activations = sae_latents.cpu()
    
    def register_hook(self):
        """Register the forward hook on the target layer."""
        layer = self._get_layer()
        self.hook_handle = layer.register_forward_hook(self._hook_fn)
    
    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def collect_activations(
        self,
        texts: list,
        tokenizer,
        max_length: int = 128,
        batch_size: int = 4,
    ) -> torch.Tensor:
        """
        Collect SAE activations for a list of texts.
        
        Args:
            texts: List of text prompts
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length (for padding/truncation)
            batch_size: Batch size for processing
            
        Returns:
            Tensor of shape (n_prompts, max_length, d_sae)
        """
        self.model.eval()
        all_activations = []
        
        self.register_hook()
        
        try:
            for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize with padding to max_length
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass (hook captures activations)
                with torch.no_grad():
                    _ = self.model(**inputs)
                
                # Get captured activations (batch_size, max_length, d_sae)
                batch_activations = self.captured_activations
                all_activations.append(batch_activations)
                
                # Clear GPU cache periodically
                if i % (batch_size * 10) == 0:
                    torch.cuda.empty_cache()
        
        finally:
            self.remove_hook()
        
        # Concatenate all batches: (n_prompts, max_length, d_sae)
        all_activations = torch.cat(all_activations, dim=0)
        return all_activations


def load_wmdp_bio_forget(max_samples: Optional[int] = None) -> List[str]:
    """Load WMDP-Bio forget set (harmful biosecurity knowledge to unlearn)."""
    print("\nLoading WMDP-Bio Forget set...")
    
    ds = load_dataset("cais/wmdp-bio-forget-corpus",split='train')
    
    # Format questions as prompts
    prompts = []
    for item in ds:
        abstract = item["abstract"]
        text = item["text"]
        prompt = f"Abstract: {abstract}\n\nText: {text}\n\n"
        prompts.append(prompt)
    
    
    print(f"  Loaded {len(prompts)} forget prompts")
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
            # Use the text directly as a prompt (or truncate if too long)
            prompts.append(text)
    
    print(f"  Loaded {len(prompts)} retain prompts")
    return prompts


def load_model_and_sae(
    model_name: str = "google/gemma-2-2b",
    sae_release: str = "gemma-scope-2b-pt-res-canonical",
    sae_id: str = "layer_9/width_16k/canonical",
    device: torch.device = None,
):
    """
    Load the Gemma model and SAE.
    
    Args:
        model_name: HuggingFace model name
        sae_release: SAE lens release name
        sae_id: SAE identifier within the release
        device: Device to load models on
        
    Returns:
        Tuple of (model, tokenizer, sae)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    if not torch.cuda.is_available():
        model = model.to(device)
    
    print(f"Model loaded. Hidden size: {model.config.hidden_size}")
    
    print(f"Loading SAE: {sae_release} / {sae_id}")
    sae_result = SAE.from_pretrained(release=sae_release, sae_id=sae_id)
    
    # Handle both old and new sae_lens API
    if isinstance(sae_result, tuple):
        sae = sae_result[0]
    else:
        sae = sae_result
    
    sae = sae.to(device)
    print(f"SAE loaded. d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    
    return model, tokenizer, sae


def main():
    parser = argparse.ArgumentParser(description="Collect SAE activations for all tokens")
    parser.add_argument("--model", type=str, default="google/gemma-2-2b", help="Model name")
    parser.add_argument("--layer", type=int, default=7, help="Layer index to hook")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per dataset (None=all)")
    parser.add_argument("--output-dir", type=str, default="sae_activations", help="Output directory")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and SAE
    sae_id = f"layer_{args.layer}/width_16k/canonical"
    model, tokenizer, sae = load_model_and_sae(
        model_name=args.model,
        sae_id=sae_id,
        device=device,
    )
    
    # Create collector
    collector = SAEActivationCollector(
        model=model,
        sae=sae,
        layer_idx=args.layer,
        device=device,
    )
    
    # Load datasets
    print("\n" + "=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    
    wmdp_dataset_forget = load_wmdp_bio_forget()
    wmdp_dataset_retain = load_wmdp_bio_retain()
    
    forget_texts = wmdp_dataset_forget
    retain_texts = wmdp_dataset_retain
    
    forget_texts = forget_texts[:2000]
    retain_texts = retain_texts[:2000]
    
    print(f"Forget set (WMDP-Bio-forget): {len(forget_texts)} samples")
    print(f"Retain set (WMDP-Bio-retain): {len(retain_texts)} samples")
    
    # Collect forget activations
    print("\n" + "=" * 60)
    print("Collecting FORGET set activations...")
    print("=" * 60)
    
    forget_activations = collector.collect_activations(
        texts=forget_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print(f"Forget activations shape: {forget_activations.shape}")
    
    # Save forget activations
    forget_path = os.path.join(args.output_dir, f"forget_sae_activations_layer{args.layer}.pt")
    print(forget_activations.shape)
    torch.save(forget_activations, forget_path)
    print(f"Saved: {forget_path}")
    
    # Clear some memory
    del forget_activations
    torch.cuda.empty_cache()
    
    # Collect retain activations
    print("\n" + "=" * 60)
    print("Collecting RETAIN set activations...")
    print("=" * 60)
    
    retain_activations = collector.collect_activations(
        texts=retain_texts,
        tokenizer=tokenizer,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    print(f"Retain activations shape: {retain_activations.shape}")
    
    # Save retain activations
    retain_path = os.path.join(args.output_dir, f"retain_sae_activations_layer{args.layer}.pt")
    print(retain_activations.shape)
    torch.save(retain_activations, retain_path)
    print(f"Saved: {retain_path}")
    
    # Save metadata
    metadata = {
        "model": args.model,
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": sae_id,
        "layer_idx": args.layer,
        "max_length": args.max_length,
        "d_sae": sae.cfg.d_sae,
        "d_model": sae.cfg.d_in,
        "forget_shape": list(torch.load(forget_path).shape),
        "retain_shape": list(retain_activations.shape),
        "num_forget_samples": len(forget_texts),
        "num_retain_samples": len(retain_texts),
    }
    
    metadata_path = os.path.join(args.output_dir, "sae_activations_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {metadata_path}")
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Forget activations: {forget_path}")
    print(f"Retain activations: {retain_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()