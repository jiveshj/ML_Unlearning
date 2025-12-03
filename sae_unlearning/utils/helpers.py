"""
Helper utilities for SAE unlearning.

Provides environment setup, activation collection, and other utility functions.
"""

import os
import torch
from datetime import datetime
from typing import List, Dict, Optional
from tqdm.auto import tqdm


def setup_environment(
    hf_token: Optional[str] = None,
    seed: int = 42,
    output_dir: Optional[str] = None
) -> str:
    """
    Setup the environment for running experiments.
    
    - Logs into HuggingFace Hub
    - Sets random seeds for reproducibility
    - Creates output directory
    
    Args:
        hf_token: HuggingFace token (uses HF_TOKEN env var if None)
        seed: Random seed for reproducibility
        output_dir: Output directory name (auto-generated if None)
        
    Returns:
        Path to the created output directory
    """
    from huggingface_hub import login
    
    # Login to HuggingFace
    token = hf_token or os.getenv("HF_TOKEN")
    if token:
        login(token=token)
        print("✓ Logged into HuggingFace Hub")
    else:
        print("⚠ No HF_TOKEN found. Some models may require authentication.")
    
    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"✓ Random seed set to {seed}")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"unlearning_results_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    return output_dir


def collect_activations_for_texts(
    model,
    tokenizer,
    texts: List[str],
    layer_indices: List[int],
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
    max_length: int = 512
) -> Dict[int, torch.Tensor]:
    """
    Collect activations from a list of texts.
    
    Processes texts in batches and collects activations from specified layers
    using forward hooks.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        texts: List of input texts
        layer_indices: Which layers to collect activations from
        batch_size: Batch size for processing
        max_samples: Optional limit on number of samples
        device: Device for computation
        max_length: Maximum sequence length for tokenization
        
    Returns:
        Dict mapping layer_idx -> activations tensor [total_tokens, hidden_dim]
    """
    from ..models.activation_collector import ActivationCollector
    
    if device is None:
        device = next(model.parameters()).device
    
    if max_samples:
        texts = texts[:max_samples]
    
    collector = ActivationCollector(model, layer_indices)
    collector.register_hooks()
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Collecting activations"):
            batch = texts[i:i+batch_size]
            
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            _ = model(**inputs)
    
    result = {}
    for layer_idx in layer_indices:
        result[layer_idx] = collector.get_activations(layer_idx)
    
    collector.remove_hooks()
    return result


def collect_activations_per_sample(
    model,
    tokenizer,
    texts: List[str],
    layer_indices: List[int],
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
    max_length: int = 512
) -> Dict[int, torch.Tensor]:
    """
    Collect activations one sample at a time (avoids padding issues).
    
    Slower than batch processing but handles variable-length sequences
    more precisely.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        texts: List of input texts
        layer_indices: Which layers to collect activations from
        max_samples: Optional limit on number of samples
        device: Device for computation
        max_length: Maximum sequence length
        
    Returns:
        Dict mapping layer_idx -> activations tensor [total_tokens, hidden_dim]
    """
    from ..models.activation_collector import ActivationCollector
    
    if device is None:
        device = next(model.parameters()).device
    
    if max_samples:
        texts = texts[:max_samples]
    
    all_activations = {idx: [] for idx in layer_indices}
    
    model.eval()
    with torch.no_grad():
        for text in tqdm(texts, desc="Collecting activations"):
            collector = ActivationCollector(model, layer_indices)
            collector.register_hooks()
            
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            _ = model(**inputs)
            
            for layer_idx in layer_indices:
                acts = collector.get_activations(layer_idx)
                if acts is not None:
                    all_activations[layer_idx].append(acts)
            
            collector.remove_hooks()
    
    # Concatenate all samples
    result = {}
    for layer_idx in layer_indices:
        if all_activations[layer_idx]:
            result[layer_idx] = torch.cat(all_activations[layer_idx], dim=0)
        else:
            result[layer_idx] = None
    
    return result


def get_device_info() -> Dict:
    """
    Get information about available compute devices.
    
    Returns:
        Dict with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['total_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3
        info['allocated_memory_gb'] = torch.cuda.memory_allocated(0) / 1024**3
        info['reserved_memory_gb'] = torch.cuda.memory_reserved(0) / 1024**3
    
    return info


def print_device_info():
    """Print formatted device information."""
    info = get_device_info()
    
    print("\n" + "="*50)
    print("DEVICE INFORMATION")
    print("="*50)
    
    if info['cuda_available']:
        print(f"GPU Available: Yes")
        print(f"Device: {info['device_name']}")
        print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
        print(f"Allocated: {info['allocated_memory_gb']:.2f} GB")
        print(f"Reserved: {info['reserved_memory_gb']:.2f} GB")
    else:
        print("GPU Available: No (using CPU)")
    
    print("="*50 + "\n")


def save_results(
    results: Dict,
    output_dir: str,
    filename: str = "results.json"
):
    """
    Save results dictionary to JSON file.
    
    Args:
        results: Results dictionary to save
        output_dir: Output directory
        filename: Output filename
    """
    import json
    
    filepath = os.path.join(output_dir, filename)
    
    # Convert any non-serializable types
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, torch.device):
            return str(obj)
        return obj
    
    # Recursively convert
    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict(item) for item in d]
        else:
            return convert(d)
    
    serializable = convert_dict(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"✓ Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded results dictionary
    """
    import json
    
    with open(filepath, 'r') as f:
        return json.load(f)


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU memory cache cleared")


def estimate_memory_usage(
    model_size_params: int,
    batch_size: int,
    seq_length: int,
    dtype_bytes: int = 4
) -> Dict[str, float]:
    """
    Estimate memory usage for running inference.
    
    Args:
        model_size_params: Number of model parameters
        batch_size: Batch size
        seq_length: Sequence length
        dtype_bytes: Bytes per parameter (4 for fp32, 2 for fp16)
        
    Returns:
        Dict with memory estimates in GB
    """
    # Model weights
    model_memory = model_size_params * dtype_bytes / (1024**3)
    
    # Activations (rough estimate)
    hidden_dim = int(model_size_params ** 0.5)  # Very rough estimate
    activation_memory = batch_size * seq_length * hidden_dim * dtype_bytes / (1024**3)
    
    # Gradients (if training)
    gradient_memory = model_memory  # Same as model for full gradients
    
    return {
        'model_weights_gb': model_memory,
        'activations_gb': activation_memory,
        'gradients_gb': gradient_memory,
        'total_inference_gb': model_memory + activation_memory,
        'total_training_gb': model_memory + activation_memory + gradient_memory
    }

