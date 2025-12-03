"""
Activation Collector module for capturing model activations during forward pass.

Provides utilities for collecting activations from specific transformer layers
using PyTorch forward hooks.
"""

import torch
from typing import List, Dict, Optional


class ActivationCollector:
    """
    Collects activations from specific layers during model forward pass.
    
    Uses PyTorch forward hooks to capture hidden states from transformer layers.
    Handles variable sequence lengths by flattening tokens.
    
    Attributes:
        model: The transformer model to collect activations from
        layer_indices: List of layer indices to hook
        activations: Dictionary storing collected activations per layer
        hooks: List of registered hook handles
    """
    
    def __init__(self, model, layer_indices: List[int]):
        """
        Initialize the collector.
        
        Args:
            model: HuggingFace transformer model
            layer_indices: Which layer indices to collect activations from
        """
        self.model = model
        self.layer_indices = layer_indices
        self.activations: Dict[int, List[torch.Tensor]] = {idx: [] for idx in layer_indices}
        self.hooks: List = []
    
    def _get_layer(self, layer_idx: int):
        """
        Access layer based on model architecture.
        
        Supports common HuggingFace model architectures.
        
        Args:
            layer_idx: Index of the layer to access
            
        Returns:
            The transformer layer module
        """
        # Try different model architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Gemma, Llama, etc.
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2, GPT-Neo, etc.
            return self.model.transformer.h[layer_idx]
        elif hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'h'):
            return self.model.base_model.h[layer_idx]
        else:
            raise RuntimeError(
                "Could not find transformer layers. "
                "Adapt _get_layer() for your model architecture."
            )
    
    def register_hooks(self):
        """Register forward hooks on specified layers."""
        for layer_idx in self.layer_indices:
            layer = self._get_layer(layer_idx)
            
            def hook_fn(module, input, output, idx=layer_idx):
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output
                self.activations[idx].append(hidden_states.detach().cpu())
            
            handle = layer.register_forward_hook(hook_fn)
            self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear(self):
        """Clear collected activations."""
        self.activations = {idx: [] for idx in self.layer_indices}
    
    def get_activations(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Get collected activations for a layer, flattened across sequences.
        
        Handles variable sequence lengths by flattening all tokens into
        a single dimension: [total_tokens, hidden_dim].
        
        Args:
            layer_idx: Which layer's activations to retrieve
            
        Returns:
            Tensor of shape [total_tokens, hidden_dim] or None if empty
        """
        acts = self.activations[layer_idx]
        if not acts:
            return None
        
        # Flatten all sequences (treats each token independently)
        flattened = []
        for act in acts:
            # act shape: [batch_size, seq_len, hidden_dim]
            batch_size, seq_len, hidden_dim = act.shape
            # Flatten batch and sequence dimensions, convert to float32
            flattened.append(act.reshape(-1, hidden_dim).float())
        
        return torch.cat(flattened, dim=0)  # [total_tokens, hidden_dim]
    
    def get_activations_batched(self, layer_idx: int, max_seq_len: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Get collected activations preserving batch structure with padding.
        
        Args:
            layer_idx: Which layer's activations to retrieve
            max_seq_len: Optional max sequence length (uses max from data if None)
            
        Returns:
            Tensor of shape [total_samples, max_seq_len, hidden_dim] or None
        """
        acts = self.activations[layer_idx]
        if not acts:
            return None
        
        if max_seq_len is None:
            max_seq_len = max(act.shape[1] for act in acts)
        
        hidden_dim = acts[0].shape[2]
        
        padded_acts = []
        for act in acts:
            batch_size, seq_len, _ = act.shape
            if seq_len < max_seq_len:
                padding = torch.zeros(batch_size, max_seq_len - seq_len, hidden_dim)
                act = torch.cat([act, padding], dim=1)
            elif seq_len > max_seq_len:
                act = act[:, :max_seq_len, :]
            padded_acts.append(act.float())
        
        return torch.cat(padded_acts, dim=0)
    
    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks and clear."""
        self.remove_hooks()
        return False

