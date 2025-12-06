"""
Clamp Prime Implementation for SAE-based Unlearning.

Based on "Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning"
https://arxiv.org/abs/2503.11127

Clamp Prime Method:
    When harmful features are detected as active (above threshold), they are
    clamped to a negative value, effectively suppressing that direction in
    the model's activation space.

Usage:
    from verified_implementation import ClampConfig, ClampPrimeHook
    
    config = ClampConfig()
    config.load_features_from_file("features_to_clamp.txt")
    
    hook = ClampPrimeHook(sae_wrapper, config)
    handle = model.model.layers[config.sae_layer].register_forward_hook(hook)
"""

import torch
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass

# Handle imports for both module and direct execution
try:
    from .config import ClampConfig
except ImportError:
    from config import ClampConfig


class GemmaScopeWrapper:
    """
    Wrapper around SAE Lens pre-trained SAE for Gemma Scope models.
    
    Provides a consistent interface for encoding/decoding activations
    through the Sparse Autoencoder.
    
    Attributes:
        sae: The underlying SAE Lens model
        device: Device the model is on
        d_model: Input dimension (model hidden size)
        d_sae: SAE latent dimension
    """
    
    def __init__(self, sae_lens_model, device: Optional[torch.device] = None):
        """
        Initialize the wrapper.
        
        Args:
            sae_lens_model: Pre-trained SAE from SAE Lens
            device: Device to move the model to (auto-detected if None)
        """
        self.sae = sae_lens_model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sae = self.sae.to(self.device)
        self.d_model = self.sae.cfg.d_in
        self.d_sae = self.sae.cfg.d_sae
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse latent space.
        
        Args:
            x: Input activations of shape [batch, seq, hidden_dim]
            
        Returns:
            Sparse latent activations of shape [batch, seq, d_sae]
        """
        x = x.to(self.device).to(torch.float32)
        return self.sae.encode(x)
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents back to activation space.
        
        Args:
            latents: Sparse latent activations
            
        Returns:
            Reconstructed activations in the original space
        """
        latents = latents.to(self.device)
        return self.sae.decode(latents)


class ClampPrimeHook:
    """
    Forward hook that applies Clamp Prime intervention.
    
    When harmful features are active (above threshold), clamp them to
    a negative value to suppress harmful knowledge retrieval.
    
    This hook should be registered on the target transformer layer:
        handle = model.model.layers[layer_idx].register_forward_hook(hook)
    
    Attributes:
        sae_wrapper: GemmaScopeWrapper for encoding/decoding
        config: ClampConfig with hyperparameters
        call_count: Number of times the hook has been called (for debugging)
    """
    
    def __init__(
        self,
        sae_wrapper: GemmaScopeWrapper,
        config: ClampConfig,
        verbose: bool = False
    ):
        """
        Initialize the Clamp Prime hook.
        
        Args:
            sae_wrapper: GemmaScopeWrapper for the SAE
            config: ClampConfig with clamping parameters
            verbose: If True, print debug information
        """
        self.sae_wrapper = sae_wrapper
        self.config = config
        self.verbose = verbose
        self._call_count = 0
    
    @property
    def harmful_features(self) -> List[int]:
        """Get the list of harmful feature indices."""
        return self.config.harmful_features
    
    @property
    def activation_threshold(self) -> float:
        """Get the activation threshold."""
        return self.config.activation_threshold
    
    @property
    def clamp_coefficient(self) -> float:
        """Get the clamp coefficient."""
        return self.config.clamp_coefficient
    
    def __call__(
        self,
        module: torch.nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Any
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Apply Clamp Prime clamping intervention to layer output.
        
        Args:
            module: The hooked module (transformer layer)
            input: Input to the module
            output: Output from the module (hidden_states, ...)
            
        Returns:
            None (modifies activations in-place for compatibility with device_map)
        """
        self._call_count += 1
        
        # Handle tuple output (common in transformer layers)
        if isinstance(output, tuple):
            activations = output[0]
        else:
            activations = output
        
        # Store original dtype and device for restoration
        original_dtype = activations.dtype
        original_device = activations.device
        
        if self.verbose and self._call_count <= 3:
            print(f"\n[ClampPrime] Hook called #{self._call_count}")
            print(f"  Activations shape: {activations.shape}")
            print(f"  Device: {activations.device}, dtype: {activations.dtype}")
        
        # Encode to SAE latent space
        latents = self.sae_wrapper.encode(activations)
        
        if self.verbose and self._call_count <= 3:
            print(f"  Latents shape: {latents.shape}")
        
        # Apply conditional clamping to each harmful feature
        clamped_count = 0
        for feat_idx in self.harmful_features:
            # Find positions where this feature is active
            active_mask = latents[..., feat_idx] > self.activation_threshold
            
            # Clamp active features to negative coefficient
            latents[..., feat_idx] = torch.where(
                active_mask,
                torch.full_like(latents[..., feat_idx], self.clamp_coefficient),
                latents[..., feat_idx]
            )
            
            if self.verbose and self._call_count <= 3:
                clamped_count += active_mask.sum().item()
        
        if self.verbose and self._call_count <= 3:
            print(f"  Total positions clamped: {clamped_count}")
        
        # Decode back to activation space
        modified_activations = self.sae_wrapper.decode(latents)
        modified_activations = modified_activations.to(
            dtype=original_dtype,
            device=original_device
        )
        
        if self.verbose and self._call_count <= 3:
            diff_norm = (activations - modified_activations).norm().item()
            print(f"  Modification norm: {diff_norm:.4f}")
        
        # CRITICAL: Modify in-place for compatibility with device_map="auto"
        # Forward hook return values can be ignored with accelerate's dispatch hooks
        activations.copy_(modified_activations)
        
        return None
    
    def reset_count(self):
        """Reset the call counter."""
        self._call_count = 0


def load_sae_and_wrapper(
    config: ClampConfig
) -> Tuple[Any, GemmaScopeWrapper]:
    """
    Load the SAE model and create a wrapper.
    
    Args:
        config: ClampConfig with SAE configuration
        
    Returns:
        Tuple of (raw_sae, wrapped_sae)
    """
    from sae_lens import SAE
    
    print(f"Loading SAE: {config.sae_release}, {config.sae_id}...")
    
    sae_result = SAE.from_pretrained(
        release=config.sae_release,
        sae_id=config.sae_id,
    )
    
    # Handle both old API (tuple) and new API (just SAE)
    if isinstance(sae_result, tuple):
        sae_model = sae_result[0]
    else:
        sae_model = sae_result
    
    # Create wrapper
    wrapper = GemmaScopeWrapper(sae_model, device=config.device)
    print(f"SAE loaded: d_model={wrapper.d_model}, d_sae={wrapper.d_sae}")
    
    return sae_model, wrapper


def create_clamp_prime_hook(
    model: torch.nn.Module,
    config: ClampConfig,
    verbose: bool = False
) -> Tuple[ClampPrimeHook, Any]:
    """
    Create and register a Clamp Prime hook on the model.
    
    Args:
        model: The transformer model (e.g., Gemma)
        config: ClampConfig with all parameters
        verbose: If True, print debug information
        
    Returns:
        Tuple of (hook, hook_handle)
    """
    # Load SAE
    _, sae_wrapper = load_sae_and_wrapper(config)
    
    # Create hook
    hook = ClampPrimeHook(sae_wrapper, config, verbose=verbose)
    
    # Register on target layer
    target_layer = model.model.layers[config.sae_layer]
    hook_handle = target_layer.register_forward_hook(hook)
    
    print(f"Clamp Prime hook registered on layer {config.sae_layer}")
    print(f"  Features to clamp: {len(config.harmful_features)}")
    print(f"  Activation threshold: {config.activation_threshold}")
    print(f"  Clamp coefficient: {config.clamp_coefficient}")
    
    return hook, hook_handle

