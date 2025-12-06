"""
Refusal Clamp Implementation for SAE-based Unlearning.

Based on "Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning"
https://arxiv.org/abs/2503.11127

Refusal Clamp Method:
    A more aggressive intervention that combines two actions:
    1. Clamps harmful features to negative values (like Clamp Prime)
    2. Boosts the refusal feature to trigger refusal behavior
    
    When any harmful feature is detected as active, both interventions
    are applied simultaneously.

Usage:
    from verified_implementation import ClampConfig, RefusalClampHook
    
    config = ClampConfig(refusal_feature=12345)
    config.load_features_from_file("features_to_clamp.txt")
    
    hook = RefusalClampHook(sae_wrapper, config)
    handle = model.model.layers[config.sae_layer].register_forward_hook(hook)
"""

import torch
from typing import List, Optional, Tuple, Any

# Handle imports for both module and direct execution
try:
    from .config import ClampConfig
    from .clamp_prime import GemmaScopeWrapper, load_sae_and_wrapper
except ImportError:
    from config import ClampConfig
    from clamp_prime import GemmaScopeWrapper, load_sae_and_wrapper


class RefusalClampHook:
    """
    Forward hook that applies Refusal Clamp intervention.
    
    This combines harmful feature clamping with refusal feature boosting:
    1. When harmful features are active, clamp them to negative values
    2. Simultaneously boost the refusal feature to encourage refusal behavior
    
    This is a stronger intervention than Clamp Prime alone, as it actively
    encourages the model to refuse harmful requests.
    
    Attributes:
        sae_wrapper: GemmaScopeWrapper for encoding/decoding
        config: ClampConfig with hyperparameters (including refusal_feature)
        mode: One of "clamp_only", "refusal_only", or "combined"
        call_count: Number of times the hook has been called (for debugging)
    """
    
    def __init__(
        self,
        sae_wrapper: GemmaScopeWrapper,
        config: ClampConfig,
        mode: str = "combined",
        verbose: bool = False
    ):
        """
        Initialize the Refusal Clamp hook.
        
        Args:
            sae_wrapper: GemmaScopeWrapper for the SAE
            config: ClampConfig with clamping parameters and refusal_feature
            mode: Intervention mode:
                  - "clamp_only": Only clamp harmful features (same as ClampPrime)
                  - "refusal_only": Only boost refusal feature when harmful detected
                  - "combined": Both clamp harmful and boost refusal (default)
            verbose: If True, print debug information
        """
        self.sae_wrapper = sae_wrapper
        self.config = config
        self.mode = mode
        self.verbose = verbose
        self._call_count = 0
        
        if mode in ("refusal_only", "combined") and config.refusal_feature is None:
            raise ValueError(
                f"refusal_feature must be specified in config for mode '{mode}'"
            )
    
    @property
    def harmful_features(self) -> List[int]:
        """Get the list of harmful feature indices."""
        return self.config.harmful_features
    
    @property
    def refusal_feature(self) -> Optional[int]:
        """Get the refusal feature index."""
        return self.config.refusal_feature
    
    @property
    def activation_threshold(self) -> float:
        """Get the activation threshold."""
        return self.config.activation_threshold
    
    @property
    def clamp_coefficient(self) -> float:
        """Get the clamp coefficient (negative value for harmful features)."""
        return self.config.clamp_coefficient
    
    @property
    def refusal_coefficient(self) -> float:
        """Get the refusal coefficient (positive value to boost refusal)."""
        return self.config.refusal_coefficient
    
    def __call__(
        self,
        module: torch.nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Any
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Apply Refusal Clamp intervention to layer output.
        
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
            print(f"\n[RefusalClamp] Hook called #{self._call_count}")
            print(f"  Mode: {self.mode}")
            print(f"  Activations shape: {activations.shape}")
        
        # Encode to SAE latent space
        latents = self.sae_wrapper.encode(activations)
        
        # Track which positions have ANY harmful feature active
        harmful_active = torch.zeros(
            latents.shape[:-1],
            dtype=torch.bool,
            device=latents.device
        )
        
        # Process harmful features
        clamped_count = 0
        for feat_idx in self.harmful_features:
            active_mask = latents[..., feat_idx] > self.activation_threshold
            harmful_active = harmful_active | active_mask
            
            # Apply clamping if in clamp mode
            if self.mode in ("clamp_only", "combined"):
                latents[..., feat_idx] = torch.where(
                    active_mask,
                    torch.full_like(latents[..., feat_idx], self.clamp_coefficient),
                    latents[..., feat_idx]
                )
                clamped_count += active_mask.sum().item()
        
        # Boost refusal feature if in refusal mode and harmful features detected
        refusal_boosted = 0
        if self.mode in ("refusal_only", "combined") and self.refusal_feature is not None:
            latents[..., self.refusal_feature] = torch.where(
                harmful_active,
                torch.full_like(
                    latents[..., self.refusal_feature],
                    self.refusal_coefficient
                ),
                latents[..., self.refusal_feature]
            )
            refusal_boosted = harmful_active.sum().item()
        
        if self.verbose and self._call_count <= 3:
            print(f"  Positions with harmful features: {harmful_active.sum().item()}")
            print(f"  Positions clamped: {clamped_count}")
            print(f"  Positions with refusal boost: {refusal_boosted}")
        
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
        activations.copy_(modified_activations)
        
        return None
    
    def reset_count(self):
        """Reset the call counter."""
        self._call_count = 0


class CombinedClampHook:
    """
    A more flexible hook that can switch between Clamp Prime and Refusal Clamp.
    
    This is useful for A/B testing different intervention strategies during
    evaluation without recreating hooks.
    """
    
    def __init__(
        self,
        sae_wrapper: GemmaScopeWrapper,
        config: ClampConfig,
        verbose: bool = False
    ):
        """
        Initialize the combined hook.
        
        Args:
            sae_wrapper: GemmaScopeWrapper for the SAE
            config: ClampConfig with all parameters
            verbose: If True, print debug information
        """
        self.sae_wrapper = sae_wrapper
        self.config = config
        self.verbose = verbose
        self._call_count = 0
        
        # Start with Clamp Prime (no refusal)
        self._use_refusal = False
    
    @property
    def use_refusal(self) -> bool:
        """Whether to use refusal clamping."""
        return self._use_refusal
    
    @use_refusal.setter
    def use_refusal(self, value: bool):
        """Set whether to use refusal clamping."""
        if value and self.config.refusal_feature is None:
            raise ValueError("refusal_feature must be set in config to use refusal mode")
        self._use_refusal = value
    
    def set_mode(self, use_refusal: bool):
        """
        Set the intervention mode.
        
        Args:
            use_refusal: If True, use Refusal Clamp; if False, use Clamp Prime
        """
        self.use_refusal = use_refusal
    
    def __call__(
        self,
        module: torch.nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: Any
    ) -> Optional[Tuple[torch.Tensor, ...]]:
        """Apply the appropriate clamping intervention."""
        self._call_count += 1
        
        if isinstance(output, tuple):
            activations = output[0]
        else:
            activations = output
        
        original_dtype = activations.dtype
        original_device = activations.device
        
        # Encode
        latents = self.sae_wrapper.encode(activations)
        
        # Track harmful activation for refusal mode
        harmful_active = torch.zeros(
            latents.shape[:-1],
            dtype=torch.bool,
            device=latents.device
        )
        
        # Clamp harmful features
        for feat_idx in self.config.harmful_features:
            active_mask = latents[..., feat_idx] > self.config.activation_threshold
            harmful_active = harmful_active | active_mask
            
            latents[..., feat_idx] = torch.where(
                active_mask,
                torch.full_like(latents[..., feat_idx], self.config.clamp_coefficient),
                latents[..., feat_idx]
            )
        
        # Optionally boost refusal
        if self._use_refusal and self.config.refusal_feature is not None:
            latents[..., self.config.refusal_feature] = torch.where(
                harmful_active,
                torch.full_like(
                    latents[..., self.config.refusal_feature],
                    self.config.refusal_coefficient
                ),
                latents[..., self.config.refusal_feature]
            )
        
        # Decode
        modified_activations = self.sae_wrapper.decode(latents)
        modified_activations = modified_activations.to(
            dtype=original_dtype,
            device=original_device
        )
        
        activations.copy_(modified_activations)
        return None


def create_refusal_clamp_hook(
    model: torch.nn.Module,
    config: ClampConfig,
    mode: str = "combined",
    verbose: bool = False
) -> Tuple[RefusalClampHook, Any]:
    """
    Create and register a Refusal Clamp hook on the model.
    
    Args:
        model: The transformer model (e.g., Gemma)
        config: ClampConfig with all parameters
        mode: Intervention mode ("clamp_only", "refusal_only", "combined")
        verbose: If True, print debug information
        
    Returns:
        Tuple of (hook, hook_handle)
    """
    # Load SAE
    _, sae_wrapper = load_sae_and_wrapper(config)
    
    # Create hook
    hook = RefusalClampHook(sae_wrapper, config, mode=mode, verbose=verbose)
    
    # Register on target layer
    target_layer = model.model.layers[config.sae_layer]
    hook_handle = target_layer.register_forward_hook(hook)
    
    print(f"Refusal Clamp hook registered on layer {config.sae_layer}")
    print(f"  Mode: {mode}")
    print(f"  Harmful features: {len(config.harmful_features)}")
    print(f"  Refusal feature: {config.refusal_feature}")
    print(f"  Activation threshold: {config.activation_threshold}")
    print(f"  Clamp coefficient: {config.clamp_coefficient}")
    print(f"  Refusal coefficient: {config.refusal_coefficient}")
    
    return hook, hook_handle

