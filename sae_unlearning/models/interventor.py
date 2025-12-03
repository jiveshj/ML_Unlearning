"""
Conditional Clamping Interventor module.

Implements the Clamp Prime and Refusal Clamp methods from the paper for
modifying SAE latent activations during inference.
"""

import torch
from typing import List, Optional

from .sae_wrapper import GemmaScopeWrapper
from ..config import UnlearningConfig


class ConditionalClampingInterventor:
    """
    Implements conditional clamping during inference.
    
    Two main methods from the paper:
    1. Clamp Prime: Clamps harmful features to negative values when active
    2. Refusal Clamp: Additionally boosts refusal feature when harmful features detected
    
    Attributes:
        sae: GemmaScopeWrapper for encoding/decoding
        harmful_features: List of feature indices identified as harmful
        refusal_feature: Index of the refusal feature (optional)
        config: UnlearningConfig with hyperparameters
    """
    
    def __init__(
        self,
        sae_wrapper: GemmaScopeWrapper,
        harmful_features: List[int],
        config: UnlearningConfig,
        refusal_feature: Optional[int] = None
    ):
        """
        Initialize the interventor.
        
        Args:
            sae_wrapper: GemmaScopeWrapper for the SAE
            harmful_features: List of harmful feature indices to clamp
            config: UnlearningConfig with clamping parameters
            refusal_feature: Optional refusal feature index for refusal clamping
        """
        self.sae = sae_wrapper
        self.harmful_features = harmful_features
        self.refusal_feature = refusal_feature
        self.config = config
    
    def clamp_prime(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Clamp Prime method: Set harmful features to negative values when active.
        
        When a harmful feature is active (above threshold), it is clamped to
        the negative clamp_coefficient, effectively suppressing that direction.
        
        Args:
            activations: Original model activations [batch, seq_len, d_model]
            
        Returns:
            Modified activations with harmful features clamped
        """
        activations = activations.to(self.config.device)
        latents = self.sae.encode(activations)
        
        # Clamp harmful features to negative coefficient
        for feat_idx in self.harmful_features:
            # Only clamp if feature is active (above threshold)
            active_mask = latents[..., feat_idx] > self.config.activation_threshold
            latents[..., feat_idx] = torch.where(
                active_mask,
                torch.full_like(latents[..., feat_idx], self.config.clamp_coefficient),
                latents[..., feat_idx]
            )
        
        # Decode back to activation space
        modified_activations = self.sae.decode(latents)
        return modified_activations
    
    def refusal_clamp(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Refusal Clamp method: Boost refusal feature when harmful features detected.
        
        This is a more aggressive intervention - whenever harmful features are
        detected as active, the refusal feature is boosted to trigger a refusal
        response from the model.
        
        Args:
            activations: Original model activations [batch, seq_len, d_model]
            
        Returns:
            Modified activations with refusal boost applied
            
        Raises:
            ValueError: If refusal_feature is not specified
        """
        if self.refusal_feature is None:
            raise ValueError("Refusal feature must be specified for refusal_clamp")
        
        activations = activations.to(self.config.device)
        latents = self.sae.encode(activations)
        
        # Track which positions have any harmful feature active
        harmful_active = torch.zeros(
            latents.shape[:-1], 
            dtype=torch.bool, 
            device=latents.device
        )
        
        for feat_idx in self.harmful_features:
            active = latents[..., feat_idx] > self.config.activation_threshold
            harmful_active = harmful_active | active
        
        # Boost refusal feature when harmful features detected
        latents[..., self.refusal_feature] = torch.where(
            harmful_active,
            torch.full_like(
                latents[..., self.refusal_feature],
                self.config.refusal_coefficient
            ),
            latents[..., self.refusal_feature]
        )
        
        modified_activations = self.sae.decode(latents)
        return modified_activations
    
    def clamp_with_refusal(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Combined method: Both clamp harmful features AND boost refusal.
        
        This applies both interventions:
        1. Clamps harmful features to negative values
        2. Boosts refusal feature when harmful features detected
        
        Args:
            activations: Original model activations
            
        Returns:
            Modified activations with both interventions
        """
        if self.refusal_feature is None:
            raise ValueError("Refusal feature must be specified")
        
        activations = activations.to(self.config.device)
        latents = self.sae.encode(activations)
        
        harmful_active = torch.zeros(
            latents.shape[:-1], 
            dtype=torch.bool, 
            device=latents.device
        )
        
        # Clamp harmful features and track which are active
        for feat_idx in self.harmful_features:
            active = latents[..., feat_idx] > self.config.activation_threshold
            harmful_active = harmful_active | active
            
            latents[..., feat_idx] = torch.where(
                active,
                torch.full_like(latents[..., feat_idx], self.config.clamp_coefficient),
                latents[..., feat_idx]
            )
        
        # Boost refusal feature
        latents[..., self.refusal_feature] = torch.where(
            harmful_active,
            torch.full_like(
                latents[..., self.refusal_feature],
                self.config.refusal_coefficient
            ),
            latents[..., self.refusal_feature]
        )
        
        modified_activations = self.sae.decode(latents)
        return modified_activations
    
    def __call__(
        self,
        activations: torch.Tensor,
        use_refusal: bool = False
    ) -> torch.Tensor:
        """
        Apply clamping intervention.
        
        Args:
            activations: Model activations to modify
            use_refusal: If True, use refusal_clamp; else use clamp_prime
            
        Returns:
            Modified activations
        """
        if use_refusal:
            return self.refusal_clamp(activations)
        else:
            return self.clamp_prime(activations)

