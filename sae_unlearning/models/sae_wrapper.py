"""
SAE Wrapper module for Gemma Scope pretrained SAEs.

Provides a unified interface for working with SAE Lens pretrained models.
"""

import torch
from typing import Tuple, Optional


class GemmaScopeWrapper:
    """
    Wrapper around SAE Lens pre-trained SAE to provide a consistent interface.
    
    This wrapper provides encode/decode functionality for Gemma Scope SAEs
    loaded via the sae_lens library.
    
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
            x: Input activations of shape [batch, hidden_dim] or [batch, seq, hidden_dim]
            
        Returns:
            Sparse latent activations of shape [batch, d_sae] or [batch, seq, d_sae]
        """
        x = x.to(self.device)
        latents = self.sae.encode(x)
        return latents
    
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
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass with reconstruction.
        
        Args:
            x: Input activations
            
        Returns:
            Tuple of (reconstruction, latents)
        """
        x = x.to(self.device)
        latents = self.encode(x)
        reconstruction = self.decode(latents)
        return reconstruction, latents
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for forward pass."""
        return self.forward(x)
    
    def get_decoder_weights(self) -> torch.Tensor:
        """
        Get the decoder weight matrix (steering vectors).
        
        Returns:
            Decoder weights of shape [d_sae, d_model]
        """
        return self.sae.W_dec
    
    def get_steering_vector(self, latent_idx: int) -> torch.Tensor:
        """
        Get the steering vector for a specific latent.
        
        Args:
            latent_idx: Index of the latent feature
            
        Returns:
            Steering vector of shape [d_model]
        """
        return self.sae.W_dec[latent_idx]

