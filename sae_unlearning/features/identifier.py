"""
Feature Identifier module for identifying harmful and refusal features.

Analyzes SAE latent activation patterns to identify features that are
more active on harmful/forget data than on benign/retain data.
"""

import torch
from typing import List, Tuple

from ..config import UnlearningConfig


class FeatureIdentifier:
    """
    Identifies harmful and refusal features in SAE latent space.
    
    Analyzes activation patterns on forget vs retain datasets to identify
    features associated with harmful knowledge that should be suppressed.
    
    All methods are static for easy use without instantiation.
    """
    
    @staticmethod
    def compute_activation_frequency(
        latents: torch.Tensor, 
        threshold: float = 0.01
    ) -> torch.Tensor:
        """
        Compute frequency of non-zero activations for each feature.
        
        Calculates what proportion of samples/tokens activate each feature
        above the given threshold.
        
        Args:
            latents: SAE latent activations
                     Shape: [batch, d_sae] or [batch, seq_len, d_sae]
            threshold: Minimum value to consider a latent "active"
            
        Returns:
            Tensor of shape [d_sae] with activation frequencies per feature
            
        Raises:
            ValueError: If latents is not 2D or 3D
        """
        if latents.dim() == 2:
            # [N, d_sae]
            active = (latents.abs() > threshold).float()
            freqs = active.mean(dim=0)
            return freqs
        elif latents.dim() == 3:
            # [B, S, d_sae]
            active = (latents.abs() > threshold).float()
            freqs = active.mean(dim=(0, 1))
            return freqs
        else:
            raise ValueError(f"latents must be 2D or 3D, got {latents.dim()}D")
    
    @staticmethod
    def identify_harmful_features(
        forget_latents: torch.Tensor,
        retain_latents: torch.Tensor,
        config: UnlearningConfig
    ) -> List[int]:
        """
        Identify features more active on forget data than retain data.
        
        Algorithm:
        1. Compute activation frequency on both datasets
        2. Filter out features with high retain frequency (too general)
        3. Rank remaining by forget frequency
        4. Return top-k features
        
        Args:
            forget_latents: Latent activations on harmful/forget dataset
            retain_latents: Latent activations on benign/retain dataset
            config: UnlearningConfig with thresholds and top_k
            
        Returns:
            List of feature indices identified as harmful
        """
        forget_freq = FeatureIdentifier.compute_activation_frequency(
            forget_latents, config.activation_threshold
        )
        retain_freq = FeatureIdentifier.compute_activation_frequency(
            retain_latents, config.activation_threshold
        )
        
        # Discard features with high retain frequency (too general to be harmful-specific)
        keep_mask = retain_freq <= config.retain_frequency_threshold
        candidates = torch.where(keep_mask)[0].tolist()
        
        if len(candidates) == 0:
            return []
        
        # Rank candidates by forget_freq and pick top-k
        forget_vals = forget_freq[candidates]
        sorted_idx = torch.argsort(forget_vals, descending=True)
        topk = min(config.top_k_features, len(candidates))
        selected = [candidates[i] for i in sorted_idx[:topk].tolist()]
        
        return selected
    
    @staticmethod
    def identify_refusal_feature(
        refusal_latents: torch.Tensor,
        threshold: float = 0.01
    ) -> int:
        """
        Identify the primary refusal feature.
        
        Finds the feature most frequently activated when the model
        produces refusal responses (e.g., "I cannot help with that").
        
        Args:
            refusal_latents: Activations when model produces refusal responses
            threshold: Activation threshold
            
        Returns:
            Index of the most frequently activated refusal feature
        """
        frequencies = FeatureIdentifier.compute_activation_frequency(
            refusal_latents, threshold
        )
        refusal_feature = torch.argmax(frequencies).item()
        return refusal_feature
    
    @staticmethod
    def get_feature_statistics(
        latents: torch.Tensor,
        feature_indices: List[int],
        threshold: float = 0.01
    ) -> dict:
        """
        Get detailed statistics for specific features.
        
        Args:
            latents: SAE latent activations
            feature_indices: Which features to analyze
            threshold: Activation threshold
            
        Returns:
            Dictionary with statistics per feature
        """
        stats = {}
        for idx in feature_indices:
            feat_vals = latents[..., idx]
            active_mask = feat_vals.abs() > threshold
            active_vals = feat_vals[active_mask]
            
            stats[idx] = {
                'frequency': active_mask.float().mean().item(),
                'mean_when_active': active_vals.mean().item() if len(active_vals) > 0 else 0.0,
                'max_activation': feat_vals.max().item(),
                'min_activation': feat_vals.min().item(),
            }
        
        return stats
    
    @staticmethod
    def compute_feature_differential(
        forget_latents: torch.Tensor,
        retain_latents: torch.Tensor,
        threshold: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the differential between forget and retain activation frequencies.
        
        Useful for visualization and understanding which features are
        most differentially activated.
        
        Args:
            forget_latents: Latent activations on forget dataset
            retain_latents: Latent activations on retain dataset
            threshold: Activation threshold
            
        Returns:
            Tuple of (forget_freq, retain_freq, differential)
        """
        forget_freq = FeatureIdentifier.compute_activation_frequency(
            forget_latents, threshold
        )
        retain_freq = FeatureIdentifier.compute_activation_frequency(
            retain_latents, threshold
        )
        differential = forget_freq - retain_freq
        
        return forget_freq, retain_freq, differential

