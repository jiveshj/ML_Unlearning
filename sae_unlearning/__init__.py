"""
SAE Unlearning Package

Implementation of "Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning"
https://arxiv.org/pdf/2503.11127

This package provides tools for machine unlearning using Sparse Autoencoders (SAEs)
with conditional clamping to remove harmful knowledge while preserving useful information.
"""

from .config import UnlearningConfig
from .pipeline.unlearning import UnlearningPipeline
from .models.sae_wrapper import GemmaScopeWrapper
from .models.interventor import ConditionalClampingInterventor
from .features.identifier import FeatureIdentifier

__version__ = "0.1.0"
__all__ = [
    "UnlearningConfig",
    "UnlearningPipeline", 
    "GemmaScopeWrapper",
    "ConditionalClampingInterventor",
    "FeatureIdentifier",
]

