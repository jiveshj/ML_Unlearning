"""
Verified Implementation of SAE-based Unlearning Methods.

This module implements the Clamp Prime and Refusal Clamp methods from:
"Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning"
by Matthew Khoriaty, Andrii Shportko, Gustavo Mercier, and Zach Wood-Doughty
(Northwestern University)

Paper: https://arxiv.org/abs/2503.11127
GitHub: https://github.com/AMindToThink/sae_jailbreak_unlearning

SAE Configuration:
    - Model: google/gemma-2-2b
    - SAE Release: gemma-scope-2b-pt-res-canonical
    - SAE ID: layer_7/width_16k/canonical
"""

from .config import ClampConfig
from .clamp_prime import ClampPrimeHook
from .refusal_clamp import RefusalClampHook

__all__ = [
    "ClampConfig",
    "ClampPrimeHook",
    "RefusalClampHook",
]

__version__ = "1.0.0"
__paper__ = "Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning"
__arxiv__ = "https://arxiv.org/abs/2503.11127"

