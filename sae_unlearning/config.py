"""
Configuration module for SAE-based unlearning.

Contains the UnlearningConfig dataclass with all hyperparameters for the unlearning process.
"""

import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class UnlearningConfig:
    """
    Configuration for the SAE-based unlearning process.
    
    Attributes:
        activation_threshold: Threshold for considering a latent "active" (default: 0.01)
        clamp_coefficient: Negative coefficient for clamping harmful features (default: -5.0)
        refusal_coefficient: Positive coefficient for refusal feature boosting (default: 3.0)
        layer_indices: Which transformer layers to apply SAE intervention on
        top_k_features: Number of top harmful features to identify and clamp
        retain_frequency_threshold: Discard features with retain frequency above this
        device: Device to run computations on (auto-detected if not specified)
    """
    activation_threshold: float = 0.01
    clamp_coefficient: float = -5.0
    refusal_coefficient: float = 3.0
    layer_indices: List[int] = field(default_factory=list)
    top_k_features: int = 50
    retain_frequency_threshold: float = 1e-4
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    def __post_init__(self):
        """Ensure device is a torch.device object."""
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "activation_threshold": self.activation_threshold,
            "clamp_coefficient": self.clamp_coefficient,
            "refusal_coefficient": self.refusal_coefficient,
            "layer_indices": self.layer_indices,
            "top_k_features": self.top_k_features,
            "retain_frequency_threshold": self.retain_frequency_threshold,
            "device": str(self.device),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "UnlearningConfig":
        """Create config from dictionary."""
        d = d.copy()
        if "device" in d:
            d["device"] = torch.device(d["device"])
        return cls(**d)

