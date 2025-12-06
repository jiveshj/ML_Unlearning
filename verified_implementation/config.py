"""
Configuration for SAE-based Unlearning Methods.

Based on "Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning"
https://arxiv.org/abs/2503.11127

This module provides the ClampConfig dataclass with all hyperparameters
for the Clamp Prime and Refusal Clamp methods.
"""

import torch
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ClampConfig:
    """
    Configuration for SAE clamping interventions.
    
    Attributes:
        model_name: HuggingFace model identifier
        sae_release: SAE Lens release name for Gemma Scope
        sae_layer: Transformer layer to apply SAE intervention
        sae_width: SAE latent dimension identifier (e.g., "16k", "131k")
        
        activation_threshold: Threshold for considering a feature "active"
        clamp_coefficient: Negative value to clamp harmful features to
        refusal_coefficient: Positive value to boost refusal feature
        
        batch_size: Batch size for inference
        max_length: Maximum sequence length for tokenization
        dtype: Model precision (bfloat16 recommended for Gemma)
        
        features_file: Path to file containing harmful feature indices
        refusal_feature: Index of the refusal feature (if known)
    """
    
    # Model configuration
    model_name: str = "google/gemma-2-2b"
    sae_release: str = "gemma-scope-2b-pt-res-canonical"
    sae_layer: int = 7
    sae_width: str = "16k"
    
    # Clamping hyperparameters (paper defaults)
    activation_threshold: float = 0.0001
    clamp_coefficient: float = -300.0
    refusal_coefficient: float = 3.0
    
    # Inference configuration
    batch_size: int = 4
    max_length: int = 512
    max_new_tokens: int = 1
    dtype: str = "bfloat16"
    
    # Feature configuration
    features_file: Optional[str] = None
    refusal_feature: Optional[int] = None
    harmful_features: List[int] = field(default_factory=list)
    
    # Device
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    def __post_init__(self):
        """Post-initialization processing."""
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
    
    @property
    def sae_id(self) -> str:
        """Get the full SAE identifier for SAE Lens."""
        return f"layer_{self.sae_layer}/width_{self.sae_width}/canonical"
    
    @property
    def torch_dtype(self) -> torch.dtype:
        """Get the torch dtype from string."""
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "sae_release": self.sae_release,
            "sae_layer": self.sae_layer,
            "sae_width": self.sae_width,
            "sae_id": self.sae_id,
            "activation_threshold": self.activation_threshold,
            "clamp_coefficient": self.clamp_coefficient,
            "refusal_coefficient": self.refusal_coefficient,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "dtype": self.dtype,
            "features_file": self.features_file,
            "refusal_feature": self.refusal_feature,
            "num_harmful_features": len(self.harmful_features),
            "device": str(self.device),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ClampConfig":
        """Create config from dictionary."""
        d = d.copy()
        # Remove computed properties
        d.pop("sae_id", None)
        d.pop("num_harmful_features", None)
        if "device" in d:
            d["device"] = torch.device(d["device"])
        return cls(**d)
    
    def load_features_from_file(self, filepath: str = None) -> List[int]:
        """
        Load harmful feature indices from a file.
        
        Args:
            filepath: Path to file with one feature index per line.
                     Uses self.features_file if not provided.
        
        Returns:
            List of feature indices
        """
        filepath = filepath or self.features_file
        if filepath is None:
            raise ValueError("No features file specified")
        
        features = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    features.append(int(line))
        
        self.harmful_features = features
        return features


# Default MMLU subjects used in the paper
MMLU_SUBJECTS = [
    "high_school_us_history",
    "high_school_geography",
    "human_aging",
    "college_computer_science",
]

# Answer choice letters
CHOICE_LETTERS = ["A", "B", "C", "D"]

