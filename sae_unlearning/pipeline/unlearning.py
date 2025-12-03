"""
Unlearning Pipeline module.

Provides the complete end-to-end pipeline for SAE-based unlearning,
orchestrating SAE loading, feature identification, and intervention setup.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from tqdm.auto import tqdm
from sae_lens import SAE

from ..config import UnlearningConfig
from ..models.sae_wrapper import GemmaScopeWrapper
from ..models.interventor import ConditionalClampingInterventor
from ..features.identifier import FeatureIdentifier


class UnlearningPipeline:
    """
    Complete pipeline for SAE-based unlearning using pre-trained Gemma Scope SAEs.
    
    This class orchestrates the full unlearning workflow:
    1. Loading pre-trained SAEs for specified layers
    2. Identifying harmful features from forget/retain data
    3. Setting up conditional clamping interventions
    4. Applying hooks during inference
    
    Attributes:
        model: The LLM to apply unlearning to
        layer_indices: Which transformer layers to intervene on
        config: UnlearningConfig with hyperparameters
        saes: Dictionary of GemmaScopeWrapper objects per layer
        interventors: Dictionary of ConditionalClampingInterventor objects
        hooks: List of registered forward hook handles
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
        config: UnlearningConfig,
        sae_release: str = "gemma-scope-2b-pt-res-canonical"
    ):
        """
        Initialize the pipeline.
        
        Args:
            model: The LLM to apply unlearning to
            layer_indices: Which transformer layers to intervene on
            config: Unlearning configuration
            sae_release: Gemma Scope release name for SAE loading
        """
        self.model = model
        self.layer_indices = layer_indices
        self.config = config
        self.sae_release = sae_release
        
        # Storage for activations (set externally)
        self.forget_acts: Optional[Dict[int, torch.Tensor]] = None
        self.retain_acts: Optional[Dict[int, torch.Tensor]] = None
        
        # Load pre-trained SAEs for each layer
        self.saes: Dict[str, GemmaScopeWrapper] = {}
        print("Loading pre-trained SAEs from Gemma Scope...")
        for layer_idx in layer_indices:
            sae_id = f"layer_{layer_idx}/width_16k/canonical"
            
            # Handle deprecation warning - new API returns just the SAE
            try:
                sae_result = SAE.from_pretrained(
                    release=sae_release,
                    sae_id=sae_id,
                )
                # Check if it's a tuple (old API) or just the SAE (new API)
                if isinstance(sae_result, tuple):
                    sae_model = sae_result[0]
                else:
                    sae_model = sae_result
            except Exception as e:
                print(f"Error loading SAE for layer {layer_idx}: {e}")
                raise
            
            wrapper = GemmaScopeWrapper(sae_model, device=config.device)
            self.saes[str(layer_idx)] = wrapper
            print(f"  Layer {layer_idx}: d_model={wrapper.d_model}, d_sae={wrapper.d_sae}")
        
        print("✓ All SAEs loaded")
        
        self.interventors: Dict[int, ConditionalClampingInterventor] = {}
        self.hooks: List = []
    
    def identify_features(
        self,
        layer_idx: int,
        forget_data: torch.Tensor,
        retain_data: torch.Tensor,
        refusal_data: Optional[torch.Tensor] = None,
        batch_size: int = 1000
    ) -> Tuple[List[int], Optional[int]]:
        """
        Identify harmful and refusal features for a layer.
        
        Processes activations through the SAE and uses FeatureIdentifier
        to find features that are differentially activated.
        
        Args:
            layer_idx: Which layer to analyze
            forget_data: Activations on harmful/forget dataset [N, hidden_dim]
            retain_data: Activations on benign/retain dataset [N, hidden_dim]
            refusal_data: Optional activations on refusal responses
            batch_size: Batch size for processing to avoid OOM
            
        Returns:
            Tuple of (harmful_features list, refusal_feature or None)
        """
        sae_wrapper = self.saes[str(layer_idx)]
        
        print(f"Forget_data shape: {forget_data.shape}")
        print(f"Retain_data shape: {retain_data.shape}")
        
        with torch.no_grad():
            forget_data = forget_data.float().to(self.config.device)
            retain_data = retain_data.float().to(self.config.device)
            
            # Process in batches to avoid OOM
            forget_latents_list = []
            retain_latents_list = []
            
            for i in tqdm(range(0, forget_data.shape[0], batch_size), desc="Processing forget data"):
                batch = forget_data[i:i+batch_size]
                latents = sae_wrapper.encode(batch)
                forget_latents_list.append(latents.cpu())
            
            for i in tqdm(range(0, retain_data.shape[0], batch_size), desc="Processing retain data"):
                batch = retain_data[i:i+batch_size]
                latents = sae_wrapper.encode(batch)
                retain_latents_list.append(latents.cpu())
            
            forget_latents = torch.cat(forget_latents_list, dim=0)
            retain_latents = torch.cat(retain_latents_list, dim=0)
            
            print(f"Forget latents shape: {forget_latents.shape}")
            print(f"Retain latents shape: {retain_latents.shape}")
            
            # Add sequence dimension if needed
            if forget_latents.dim() == 2:
                forget_latents = forget_latents.unsqueeze(1)
                retain_latents = retain_latents.unsqueeze(1)
            
            # Identify harmful features
            harmful_features = FeatureIdentifier.identify_harmful_features(
                forget_latents,
                retain_latents,
                self.config
            )
            
            print(f"Identified {len(harmful_features)} harmful features")
            
            # Identify refusal feature if data provided, else use default
            refusal_feature = 15864  # Default for gemma-2-2b layer 7
            if refusal_data is not None:
                refusal_data = refusal_data.float().to(self.config.device)
                
                refusal_latents_list = []
                for i in range(0, refusal_data.shape[0], batch_size):
                    batch = refusal_data[i:i+batch_size]
                    latents = sae_wrapper.encode(batch)
                    refusal_latents_list.append(latents.cpu())
                
                refusal_latents = torch.cat(refusal_latents_list, dim=0)
                
                if refusal_latents.dim() == 2:
                    refusal_latents = refusal_latents.unsqueeze(1)
                
                refusal_feature = FeatureIdentifier.identify_refusal_feature(
                    refusal_latents,
                    threshold=self.config.activation_threshold
                )
                print(f"Identified refusal feature: {refusal_feature}")
        
        return harmful_features, refusal_feature
    
    def setup_interventions(
        self,
        layer_idx: int,
        harmful_features: List[int],
        refusal_feature: Optional[int] = None
    ):
        """
        Setup interventor for a specific layer.
        
        Creates a ConditionalClampingInterventor configured with the
        identified harmful and refusal features.
        
        Args:
            layer_idx: Which layer to setup intervention for
            harmful_features: List of harmful feature indices
            refusal_feature: Optional refusal feature index
        """
        sae_wrapper = self.saes[str(layer_idx)]
        interventor = ConditionalClampingInterventor(
            sae_wrapper=sae_wrapper,
            harmful_features=harmful_features,
            refusal_feature=refusal_feature,
            config=self.config
        )
        self.interventors[layer_idx] = interventor
    
    def apply_hooks(self, use_refusal: bool = True):
        """
        Apply forward hooks to intervene on model activations during inference.
        
        Registers hooks on the specified layers that apply the clamping
        intervention during the forward pass.
        
        Args:
            use_refusal: Whether to use refusal clamping (more aggressive)
        """
        self.remove_hooks()
        
        for layer_idx in self.layer_indices:
            if layer_idx not in self.interventors:
                continue
            
            interventor = self.interventors[layer_idx]
            
            def hook_fn(module, input, output, interventor=interventor, use_refusal=use_refusal):
                if isinstance(output, tuple):
                    activations = output[0]
                else:
                    activations = output
                
                modified = interventor(activations, use_refusal=use_refusal)
                
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                else:
                    return modified
            
            layer = self._get_layer(layer_idx)
            handle = layer.register_forward_hook(hook_fn)
            self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def _get_layer(self, layer_idx: int):
        """
        Get the transformer layer by index.
        
        Supports common HuggingFace model architectures.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            The transformer layer module
        """
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h[layer_idx]
        else:
            raise RuntimeError("Could not find transformer layers")
    
    def get_sae(self, layer_idx: int) -> GemmaScopeWrapper:
        """Get the SAE wrapper for a specific layer."""
        return self.saes[str(layer_idx)]
    
    def save_config(self, path: str):
        """Save the pipeline configuration to a JSON file."""
        import json
        
        config_dict = {
            'config': self.config.to_dict(),
            'layer_indices': self.layer_indices,
            'sae_release': self.sae_release,
        }
        
        # Add harmful features if identified
        if self.interventors:
            config_dict['interventors'] = {
                str(k): {
                    'harmful_features': v.harmful_features,
                    'refusal_feature': v.refusal_feature
                }
                for k, v in self.interventors.items()
            }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)

