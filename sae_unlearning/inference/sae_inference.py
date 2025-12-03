"""
SAE Inference module.

Provides utilities for running inference with SAE-based interventions applied.
Supports both single prompt and batch generation with clamping hooks active.
"""

import torch
from typing import List, Dict, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer


class SAEInferenceEngine:
    """
    Engine for running inference with SAE interventions.
    
    Wraps a model and pipeline to provide convenient methods for
    generating text with clamping interventions applied.
    
    Attributes:
        model: The language model
        tokenizer: Tokenizer
        pipeline: UnlearningPipeline with interventions configured
        device: Device for inference
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        pipeline=None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the inference engine.
        
        Args:
            model: HuggingFace model
            tokenizer: Tokenizer
            pipeline: UnlearningPipeline with interventions setup (optional)
            device: Device for inference (auto-detected if None)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.device = device or next(model.parameters()).device
        
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        use_intervention: bool = True,
        use_refusal: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **generate_kwargs
    ) -> str:
        """
        Generate text for a single prompt with optional intervention.
        
        Args:
            prompt: Input prompt text
            max_new_tokens: Maximum tokens to generate
            use_intervention: Whether to apply SAE intervention
            use_refusal: Whether to use refusal clamping (vs clamp prime)
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling (vs greedy)
            **generate_kwargs: Additional kwargs for model.generate()
            
        Returns:
            Generated text (including prompt)
        """
        # Apply hooks if requested
        if use_intervention and self.pipeline is not None:
            self.pipeline.apply_hooks(use_refusal=use_refusal)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
        finally:
            # Remove hooks
            if use_intervention and self.pipeline is not None:
                self.pipeline.remove_hooks()
        
        return generated_text
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        use_intervention: bool = True,
        use_refusal: bool = False,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **generate_kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts with optional intervention.
        
        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            use_intervention: Whether to apply SAE intervention
            use_refusal: Whether to use refusal clamping
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **generate_kwargs: Additional kwargs for model.generate()
            
        Returns:
            List of generated texts
        """
        # Apply hooks if requested
        if use_intervention and self.pipeline is not None:
            self.pipeline.apply_hooks(use_refusal=use_refusal)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)
            
            # Generate
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generate_kwargs
                )
            
            # Decode all
            generated_texts = [
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            ]
            
        finally:
            # Remove hooks
            if use_intervention and self.pipeline is not None:
                self.pipeline.remove_hooks()
        
        return generated_texts
    
    def compare_generations(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        **generate_kwargs
    ) -> Dict[str, str]:
        """
        Generate text with and without intervention for comparison.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **generate_kwargs: Additional generation kwargs
            
        Returns:
            Dict with 'baseline', 'clamp_prime', and 'refusal_clamp' generations
        """
        results = {}
        
        # Baseline (no intervention)
        results['baseline'] = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            use_intervention=False,
            temperature=temperature,
            **generate_kwargs
        )
        
        if self.pipeline is not None:
            # Clamp Prime
            results['clamp_prime'] = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                use_intervention=True,
                use_refusal=False,
                temperature=temperature,
                **generate_kwargs
            )
            
            # Refusal Clamp
            results['refusal_clamp'] = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                use_intervention=True,
                use_refusal=True,
                temperature=temperature,
                **generate_kwargs
            )
        
        return results
    
    def get_response_only(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        use_intervention: bool = True,
        use_refusal: bool = False,
        **generate_kwargs
    ) -> str:
        """
        Generate text and return only the new tokens (excluding prompt).
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            use_intervention: Whether to apply intervention
            use_refusal: Whether to use refusal clamping
            **generate_kwargs: Additional generation kwargs
            
        Returns:
            Generated text without the original prompt
        """
        full_text = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            use_intervention=use_intervention,
            use_refusal=use_refusal,
            **generate_kwargs
        )
        
        # Remove the prompt from the output
        if full_text.startswith(prompt):
            return full_text[len(prompt):].strip()
        return full_text


def generate_with_intervention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    pipeline=None,
    use_refusal: bool = False,
    max_new_tokens: int = 100,
    **generate_kwargs
) -> str:
    """
    Convenience function for single generation with intervention.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompt: Input prompt
        pipeline: UnlearningPipeline (optional)
        use_refusal: Whether to use refusal clamping
        max_new_tokens: Maximum tokens to generate
        **generate_kwargs: Additional generation kwargs
        
    Returns:
        Generated text
    """
    engine = SAEInferenceEngine(model, tokenizer, pipeline)
    return engine.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        use_intervention=(pipeline is not None),
        use_refusal=use_refusal,
        **generate_kwargs
    )


def batch_generate_with_intervention(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    pipeline=None,
    use_refusal: bool = False,
    max_new_tokens: int = 100,
    **generate_kwargs
) -> List[str]:
    """
    Convenience function for batch generation with intervention.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        prompts: List of input prompts
        pipeline: UnlearningPipeline (optional)
        use_refusal: Whether to use refusal clamping
        max_new_tokens: Maximum tokens to generate
        **generate_kwargs: Additional generation kwargs
        
    Returns:
        List of generated texts
    """
    engine = SAEInferenceEngine(model, tokenizer, pipeline)
    return engine.generate_batch(
        prompts,
        max_new_tokens=max_new_tokens,
        use_intervention=(pipeline is not None),
        use_refusal=use_refusal,
        **generate_kwargs
    )

