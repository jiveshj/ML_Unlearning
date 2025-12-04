from datasets import load_dataset
from typing import List, Optional


def load_wmdp_bio_forget(max_samples: Optional[int] = None) -> List[str]:
    """Load WMDP-Bio forget set (harmful biosecurity knowledge to unlearn)."""
    print("\nLoading WMDP-Bio Forget set...")
    
    ds = load_dataset("cais/wmdp-bio-forget-corpus",split='train')
    
    # Format questions as prompts
    prompts = []
    for item in ds:
        abstract = item["abstract"]
        text = item["text"]
        prompt = f"Abstract: {abstract}\n\nText: {text}\n\n"
        prompts.append(prompt)
    
    
    print(f"  Loaded {len(prompts)} forget prompts")
    return prompts


load_wmdp_bio_forget()