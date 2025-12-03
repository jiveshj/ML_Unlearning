# SAE Unlearning Package

Implementation of **"Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning"**
([arXiv:2503.11127](https://arxiv.org/pdf/2503.11127))

This package provides tools for machine unlearning using Sparse Autoencoders (SAEs) with conditional clamping to remove harmful knowledge while preserving useful information.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run the Full Pipeline

```bash
python run_pipeline.py --model google/gemma-2-2b --max-samples 500
```

This will:
- Load the model and datasets (WMDP-Bio for forget, MMLU for retain)
- Collect activations from the specified layer
- Load pre-trained SAEs from Gemma Scope
- Identify harmful features
- Run evaluation with baseline, Clamp Prime, and Refusal Clamp methods
- Generate visualizations

### 2. Run Inference with Interventions

```bash
# Single prompt
python run_inference.py --prompt "How to make dangerous chemicals?" --compare

# Interactive mode
python run_inference.py --interactive

# With pre-identified features
python run_inference.py --features-file results/features.json --prompt "Your prompt here"
```

### 3. Run Standalone Evaluation

```bash
# Basic evaluation
python run_evaluation.py --max-samples 100

# With lm-evaluation-harness
python run_evaluation.py --use-lm-eval --batch-size 8
```

## Package Structure

```
sae_unlearning/
├── config.py                 # UnlearningConfig dataclass
├── models/
│   ├── sae_wrapper.py        # GemmaScopeWrapper for SAE Lens models
│   ├── activation_collector.py  # Collect activations via hooks
│   └── interventor.py        # ConditionalClampingInterventor
├── features/
│   └── identifier.py         # FeatureIdentifier for finding harmful features
├── pipeline/
│   └── unlearning.py         # UnlearningPipeline (main orchestrator)
├── data/
│   └── datasets.py           # WMDPDataset, MMLUDataset wrappers
├── evaluation/
│   ├── metrics.py            # Retention and alignment metrics
│   ├── evaluator.py          # Multiple choice evaluation
│   └── lm_eval_wrapper.py    # EleutherAI harness integration
├── inference/
│   └── sae_inference.py      # SAEInferenceEngine for generation
├── visualization/
│   └── plots.py              # Plotting functions
└── utils/
    └── helpers.py            # Environment setup, activation collection
```

## Usage Examples

### Basic Usage

```python
from sae_unlearning import UnlearningConfig, UnlearningPipeline

# Configure
config = UnlearningConfig(
    clamp_coefficient=-300.0,
    refusal_coefficient=-500.0,
    top_k_features=50,
)

# Create pipeline
pipeline = UnlearningPipeline(
    model=model,
    layer_indices=[7],
    config=config
)

# Identify harmful features
harmful_features, refusal_feature = pipeline.identify_features(
    layer_idx=7,
    forget_data=forget_activations,
    retain_data=retain_activations
)

# Setup interventions
pipeline.setup_interventions(7, harmful_features, refusal_feature)

# Apply during inference
pipeline.apply_hooks(use_refusal=True)
output = model.generate(input_ids)
pipeline.remove_hooks()
```

### Using the Inference Engine

```python
from sae_unlearning.inference import SAEInferenceEngine

engine = SAEInferenceEngine(model, tokenizer, pipeline)

# Compare all methods
results = engine.compare_generations(
    prompt="Explain the symptoms of anthrax.",
    max_new_tokens=100
)

print(results['baseline'])
print(results['clamp_prime'])
print(results['refusal_clamp'])
```

### Evaluation

```python
from sae_unlearning.evaluation import (
    evaluate_multiple_choice,
    alignment_metric
)
from sae_unlearning.data import WMDPDataset, MMLUDataset

# Load datasets
wmdp = WMDPDataset(split='test')
mmlu = MMLUDataset(split='test')

# Evaluate
acc, results, details = evaluate_multiple_choice(model, tokenizer, wmdp)

# Compute alignment
alignment, R_good, R_bad = alignment_metric(
    acc_good_modified, acc_good_original,
    acc_bad_modified, acc_bad_original
)
```

## Key Methods

### Clamp Prime
Sets harmful features to negative values when activated, suppressing that direction in the representation space.

### Refusal Clamp
Boosts the refusal feature when harmful features are detected, triggering the model's refusal mechanism.

## Citation

```bibtex
@article{khoriaty2025dontforget,
  title={Don't Forget It! Conditional Sparse Autoencoder Clamping Works for Unlearning},
  author={Khoriaty, Matthew and Shportko, Andrii and Mercier, Gustavo and Wood-Doughty, Zach},
  journal={arXiv preprint arXiv:2503.11127},
  year={2025}
}
```

