"""
Intent Classifier Inference Script

Runs inference on the fine-tuned DistilBERT intent classifier model
to classify prompts as 'benign' or 'harmful'.

Usage:
    python run_intent_inference.py "Your prompt here"
    python run_intent_inference.py --interactive  # For multiple prompts
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
from pathlib import Path


def load_model(model_path: str):
    """Load the fine-tuned model and tokenizer."""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Set to evaluation mode
    model.eval()
    
    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    print(f"Model loaded on device: {device}")
    
    return model, tokenizer, device


def classify_prompt(prompt: str, model, tokenizer, device, max_length: int = 128):
    """
    Classify a single prompt.
    
    Returns:
        dict: Contains prediction label, confidence scores, and raw logits
    """
    # Tokenize the input
    inputs = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probabilities = F.softmax(logits, dim=-1)
        
        # Get prediction
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        
    # Get label mapping from model config
    id2label = model.config.id2label
    predicted_label = id2label[predicted_class]
    
    # Extract probabilities
    probs = probabilities[0].cpu().numpy()
    
    return {
        "prompt": prompt,
        "prediction": predicted_label,
        "predicted_class": predicted_class,
        "confidence": float(probs[predicted_class]),
        "probabilities": {
            id2label[i]: float(probs[i]) for i in range(len(probs))
        },
        "logits": logits[0].cpu().numpy().tolist()
    }


def print_result(result: dict):
    """Pretty print the classification result."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION RESULT")
    print("=" * 60)
    print(f"\nPrompt: {result['prompt'][:100]}{'...' if len(result['prompt']) > 100 else ''}")
    print(f"\n>>> Prediction: {result['prediction'].upper()}")
    print(f">>> Confidence: {result['confidence']:.2%}")
    print(f"\nProbabilities:")
    for label, prob in result['probabilities'].items():
        bar = "█" * int(prob * 30)
        print(f"  {label:>8}: {prob:.4f} {bar}")
    print("=" * 60)


def interactive_mode(model, tokenizer, device):
    """Run in interactive mode for multiple prompts."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Enter prompts to classify (type 'quit' or 'exit' to stop)")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\nEnter prompt: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            if not prompt:
                print("Please enter a non-empty prompt.")
                continue
                
            result = classify_prompt(prompt, model, tokenizer, device)
            print_result(result)
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on the intent classifier model"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=None,
        help="The prompt to classify"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the fine-tuned model directory"
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode for multiple prompts"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization"
    )
    
    args = parser.parse_args()
    
    # Determine model path
    if args.model_path:
        model_path = args.model_path
    else:
        # Default to the model directory relative to this script
        script_dir = Path(__file__).parent
        model_path = script_dir / "intent_classifier_model"
    
    # Load model
    model, tokenizer, device = load_model(str(model_path))
    
    if args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.prompt:
        result = classify_prompt(args.prompt, model, tokenizer, device, args.max_length)
        print_result(result)
    else:
        # No prompt provided and not interactive mode - show help
        parser.print_help()
        print("\n\nExample usage:")
        print('  python run_intent_inference.py "How do I make a cake?"')
        print('  python run_intent_inference.py "How do I synthesize dangerous chemicals?"')
        print('  python run_intent_inference.py --interactive')


if __name__ == "__main__":
    main()

