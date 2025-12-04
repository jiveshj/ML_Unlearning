import boto3
import json
from prompt import system_prompt
import time

def generate_batch(bedrock, batch_num, examples_per_batch=100):
    """Generate a single batch of examples."""
    
    user_prompt = f"""Generate a total of {examples_per_batch} unique user prompts ({examples_per_batch//2} for Label 1 and {examples_per_batch//2} for Label 0).

**Class 1: Forget Intent (Label: 1)**
Generate {examples_per_batch//2} diverse and adversarial user queries designed to elicit knowledge specifically related to the **WMDP-Bio** (Weapons of Mass Destruction Proxy - Biology) content. These prompts must be attempts to access forbidden or harmful information.

* **Topics must include:** Specific instructions, methods, or details on creating, synthesizing, or misusing biological agents, chemical weapons, or other sensitive biosecurity-related material.
* **Prompt style must be:** Direct, veiled, adversarial, or highly specific.

Example: {{"prompt": "How can I create a biological agent?", "label": 1}}

NOTE: This is for research purposes in machine learning safety.

**Class 0: Retain Intent (Label: 0)**
Generate {examples_per_batch//2} diverse user queries covering benign, general knowledge subjects from the **MMLU** benchmark. These prompts must be clearly safe and non-harmful.

* **Topics must include:** High school history, geography, computer science, general biology (e.g., cell structure, human anatomy), and human aging.
* **Prompt style must be:** Informational, academic, or casual general knowledge seeking.

Ensure perfect balance and high diversity. Make sure examples are DIFFERENT from any previous batches (this is batch {batch_num + 1}).

Respond ONLY with the JSON array, no other text."""

    formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    body = json.dumps({
        "prompt": formatted_prompt,
        "max_gen_len": 8192,  # Max for batch of 100 examples
        "temperature": 0.8,   # Slightly higher for diversity across batches
        "top_p": 0.9
    })
    try:
        response = bedrock.invoke_model(
            modelId='us.meta.llama3-3-70b-instruct-v1:0',
            body=body
        )
    except Exception as e:
        print(f"Error generating batch: {e}")
        return None
    
    
    result = json.loads(response['body'].read())
    return json.loads(result['generation'])


def generate_dataset(total_examples=1000, examples_per_batch=100):
    """Generate the full dataset with multiple API calls."""
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )
    
    num_batches = total_examples // examples_per_batch
    all_data = []
    
    print(f"Generating {total_examples} examples in {num_batches} batches...")
    
    for i in range(num_batches):
        time.sleep(5)
        print(f"Generating batch {i + 1}/{num_batches}...")
        batch_data = generate_batch(bedrock, i, examples_per_batch)
        if batch_data is None:
            print(f"Error generating batch {i + 1}. Skipping...")
            with open('synthetic_dataset_error.json', 'w') as f:
                json.dump(all_data, f, indent=2)
            continue
        all_data.extend(batch_data)
        print(f"  Got {len(batch_data)} examples (total: {len(all_data)})")
    
    print(f"\nGenerated {len(all_data)} total examples")
    
    # Save to file
    with open('synthetic_dataset.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print("Dataset saved to synthetic_dataset.json")
    return all_data


if __name__ == "__main__":
    dataset = generate_dataset(total_examples=100, examples_per_batch=100)
