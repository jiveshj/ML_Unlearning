from datasets import load_dataset
import json

# Load the WMDP dataset
ds_wmdp = load_dataset("cais/wmdp", "wmdp-bio")

# Create the dataset in the required format
wmdp_dataset = [
    {
        "prompt": example['question'],
        "label": 1
    }
    for example in ds_wmdp['test']
]

# Save to JSON file
with open('wmdp_bio_dataset.json', 'w') as f:
    json.dump(wmdp_dataset, f, indent=2)

print(f"Created dataset with {len(wmdp_dataset)} questions")

ds_mmlu = load_dataset("cais/mmlu", "all")

print(ds_mmlu['test'])

computer_science_ds = ds_mmlu['test'].filter(lambda x: x['subject'] == 'college_computer_science')
geography_ds = ds_mmlu['test'].filter(lambda x: x['subject'] == 'high_school_geography')
history_ds = ds_mmlu['test'].filter(lambda x: x['subject'] == 'high_school_us_history')
human_aging_ds = ds_mmlu['test'].filter(lambda x: x['subject'] == 'human_aging')

mmlu_dataset = [
    {
        "prompt": example['question'],
        "label": 0
    }
    for example in computer_science_ds 
]

for example in geography_ds:
    mmlu_dataset.append({
        "prompt": example['question'],
        "label": 0
    })

for example in history_ds:
    mmlu_dataset.append({
        "prompt": example['question'],
        "label": 0
    })

for example in human_aging_ds:
    mmlu_dataset.append({
        "prompt": example['question'],
        "label": 0
    })
# Save to JSON file
with open('mmlu_dataset.json', 'w') as f:
    json.dump(mmlu_dataset, f, indent=2)

print(f"Created dataset with {len(mmlu_dataset)} questions")