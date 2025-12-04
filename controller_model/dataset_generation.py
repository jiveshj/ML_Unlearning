import json
import glob

# Load all dataset files
all_data = []
for filepath in sorted(glob.glob('synthetic_dataset_*.json')):
    if 'some_samples' in filepath or 'error' in filepath:
        continue  # Skip non-numbered files
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
        all_data.extend(data)

print(f"\nTotal entries loaded: {len(all_data)}")

# Deduplicate: keep unique prompts with their labels
# Using dict to preserve first occurrence
unique_prompts = {}
duplicates = 0

for item in all_data:
    prompt = item['prompt']
    label = item['label']
    
    if prompt not in unique_prompts:
        unique_prompts[prompt] = label
    else:
        duplicates += 1
        # Check for label conflicts
        if unique_prompts[prompt] != label:
            print(f"Warning: Label conflict for prompt: '{prompt[:50]}...'")
            print(f"  Existing label: {unique_prompts[prompt]}, New label: {label}")

# Create final dataset
final_dataset = [
    {"prompt": prompt, "label": label}
    for prompt, label in unique_prompts.items()
]

# Count labels
label_0_count = sum(1 for item in final_dataset if item['label'] == 0)
label_1_count = sum(1 for item in final_dataset if item['label'] == 1)

print(f"\n--- Summary ---")
print(f"Total prompts loaded: {len(all_data)}")
print(f"Duplicates removed: {duplicates}")
print(f"Unique prompts: {len(final_dataset)}")
print(f"  Label 0 (Retain): {label_0_count}")
print(f"  Label 1 (Forget): {label_1_count}")

# Save the final dataset
# with open('final_unique_dataset.json', 'w') as f:
#     json.dump(final_dataset, f, indent=2)

print(f"\nSaved to final_unique_dataset.json")
