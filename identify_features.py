import torch


def compute_normalized_frequency(tensor_path: str) -> torch.Tensor:
    """
    Load a tensor and compute normalized frequency for each feature.
    
    Args:
        tensor_path: Path to the .pt file containing the tensor
        
    Returns:
        Tensor of shape (16384,) containing normalized frequencies
    """
    # Load the tensor
    print(f"Loading tensor from {tensor_path}...")
    tensor = torch.load(tensor_path)
    print(f"Original shape: {tensor.shape}")
    
    # Reshape from (2000, 128, 16384) to (2000*128, 16384)
    num_samples = tensor.shape[0]
    seq_len = tensor.shape[1]
    num_features = tensor.shape[2]
    
    reshaped = tensor.reshape(num_samples * seq_len, num_features)
    print(f"Reshaped to: {reshaped.shape}")
    
    total_positions = reshaped.shape[0]  # 2000 * 128 = 256000
    
    # Count activations greater than 0 for each feature
    activations_gt_zero = (reshaped > 0).sum(dim=0).float()
    
    # Compute normalized frequency
    normalized_freq = activations_gt_zero / total_positions

    
    # print(f"Normalized frequency shape: {normalized_freq.shape}")
    
    #Read frequent features from file
    with open("frequencies_9/retain_missing_features_layer9_machine_readable.txt", "r") as f:
        frequent_features = [int(line.strip()) for line in f]
    
    # # Create tensor of selected activations (frequencies at frequent_features indices)
    frequent_features_tensor = torch.tensor(frequent_features)
    selected_activations = normalized_freq[frequent_features_tensor]
    print(f"Selected activations shape: {selected_activations.shape}")
    
    # # # Get topk from selected activations
    topk_values, topk_indices = selected_activations.topk(k=50)
    #Map back to original feature indices
    topk_features = frequent_features_tensor[topk_indices].tolist()
    
    print(f"Top 50 features from frequent features list:")
    for feat, val in zip(topk_features, topk_values.tolist()):
        print(f"  Feature {feat}: {val:.6f}")
    
    with open("frequencies_9/features_to_clamp_layer9.txt", "w") as f:
        for feature in topk_features:
            f.write(f"{feature}\n")
    # print(f"Min frequency: {normalized_freq.min().item():.6f}")
    # print(f"Max frequency: {normalized_freq.max().item():.6f}")
    # print(f"Mean frequency: {normalized_freq.mean().item():.6f}")
    
    # # Count features by frequency ranges
    # num_never_active = (normalized_freq == 0).sum().item()
    # num_rare = ((normalized_freq > 0) & (normalized_freq < 0.0001)).sum().item()
    # num_frequent = (normalized_freq >= 0.0001).sum().item()
    
    # print(f"\nFeature frequency distribution:")
    # print(f"  Never active (freq = 0): {num_never_active}")
    # print(f"  Rare (0 < freq < 0.0001): {num_rare}")
    # print(f"  Frequent (freq >= 0.0001): {num_frequent}")
    
    # # Get features with frequency > 0.0001
    # frequent_mask = normalized_freq > 0.0001
    # frequent_features = torch.nonzero(frequent_mask).squeeze(-1)
    # frequent_freqs = normalized_freq[frequent_mask]
    
    # print(f"\nFeatures with frequency > 0.0001 ({len(frequent_features)} total):")
    # with open("frequencies_9/retain_frequent_features_greater_than_0.0001_layer9_machine_readable.txt", "w") as f:
    #     for idx, freq in zip(frequent_features.tolist(), frequent_freqs.tolist()):
    #         f.write(f"{idx}\n")
    # for idx, freq in zip(frequent_features.tolist(), frequent_freqs.tolist()):
    #     print(f"  Feature {idx}: {freq:.6f}")
    
    return normalized_freq


def main():
    tensor_forget_path = "/home/ubuntu/ML_Unlearning/sae_activations/forget_sae_activations_layer9.pt"
    tensor_retain_path = "/home/ubuntu/ML_Unlearning/sae_activations/retain_sae_activations_layer9.pt"
  
    
    # Compute normalized frequencies
    normalized_freq = compute_normalized_frequency(tensor_forget_path)
    
    
    return normalized_freq


if __name__ == "__main__":
    main()

