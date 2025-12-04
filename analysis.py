with  open("/home/ubuntu/ML_Unlearning/frequencies_second_time/top50_features_layer7.txt", "r") as f:
    top50_features = [int(line.strip()) for line in f]


with open("/home/ubuntu/ML_Unlearning/frequencies_second_time/features_to_clamp_layer7.txt", "r") as f:
    features_to_clamp = [int(line.strip()) for line in f]



common_features = set(top50_features) & set(features_to_clamp)

print(len(common_features))