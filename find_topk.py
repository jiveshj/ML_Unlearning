total_features = [i for i in range(16384)]

with open("/home/ubuntu/ML_Unlearning/frequencies_9/retain_frequent_features_greater_than_0.0001_layer9_machine_readable.txt", "r") as f:
    frequent_features = [int(line.strip()) for line in f]



set_of_frequent_features = set(frequent_features)
set_of_total_features = set(total_features)

missing_features = set_of_total_features - set_of_frequent_features

with open("/home/ubuntu/ML_Unlearning/frequencies_9/retain_missing_features_layer9_machine_readable.txt", "w") as f:
    for feature in missing_features:
        f.write(f"{feature}\n")
print(len(missing_features))