import pickle
import copy

# Load the original data
input_path = "/BEVFusionAD/data/mini/nuscenes_infos_val.pkl"
output_path = "/BEVFusionAD/data/mini/nuscenes_infos_val_cleaned.pkl"

with open(input_path, 'rb') as f:
    nuscenes_infos = pickle.load(f)

# Get original data list
original_data = nuscenes_infos["infos"]
print(f"Original dataset contains {len(original_data)} samples")

# Define a function to identify corrupted samples
def is_corrupted(sample):
    # Example condition: Check for a specific token
    # Replace this with your actual condition to identify the corrupted sample
    corrupted_token = '3e8750f331d7499e9b5123e9eb70f2e2' # actual one: "a1165ef34d62441db1047c17865c5797"
    return sample.get("token") == corrupted_token

# Filter out corrupted samples
filtered_data = [sample for sample in original_data if not is_corrupted(sample)]
print(f"Filtered dataset contains {len(filtered_data)} samples")
print(f"Removed {len(original_data) - len(filtered_data)} samples")

# Replace the original infos with the filtered ones
nuscenes_infos["infos"] = filtered_data

# Save modified file
with open(output_path, 'wb') as f:
    pickle.dump(nuscenes_infos, f)

print(f"Cleaned dataset saved to {output_path}") 