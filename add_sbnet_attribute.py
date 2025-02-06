import pickle
import copy

# Load the original data
with open("/BEVFusionAD/data/nuscenes/nuscenes_infos_train.pkl", 'rb') as f:
    nuscenes_infos = pickle.load(f)


import pdb; pdb.set_trace()
# Get original data list
original_data = nuscenes_infos["data_list"]
sample_indices = [eg["sample_idx"] for eg in original_data]
total_samples = max(sample_indices) + 1  # Add 1 since indices are 0-based

# Assert that there are no holes in the sample indices
assert set(range(total_samples)) == set(sample_indices), "Sample indices are not continuous"

# Create a new list with three versions of each entry
new_data_list = [None] * (total_samples * 3)  # Pre-allocate list for img, lidar, and both

for entry in original_data:
    # Add image modality entry
    img_entry = copy.deepcopy(entry)
    img_entry['sbnet_modality'] = 'img'
    new_data_list[entry['sample_idx']] = img_entry
    
    # Add lidar modality entry
    lidar_entry = copy.deepcopy(entry)
    lidar_entry['sbnet_modality'] = 'lidar'
    lidar_entry['sample_idx'] += total_samples
    new_data_list[lidar_entry['sample_idx']] = lidar_entry

# Remove any None entries (if there were gaps in sample_idx)
new_data_list = [entry for entry in new_data_list if entry is not None]

# Replace the original data_list with the new one
nuscenes_infos["data_list"] = new_data_list

# Verify the changes
for i, entry in enumerate(nuscenes_infos["data_list"]):
    print(f"Index: {i}, Sample Index: {entry['sample_idx']}, Modality: {entry['sbnet_modality']}")

# Save modified file
with open("/BEVFusionAD/data/nuscenes/nuscenes_infos_train.pkl", 'wb') as f:
    pickle.dump(nuscenes_infos, f)