import pickle
import copy

# Load the modified data
with open("/BEVFusionAD/data/nuscenes/nuscenes_infos_train.pkl", 'rb') as f:
    nuscenes_infos = pickle.load(f)

# Get the modified data list
modified_data = nuscenes_infos["infos"]

# Create a new list with only one version of each entry (removing duplicates)
# We'll keep only the entries with 'lidar' modality and remove the 'sbnet_modality' field
original_data_list = []

# Process every other entry (since each original entry was duplicated)
for i in range(0, len(modified_data), 2):
    # Get the lidar entry (which should be the first of each pair)
    entry = copy.deepcopy(modified_data[i])
    
    # Verify this is a lidar entry, otherwise use the next one
    if entry.get('sbnet_modality') != 'lidar':
        entry = copy.deepcopy(modified_data[i+1])
        assert entry.get('sbnet_modality') == 'lidar', f"Expected lidar modality at index {i+1}"
    
    # Remove the sbnet_modality field
    if 'sbnet_modality' in entry:
        del entry['sbnet_modality']
    
    original_data_list.append(entry)

# Replace the modified infos with the restored original
nuscenes_infos["infos"] = original_data_list

# Save restored file
with open("/BEVFusionAD/data/nuscenes/nuscenes_infos_train_bev.pkl", 'wb') as f:
    pickle.dump(nuscenes_infos, f)

print(f"Restored original data: removed {len(modified_data) - len(original_data_list)} duplicate entries")
print(f"Original entry count: {len(original_data_list)}")