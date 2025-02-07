import pickle
import copy

# Load the original data
# Save mod
with open("/BEVFusionAD/data/nuscenes/nuscenes_infos_train.pkl", 'rb') as f:
    nuscenes_infos = pickle.load(f)

# Get original data list
original_data = nuscenes_infos["infos"]

# Create a new list with two versions of each entry (camera and lidar)
new_data_list = []

for entry in original_data:
    # Add original entry as lidar modality
    lidar_entry = copy.deepcopy(entry)
    lidar_entry['sbnet_modality'] = 'lidar'
    new_data_list.append(lidar_entry)
    
    # Add camera modality entry
    camera_entry = copy.deepcopy(entry)
    camera_entry['sbnet_modality'] = 'camera'
    new_data_list.append(camera_entry)

# Replace the original infos with the new one
nuscenes_infos["infos"] = new_data_list

# Save modified file
with open("/BEVFusionAD/data/nuscenes/nuscenes_infos_train.pkl", 'wb') as f:
    pickle.dump(nuscenes_infos, f)