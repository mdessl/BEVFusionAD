import os
import re
import pandas as pd
from functools import reduce # Import reduce for merging

def extract_map_scores(base_dir, model_name):
    """
    Extracts mAP scores from log files in subdirectories of base_dir for a specific model.

    Args:
        base_dir (str): The path to the directory containing the evaluation subdirectories
                        (e.g., 'BEVFusionAD/work_dirs/eval_bev_mini').
        model_name (str): The name to use for the mAP column (e.g., 'BEVFusion').

    Returns:
        pandas.DataFrame: A DataFrame containing Corruption, Severity, and mAP score
                          under the column specified by model_name.
                          Returns an empty DataFrame if the base directory doesn't exist
                          or no log files are found.
    """
    results = []
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory '{base_dir}' not found for model '{model_name}'.")
        # Return DataFrame with expected columns for merging
        return pd.DataFrame(columns=['Corruption', 'Severity', model_name])

    # Regex to find the mAP line and extract the score
    map_regex = re.compile(r"^\s*mAP:\s*(\d+\.\d+)")
    # Regex to parse corruption and severity from the directory name
    dir_regex = re.compile(r"^(.*?)_sev(\d+)$")

    for subdir_name in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir_name)

        if os.path.isdir(subdir_path):
            # Parse corruption and severity from subdir name
            dir_match = dir_regex.match(subdir_name)
            if not dir_match:
                print(f"Skipping directory with unexpected name format: {subdir_name} in {base_dir}")
                continue

            corruption_type = dir_match.group(1)
            # Handle potential 'clean' case where severity might not be present or is 0
            severity_level_str = dir_match.group(2)
            severity_level = int(severity_level_str) if severity_level_str else 0


            # Construct the expected log file name
            log_file_name = f"test_output_{subdir_name}.log"
            log_file_path = os.path.join(subdir_path, log_file_name)

            map_score = None
            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, 'r') as f:
                        for line in f:
                            match = map_regex.search(line)
                            if match:
                                map_score = float(match.group(1))
                                break # Found the score, no need to read further
                except Exception as e:
                    print(f"Error reading file {log_file_path}: {e}")
            else:
                # Check for the clean log file name format as well
                clean_log_file_name = "test_output_clean.log"
                clean_log_file_path = os.path.join(subdir_path, clean_log_file_name)
                if subdir_name == "clean" and os.path.exists(clean_log_file_path):
                     try:
                        with open(clean_log_file_path, 'r') as f:
                            for line in f:
                                match = map_regex.search(line)
                                if match:
                                    map_score = float(match.group(1))
                                    severity_level = 0 # Explicitly set severity for clean
                                    break
                     except Exception as e:
                        print(f"Error reading file {clean_log_file_path}: {e}")
                else:
                    print(f"Warning: Log file not found: {log_file_path} (and not clean case)")


            if map_score is not None:
                results.append({
                    'Corruption': corruption_type,
                    'Severity': severity_level,
                    model_name: map_score # Use model_name as the key for the score
                })
            else:
                 # Only print warning if it wasn't the clean case we already checked
                 if not (subdir_name == "clean" and os.path.exists(os.path.join(subdir_path, "test_output_clean.log"))):
                     print(f"Warning: mAP score not found in {log_file_path} or corresponding clean log")


    if not results:
        print(f"No mAP scores found for model '{model_name}' in {base_dir}.")
        # Return DataFrame with expected columns for merging
        return pd.DataFrame(columns=['Corruption', 'Severity', model_name])

    # Create a pandas DataFrame
    df = pd.DataFrame(results)
    # Ensure correct types before sorting/merging
    df['Severity'] = pd.to_numeric(df['Severity'])
    df[model_name] = pd.to_numeric(df[model_name])
    df = df.sort_values(by=['Corruption', 'Severity']).reset_index(drop=True)
    return df

# --- Usage Example ---

# Define models and their corresponding evaluation directories
#     'BEVFusion': '/BEVFusionAD/work_dirs/eval_bev_mini', # Model name and path

model_configs = {
    'SBNet_Avg': '/BEVFusionAD/work_dirs/eval_sbnet_avg_07lidar_03img' # Add more models here if needed
    # 'AnotherModel': 'path/to/another/eval/dir'
}

# Extract scores for each model
all_scores_dfs = []
for model_name, base_dir in model_configs.items():
    print(f"\n--- Processing Model: {model_name} ---")
    scores_df = extract_map_scores(base_dir, model_name)
    if not scores_df.empty:
        all_scores_dfs.append(scores_df)
    else:
        print(f"No scores found or error processing {model_name}, skipping merge for this model.")

# Merge the DataFrames if we have results from multiple models
if len(all_scores_dfs) > 1:
    # Use reduce to iteratively merge DataFrames on Corruption and Severity
    # 'outer' merge keeps all rows, filling missing scores with NaN
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=['Corruption', 'Severity'], how='outer'), all_scores_dfs)
    # Sort the final merged DataFrame
    merged_df = merged_df.sort_values(by=['Corruption', 'Severity']).reset_index(drop=True)
elif len(all_scores_dfs) == 1:
    merged_df = all_scores_dfs[0] # Only one model processed successfully
else:
    merged_df = pd.DataFrame() # No results found for any model

# Print the combined results
if not merged_df.empty:
    print("\n--- Combined mAP Scores ---")
    # Fill NaN values for display if desired, e.g., with 0 or '-'
    # print(merged_df.fillna('-').to_string())
    print(merged_df.to_string()) # Use to_string() to print the full DataFrame

    # Optional: Save the combined results to CSV in a common directory or the first model's dir
    # Choose a suitable output path
    output_dir = os.path.dirname(list(model_configs.values())[0]) # e.g., BEVFusionAD/work_dirs/
    csv_path = os.path.join(output_dir, 'eval_sbnet_avg_07lidar_03img.csv')
    try:
        #merged_df.to_csv(csv_path, index=False, float_format='%.2f') # Format float precision
        print(f"\nCombined results saved to {csv_path}")
    except Exception as e:
        print(f"\nError saving combined results to {csv_path}: {e}")
else:
    print("\nNo combined results to display or save.")


# --- Remove the old single-model processing ---
# base_evaluation_dir = 'BEVFusionAD/work_dirs/eval_bev_mini' # Adjust this path if needed
# scores_df = extract_map_scores(base_evaluation_dir)
# ... (rest of old printing/saving logic) ...

# base_evaluation_dir = 'BEVFusionAD/work_dirs/eval_sbnet_avg_mini' # Adjust this path if needed
# scores_df = extract_map_scores(base_evaluation_dir)
# ... (rest of old printing/saving logic) ...
