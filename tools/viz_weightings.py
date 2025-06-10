import re
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import logging

# --- Configuration ---
WORK_DIRS_BASE = "work_dirs"
# Assuming the model name used in the directory structure is known
# Extract from the previous script's CONFIGS_AND_CHECKPOINTS if needed
MODEL_NAME = "eval_sbnet_avg_iterate" # Or "bevfusion_tf" etc.
OUTPUT_PLOT_DIR = "plots"
LOG_FILE_PATTERN = "test_output_*.log"
MAP_REGEX = r"'pts_bbox_NuScenes/mAP':\s*([0-9.]+)"

# Baseline mAP scores for 0.5/0.5 weighting
BASELINE_MAPS = {
    "beamsreducing_sev3": 0.2137,
    "motionblur_sev3": 0.3861,
    "fog_sev3": 0.1911,
    # Add more baseline values here if available for other corruptions/severities
    # e.g., "fog_sev1": 0.XXXX,
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Configuration ---

def parse_log_file(log_path: Path):
    """Parses a log file to extract corruption, severity, weight, and mAP."""
    try:
        # Extract info from the directory structure and filename
        # Example path: work_dirs/eval_sbnet_avg_iterate/fog_sev3/img0p35_lidar0p65/test_output_fog_sev3_img0p35_lidar0p65.log
        parts = log_path.parts
        weight_dir_name = parts[-2] # e.g., img0p35_lidar0p65
        corruption_sev_dir_name = parts[-3] # e.g., fog_sev3

        # Extract corruption and severity
        match_corr_sev = re.match(r"(.+)_sev(\d+)", corruption_sev_dir_name)
        if not match_corr_sev:
            logger.warning(f"Could not parse corruption/severity from dir: {corruption_sev_dir_name} in path {log_path}")
            return None
        corruption_type = match_corr_sev.group(1)
        severity = int(match_corr_sev.group(2))

        # Extract image weight
        match_weight = re.search(r"img([\d.]+)p([\d]+)", weight_dir_name) # Handle float like 0p35
        if not match_weight:
             match_weight = re.search(r"img([\d]+)", weight_dir_name) # Handle integer like img1
             if not match_weight:
                 logger.warning(f"Could not parse image weight from dir: {weight_dir_name} in path {log_path}")
                 return None
             img_weight_str = match_weight.group(1)
             img_weight = float(img_weight_str)
        else:
            img_weight_str = f"{match_weight.group(1)}.{match_weight.group(2)}" # Reconstruct float string
            img_weight = float(img_weight_str)


        # Extract mAP from file content
        map_score = None
        with open(log_path, 'r') as f:
            content = f.read()
            # Find the last occurrence of the mAP pattern
            matches = list(re.finditer(MAP_REGEX, content))
            if matches:
                map_score = float(matches[-1].group(1))
            else:
                logger.warning(f"mAP score not found in log file: {log_path}")
                return None

        return {
            "corruption": corruption_type,
            "severity": severity,
            "img_weight": img_weight,
            "map": map_score,
            "path": log_path
        }

    except Exception as e:
        logger.error(f"Error parsing log file {log_path}: {e}", exc_info=True)
        return None

def plot_map_scores(data, output_dir: Path):
    """Generates plots for mAP vs image weight for each corruption/severity."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group data by corruption and severity
    grouped_data = {}
    for item in data:
        key = (item['corruption'], item['severity'])
        if key not in grouped_data:
            grouped_data[key] = []
        grouped_data[key].append((item['img_weight'], item['map']))

    # Create plots
    for (corruption, severity), scores in grouped_data.items():
        if not scores:
            continue

        # Sort by image weight
        scores.sort(key=lambda x: x[0])
        img_weights = [s[0] for s in scores]
        map_values = [s[1] for s in scores]

        plt.figure(figsize=(10, 6))
        plt.plot(img_weights, map_values, marker='o', linestyle='-', label='mAP Score')

        # Add baseline if available
        baseline_key = f"{corruption}_sev{severity}"
        if baseline_key in BASELINE_MAPS:
            baseline_map = BASELINE_MAPS[baseline_key]
            plt.axhline(y=baseline_map, color='r', linestyle='--', label=f'Baseline (0.5/0.5) mAP: {baseline_map:.4f}')

        plt.xlabel("Image Weight")
        plt.ylabel("mAP Score")
        plt.title(f"mAP vs. Image Weight for {corruption.capitalize()} Severity {severity}")
        plt.grid(True)
        plt.legend()
        plt.ylim(bottom=0) # Ensure y-axis starts at 0

        plot_filename = output_dir / f"{corruption}_sev{severity}_map_vs_weight.png"
        plt.savefig(plot_filename)
        plt.close() # Close the figure to free memory
        logger.info(f"Saved plot: {plot_filename}")

def main():
    base_path = Path(WORK_DIRS_BASE) / MODEL_NAME
    if not base_path.is_dir():
        logger.error(f"Base directory not found: {base_path}")
        return

    all_log_files = list(base_path.glob(f"*/{LOG_FILE_PATTERN}")) # Initial search one level deep
    all_log_files.extend(list(base_path.glob(f"*/*/{LOG_FILE_PATTERN}"))) # Search two levels deep for weight dirs

    if not all_log_files:
        logger.warning(f"No log files found matching pattern '{LOG_FILE_PATTERN}' in subdirectories of {base_path}")
        return

    logger.info(f"Found {len(all_log_files)} potential log files. Parsing...")

    parsed_data = []
    for log_file in all_log_files:
        data = parse_log_file(log_file)
        if data:
            parsed_data.append(data)

    if not parsed_data:
        logger.error("No valid data could be parsed from the log files.")
        return

    logger.info(f"Successfully parsed data from {len(parsed_data)} log files.")

    # Optional: Print parsed data summary using pandas
    try:
        df = pd.DataFrame(parsed_data)
        print("\n--- Parsed Data Summary ---")
        print(df.groupby(['corruption', 'severity', 'img_weight']).size())
        print("-------------------------\n")
    except ImportError:
        logger.warning("Pandas not installed. Skipping summary table.")


    plot_map_scores(parsed_data, Path(OUTPUT_PLOT_DIR))
    logger.info("Plot generation complete.")

if __name__ == "__main__":
    main()