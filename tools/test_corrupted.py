from contextlib import contextmanager
from pathlib import Path
import shutil
import os
import subprocess
import logging
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import numpy as np
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("directory_operations.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
MULTICORRUPT_DIR = "/BEVFusionAD/data/fog_data/multicorrupt"
TARGET_DIR = "/BEVFusionAD/data/nuscenes"

CONFIGS_AND_CHECKPOINTS = {
    "configs/bevfusion/sbnet.py": (
        '/BEVFusionAD/work_dirs/3103_sbnet_avg/iter_1.pth',
        "eval_sbnet_avg_optimal_weights_full"
    )
}
#    "/BEVFusionAD/configs/bevfusion/bevf_tf_4x8_6e_nusc.py": (
#        "/BEVFusionAD/data/transfusion_train/bevfusion_tf.pth",
#        "bevfusion_tf"
#    )

@contextmanager
def directory_link_manager(src_dir: str, target_dir: str, dry_run: bool = False):
    """
    Safely manage directory symlinks with backups using rename operations.
    
    Args:
        src_dir: Source directory to link from
        target_dir: Target directory to create links in
        dry_run: If True, only log actions without executing them
    """
    src_path = Path(src_dir).resolve()
    target_path = Path(target_dir)
    original_dirs = {}
    modified_targets = []
    
    if not src_path.exists():
        logger.error(f"Source directory {src_path} does not exist.")
        raise FileNotFoundError(f"Source directory {src_path} does not exist")
    
    # Create a unique session ID for this operation
    session_id = int(time.time())
    logger.info(f"Starting directory link operation [Session ID: {session_id}]")
    logger.info(f"Source: {src_path}, Target: {target_path}")
    
    if dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    
    try:
        # Process sweeps and samples directories
        for data_type in ['sweeps', 'samples']:
            src_data_dir = src_path / data_type
            if not src_data_dir.exists():
                logger.warning(f"Source data directory {src_data_dir} does not exist, skipping")
                continue

            # Process all subdirectories in sweeps/samples
            for dir_path in src_data_dir.iterdir():
                if not dir_path.is_dir():
                    continue

                # Create target path
                target = target_path / data_type / dir_path.name
                
                # Track this target as being modified
                modified_targets.append(target)

                if target.exists():
                    # --- Add diagnostic logging ---
                    target_type = "directory" if target.is_dir() else "symlink" if target.is_symlink() else "file" if target.is_file() else "other"
                    logger.info(f"Target {target} exists. Type: {target_type}. Is symlink: {target.is_symlink()}")
                    # --- End diagnostic logging ---

                    if not target.is_symlink():
                        # Create backup with _original suffix and session ID
                        backup_dir = target.parent / f"{target.name}_original_{session_id}"
                        logger.info(f"Attempting backup: {target} exists and is not a symlink. Will back up to {backup_dir}") # Modified log
                        
                        if not dry_run:
                            # Ensure parent directory exists
                            backup_dir.parent.mkdir(parents=True, exist_ok=True)
                            
                            # Use rename (move) operation which is atomic on same filesystem
                            logger.info(f"Executing: shutil.move({target}, {backup_dir})") # Add log before move
                            shutil.move(target, backup_dir)
                            logger.info(f"Successfully moved {target} to {backup_dir}") # Add log after move
                            
                            # Store path for restoration
                            original_dirs[str(target)] = backup_dir
                    else: # target exists and IS a symlink
                        logger.info(f"Target {target} exists but is a symlink. Removing existing symlink.")
                        if not dry_run:
                            target.unlink()
                else: # target does not exist
                     logger.info(f"Target {target} does not exist. No backup needed.")

                # Create symlink
                logger.info(f"Creating symlink from {dir_path} to {target}")
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(dir_path)

        logger.info(f"Successfully set up all symlinks for test [Session ID: {session_id}]")
        yield

    except Exception as e:
        logger.error(f"Error during directory linking: {e}")
        # Immediately try to restore on error
        logger.warning(f"Exception occurred - attempting to restore original directories")
        restore_original_dirs(original_dirs, dry_run)
        raise

    finally:
        logger.info(f"Cleaning up after test - restoring original directories [Session ID: {session_id}]")
        # Restore original directories
        restore_original_dirs(original_dirs, dry_run)
        
        # Verify all modified targets are back to normal (either restored or removed)
        verification_failed = False
        for target in modified_targets:
            if target.is_symlink():
                logger.error(f"Failed restoration: {target} is still a symlink")
                verification_failed = True
        
        if verification_failed:
            logger.critical(f"CRITICAL: Some directories were not properly restored! Manual inspection required!")
        else:
            logger.info(f"All directories successfully restored [Session ID: {session_id}]")

def restore_original_dirs(original_dirs, dry_run=False):
    """Helper function to restore original directories using rename operations."""
    for target_path_str, backup_dir in original_dirs.items():
        target = Path(target_path_str)
        if not backup_dir.exists():
            logger.error(f"Backup directory {backup_dir} doesn't exist! Cannot restore {target}")
            continue
            
        logger.info(f"Restoring {target} from {backup_dir} using rename")
        if not dry_run:
            if target.exists():
                if target.is_symlink():
                    target.unlink()
                else:
                    shutil.rmtree(target)
                    
            # Restore from backup using rename (move) operation
            shutil.move(backup_dir, target)
            logger.info(f"Successfully restored {target}")

def run_test(config: str, checkpoint: str, gpu_id: int, img_weight: float, lidar_weight: float, output_log_file: str = None):
    """
    Run corruption test using the dist_test.sh script on a specific GPU,
    stream output to console, and save it to a specified log file.

    Args:
        config: Path to the model config file.
        checkpoint: Path to the model checkpoint file.
        gpu_id: The specific GPU ID to run the test on.
        img_weight: Weight for image data.
        lidar_weight: Weight for lidar data.
        output_log_file: Optional path to a file where stdout and stderr
                         of the command should be appended.
    """
    # Each process runs on 1 GPU
    gpus_for_script = 1
    cmd = [
        "bash",
        "./tools/dist_test.sh",
        config,
        checkpoint,
        str(gpus_for_script), # Use 1 GPU for this specific process
        "--eval", "bbox"
    ]

    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) # Assign specific GPU
    # Assign unique port per process to avoid conflicts
    base_port = 29500 # Base port number
    env["PORT"] = str(base_port + gpu_id) # Assign unique port
    # --- Add weight environment variables ---
    env["IMG_WEIGHT"] = str(img_weight)
    env["LIDAR_WEIGHT"] = str(lidar_weight)
    # --- End Add ---

    logger.info(f"Running test command on GPU {gpu_id} (Port: {env['PORT']}, IMG_W: {img_weight:.2f}, LIDAR_W: {lidar_weight:.2f}): {' '.join(cmd)}")
    if output_log_file:
        # Ensure the directory for the log file exists
        try:
            os.makedirs(os.path.dirname(output_log_file), exist_ok=True)
            logger.info(f"Appending output to: {output_log_file}")
        except OSError as e:
            logger.error(f"Failed to create directory for log file {output_log_file}: {e}")
            # Decide if you want to proceed without logging or return False
            # Proceeding without file logging for now:
            output_log_file = None
            logger.warning("Proceeding without saving output to file.")


    log_file_handle = None
    try:
        # Open log file in append mode if specified
        if output_log_file:
            log_file_handle = open(output_log_file, 'a', encoding='utf-8')

        # Start the subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,       # Capture stdout
            stderr=subprocess.STDOUT,      # Redirect stderr to stdout (like 2>&1)
            text=True,                   # Decode output as text
            encoding='utf-8',            # Specify encoding
            errors='replace',            # Handle potential decoding errors
            bufsize=1,                   # Line-buffered output
            env=env
        )

        # Read output line by line while the process is running
        if process.stdout:
            while True:
                line = process.stdout.readline()
                if not line: # readline returns empty string on EOF
                    break
                # Write to console (real-time)
                sys.stdout.write(line)
                sys.stdout.flush() # Ensure it's displayed immediately
                # Write to log file if open
                if log_file_handle:
                    log_file_handle.write(line)
                    log_file_handle.flush() # Ensure it's written immediately

        # Wait for the process to complete and get the return code
        return_code = process.wait()

        if return_code == 0:
            logger.info(f"Test completed successfully (Return Code: {return_code})")
            return True
        else:
            logger.error(f"Test failed (Return Code: {return_code})")
            return False

    except FileNotFoundError:
        logger.error(f"Error: The command 'bash' or script './tools/dist_test.sh' was not found.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during test execution: {e}", exc_info=True)
        return False
    finally:
        # Ensure the log file is closed if it was opened
        if log_file_handle:
            log_file_handle.close()

# --- Define run_test_wrapper outside the loop ---
def run_test_wrapper(task_params, dry_run_flag):
    """Wrapper function for ProcessPoolExecutor."""
    # Ensure work_dir exists before logging
    # This needs to be done in the child process
    if not dry_run_flag:
         os.makedirs(task_params["work_dir"], exist_ok=True)
    logger.info(f"Starting test process for {task_params['model_name']} on GPU {task_params['gpu_id']}...")
    try:
        success = run_test(
            config=task_params["config"],
            checkpoint=task_params["checkpoint"],
            gpu_id=task_params["gpu_id"], # Pass GPU ID
            # --- Pass weights ---
            img_weight=task_params["img_weight"],
            lidar_weight=task_params["lidar_weight"],
            # --- End Pass ---
            output_log_file=task_params["output_log"]
        )
        return task_params, success # Return params to identify which task finished
    except Exception as e:
        logger.error(f"Exception in run_test_wrapper for {task_params['model_name']}: {e}", exc_info=True)
        return task_params, False # Indicate failure
# --- End Define ---

# --- Function to get optimal weights ---
def get_optimal_img_weight(corruption_type: str, severity: str) -> float:
    """
    Returns the optimal image weight based on corruption type and severity
    using a predefined dictionary. Returns 0.5 as a default if no specific
    weight is defined for the given combination.
    """
    # Convert severity to string just in case it's passed as int
    severity_str = str(severity)

    # --- Define optimal weights in a nested dictionary ---
    # Structure: { corruption_type: { severity_level: img_weight } }
    OPTIMAL_WEIGHTS = {
        "beamsreducing": {
            "3": 0.425,
            # Add other severities for beamsreducing if known, e.g.:
            # "1": 0.5,
            # "5": 0.4,
        },
        "fog": {
            "3": 0.375,
            # Add other severities for fog if known
        },
        "motionblur": {
            "3": 0.375,
            # Add other severities for motionblur if known
        },
        # Add other corruption types and their severities/weights here
        # "snow": {
        #     "1": 0.6,
        #     "3": 0.55,
        #     "5": 0.5,
        # }
    }
    # --- End weight definition ---

    # Default weight if no specific rule applies
    default_weight = 0.5

    # Look up the weight
    if corruption_type in OPTIMAL_WEIGHTS:
        if severity_str in OPTIMAL_WEIGHTS[corruption_type]:
            weight = OPTIMAL_WEIGHTS[corruption_type][severity_str]
            logger.info(f"Using specific img_weight {weight} for {corruption_type} severity {severity_str}")
            return weight
        else:
            logger.warning(f"Severity level '{severity_str}' not defined for corruption type '{corruption_type}'. Using default: {default_weight}")
            return default_weight
    else:
        logger.warning(f"Corruption type '{corruption_type}' not defined in OPTIMAL_WEIGHTS. Using default: {default_weight}")
        return default_weight
# --- End function ---

# --- Function containing the original weight iteration logic (currently unused) ---
def run_all_tests_with_weight_iteration(args):
    """
    Runs tests iterating through all found corruptions, severities,
    and a predefined range of image weights.
    This function encapsulates the original weight iteration behavior.
    """
    logger.info("Starting test execution with WEIGHT ITERATION.")
    img_weights = np.arange(0.35, 0.65 + 0.01, 0.02) # Use 0.01 in stop to include 0.65

    # Process each corruption directory
    for corrupt_dir in Path(MULTICORRUPT_DIR).iterdir():
        if not corrupt_dir.is_dir():
            continue

        # Process each severity version
        severity_dirs = sorted([d for d in corrupt_dir.iterdir() if d.name.isdigit()])
        if not severity_dirs:
            logger.warning(f"No severity directories found in {corrupt_dir}")
            continue

        for version_dir in severity_dirs:
            # --- Loop through weights ---
            for img_weight in img_weights:
                lidar_weight = 1.0 - img_weight
                logger.info(f"Preparing tests for {corrupt_dir.name} version {version_dir.name} with IMG_W={img_weight:.2f}, LIDAR_W={lidar_weight:.2f} using GPUs: {args.gpus}")

                tasks_to_run = []
                all_logs_exist_for_level_and_weight = True # Check logs for this specific weight combo
                gpu_assignment_index = 0

                # Prepare tasks and check for existing logs for this severity level and weight
                for config, (checkpoint, model_name) in CONFIGS_AND_CHECKPOINTS.items():
                    weight_suffix = f"img{img_weight:.2f}_lidar{lidar_weight:.2f}".replace('.', 'p')
                    work_dir = f"work_dirs/{model_name}/{corrupt_dir.name}_sev{version_dir.name}/{weight_suffix}"
                    output_log = os.path.join(work_dir, f"test_output_{corrupt_dir.name}_sev{version_dir.name}_{weight_suffix}.log")

                    if os.path.exists(output_log):
                        logger.info(f"Log file {output_log} already exists for {model_name} with these weights. Will skip this specific test if others run.")
                    else:
                        all_logs_exist_for_level_and_weight = False # At least one test needs to run
                        gpu_id = args.gpus[gpu_assignment_index % len(args.gpus)]
                        tasks_to_run.append({
                            "config": config, "checkpoint": checkpoint, "model_name": model_name,
                            "corrupt_name": corrupt_dir.name, "severity": version_dir.name,
                            "gpu_id": gpu_id, "output_log": output_log, "work_dir": work_dir,
                            "img_weight": img_weight, "lidar_weight": lidar_weight,
                        })
                        gpu_assignment_index += 1

                if all_logs_exist_for_level_and_weight:
                    logger.info(f"All logs already exist for {corrupt_dir.name} version {version_dir.name} with IMG_W={img_weight:.2f}. Skipping actual execution.")
                    continue # Skip actual execution if logs exist for this weight

                logger.info(f"Running tests for {len(tasks_to_run)} model(s) on {corrupt_dir.name} severity {version_dir.name} with IMG_W={img_weight:.2f}")

                # Use the specific corruption version directory as the source
                corruption_source_dir = str(version_dir)
                with directory_link_manager(corruption_source_dir, TARGET_DIR, args.dry_run):
                    logger.info(f"Directory links established for {corruption_source_dir}")

                    # Run tests with managed data state using ProcessPoolExecutor
                    results = {}
                    num_workers = min(len(args.gpus), len(tasks_to_run))
                    if not tasks_to_run:
                         logger.warning(f"No tasks to run for {corrupt_dir.name} severity {version_dir.name} with IMG_W={img_weight:.2f} after filtering, but logs were missing. Check configuration.")
                         continue

                    logger.info(f"Using {num_workers} worker processes on GPUs: {args.gpus[:num_workers]}")
                    with ProcessPoolExecutor(max_workers=num_workers) as executor:
                        futures = {executor.submit(run_test_wrapper, params, args.dry_run): params for params in tasks_to_run}
                        for future in as_completed(futures):
                            try:
                                task_params, success = future.result()
                                results[f"{task_params['model_name']}_img{task_params['img_weight']:.2f}"] = success
                            except Exception as exc:
                                failed_params = futures[future]
                                logger.error(f'Task for {failed_params["model_name"]} (IMG_W={failed_params["img_weight"]:.2f}) generated an exception: {exc}')
                                results[f"{failed_params['model_name']}_img{failed_params['img_weight']:.2f}"] = False

                    logger.info(f"Test results for {corrupt_dir.name} sev {version_dir.name} IMG_W={img_weight:.2f}: {results}")
                # --- End directory_link_manager context ---
            # --- End weight loop ---
        # --- End severity loop ---
    # --- End corruption loop ---
    logger.info("Weight iteration testing finished.")
# --- End weight iteration function ---


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run corruption tests with data linking')
    parser.add_argument('--dry-run', action='store_true', help='Perform a dry run without making changes')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0],
                        help='List of specific GPU IDs to use for concurrent testing (e.g., --gpus 0 1 3)')
    # --- Add arguments for specific runs ---
    parser.add_argument('--specific-corruptions', type=str, nargs='+', default=None,
                        help='Run only for these specific corruption types (e.g., --specific-corruptions beamsreducing fog)')
    parser.add_argument('--specific-severities', type=str, nargs='+', default=None,
                        help='Run only for these specific severity levels (e.g., --specific-severities 3 5)')
    # --- End Add arguments ---
    args = parser.parse_args()

    # Log the specific GPUs being used
    logger.info(f"Using specific GPU IDs: {args.gpus}")

    # --- Determine which corruption/severity combinations to run ---
    combinations_to_run = []
    if args.specific_corruptions and args.specific_severities:
        logger.info(f"Running ONLY for specific corruptions: {args.specific_corruptions} and severities: {args.specific_severities}")
        for corruption_name in args.specific_corruptions:
            corrupt_dir = Path(MULTICORRUPT_DIR) / corruption_name
            if not corrupt_dir.is_dir():
                logger.warning(f"Specified corruption directory {corrupt_dir} not found. Skipping.")
                continue
            for severity in args.specific_severities:
                version_dir = corrupt_dir / severity
                if not version_dir.is_dir():
                    logger.warning(f"Specified severity directory {version_dir} not found for {corruption_name}. Skipping.")
                    continue
                combinations_to_run.append({
                    "name": corruption_name,
                    "severity": severity,
                    "path": version_dir
                })
    else:
        logger.info(f"Scanning {MULTICORRUPT_DIR} for all available corruptions and severities.")
        # Process each corruption directory found
        for corrupt_dir in Path(MULTICORRUPT_DIR).iterdir():
            if not corrupt_dir.is_dir():
                continue
            # Process each severity version found
            severity_dirs = sorted([d for d in corrupt_dir.iterdir() if d.name.isdigit()])
            if not severity_dirs:
                logger.warning(f"No severity directories found in {corrupt_dir}")
                continue
            for version_dir in severity_dirs:
                combinations_to_run.append({
                    "name": corrupt_dir.name,
                    "severity": version_dir.name,
                    "path": version_dir
                })

    if not combinations_to_run:
        logger.error("No valid corruption/severity combinations found to run. Exiting.")
        return
    # --- End determining combinations ---


    # --- Process the selected combinations using optimal weights ---
    logger.info(f"Processing {len(combinations_to_run)} corruption/severity combinations using OPTIMAL weights.")
    for combo in combinations_to_run:
        corruption_name = combo["name"]
        severity = combo["severity"]
        version_dir = combo["path"] # Path object for the specific version dir

        # Get specific weight for this combination
        img_weight = get_optimal_img_weight(corruption_name, severity)
        lidar_weight = 1.0 - img_weight
        # Log the weights being used, but they won't be in the path anymore
        logger.info(f"Preparing tests for {corruption_name} version {severity} with optimal IMG_W={img_weight:.3f}, LIDAR_W={lidar_weight:.3f} using GPUs: {args.gpus}") # Use .3f for potentially more precise weights

        tasks_to_run = []
        log_exists_for_level = True # Check if log exists for this specific combo
        gpu_assignment_index = 0

        # Prepare tasks and check for existing logs for this severity level
        for config, (checkpoint, model_name) in CONFIGS_AND_CHECKPOINTS.items():
            # --- Simplify work_dir and log file name (remove weight suffix) ---
            # weight_suffix = f"img{img_weight:.2f}_lidar{lidar_weight:.2f}".replace('.', 'p') # Removed
            work_dir = f"work_dirs/{model_name}/{corruption_name}_sev{severity}" # Simplified path
            output_log = os.path.join(work_dir, f"test_output_{corruption_name}_sev{severity}.log") # Simplified log name
            # --- End Simplify ---

            if os.path.exists(output_log):
                logger.info(f"Log file {output_log} already exists for {model_name}. Skipping this test.")
            else:
                log_exists_for_level = False # At least one test needs to run
                gpu_id = args.gpus[gpu_assignment_index % len(args.gpus)]
                tasks_to_run.append({
                    "config": config,
                    "checkpoint": checkpoint,
                    "model_name": model_name,
                    "corrupt_name": corruption_name,
                    "severity": severity,
                    "gpu_id": gpu_id,
                    "output_log": output_log, # Pass simplified log path
                    "work_dir": work_dir,     # Pass simplified work_dir path
                    "img_weight": img_weight,
                    "lidar_weight": lidar_weight,
                })
                gpu_assignment_index += 1

        if log_exists_for_level and not (args.specific_corruptions and args.specific_severities):
             # Only skip based on existing logs if we are NOT running specific combinations
            logger.info(f"Log already exists for {corruption_name} version {severity} at {output_log}. Skipping execution.") # Updated log message
            continue # Skip actual execution if log exists

        if not tasks_to_run:
             # Log if no tasks are generated (e.g., all logs existed or no models configured)
             if not log_exists_for_level: # Only warn if logs were missing but still no tasks
                 logger.warning(f"No tasks to run for {corruption_name} severity {severity}, and log was missing. Check configuration.")
             continue # Skip if no tasks

        logger.info(f"Running tests for {len(tasks_to_run)} model(s) on {corruption_name} severity {severity}") # Removed weight mention here as it's fixed per combo

        # Use the specific corruption version directory as the source
        corruption_source_dir = str(version_dir)
        with directory_link_manager(corruption_source_dir, TARGET_DIR, args.dry_run):
            logger.info(f"Directory links established for {corruption_source_dir}")

            # Run tests with managed data state using ProcessPoolExecutor
            results = {}
            num_workers = min(len(args.gpus), len(tasks_to_run))

            logger.info(f"Using {num_workers} worker processes on GPUs: {args.gpus[:num_workers]}")
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                futures = {executor.submit(run_test_wrapper, params, args.dry_run): params for params in tasks_to_run}

                # Process completed tasks
                for future in as_completed(futures):
                    try:
                        task_params, success = future.result()
                        # Use model name as key, weight is fixed for this run
                        results[task_params['model_name']] = success
                    except Exception as exc:
                        failed_params = futures[future]
                        logger.error(f'Task for {failed_params["model_name"]} (IMG_W={failed_params["img_weight"]:.2f}) generated an exception: {exc}')
                        results[failed_params['model_name']] = False # Mark as failed

            # Log results without weights in the message, as they are implicit for the combo
            logger.info(f"Test results for {corruption_name} sev {severity}: {results}")
        # --- End directory_link_manager context ---
    # --- End loop over selected combinations ---

    logger.info("Optimal weight testing finished.")


if __name__ == "__main__":
    main()
    # If you want to run the weight iteration instead, you could potentially call:
    # run_all_tests_with_weight_iteration(args)
    # after parsing args, perhaps controlled by another argument.
