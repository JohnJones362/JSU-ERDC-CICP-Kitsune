# old code: 6/24/25 (9:19 AM)
from Kitsune import Kitsune
import numpy as np
import time
from KitNET.KitNET import KitNET # Import KitNET directly for loading
import os
import shutil
import psutil
from chunked_kitsune import process_in_chunks
import pandas as pd
from datetime import datetime
import multiprocessing
import csv
import cProfile
import pstats
from tqdm import tqdm
import gc
import subprocess
import sys
import collections # For deque (rolling window)
import pickle # Explicitly import pickle for saving/loading combined state
import logging # Added for logging warnings and errors 

# Set up logging for example.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import the simulator parameters and function
from realtime_data_simulator import simulate_data_pipeline, INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE, SLEEP_TIME

# --- Centralized Configuration Class ---
class AppConfig:
    # General Run Settings
    RUN_REALTIME_SIMULATION = True # Set to True for real-time, False for offline chunked processing
    INPUT_DELIMITER = '\t' # Set to '\t' for tab-separated, ',' for comma-separated
    PACKET_LIMIT = None # Set to an int (e.g., 1000) for limited packets, None for unlimited

    # Kitsune Model Parameters (used for both training and defining load path)
    MAX_AE_SIZE = 10 # Maximum size for any autoencoder
    FM_GRACE_PERIOD = 10 # Feature mapping grace period (samples)
    AD_GRACE_PERIOD = 15 # Anomaly detection grace period (samples)

    # PCA Optimization Parameters
    PCA_COMPONENTS = None # Set to an int (e.g., 20) to enable PCA, None to disable
    PCA_GRACE_PERIOD = FM_GRACE_PERIOD # Grace period for PCA fitting

    # Batch Processing Parameter
    BATCH_SIZE = 5 # Number of samples to process in each batch

    # Adaptive Thresholding Configuration
    ADAPTIVE_THRESHOLD_WINDOW_SIZE = 5 # Number of recent "benign" RMSEs to keep in rolling window
    ADAPTIVE_THRESHOLD_UPDATE_INTERVAL = 1 # Recalculate threshold every X packets
    ADAPTIVE_THRESHOLD_PERCENTILE = 99.9 # Percentile to use for adaptive threshold calculation

    # Main Loop Waiting Time & Timeout
    MAIN_LOOP_WAIT_TIME = 0.5 # Seconds to sleep if no new data (real-time mode)
    LIVE_STREAM_MAX_NO_DATA_TIMEOUT = 10 # Max seconds to wait for new data before concluding stream

    # Model Saving/Loading Configuration
    SAVE_KITSUNE_MODEL = True # Set to True to save the model after AD grace period
    LOAD_KITSUNE_MODEL = False # Set to True to load a pre-trained model and skip training
    # Set to exact timestamp (e.g., "20250812_151801") if loading, or None if saving new.
    MODEL_TIMESTAMP_TO_LOAD = None

    # Offline Processing Specifics (only relevant if RUN_REALTIME_SIMULATION is False)
    OFFLINE_NUM_CHUNKS = None # Number of chunks for offline processing, calculated automatically if None
    OFFLINE_IMPORT_FILE = "CTU-IoT-Malware-Capture-1-1conn.log.labeled" # Default offline input file
    OFFLINE_USE_TSV = True # True if OFFLINE_IMPORT_FILE is TSV/CSV, False if PCAP

    # Output Directory Base
    RESULTS_DIR = "Results"

# --- End Centralized Configuration Class ---

def check_system_resources(file_path):
    """Check if system has enough resources to process the file"""
    try:
        # First check if file exists
        if not os.path.exists(file_path):
            logging.error(f"Error: Input file {file_path} not found!")
            return False

        # Check available disk space (need at least 3x file size)
        file_size = os.path.getsize(file_path)
        free_space = shutil.disk_usage(os.path.dirname(os.path.abspath(file_path))).free
        if free_space < file_size * 3:
            raise RuntimeError(f"Not enough disk space. Need at least {file_size * 3 / (1024**3):.2f} GB, but only {free_space / (1024**3):.2f} GB available")

        # Calculate required RAM based on packet limit and chunk size
        required_ram = 400 * 1024 * 1024 # Example: Try 400 MB. You can try 300 if needed.
        required_ram_gb = required_ram / (1024**3)
        min_ram_gb = max(0.5, required_ram_gb)

        free_ram = psutil.virtual_memory().available
        if free_ram < min_ram_gb * (1024**3):
            raise RuntimeError(f"Not enough RAM. Need at least {min_ram_gb:.2f} GB, but only {free_ram / (1024**3):.2f} GB available")

        return True
    except Exception as e:
        logging.error(f"Resource check failed: {str(e)}")
        return False

def attempt_file_delete(file_path, max_retries=5, retry_delay=0.5):
    """Attempts to delete a file with retries in case of PermissionError."""
    for i in range(max_retries):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logging.info(f"Successfully deleted existing file: {file_path}")
                return True
            except PermissionError as e:
                logging.warning(f"Attempt {i+1}/{max_retries}: Could not delete '{file_path}' due to permission error: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            except Exception as e:
                logging.error(f"Error during file deletion of '{file_path}': {e}")
                raise
        else:
            logging.info(f"File '{file_path}' does not exist. No need to delete.")
            return True

    logging.error(f"Failed to delete '{file_path}' after {max_retries} attempts. Please close any programs using this file.")
    return False

def wait_for_file_to_exist(file_path, timeout=10, check_interval=0.1):
    """Waits for a file to exist on disk within a timeout period."""
    start_time = time.time()
    while not os.path.exists(file_path):
        if time.time() - start_time > timeout:
            logging.error(f"Timeout: File '{file_path}' did not appear within {timeout} seconds.")
            return False
        time.sleep(check_interval)
    logging.info(f"File '{file_path}' found after {time.time() - start_time:.2f} seconds.")
    return True


def main():
    profiler = cProfile.Profile()
    profiler.enable()

    path = ""
    realtime_sim_process = None # Initialize to None outside try/except for finally block

    # --- Centralized File Cleanup for simulated_data.tsv ---
    if AppConfig.RUN_REALTIME_SIMULATION:
        if not attempt_file_delete(OUTPUT_FILE):
            sys.exit(1)

    # --- Start Real-time Simulation or Setup Offline Processing ---
    if AppConfig.RUN_REALTIME_SIMULATION:
        logging.info("Starting real-time data simulation...")
        try:
            realtime_sim_process = subprocess.Popen([sys.executable, 'realtime_data_simulator.py'])
            logging.info(f"Giving simulator {SLEEP_TIME * 2} seconds to start writing data and ensure file creation...")
            if not wait_for_file_to_exist(OUTPUT_FILE, timeout=10):
                logging.error(f"Error: Simulator did not create '{OUTPUT_FILE}' in time. Aborting.")
                if realtime_sim_process:
                    realtime_sim_process.terminate()
                sys.exit(1)
            logging.info("Simulator should be running and file created.")
        except Exception as e:
            logging.error(f"Error starting real-time simulator: {e}")
            sys.exit(1)
        
        path = OUTPUT_FILE
        # The input_delimiter is already defined in AppConfig
    else:
        # --- Offline Analysis Setup ---
        cpu_count = os.cpu_count()
        num_chunks = AppConfig.OFFLINE_NUM_CHUNKS if AppConfig.OFFLINE_NUM_CHUNKS is not None else max(6, min(cpu_count * 2, 12))
        logging.info(f"Using {num_chunks} chunks based on {cpu_count} CPU cores (or configured).")

        logging.info(f"Packet limit: {'Unlimited' if AppConfig.PACKET_LIMIT is None else AppConfig.PACKET_LIMIT}")

        import_file = AppConfig.OFFLINE_IMPORT_FILE
        use_tsv = AppConfig.OFFLINE_USE_TSV

        if not use_tsv:
            path = f"{import_file}.pcap"
        else:
            path = f"{import_file}.csv"
            # Specific delimiter for CTU-IoT-Malware-Capture-1-1conn.log.labeled if using it offline
            if import_file == "CTU-IoT-Malware-Capture-1-1conn.log.labeled":
                AppConfig.INPUT_DELIMITER = '|' # Override for this specific file

        logging.info(f"Input file for offline analysis: {path}")

    # Create the results folder structure
    anomalies_folder = os.path.join(AppConfig.RESULTS_DIR, "anomalies")
    confidence_folder = os.path.join(AppConfig.RESULTS_DIR, "confidence")
    rmse_folder = os.path.join(AppConfig.RESULTS_DIR, "rmse")
    pickle_folder = os.path.join(AppConfig.RESULTS_DIR, "pickle")
    logs_folder = os.path.join(AppConfig.RESULTS_DIR, "logs")
    chunks_folder = os.path.join(AppConfig.RESULTS_DIR, "chunks")
    thresholds_folder = os.path.join(AppConfig.RESULTS_DIR, "thresholds")

    for folder in [anomalies_folder, confidence_folder, rmse_folder, pickle_folder, logs_folder, chunks_folder, thresholds_folder]:
        os.makedirs(folder, exist_ok=True)

    # Check system resources
    logging.info("\nChecking system resources...")
    if not check_system_resources(path):
        if AppConfig.RUN_REALTIME_SIMULATION and realtime_sim_process:
            realtime_sim_process.terminate()
        sys.exit(1)

    logging.info("\nStarting processing with Kitsune...")
    start_time_overall = time.time()

    K = None
    loaded_kitnet_model = None
    initial_threshold_from_load = None
    pca_fitted_from_load = False

    # Initialize threshold_calculated before any conditional blocks
    threshold_calculated = False 

    # --- Construct Model File Path ---
    current_save_timestamp = None
    
    model_file_name_suffix = f"maxAE={AppConfig.MAX_AE_SIZE}_FMgrace={AppConfig.FM_GRACE_PERIOD}_ADgrace={AppConfig.AD_GRACE_PERIOD}"
    if AppConfig.PCA_COMPONENTS is not None:
        model_file_name_suffix += f"_PCA={AppConfig.PCA_COMPONENTS}"
    
    if AppConfig.SAVE_KITSUNE_MODEL and not AppConfig.LOAD_KITSUNE_MODEL:
        current_save_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file_name_suffix += f"_TS={current_save_timestamp}"
        logging.info(f"Model will be saved with timestamp: {current_save_timestamp}")
    elif AppConfig.LOAD_KITSUNE_MODEL and AppConfig.MODEL_TIMESTAMP_TO_LOAD:
        model_file_name_suffix += f"_TS={AppConfig.MODEL_TIMESTAMP_TO_LOAD}"
        logging.info(f"Attempting to load model with timestamp: {AppConfig.MODEL_TIMESTAMP_TO_LOAD}")
    else:
        current_save_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_file_name_suffix += f"_TS={current_save_timestamp}"
        logging.warning("Warning: Model timestamp for path not explicitly set for load or save. Using current timestamp.")

    model_filepath = os.path.join(pickle_folder, f"kitnet_state_{model_file_name_suffix}.pkl")
    logging.info(f"Model file path: {model_filepath}")


    # --- Model Loading Logic ---
    if AppConfig.LOAD_KITSUNE_MODEL and os.path.exists(model_filepath):
        try:
            logging.info(f"Attempting to load pre-trained model state from: {model_filepath}")
            with open(model_filepath, 'rb') as f:
                loaded_state = pickle.load(f)
            
            # Reconstruct KitNET instance
            loaded_kitnet_model = KitNET(n=loaded_state['kitnet_model_state']['n'],
                                         m=loaded_state['kitnet_model_state']['m'],
                                         FM_grace_period=loaded_state['kitnet_model_state']['FM_grace_period'],
                                         AD_grace_period=loaded_state['kitnet_model_state']['AD_grace_period'],
                                         learning_rate=loaded_state['kitnet_model_state']['lr'],
                                         hidden_ratio=loaded_state['kitnet_model_state']['hr'])
            loaded_kitnet_model.set_weights(loaded_state['kitnet_model_state'])
            
            initial_threshold_from_load = loaded_state['current_threshold']
            threshold_calculated = loaded_state['threshold_calculated']
            pca_fitted_from_load = loaded_state.get('pca_fitted', False)
            pca_components_from_load = loaded_state.get('pca_components', None)
            pca_grace_period_from_load = loaded_state.get('pca_grace_period', AppConfig.FM_GRACE_PERIOD) # Use AppConfig default if not in state

            logging.info(f"Pre-trained model and threshold ({initial_threshold_from_load:.6f}) loaded successfully.")
            if pca_fitted_from_load:
                logging.info(f"PCA was fitted with {pca_components_from_load} components in the loaded model.")
            else:
                logging.info("PCA was not fitted or not enabled in the loaded model's saved state.")

            # Update Kitsune parameters to match loaded model for consistency
            AppConfig.MAX_AE_SIZE = loaded_state['kitnet_model_state']['m']
            AppConfig.FM_GRACE_PERIOD = loaded_state['kitnet_model_state']['FM_grace_period']
            AppConfig.AD_GRACE_PERIOD = loaded_state['kitnet_model_state']['AD_grace_period']
            AppConfig.PCA_COMPONENTS = pca_components_from_load
            AppConfig.PCA_GRACE_PERIOD = pca_grace_period_from_load

        except Exception as e:
            logging.error(f"Error loading pre-trained model and threshold from {model_filepath}: {e}. Proceeding with training from scratch.")
            loaded_kitnet_model = None
            initial_threshold_from_load = None
            threshold_calculated = False
            pca_fitted_from_load = False

    # Initialize file writers and their underlying file objects to None
    rmse_file_obj, ms_conf_file_obj, sec_conf_file_obj, anomaly_file_obj = None, None, None, None
    rmse_writer, ms_conf_writer, sec_conf_writer, anomaly_writer = None, None, None, None

    try:
        # Initialize Kitsune with or without the pre-trained model
        K = Kitsune(file_path=path,
                    limit=AppConfig.PACKET_LIMIT,
                    max_autoencoder_size=AppConfig.MAX_AE_SIZE,
                    FM_grace_period=AppConfig.FM_GRACE_PERIOD,
                    AD_grace_period=AppConfig.AD_GRACE_PERIOD,
                    pca_components=AppConfig.PCA_COMPONENTS,
                    pca_grace_period=AppConfig.PCA_GRACE_PERIOD,
                    live_stream=AppConfig.RUN_REALTIME_SIMULATION,
                    input_delimiter=AppConfig.INPUT_DELIMITER,
                    pretrained_anom_detector=loaded_kitnet_model
                   )
        # Store initial feature count for logging purposes
        num_features = K.FE.get_num_features()
        logging.info(f"Feature Extractor initialized. Detected {num_features} features.")


        if pca_fitted_from_load and K.pca is not None:
            K.pca_fitted = True
            # TODO: For a complete PCA load, you'd need to load its components and mean here too
            # K.pca.components_ = loaded_state['pca_components_data']
            # K.pca.mean_ = loaded_state['pca_mean_data']


        # Prepare output files for continuous writing
        base_name = os.path.splitext(os.path.basename(path))[0]
        current_datetime_str_for_files = datetime.now().strftime('%Y%m%d_%H%M%S')
        rmse_output_file = os.path.join(rmse_folder, f"rmse_realtime_{base_name}_{current_datetime_str_for_files}.csv")
        ms_conf_output_file = os.path.join(confidence_folder, f"confidence_ms_realtime_{base_name}_{current_datetime_str_for_files}.csv")
        sec_conf_output_file = os.path.join(confidence_folder, f"confidence_sec_realtime_{base_name}_{current_datetime_str_for_files}.csv")
        anomaly_output_file = os.path.join(anomalies_folder, f"anomalies_realtime_{base_name}_{current_datetime_str_for_files}.csv")
        threshold_output_file = os.path.join(thresholds_folder, f"thresholds_realtime_{base_name}_{current_datetime_str_for_files}.csv")

        # Open files and create writers, storing file objects
        rmse_file_obj = open(rmse_output_file, 'w', newline='')
        ms_conf_file_obj = open(ms_conf_output_file, 'w', newline='')
        sec_conf_file_obj = open(sec_conf_output_file, 'w', newline='')
        anomaly_file_obj = open(anomaly_output_file, 'w', newline='')

        rmse_writer = csv.writer(rmse_file_obj)
        ms_conf_writer = csv.writer(ms_conf_file_obj)
        sec_conf_writer = csv.writer(sec_conf_file_obj)
        anomaly_writer = csv.writer(anomaly_file_obj)

        # Write headers
        rmse_writer.writerow(["packet_idx", "timestamp", "rmse"])
        ms_conf_writer.writerow(["packet_idx", "timestamp_ms", "confidence_score"])
        sec_conf_writer.writerow(["packet_idx", "timestamp_s", "confidence_score"])
        anomaly_writer.writerow(["PacketIndex", "Timestamp", "RMSE", "FeatureVector_Summary"])

        packet_count = 0
        grace_period_rmses = []
        adaptive_rmses_buffer = collections.deque(maxlen=AppConfig.ADAPTIVE_THRESHOLD_WINDOW_SIZE)
        
        current_threshold = initial_threshold_from_load if initial_threshold_from_load is not None else 0.01

        total_grace_packets = 0
        if not K.AnomDetector.FM_train_complete or not K.AnomDetector.AD_train_complete:
            # Determine effective grace period based on PCA inclusion
            effective_fm_grace = AppConfig.FM_GRACE_PERIOD
            effective_ad_grace = AppConfig.AD_GRACE_PERIOD
            effective_pca_grace = AppConfig.PCA_GRACE_PERIOD if AppConfig.PCA_COMPONENTS is not None else 0

            # PCA training happens before FM
            if K.pca is not None and not K.pca_fitted:
                total_grace_packets = effective_pca_grace
                logging.info(f"Grace Period Status: Starting PCA training for {effective_pca_grace} samples. Total grace period packets: {total_grace_packets}.")
            elif not K.AnomDetector.FM_train_complete:
                total_grace_packets = effective_fm_grace + effective_pca_grace # FM starts after PCA if PCA is enabled
                logging.info(f"Grace Period Status: PCA training complete. Starting Feature Mapping (FM) for {effective_fm_grace} samples. Total grace period packets: {total_grace_packets}.")
            elif not K.AnomDetector.AD_train_complete:
                total_grace_packets = effective_ad_grace + effective_fm_grace + effective_pca_grace # AD starts after FM and PCA
                logging.info(f"Grace Period Status: Feature Mapping (FM) complete. Starting Anomaly Detection (AD) training for {effective_ad_grace} samples. Total grace period packets: {total_grace_packets}.")
            
            # This is the total number of packets expected to be in a grace period for the ENTIRE training
            # A packet might be consumed by PCA, then FM, then AD, or just FM then AD if no PCA.
            # So `total_grace_packets` should reflect the highest cumulative grace period.
            total_grace_packets = max(effective_pca_grace, effective_fm_grace + effective_pca_grace, effective_ad_grace + effective_fm_grace + effective_pca_grace)

        else:
            logging.info(f"Grace Period Status: All grace periods (PCA, FM, AD) are considered complete. Starting with threshold: {current_threshold:.6f}")
            threshold_calculated = True
            if AppConfig.LOAD_KITSUNE_MODEL and initial_threshold_from_load is not None:
                # Pre-populate buffer for adaptive thresholding if loading model
                for _ in range(AppConfig.ADAPTIVE_THRESHOLD_WINDOW_SIZE // 5):
                    adaptive_rmses_buffer.append(initial_threshold_from_load * 0.8)


        loop_start_time = time.time()
        no_data_start_time = None # New: Timer for how long no data has been received

        while True:
            # Get current process memory usage and total CPU usage for comprehensive progress updates
            process = psutil.Process(os.getpid())
            current_memory_mib = process.memory_info().rss / (1024 * 1024)
            current_cpu_percent = process.cpu_percent(interval=None) # Non-blocking

            feature_vectors_batch, timestamps_batch = K.FE.get_next_batch_vectors(AppConfig.BATCH_SIZE)

            if feature_vectors_batch.size == 0:
                # If no data, check if we've just started waiting
                if no_data_start_time is None:
                    no_data_start_time = time.time()
                    logging.info("No new data from simulator. Starting no-data timeout countdown...")
                
                elapsed_no_data_time = time.time() - no_data_start_time

                # Check if simulator process has terminated
                simulator_terminated = False
                if AppConfig.RUN_REALTIME_SIMULATION and realtime_sim_process:
                    simulator_terminated = realtime_sim_process.poll() is not None
                elif not AppConfig.RUN_REALTIME_SIMULATION: # Offline mode, no simulator process
                    simulator_terminated = True # Assume data source is exhausted if get_next_batch_vectors returns empty

                if simulator_terminated and elapsed_no_data_time >= AppConfig.LIVE_STREAM_MAX_NO_DATA_TIMEOUT:
                    logging.info(f"Real-time simulator process terminated and no new data for {elapsed_no_data_time:.2f} seconds. Concluding stream.")
                    break # Exit main processing loop
                elif elapsed_no_data_time >= AppConfig.LIVE_STREAM_MAX_NO_DATA_TIMEOUT:
                    logging.info(f"No new data for {elapsed_no_data_time:.2f} seconds (timeout). Concluding stream.")
                    break # Exit main processing loop due to no data, even if simulator is still technically running
                elif simulator_terminated:
                    # If simulator terminated but not yet past timeout, give FeatureExtractor a bit more time
                    logging.info(f"Simulator process terminated, waiting for {AppConfig.LIVE_STREAM_MAX_NO_DATA_TIMEOUT - elapsed_no_data_time:.2f} more seconds for any straggling data...")
                else:
                    # Simulator still running, but no data available yet. Wait.
                    logging.info(f"No new data from simulator. Waiting for {AppConfig.MAIN_LOOP_WAIT_TIME} seconds. Total no-data time: {elapsed_no_data_time:.2f}s")
                
                time.sleep(AppConfig.MAIN_LOOP_WAIT_TIME) # Use the dedicated wait time
                continue # Try fetching data again
            else:
                # Data was received, reset the no-data timer
                if no_data_start_time is not None:
                    logging.info(f"Received data after {time.time() - no_data_start_time:.2f}s of no data. Resetting no-data timeout.")
                no_data_start_time = None


            RMSES_batch = K.proc_batch(feature_vectors_batch, timestamps_batch)

            for i_in_batch, RMSE in enumerate(RMSES_batch):
                current_timestamp = timestamps_batch[i_in_batch]
                current_feature_vector = feature_vectors_batch[i_in_batch]
                packet_count += 1

                if AppConfig.PACKET_LIMIT is not None and packet_count > AppConfig.PACKET_LIMIT:
                    logging.info(f"Packet limit ({AppConfig.PACKET_LIMIT}) reached. Stopping processing.")
                    break

                # --- Grace Period Status & Logging ---
                if RMSE == -2: # Indicates PCA or FM training in progress (from Kitsune.py)
                    # Check if PCA training is complete
                    if K.pca is not None and K.pca_fitted and not K.AnomDetector.FM_train_complete and packet_count >= AppConfig.PCA_GRACE_PERIOD:
                        logging.info(f"Grace Period Update: PCA training completed at packet {packet_count}. Moving to Feature Mapping training.")
                    
                    # Check if FM training is complete
                    if K.AnomDetector.FM_train_complete and not K.AnomDetector.AD_train_complete and packet_count >= (AppConfig.PCA_GRACE_PERIOD + AppConfig.FM_GRACE_PERIOD):
                        logging.info(f"Grace Period Update: Feature Mapping (FM) training completed at packet {packet_count}. Moving to Anomaly Detection (AD) training.")

                    # Check if AD training is complete
                    if K.AnomDetector.FM_train_complete and K.AnomDetector.AD_train_complete and not threshold_calculated and packet_count >= total_grace_packets:
                        logging.info(f"Grace Period Update: Anomaly Detection (AD) training completed at packet {packet_count}. Preparing to calculate initial threshold.")
                        
                    # Detailed progress for grace periods
                    if packet_count % 100 == 0 or packet_count == total_grace_packets: # Log more frequently during grace period
                        current_training_phase = "PCA/FM Training"
                        if K.AnomDetector.FM_train_complete and not K.AnomDetector.AD_train_complete:
                            current_training_phase = "AD Training"
                        elif K.pca is not None and not K.pca_fitted:
                            current_training_phase = "PCA Training"

                        packets_remaining = total_grace_packets - packet_count
                        logging.info(
                            f"Progress: Packet {packet_count} (Time: {current_timestamp:.3f}, Mem: {current_memory_mib:.2f} MiB, CPU: {current_cpu_percent:.2f}%). "
                            f"Phase: {current_training_phase}. Packets remaining in grace period: {max(0, packets_remaining)}."
                        )
                    
                    if not threshold_calculated and K.AnomDetector.FM_train_complete and K.AnomDetector.AD_train_complete and K.pca_fitted:
                         # This block handles the actual initial threshold calculation right after all grace periods are met
                        logging.info(f"\nGrace Period Concluded: All training phases complete at packet {packet_count}. Calculating initial anomaly threshold...")
                        if len(grace_period_rmses) > 0:
                            benign_rmses_for_initial_threshold = [r for r in grace_period_rmses if r >= 0]
                            if len(benign_rmses_for_initial_threshold) > 0:
                                sorted_rmses = np.sort(benign_rmses_for_initial_threshold)
                                percentile_idx = min(int(len(sorted_rmses) * (AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE / 100.0)), len(sorted_rmses) - 1)
                                initial_calculated_threshold = sorted_rmses[percentile_idx]
                                
                                old_threshold_for_log = current_threshold # Capture current value before update
                                current_threshold = initial_calculated_threshold
                                logging.info(
                                    f"Threshold Calculated: Initial {AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE}th percentile threshold set to {current_threshold:.6f} "
                                    f"(previously {old_threshold_for_log:.6f}). Used {len(benign_rmses_for_initial_threshold)} samples."
                                )
                                for r in benign_rmses_for_initial_threshold:
                                    adaptive_rmses_buffer.append(r)
                                threshold_calculated = True
                                grace_period_rmses = []
                            else:
                                logging.warning("Warning: Not enough valid RMSEs after grace period to calculate a robust initial threshold. Using default.")
                                threshold_calculated = True # Still mark as calculated to proceed
                        else:
                            logging.warning("Warning: No RMSEs collected during grace period for initial threshold. Using default/loaded threshold.")
                            threshold_calculated = True # Still mark as calculated to proceed

                        if AppConfig.SAVE_KITSUNE_MODEL and not loaded_kitnet_model:
                            try:
                                model_and_threshold_state = {
                                    'kitnet_model_state': K.AnomDetector.get_weights(),
                                    'current_threshold': current_threshold,
                                    'threshold_calculated': threshold_calculated,
                                    'pca_fitted': K.pca_fitted if K.pca else False,
                                    'pca_components': AppConfig.PCA_COMPONENTS,
                                    'pca_grace_period': AppConfig.PCA_GRACE_PERIOD
                                }
                                specific_model_filepath = os.path.join(pickle_folder, f"kitnet_state_{model_file_name_suffix}.pkl")
                                with open(specific_model_filepath, 'wb') as f:
                                    pickle.dump(model_and_threshold_state, f)
                                logging.info(f"Model Saved: Trained Kitsune model and initial threshold saved to: {specific_model_filepath}")
                            except Exception as e:
                                logging.error(f"Error saving model: {e}")
                    continue # Continue to next packet if still in grace period or just finished it
                
                # If not in grace period and threshold is not calculated, accumulate for initial threshold
                if not threshold_calculated:
                    if RMSE is not None and RMSE >= 0:
                        grace_period_rmses.append(RMSE)

                    should_calculate_initial_threshold = False
                    if total_grace_packets > 0 and packet_count >= total_grace_packets:
                        should_calculate_initial_threshold = True
                    # This elif is for edge cases where grace periods might be met earlier or out of order
                    elif K.AnomDetector.FM_train_complete and K.AnomDetector.AD_train_complete and K.pca_fitted and len(grace_period_rmses) >= AppConfig.FM_GRACE_PERIOD: # Use FMgrace as a minimum for collecting "benign" RMSEs
                         should_calculate_initial_threshold = True

                    if should_calculate_initial_threshold:
                        logging.info("\nGrace Period Ended: Sufficient initial samples collected. Calculating initial anomaly threshold...")
                        benign_rmses_for_initial_threshold = [r for r in grace_period_rmses if r >= 0]
                        
                        if len(benign_rmses_for_initial_threshold) > 0:
                            sorted_rmses = np.sort(benign_rmses_for_initial_threshold)
                            percentile_idx = min(int(len(sorted_rmses) * (AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE / 100.0)), len(sorted_rmses) - 1)
                            percentile_threshold = sorted_rmses[percentile_idx]

                            old_threshold = current_threshold
                            current_threshold = percentile_threshold
                            logging.info(f"Threshold Calculated: Initial {AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE}th percentile threshold set to {current_threshold:.6f} (was {old_threshold:.6f}).")

                            for r in benign_rmses_for_initial_threshold:
                                adaptive_rmses_buffer.append(r)

                            benignSample_log = [np.log(r) for r in benign_rmses_for_initial_threshold if r > 0]
                            statistical_threshold = 0.1 # Default if no positive RMSEs
                            if len(benignSample_log) > 0:
                                statistical_threshold = np.exp(np.mean(benignSample_log) + 2 * np.std(benignSample_log))
                            elif len(benign_rmses_for_initial_threshold) > 0: # If log transformation not possible, use raw
                                statistical_threshold = np.mean(benign_rmses_for_initial_threshold) + 2 * np.std(benign_rmses_for_initial_threshold)

                            with open(threshold_output_file, "w", newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(["ThresholdType", "Value", "Description"])
                                writer.writerow([f"{AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE}th Percentile RMSE (Initial)", percentile_threshold, f"{AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE}th percentile RMSE from benign samples during initial phase"])
                                writer.writerow(["Statistical (for reference)", statistical_threshold, "Log-normal distribution based threshold (mean + 2*std)"])
                                writer.writerow(["BenignSamplesUsed (Initial)", len(benign_rmses_for_initial_threshold), "Number of benign samples used for initial threshold calculation"])
                                writer.writerow(["TotalGracePeriodPackets", total_grace_packets, "Total packets in all grace periods (PCA + FM + AD)"])
                            logging.info(f"Thresholds saved to {threshold_output_file}")
                        else:
                            logging.warning("Warning: Not enough valid RMSEs during initial phase to calculate a robust threshold. Using default.")

                        threshold_calculated = True
                        grace_period_rmses = []

                        if AppConfig.SAVE_KITSUNE_MODEL and not loaded_kitnet_model:
                            try:
                                model_and_threshold_state = {
                                    'kitnet_model_state': K.AnomDetector.get_weights(),
                                    'current_threshold': current_threshold,
                                    'threshold_calculated': threshold_calculated,
                                    'pca_fitted': K.pca_fitted if K.pca else False,
                                    'pca_components': AppConfig.PCA_COMPONENTS,
                                    'pca_grace_period': AppConfig.PCA_GRACE_PERIOD
                                }
                                specific_model_filepath = os.path.join(pickle_folder, f"kitnet_state_{model_file_name_suffix}.pkl")
                                with open(specific_model_filepath, 'wb') as f:
                                    pickle.dump(model_and_threshold_state, f)
                                logging.info(f"Model Saved: Trained Kitsune model and initial threshold saved to: {specific_model_filepath}")
                            except Exception as e:
                                logging.error(f"Error saving model: {e}")

                if not threshold_calculated:
                    continue # If still in grace period or threshold not calculated, continue to next packet


                if RMSE is not None and RMSE >= 0 and RMSE <= current_threshold * 1.5: # Only add "benign" like RMSEs to buffer
                    adaptive_rmses_buffer.append(RMSE)

                # --- Adaptive Threshold Update Logging ---
                if packet_count % AppConfig.ADAPTIVE_THRESHOLD_UPDATE_INTERVAL == 0 and \
                   len(adaptive_rmses_buffer) >= AppConfig.ADAPTIVE_THRESHOLD_WINDOW_SIZE // 2: # Ensure enough samples in buffer
                    
                    new_threshold = np.percentile(list(adaptive_rmses_buffer), AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE)
                    # Only update if the new threshold is significantly different or higher
                    if new_threshold > 0:
                        if (abs(new_threshold - current_threshold) / current_threshold > 0.01) or (new_threshold > current_threshold):
                            old_threshold = current_threshold
                            current_threshold = new_threshold
                            logging.info(
                                f"Threshold Updated: Packet {packet_count}. Old: {old_threshold:.6f}, New: {current_threshold:.6f}. "
                                f"Reason: Adaptive update based on {len(adaptive_rmses_buffer)} recent samples from buffer."
                            )
                    else:
                        logging.warning(
                            f"Warning: Adaptive threshold calculated to {new_threshold:.6f} at packet {packet_count}. "
                            f"Keeping previous threshold {current_threshold:.6f}. (Buffer size: {len(adaptive_rmses_buffer)})"
                        )
            
            # --- Anomaly Details Logging ---
            if threshold_calculated and RMSE is not None and RMSE >= 0: # Only check for anomalies after threshold is set
                if RMSE > current_threshold:
                    # Summarize feature vector for logging and output file
                    feature_vector_summary = "N/A" # Default if vector is empty or malformed
                    if isinstance(current_feature_vector, np.ndarray) and current_feature_vector.size > 0:
                        if len(current_feature_vector) <= 6: # Small enough to show fully
                            feature_vector_summary = str(np.round(current_feature_vector, 2).tolist())
                        else: # Summarize large vectors
                            feature_vector_summary = (
                                f"[{current_feature_vector[0]:.2f}, {current_feature_vector[1]:.2f}, {current_feature_vector[2]:.2f}, "
                                f"..., {current_feature_vector[-3]:.2f}, {current_feature_vector[-2]:.2f}, {current_feature_vector[-1]:.2f}] "
                                f"(len: {len(current_feature_vector)})"
                            )
                    
                    anomaly_writer.writerow([packet_count, current_timestamp, RMSE, feature_vector_summary])
                    logging.info(
                        f"*** ANOMALY DETECTED! *** Packet: {packet_count}, Time: {current_timestamp:.3f}, "
                        f"RMSE: {RMSE:.4f} (Threshold: {current_threshold:.6f}). Features: {feature_vector_summary}"
                    )

                # --- Progress Updates Logging ---
                # Log progress more frequently if not in grace period
                if packet_count % 100 == 0: # Log every 100 packets during normal operation
                    elapsed_time = time.time() - loop_start_time
                    processing_rate = 0.0
                    if elapsed_time > 0:
                        processing_rate = packet_count / elapsed_time
                    
                    logging.info(
                        f"Progress: Packet {packet_count} (Time: {current_timestamp:.3f}, Mem: {current_memory_mib:.2f} MiB, CPU: {current_cpu_percent:.2f}%). "
                        f"Latest RMSE: {RMSE:.4f}. Rate: {processing_rate:.2f} packets/sec."
                    )
            
            # Check packet limit and break
            if AppConfig.PACKET_LIMIT is not None and packet_count >= AppConfig.PACKET_LIMIT:
                logging.info(f"Packet limit ({AppConfig.PACKET_LIMIT}) reached. Stopping processing.")
                break

        # --- Final Threshold Calculation if Grace Period Not Fully Met ---
        if not threshold_calculated and len(grace_period_rmses) > 0 and (packet_count > 0):
            logging.info("\nCalculating final anomaly threshold (due to early termination/small dataset)...")
            benign_rmses_for_final_threshold = [r for r in grace_period_rmses if r >= 0]
            if len(benign_rmses_for_final_threshold) > 0:
                sorted_rmses = np.sort(benign_rmses_for_final_threshold)
                percentile_idx = min(int(len(sorted_rmses) * (AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE / 100.0)), len(sorted_rmses) - 1)
                final_threshold = sorted_rmses[percentile_idx]
                logging.info(f"Final Threshold: Calculated {AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE}th percentile final threshold: {final_threshold:.6f}")
                with open(threshold_output_file, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["--- Final Threshold ---", "", ""])
                    writer.writerow([f"{AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE}th Percentile RMSE (Final)", final_threshold, f"Final {AppConfig.ADAPTIVE_THRESHOLD_PERCENTILE}th percentile RMSE based on available benign samples"])
                    writer.writerow(["Samples Used (Final)", len(benign_rmses_for_final_threshold), "Number of benign samples used for final threshold calculation"])
                logging.info(f"Final threshold appended to {threshold_output_file}")
            else:
                logging.warning("Warning: Not enough valid RMSEs to calculate a robust final threshold. Default used.")


    except Exception as e:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log = os.path.join(logs_folder, f"error_realtime_{timestamp_str}.txt") if AppConfig.RUN_REALTIME_SIMULATION else os.path.join(logs_folder, f"error_offline_{timestamp_str}.txt")
        with open(error_log, "w") as f:
            f.write(f"Error occurred at {timestamp_str}\n")
            f.write(f"Error: {str(e)}\n")
        logging.error(f"\nCRITICAL ERROR: {e}. Check {error_log} for details.")
        raise # Re-raise the exception after logging

    finally:
        # Ensure realtime_sim_process is defined to avoid UnboundLocalError
        # Use locals().get for safe access if it might not be initialized due to an early error
        realtime_sim_process = locals().get('realtime_sim_process', None)

        if realtime_sim_process and AppConfig.RUN_REALTIME_SIMULATION:
            logging.info("Cleanup: Terminating real-time simulator process...")
            realtime_sim_process.terminate()
            realtime_sim_process.wait()
            logging.info("Cleanup: Simulator process terminated.")
            if rmse_file_obj and not rmse_file_obj.closed: rmse_file_obj.close()
            if ms_conf_file_obj and not ms_conf_file_obj.closed: ms_conf_file_obj.close()
            if sec_conf_file_obj and not sec_conf_file_obj.closed: sec_conf_file_obj.close()
            if anomaly_file_obj and not anomaly_file_obj.closed: anomaly_file_obj.close()
        elif not AppConfig.RUN_REALTIME_SIMULATION: # For offline mode, clean chunks
            if os.path.exists(chunks_folder):
                try:
                    shutil.rmtree(chunks_folder)
                    logging.info("\nCleanup: Temporary chunks cleaned up successfully.")
                except Exception as e:
                    logging.warning(f"\nCleanup Warning: Could not clean up chunks folder: {str(e)}")

        profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(50) # Uncomment to print profiling stats if needed after profiling run

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
