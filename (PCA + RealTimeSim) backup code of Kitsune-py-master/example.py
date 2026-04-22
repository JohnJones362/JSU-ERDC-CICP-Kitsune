# old code: 6/24/25 (9:19 AM)
from Kitsune import Kitsune
import numpy as np
import time
from KitNET.KitNET import KitNET
from scapy.all import rdpcap, wrpcap # Keep for potential future use or if split_pcap is ever used
import os
import shutil
import psutil
from chunked_kitsune import process_in_chunks # Keep this for chunked processing of static files
import pandas as pd
from datetime import datetime
import multiprocessing
import csv
import cProfile
import pstats
from tqdm import tqdm
import gc
import subprocess # Added for running the simulator
import sys # Added for sys.exit

# Import the simulator parameters and function
from realtime_data_simulator import simulate_data_pipeline, INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE, SLEEP_TIME

def check_system_resources(file_path):
    """Check if system has enough resources to process the file"""
    try:
        # First check if file exists
        if not os.path.exists(file_path):
            print(f"Error: Input file {file_path} not found!")
            return False

        # Check available disk space (need at least 3x file size)
        file_size = os.path.getsize(file_path)
        free_space = shutil.disk_usage(os.path.dirname(os.path.abspath(file_path))).free
        if free_space < file_size * 3:
            raise RuntimeError(f"Not enough disk space. Need at least {file_size * 3 / (1024**3):.2f} GB, but only {free_space / (1024**3):.2f} GB available")

        # Calculate required RAM based on packet limit and chunk size
        # Assuming ~1KB per packet in memory for processing
        packet_memory = 1000  # bytes per packet
        # Use a reasonable default if packet_limit is not yet defined or is None
        # Access packet_limit from globals or define a default for this check if it's not passed
        # For this function, it's safer to not rely on a global 'packet_limit' that might not be set yet.
        # Instead, assume a typical processing memory footprint.
        required_ram = 500 * 1024 * 1024 # 500 MB for Kitsune and processing
        required_ram_gb = required_ram / (1024**3)
        min_ram_gb = max(0.5, required_ram_gb)  # At least 0.5GB free RAM required

        free_ram = psutil.virtual_memory().available
        if free_ram < min_ram_gb * (1024**3):  # Convert GB to bytes for comparison
            raise RuntimeError(f"Not enough RAM. Need at least {min_ram_gb:.2f} GB, but only {free_ram / (1024**3):.2f} GB available")

        return True
    except Exception as e:
        print(f"Resource check failed: {str(e)}")
        return False

def main():
    profiler = cProfile.Profile()
    profiler.enable()

    # --- Configuration Variables (Always defined) ---
    run_realtime_simulation = True # Set to True to enable real-time simulation
    realtime_sim_process = None
    input_delimiter = '\t' # Default delimiter for FeatureExtractor
    path = "" # Initialize path
    packet_limit = None # Initialize packet_limit, will be set specifically below
    num_chunks = None # Initialize num_chunks, will be set specifically below
    import_file = None # Initialize import_file for offline mode

    if run_realtime_simulation:
        print("Starting real-time data simulation...")
        try:
            # Start the simulation script as a separate process
            realtime_sim_process = subprocess.Popen([sys.executable, 'realtime_data_simulator.py'])
            print(f"Giving simulator {SLEEP_TIME * 2} seconds to start writing data...")
            time.sleep(SLEEP_TIME * 2)
            print("Simulator should be running.")
            path = OUTPUT_FILE # Kitsune will read from the simulated data file
            # packet_limit remains None for continuous real-time processing unless explicitly set
            input_delimiter = '\t' # Simulator outputs tab-delimited
        except Exception as e:
            print(f"Error starting real-time simulator: {e}")
            sys.exit(1)
    else:
        # --- Offline Analysis Setup ---
        # Number of chunks to split the processing into
        cpu_count = os.cpu_count()
        num_chunks = max(6, min(cpu_count * 2, 12))  # Scale with CPU cores, but cap at 12
        print(f"Using {num_chunks} chunks based on {cpu_count} CPU cores")

        # Optional packet limit (set to None or a number)
        packet_limit = 1_000_000  # Set to a number like 1_000_000 to limit processing
        print(f"Packet limit: {'Unlimited' if packet_limit is None else packet_limit}")

        # Input file name (without extension)
        import_file = "CTU-IoT-Malware-Capture-1-1conn.log.labeled" # CHANGE THIS to your file name without .pcap or .csv/.tsv
        use_tsv = True # Set to True if your static file is TSV/CSV, False for PCAP

        if not use_tsv:
            path = f"{import_file}.pcap"
        else:
            path = f"{import_file}.csv" # Or .tsv, depending on your file
            # IMPORTANT: If your original file is pipe-delimited, set the delimiter here
            input_delimiter = '|' if import_file == "CTU-IoT-Malware-Capture-1-1conn.log.labeled" else '\t' # Example for your specific file

        print(f"Input file for offline analysis: {path}")


    # KitNET params:
    maxAE = 10
    FMgrace = 5_000
    ADgrace = 10_000

    # --- PCA Optimization Parameters (from research paper) ---
    pca_components = None
    pca_grace_period = FMgrace
    if pca_components is not None:
        print(f"PCA enabled: {pca_components} components, grace period: {pca_grace_period} packets.")
    else:
        print("PCA disabled.")
    # --- End PCA Optimization Parameters ---

    # Create the results folder structure
    results_folder = "Results"
    anomalies_folder = os.path.join(results_folder, "anomalies")
    confidence_folder = os.path.join(results_folder, "confidence")
    rmse_folder = os.path.join(results_folder, "rmse")
    pickle_folder = os.path.join(results_folder, "pickle")
    logs_folder = os.path.join(results_folder, "logs")
    chunks_folder = os.path.join(results_folder, "chunks")
    thresholds_folder = os.path.join(results_folder, "thresholds")

    # Create all required folders
    for folder in [anomalies_folder, confidence_folder, rmse_folder, pickle_folder, logs_folder, chunks_folder, thresholds_folder]:
        os.makedirs(folder, exist_ok=True)

    # Check system resources
    print("\nChecking system resources...")
    # Pass path directly to check_system_resources
    if not check_system_resources(path):
        if realtime_sim_process:
            realtime_sim_process.terminate()
        sys.exit(1)

    # Save/load pickle path (adjusted for real-time scenario)
    pickle_file = os.path.join(pickle_folder, f"kitnet_state_maxAE={maxAE}_FMgrace={FMgrace}_ADgrace={ADgrace}.pkl") # Removed 'realtime' from name

    print("\nStarting processing with Kitsune...")
    start = time.time()

    K = None # Initialize K outside try block
    try:
        if run_realtime_simulation:
            # Initialize Kitsune for real-time processing
            K = Kitsune(file_path=path,
                        limit=packet_limit, # Will be None for continuous processing
                        max_autoencoder_size=maxAE,
                        FM_grace_period=FMgrace,
                        AD_grace_period=ADgrace,
                        pca_components=pca_components,
                        pca_grace_period=pca_grace_period,
                        live_stream=True, # Enable live stream mode in FeatureExtractor
                        input_delimiter=input_delimiter # Pass delimiter
                       )

            # Prepare output files for continuous writing
            base_name = os.path.splitext(os.path.basename(path))[0]
            rmse_output_file = os.path.join(rmse_folder, f"rmse_realtime_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            ms_conf_output_file = os.path.join(confidence_folder, f"confidence_ms_realtime_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            sec_conf_output_file = os.path.join(confidence_folder, f"confidence_sec_realtime_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            anomaly_output_file = os.path.join(anomalies_folder, f"anomalies_realtime_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            threshold_output_file = os.path.join(thresholds_folder, f"thresholds_realtime_{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

            # Open files for appending
            rmse_writer = csv.writer(open(rmse_output_file, 'w', newline=''))
            ms_conf_writer = csv.writer(open(ms_conf_output_file, 'w', newline=''))
            sec_conf_writer = csv.writer(open(sec_conf_output_file, 'w', newline=''))
            anomaly_writer = csv.writer(open(anomaly_output_file, 'w', newline=''))

            # Write headers
            rmse_writer.writerow(["packet_idx", "timestamp", "rmse"])
            ms_conf_writer.writerow(["packet_idx", "timestamp_ms", "confidence_score"])
            sec_conf_writer.writerow(["packet_idx", "timestamp_s", "confidence_score"])
            anomaly_writer.writerow(["PacketIndex", "Timestamp", "RMSE"])

            packet_count = 0
            grace_period_rmses = []
            threshold_calculated = False
            current_threshold = 0.1
            total_grace_packets = (pca_grace_period if pca_components is not None else 0) + FMgrace + ADgrace

            print(f"Kitsune is in training/grace period for {total_grace_packets} packets (PCA + FM + AD).")

            while True:
                RMSE = K.proc_next_packet()
                packet_count += 1

                if packet_limit is not None and packet_count > packet_limit:
                    print(f"Packet limit ({packet_limit}) reached. Stopping processing.")
                    break

                if RMSE == -1: # No more packets from FE (simulator might have finished or is slow)
                    if realtime_sim_process and realtime_sim_process.poll() is not None:
                        print("Real-time simulator process has terminated. No more data.")
                        break
                    else:
                        print("No new data from simulator. Waiting...")
                        time.sleep(SLEEP_TIME)
                        continue

                # Handle PCA grace period
                if RMSE == -2:
                    if packet_count % 1000 == 0:
                        print(f"PCA training in progress... Packet {packet_count}")
                    continue

                # Collect RMSEs for threshold calculation during the combined grace period
                if packet_count <= total_grace_packets and RMSE is not None:
                    if RMSE >= 0:
                        grace_period_rmses.append(RMSE)

                # Calculate threshold once grace period is over
                if packet_count == total_grace_packets and not threshold_calculated:
                    print("\nGrace period ended. Calculating anomaly threshold...")
                    if len(grace_period_rmses) > 0:
                        benign_rmses_for_threshold = [r for r in grace_period_rmses if r >= 0]

                        if len(benign_rmses_for_threshold) > 0:
                            max_rmse_threshold = max(benign_rmses_for_threshold)
                            benignSample_log = []
                            for rmse in benign_rmses_for_threshold:
                                if rmse > 0:
                                    benignSample_log.append(np.log(rmse))

                            if len(benignSample_log) > 0:
                                statistical_threshold = np.exp(np.mean(benignSample_log) + 2 * np.std(benignSample_log))
                            else:
                                statistical_threshold = np.mean(benign_rmses_for_threshold) + 2 * np.std(benign_rmses_for_threshold) if len(benign_rmses_for_threshold) > 0 else 0.1

                            current_threshold = max_rmse_threshold # Or statistical_threshold
                            print(f"Calculated Max RMSE threshold: {max_rmse_threshold}")
                            print(f"Calculated Statistical threshold: {statistical_threshold}")
                            print(f"Using threshold: {current_threshold}")

                            with open(threshold_output_file, "w", newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow(["ThresholdType", "Value", "Description"])
                                writer.writerow(["MaxRMSE", max_rmse_threshold, "Maximum RMSE value from benign samples during grace period"])
                                writer.writerow(["Statistical", statistical_threshold, "Log-normal distribution based threshold (mean + 2*std)"])
                                writer.writerow(["BenignSamplesUsed", len(benign_rmses_for_threshold), "Number of benign samples used for threshold calculation"])
                                writer.writerow(["TotalGracePeriodPackets", total_grace_packets, "Total packets in all grace periods (PCA + FM + AD)"])
                            print(f"Thresholds saved to {threshold_output_file}")
                        else:
                            print("Not enough valid RMSEs during grace period to calculate a robust threshold. Using default.")

                    threshold_calculated = True
                    grace_period_rmses = []

                # Only start anomaly detection and logging after the grace period
                if packet_count > total_grace_packets and RMSE is not None and RMSE >= 0:
                    if RMSE > 0:
                        ms_conf = 1.0 / RMSE
                        sec_conf = 1.0 / (RMSE * 1000)
                    else:
                        ms_conf = float('inf')
                        sec_conf = float('inf')

                    rmse_writer.writerow([packet_count, K.get_latest_packet_time(), RMSE])
                    ms_conf_writer.writerow([packet_count, K.get_latest_packet_time(), ms_conf])
                    sec_conf_writer.writerow([packet_count, K.get_latest_packet_time(), sec_conf])

                    if RMSE > current_threshold:
                        anomaly_writer.writerow([packet_count, K.get_latest_packet_time(), RMSE])
                        print(f"Anomaly detected at packet {packet_count}, RMSE: {RMSE:.4f}")

                    if packet_count % 1000 == 0:
                        print(f"Processed {packet_count} packets. Latest RMSE: {RMSE:.4f}")
        else:
            # --- Offline Analysis Execution ---
            # Ensure path is set correctly for offline mode
            # This logic is now handled above in the `else` block for `run_realtime_simulation`
            # The `path` variable is already set based on `import_file` and `use_tsv`

            rmse_file, ms_conf_file, sec_conf_file = process_in_chunks(
                input_file=path,
                num_chunks=num_chunks, # num_chunks is now defined in the else block
                maxAE=maxAE,
                FMgrace=FMgrace,
                ADgrace=ADgrace,
                output_dir=results_folder,
                chunks_dir=chunks_folder,
                packet_limit=packet_limit, # packet_limit is now defined in the else block
                pca_components=pca_components,
                pca_grace_period=pca_grace_period
            )

            print("\nProcessing results from chunked analysis...")
            batch_size = 100000
            RMSEs = []
            timestamps = []
            packet_indices = []

            with open(rmse_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)

                total_lines = sum(1 for _ in f)
                f.seek(0)
                next(reader)

                with tqdm(total=total_lines, desc="Reading RMSE values") as pbar:
                    current_batch = {'packet_idx': [], 'time': [], 'rmse': []}

                    for row in reader:
                        rmse_val = float(row[2])
                        if rmse_val == -2:
                            continue

                        current_batch['packet_idx'].append(int(row[0]))
                        current_batch['time'].append(float(row[1]))
                        current_batch['rmse'].append(rmse_val)

                        if len(current_batch['rmse']) >= batch_size:
                            packet_indices.extend(current_batch['packet_idx'])
                            timestamps.extend(current_batch['time'])
                            RMSEs.extend(current_batch['rmse'])
                            pbar.update(len(current_batch['rmse']))
                            current_batch = {'packet_idx': [], 'time': [], 'rmse': []}
                            gc.collect()

                    if current_batch['rmse']:
                        packet_indices.extend(current_batch['packet_idx'])
                        timestamps.extend(current_batch['time'])
                        RMSEs.extend(current_batch['rmse'])
                        pbar.update(len(current_batch['rmse']))

            stop = time.time()
            print("\nComplete. Time elapsed: "+ str(stop - start))

            print("\nCalculating statistics...")
            total_grace_packets = (pca_grace_period if pca_components is not None else 0) + FMgrace + ADgrace

            benign_start_idx_in_filtered_list = 0
            for i, p_idx in enumerate(packet_indices):
                if p_idx >= total_grace_packets:
                    benign_start_idx_in_filtered_list = i
                    break

            print("Calculating threshold from benign samples...")
            benign_rmses = []
            for i in range(benign_start_idx_in_filtered_list, min(benign_start_idx_in_filtered_list + 500000, len(RMSEs))):
                benign_rmses.append(RMSEs[i])

            print(f"Number of benign samples collected for threshold: {len(benign_rmses)}")
            if len(benign_rmses) > 0:
                print(f"Min RMSE: {min(benign_rmses)}")
                print(f"Max RMSE: {max(benign_rmses)}")
                print(f"Number of zeros: {sum(1 for x in benign_rmses if x == 0)}")
                print(f"Number of negative values: {sum(1 for x in benign_rmses if x < 0)}")
                print(f"Number of NaNs: {sum(1 for x in benign_rmses if np.isnan(x))}")

            max_rmse_threshold = max(benign_rmses) if len(benign_rmses) > 0 else 0.1
            print(f"Max RMSE threshold: {max_rmse_threshold}")

            benignSample = []
            for rmse in benign_rmses:
                if rmse > 0:
                    benignSample.append(np.log(rmse))

            if len(benignSample) > 0:
                statistical_threshold = np.exp(np.mean(benignSample) + 2 * np.std(benignSample))
            else:
                statistical_threshold = np.mean(benign_rmses) + 2 * np.std(benign_rmses) if len(benign_rmses) > 0 else 0.1
            print(f"Statistical threshold: {statistical_threshold}")

            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

            threshold_filename = os.path.join(thresholds_folder, f"thresholds_{os.path.splitext(os.path.basename(path))[0]}_time={timestamp_str}.csv")
            with open(threshold_filename, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ThresholdType", "Value", "Description"])
                writer.writerow(["MaxRMSE", max_rmse_threshold, "Maximum RMSE value from benign samples"])
                writer.writerow(["Statistical", statistical_threshold, "Log-normal distribution based threshold (mean + 2*std)"])
                writer.writerow(["BenignSamples", len(benign_rmses), "Number of benign samples used"])
                writer.writerow(["TotalGracePeriodPackets", total_grace_packets, "Total packets in all grace periods (PCA + FM + AD)"])
                writer.writerow(["EffectiveBenignStartIndex", benign_start_idx_in_filtered_list, "Starting index in filtered RMSE list for benign samples"])

            threshold = max_rmse_threshold

            anomalies = []
            print("Detecting anomalies...")

            for i in tqdm(range(0, len(RMSEs), batch_size), desc="Processing batches"):
                batch_end = min(i + batch_size, len(RMSEs))
                batch_rmse = np.array(RMSEs[i:batch_end])
                batch_timestamps = np.array(timestamps[i:batch_end])
                batch_packet_indices = np.array(packet_indices[i:batch_end])

                valid_batch_indices = np.where(batch_packet_indices >= total_grace_packets)[0]

                if len(valid_batch_indices) > 0:
                    filtered_batch_rmse = batch_rmse[valid_batch_indices]
                    filtered_batch_timestamps = batch_timestamps[valid_batch_indices]
                    filtered_batch_packet_indices = batch_packet_indices[valid_batch_indices]

                    anomaly_indices_in_filtered_batch = np.where(filtered_batch_rmse > threshold)[0]
                    if len(anomaly_indices_in_filtered_batch) > 0:
                        for idx in anomaly_indices_in_filtered_batch:
                            anomalies.append((
                                filtered_batch_packet_indices[idx],
                                filtered_batch_timestamps[idx],
                                filtered_batch_rmse[idx]
                            ))

                del batch_rmse
                del batch_timestamps
                del batch_packet_indices
                gc.collect()

            print("\nSaving results...")

            anomaly_filename = os.path.join(anomalies_folder, f"anomalies_{os.path.splitext(os.path.basename(path))[0]}_time={timestamp_str}.csv")
            with open(anomaly_filename, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["PacketIndex", "Timestamp", "RMSE"])
                for anomaly in anomalies:
                    writer.writerow(anomaly)

            new_rmse_file = os.path.join(rmse_folder, f"rmse_{os.path.splitext(os.path.basename(path))[0]}_time={timestamp_str}.csv")
            new_ms_conf_file = os.path.join(confidence_folder, f"confidence_ms_{os.path.splitext(os.path.basename(path))[0]}_time={timestamp_str}.csv")
            new_sec_conf_file = os.path.join(confidence_folder, f"confidence_sec_{os.path.splitext(os.path.basename(path))[0]}_time={timestamp_str}.csv")

            os.rename(rmse_file, new_rmse_file)
            os.rename(ms_conf_file, new_ms_conf_file)
            os.rename(sec_conf_file, new_sec_conf_file)

            log_file = os.path.join(logs_folder, f"process_completed_{timestamp_str}.txt")
            with open(log_file, "w") as f:
                f.write(f"Process finished at {timestamp_str}\n")
                f.write(f"\nProcessing Summary:\n")
                f.write(f"- Input file: {path}\n")
                f.write(f"- File size: {os.path.getsize(path) / (1024**2):.2f} MB\n")
                f.write(f"- Number of chunks: {num_chunks}\n")
                f.write(f"- Total packets processed (excluding PCA grace period): {len(RMSEs)}\n")
                f.write(f"- Anomalies detected: {len(anomalies)}\n")
                f.write(f"- Time elapsed: {stop - start:.2f} seconds\n")
                f.write(f"- Processing rate: {len(RMSEs)/(stop - start):.2f} packets/second\n")
                f.write(f"\nThreshold Information:\n")
                f.write(f"- Max RMSE threshold: {max_rmse_threshold}\n")
                f.write(f"- Statistical threshold: {statistical_threshold}\n")
                f.write(f"- Threshold used for detection: {threshold}\n")
                f.write(f"- Benign samples used for threshold: {len(benign_rmses)}\n")
                f.write(f"- Total grace period packets (PCA+FM+AD): {total_grace_packets}\n")
                f.write(f"\nPCA Configuration:\n")
                f.write(f"- PCA Enabled: {'Yes' if pca_components is not None else 'No'}\n")
                if pca_components is not None:
                    f.write(f"- PCA Components: {pca_components}\n")
                    f.write(f"- PCA Grace Period: {pca_grace_period} packets\n")
                f.write(f"\nSystem Information:\n")
                f.write(f"- CPU cores: {os.cpu_count()}\n")
                f.write(f"- RAM available: {psutil.virtual_memory().available / (1024**3):.2f} GB\n")
                f.write(f"- Disk space available: {shutil.disk_usage('.').free / (1024**3):.2f} GB\n")
                f.write(f"\nOutput Files:\n")
                f.write(f"- Anomalies: {anomaly_filename}\n")
                f.write(f"- RMSE values: {new_rmse_file}\n")
                f.write(f"- MS Confidence: {new_ms_conf_file}\n")
                f.write(f"- Sec Confidence: {new_sec_conf_file}\n")
                f.write(f"- Thresholds: {threshold_filename}\n")

            print(f"\nResults saved:")
            print(f"- Anomalies saved to {anomaly_filename}")
            print(f"- Millisecond confidence scores saved to {new_ms_conf_file}")
            print(f"- Second confidence scores saved to {new_sec_conf_file}")
            print(f"- RMSE values saved to {new_rmse_file}")
            print(f"- Thresholds saved to {threshold_filename}")
            print(f"- Found {len(anomalies)} anomalies")
            print(f"\nProcess completion log saved to {log_file}")


    except Exception as e:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log = os.path.join(logs_folder, f"error_realtime_{timestamp_str}.txt") if run_realtime_simulation else os.path.join(logs_folder, f"error_offline_{timestamp_str}.txt")
        with open(error_log, "w") as f:
            f.write(f"Error occurred at {timestamp_str}\n")
            f.write(f"Error: {str(e)}\n")
        print(f"\nError occurred. Check {error_log} for details.")
        raise

    finally:
        if run_realtime_simulation and realtime_sim_process:
            print("Terminating real-time simulator process...")
            realtime_sim_process.terminate()
            realtime_sim_process.wait()
            print("Simulator process terminated.")
            # Ensure file writers are closed if they were opened
            if 'rmse_writer' in locals() and not rmse_writer.closed: rmse_writer.close()
            if 'ms_conf_writer' in locals() and not ms_conf_writer.closed: ms_conf_writer.close()
            if 'sec_conf_writer' in locals() and not sec_conf_writer.closed: sec_conf_writer.close()
            if 'anomaly_writer' in locals() and not anomaly_writer.closed: anomaly_writer.close()
        else:
            if os.path.exists(chunks_folder):
                try:
                    shutil.rmtree(chunks_folder)
                    print("\nTemporary chunks cleaned up successfully")
                except Exception as e:
                    print(f"\nWarning: Could not clean up chunks folder: {str(e)}")

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(50) # Uncomment to print profiling stats

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()

