# old code: 6/24/25 (9:19 AM)
from Kitsune import Kitsune
import numpy as np
# import cupy as np
import time
from KitNET.KitNET import KitNET
from scapy.all import rdpcap, wrpcap
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
        current_packet_limit = globals().get('packet_limit', None)
        if current_packet_limit is not None:
            total_packets = min(1_000_000, current_packet_limit)  # Use packet limit if specified
        else:
            total_packets = 1_000_000  # Default to 1M packets for resource estimation

        # Calculate memory per chunk
        num_chunks = max(3, min(os.cpu_count() * 2, 12))  # Same calculation as in main()
        packets_per_chunk = total_packets // num_chunks
        required_ram = (packets_per_chunk * packet_memory) * 2  # Double for safety margin

        # Convert to GB for comparison
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

    # Number of chunks to split the processing into
    cpu_count = os.cpu_count()
    num_chunks = max(6, min(cpu_count * 2, 12))  # Scale with CPU cores, but cap at 12
    print(f"Using {num_chunks} chunks based on {cpu_count} CPU cores")

    # Optional packet limit (set to None or a number)
    packet_limit = 1_000_000  # Set to a number like 1_000_000 to limit processing
    print(f"Packet limit: {'Unlimited' if packet_limit is None else packet_limit}")

    # Always split large files
    split_pcap = True

    # Load pre-built model or nah
    load_pickle = False

    # pca/tsv file name
    import_file = "Dec2019_00002_20191206103000"

    # use pre-made tsv file or nah
    use_tsv = True

    # KitNET params:
    maxAE = 10
    FMgrace = 5_000
    ADgrace = 10_000

    # --- PCA Optimization Parameters (from research paper) ---
    # Rationale: The research paper emphasizes advanced dimensionality reduction like PCA
    # to improve robustness and efficiency in IDS. These parameters enable and configure PCA.
    pca_components = None         # Number of components for PCA. Set to None to disable PCA.
                                # If an integer, features will be reduced to this many dimensions.
    pca_grace_period = FMgrace  # Number of packets to collect for PCA fitting.
                                # It's recommended to set this to be at least FMgrace.
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
    thresholds_folder = os.path.join(results_folder, "thresholds")  # NEW: Add thresholds folder

    # Create all required folders
    for folder in [anomalies_folder, confidence_folder, rmse_folder, pickle_folder, logs_folder, chunks_folder, thresholds_folder]:
        os.makedirs(folder, exist_ok=True)

    # Input file path
    if not use_tsv:
        path = f"{import_file}.pcap"
    else:
        path = f"{import_file}.pcap.tsv"

    # Check if input file exists
    if not os.path.exists(path):
        print(f"Error: Input file {path} not found!")
        return

    # Check system resources
    print("\nChecking system resources...")
    if not check_system_resources(path):
        return

    # Save/load pickle path
    pickle_file = os.path.join(pickle_folder, f"kitnet_state_for_{import_file}_with_maxAE={maxAE}_FMgrace={FMgrace}_ADgrace={ADgrace}_packet_limit={packet_limit}.pkl")

    print("\nStarting processing...")
    start = time.time()

    try:
        # Process in chunks and get combined files
        rmse_file, ms_conf_file, sec_conf_file = process_in_chunks(
            input_file=path,
            num_chunks=num_chunks,
            maxAE=maxAE,
            FMgrace=FMgrace,
            ADgrace=ADgrace,
            output_dir=results_folder,
            chunks_dir=chunks_folder,
            packet_limit=packet_limit,
            pca_components=pca_components,       # Pass PCA components
            pca_grace_period=pca_grace_period    # Pass PCA grace period
        )

        # Process RMSEs in batches to reduce memory usage
        print("\nProcessing results...")
        batch_size = 100000  # Process 100K rows at a time
        RMSEs = []
        timestamps = []
        packet_indices = [] # To store packet_idx

        with open(rmse_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header, but store it if needed

            # Count total lines for progress bar
            total_lines = sum(1 for _ in f)
            f.seek(0)
            next(reader)  # Skip header again

            # Process in batches with progress bar
            with tqdm(total=total_lines, desc="Reading RMSE values") as pbar:
                current_batch = {'packet_idx': [], 'time': [], 'rmse': []}

                for row in reader:
                    # Rationale: Filter out RMSE values of -2, which indicate packets
                    # processed during the PCA grace period when no anomaly score is generated.
                    # This ensures that only valid RMSEs are used for threshold calculation
                    # and anomaly detection, improving the accuracy of the results.
                    rmse_val = float(row[2])
                    if rmse_val == -2:
                        continue # Skip packets during PCA grace period

                    current_batch['packet_idx'].append(int(row[0]))
                    current_batch['time'].append(float(row[1]))
                    current_batch['rmse'].append(rmse_val)

                    if len(current_batch['rmse']) >= batch_size:
                        # Process batch
                        packet_indices.extend(current_batch['packet_idx'])
                        timestamps.extend(current_batch['time'])
                        RMSEs.extend(current_batch['rmse'])

                        # Update progress
                        pbar.update(len(current_batch['rmse'])) # Update with actual processed count

                        # Clear batch
                        current_batch = {'packet_idx': [], 'time': [], 'rmse': []}
                        gc.collect()

                # Process remaining batch
                if current_batch['rmse']:
                    packet_indices.extend(current_batch['packet_idx'])
                    timestamps.extend(current_batch['time'])
                    RMSEs.extend(current_batch['rmse'])
                    pbar.update(len(current_batch['rmse'])) # Update with actual processed count

        stop = time.time()
        print("\nComplete. Time elapsed: "+ str(stop - start))

        # Calculate statistics in batches
        print("\nCalculating statistics...")
        # The benign_start index needs to account for the PCA grace period if PCA is enabled.
        # It should be the sum of pca_grace_period (if active) + FMgrace + ADgrace.
        # However, since RMSEs collected during pca_grace_period are filtered out (value -2),
        # the effective start for benign samples is simply FMgrace + ADgrace.
        # The packet_indices list will reflect the correct indices after filtering.
        
        # Determine the effective start index for benign samples in the filtered RMSEs list.
        # We need to find the first index in 'packet_indices' that is greater than or equal to
        # the sum of all grace periods (PCA + FM + AD).
        total_grace_packets = (pca_grace_period if pca_components is not None else 0) + FMgrace + ADgrace
        
        benign_start_idx_in_filtered_list = 0
        for i, p_idx in enumerate(packet_indices):
            if p_idx >= total_grace_packets:
                benign_start_idx_in_filtered_list = i
                break
        
        # Calculate threshold using all available benign samples
        print("Calculating threshold from benign samples...")
        benign_rmses = []
        # Use a subset of benign samples for threshold calculation to avoid memory issues
        # and to ensure the threshold is based on a stable period.
        # Here, we take up to 500,000 samples starting from the effective benign_start_idx.
        for i in range(benign_start_idx_in_filtered_list, min(benign_start_idx_in_filtered_list + 500000, len(RMSEs))):
            benign_rmses.append(RMSEs[i])

        # Debug prints
        print(f"Number of benign samples collected for threshold: {len(benign_rmses)}")
        if len(benign_rmses) > 0:
            print(f"Min RMSE: {min(benign_rmses)}")
            print(f"Max RMSE: {max(benign_rmses)}")
            print(f"Number of zeros: {sum(1 for x in benign_rmses if x == 0)}")
            print(f"Number of negative values: {sum(1 for x in benign_rmses if x < 0)}")
            print(f"Number of NaNs: {sum(1 for x in benign_rmses if np.isnan(x))}")

        # NEW: Calculate max RMSE threshold
        max_rmse_threshold = max(benign_rmses) if len(benign_rmses) > 0 else 0.1
        print(f"Max RMSE threshold: {max_rmse_threshold}")

        # Calculate threshold using log-normal distribution
        benignSample = []
        for rmse in benign_rmses:
            if rmse > 0:  # Only include positive RMSE values
                benignSample.append(np.log(rmse))

        if len(benignSample) > 0:
            statistical_threshold = np.exp(np.mean(benignSample) + 2 * np.std(benignSample))
        else:
            # Fallback to a simple statistical threshold if log-normal fails
            statistical_threshold = np.mean(benign_rmses) + 2 * np.std(benign_rmses) if len(benign_rmses) > 0 else 0.1
        print(f"Statistical threshold: {statistical_threshold}")

        # Get current timestamp for filenames
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # NEW: Save both thresholds to file
        threshold_filename = os.path.join(thresholds_folder, f"thresholds_{os.path.splitext(os.path.basename(path))[0]}_time={timestamp_str}.csv")
        with open(threshold_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["ThresholdType", "Value", "Description"])
            writer.writerow(["MaxRMSE", max_rmse_threshold, "Maximum RMSE value from benign samples"])
            writer.writerow(["Statistical", statistical_threshold, "Log-normal distribution based threshold (mean + 2*std)"])
            writer.writerow(["BenignSamples", len(benign_rmses), "Number of benign samples used"])
            writer.writerow(["TotalGracePeriodPackets", total_grace_packets, "Total packets in all grace periods (PCA + FM + AD)"])
            writer.writerow(["EffectiveBenignStartIndex", benign_start_idx_in_filtered_list, "Starting index in filtered RMSE list for benign samples"])


        # Use statistical threshold for anomaly detection (or change to max_rmse_threshold if preferred)
        threshold = max_rmse_threshold  # Change this to statistical_threshold if you want to use statistical threshold

        # Find anomalies in batches
        anomalies = []
        print("Detecting anomalies...")

        for i in tqdm(range(0, len(RMSEs), batch_size), desc="Processing batches"):
            batch_end = min(i + batch_size, len(RMSEs))
            batch_rmse = np.array(RMSEs[i:batch_end])
            batch_timestamps = np.array(timestamps[i:batch_end])
            batch_packet_indices = np.array(packet_indices[i:batch_end])

            # Skip grace period based on the original packet index
            # Rationale: Anomalies should only be detected after the entire system
            # (including PCA and KitNET's own grace periods) has completed its training.
            # This ensures that initial learning phases do not generate false positives.
            # We use the original packet index to ensure consistency with the grace period definition.
            
            # Find indices within the current batch that are after the total grace period
            valid_batch_indices = np.where(batch_packet_indices >= total_grace_packets)[0]

            if len(valid_batch_indices) > 0:
                # Filter batch to only include valid indices
                filtered_batch_rmse = batch_rmse[valid_batch_indices]
                filtered_batch_timestamps = batch_timestamps[valid_batch_indices]
                filtered_batch_packet_indices = batch_packet_indices[valid_batch_indices]

                # Vectorized anomaly detection on the filtered batch
                anomaly_indices_in_filtered_batch = np.where(filtered_batch_rmse > threshold)[0]
                if len(anomaly_indices_in_filtered_batch) > 0:
                    for idx in anomaly_indices_in_filtered_batch:
                        anomalies.append((
                            filtered_batch_packet_indices[idx],
                            filtered_batch_timestamps[idx],
                            filtered_batch_rmse[idx]
                        ))

            # Clear memory
            del batch_rmse
            del batch_timestamps
            del batch_packet_indices
            gc.collect()

        print("\nSaving results...")

        # Save anomalies to file with timestamp
        anomaly_filename = os.path.join(anomalies_folder, f"anomalies_{os.path.splitext(os.path.basename(path))[0]}_time={timestamp_str}.csv")
        with open(anomaly_filename, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["PacketIndex", "Timestamp", "RMSE"])
            for anomaly in anomalies:
                writer.writerow(anomaly)

        # Move and rename output files
        new_rmse_file = os.path.join(rmse_folder, f"rmse_{os.path.splitext(os.path.basename(path))[0]}_time={timestamp_str}.csv")
        new_ms_conf_file = os.path.join(confidence_folder, f"confidence_ms_{os.path.splitext(os.path.basename(path))[0]}_time={timestamp_str}.csv")
        new_sec_conf_file = os.path.join(confidence_folder, f"confidence_sec_{os.path.splitext(os.path.basename(path))[0]}_time={timestamp_str}.csv")

        # Rationale: Renaming files after processing ensures that temporary files
        # are not confused with final outputs and provides a clear, timestamped
        # record of the results for each run.
        os.rename(rmse_file, new_rmse_file)
        os.rename(ms_conf_file, new_ms_conf_file)
        os.rename(sec_conf_file, new_sec_conf_file)

        # Create completion log with timestamp
        log_file = os.path.join(logs_folder, f"process_completed_{timestamp_str}.txt")
        with open(log_file, "w") as f:
            f.write(f"Process finished at {timestamp_str}\n")
            f.write(f"\nProcessing Summary:\n")
            f.write(f"- Input file: {path}\n")
            f.write(f"- File size: {os.path.getsize(path) / (1024**2):.2f} MB\n")
            f.write(f"- Number of chunks: {num_chunks}\n")
            f.write(f"- Total packets processed (excluding PCA grace period): {len(RMSEs)}\n") # Updated count
            f.write(f"- Anomalies detected: {len(anomalies)}\n")
            f.write(f"- Time elapsed: {stop - start:.2f} seconds\n")
            f.write(f"- Processing rate: {len(RMSEs)/(stop - start):.2f} packets/second\n")
            f.write(f"\nThreshold Information:\n")
            f.write(f"- Max RMSE threshold: {max_rmse_threshold}\n")
            f.write(f"- Statistical threshold: {statistical_threshold}\n")
            f.write(f"- Threshold used for detection: {threshold}\n")
            f.write(f"- Benign samples used for threshold: {len(benign_rmses)}\n")
            f.write(f"- Total grace period packets (PCA+FM+AD): {total_grace_packets}\n")
            f.write(f"\nPCA Configuration:\n") # Added PCA config to log
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
        # Log error and cleanup
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log = os.path.join(logs_folder, f"error_{timestamp_str}.txt")
        with open(error_log, "w") as f:
            f.write(f"Error occurred at {timestamp_str}\n")
            f.write(f"Error: {str(e)}\n")
        print(f"\nError occurred. Check {error_log} for details.")
        raise

    finally:
        # Cleanup temporary chunks
        if os.path.exists(chunks_folder):
            try:
                shutil.rmtree(chunks_folder)
                print("\nTemporary chunks cleaned up successfully")
            except Exception as e:
                print(f"\nWarning: Could not clean up chunks folder: {str(e)}")

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats(50)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()