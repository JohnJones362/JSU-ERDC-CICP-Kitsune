# old code: 6/24/25 (9:19 AM)

from Kitsune import Kitsune
import numpy as np
import csv
import os
import multiprocessing
import gc
from typing import Tuple # Added for type hinting

def process_chunk(chunk_file: str, maxAE: int, FMgrace: int, ADgrace: int, output_dir: str, pca_components: int = None, pca_grace_period: int = None) -> Tuple[str, str, str]:
    """
    Process a single chunk file with Kitsune and save RMSEs and confidence scores.

    Args:
        chunk_file (str): Path to chunk file (pcap or tsv).
        maxAE (int): Maximum size for any autoencoder.
        FMgrace (int): Feature mapping grace period.
        ADgrace (int): Anomaly detection grace period.
        output_dir (str): Directory to store RMSE and confidence output.
        pca_components (int, optional): Number of components for PCA dimensionality reduction.
        pca_grace_period (int, optional): Number of packets to collect for PCA fitting.

    Returns:
        Tuple[str, str, str]: Paths to the output RMSE file, millisecond confidence file,
                              and second confidence file.
    """
    # Initialize Kitsune with PCA parameters
    # Rationale: Pass PCA configuration to Kitsune. This allows Kitsune to perform
    # dimensionality reduction as a preprocessing step, improving efficiency and
    # potentially stability as suggested by the research paper.
    K = Kitsune(chunk_file, limit=np.inf, max_autoencoder_size=maxAE, FM_grace_period=FMgrace, AD_grace_period=ADgrace, pca_components=pca_components, pca_grace_period=pca_grace_period)

    # Create output files
    base_name = os.path.splitext(os.path.basename(chunk_file))[0]
    rmse_file = os.path.join(output_dir, f"{base_name}_rmse.csv")
    ms_conf_file = os.path.join(output_dir, f"{base_name}_ms_conf.csv")
    sec_conf_file = os.path.join(output_dir, f"{base_name}_sec_conf.csv")

    # Process packets in batches
    batch_size = 1000
    current_batch = {'rmse': [], 'time': [], 'packet_idx': []} # Added packet_idx to batch

    # Open all files at once to avoid frequent open/close operations
    with open(rmse_file, 'w', newline='') as rf, \
         open(ms_conf_file, 'w', newline='') as mcf, \
         open(sec_conf_file, 'w', newline='') as scf:

        rmse_writer = csv.writer(rf)
        ms_conf_writer = csv.writer(mcf)
        sec_conf_writer = csv.writer(scf)

        # Write headers
        rmse_writer.writerow(["packet_idx", "timestamp", "rmse"])
        ms_conf_writer.writerow(["packet_idx", "timestamp_ms", "confidence_score"]) # Updated header
        sec_conf_writer.writerow(["packet_idx", "timestamp_s", "confidence_score"]) # Updated header

        packet_count = 0
        while True:
            # Process next packet
            RMSE = K.proc_next_packet()
            packet_count += 1 # Increment packet_count for every packet processed

            if RMSE == -1:
                break
            elif RMSE == -2: # PCA training in progress, skip anomaly score for now
                # Rationale: During PCA grace period, Kitsune collects data to fit PCA.
                # No anomaly scores are generated yet. Skipping recording RMSE ensures
                # that incomplete or pre-PCA scores do not corrupt results.
                continue

            # Store results in current batch
            current_batch['rmse'].append(RMSE)
            current_batch['time'].append(K.get_latest_packet_time())
            current_batch['packet_idx'].append(packet_count) # Store actual packet index

            # Write batch if full
            if len(current_batch['rmse']) >= batch_size:
                for i in range(len(current_batch['rmse'])):
                    # Calculate confidence scores
                    # Rationale: Confidence scores are derived from RMSE.
                    # A higher RMSE indicates lower confidence in normality.
                    if current_batch['rmse'][i] > 0:
                        ms_conf = 1.0 / current_batch['rmse'][i]
                        sec_conf = 1.0 / (current_batch['rmse'][i] * 1000)  # Scale for seconds
                    else:
                        ms_conf = float('inf') # Or a very large number if RMSE is zero/negative
                        sec_conf = float('inf')

                    rmse_writer.writerow([current_batch['packet_idx'][i],
                                          current_batch['time'][i],
                                          current_batch['rmse'][i]])
                    ms_conf_writer.writerow([current_batch['packet_idx'][i],
                                             current_batch['time'][i],
                                             ms_conf])
                    sec_conf_writer.writerow([current_batch['packet_idx'][i],
                                             current_batch['time'][i],
                                             sec_conf])

                # Clear batch
                current_batch = {'rmse': [], 'time': [], 'packet_idx': []}
                gc.collect()

        # Write remaining batch
        if current_batch['rmse']:
            for i in range(len(current_batch['rmse'])):
                # Calculate confidence scores for remaining batch
                if current_batch['rmse'][i] > 0:
                    ms_conf = 1.0 / current_batch['rmse'][i]
                    sec_conf = 1.0 / (current_batch['rmse'][i] * 1000)
                else:
                    ms_conf = float('inf')
                    sec_conf = float('inf')

                rmse_writer.writerow([current_batch['packet_idx'][i],
                                      current_batch['time'][i],
                                      current_batch['rmse'][i]])
                ms_conf_writer.writerow([current_batch['packet_idx'][i],
                                         current_batch['time'][i],
                                         ms_conf])
                sec_conf_writer.writerow([current_batch['packet_idx'][i],
                                         current_batch['time'][i],
                                         sec_conf])

    # Clean up
    del K
    gc.collect()

    return rmse_file, ms_conf_file, sec_conf_file

if __name__ == '__main__':
    multiprocessing.freeze_support()