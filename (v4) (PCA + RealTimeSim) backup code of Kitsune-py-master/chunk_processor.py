from Kitsune import Kitsune
import numpy as np
import csv
import os
import multiprocessing
import gc
from typing import Tuple # Added for type hinting
import logging # Added for logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    K = Kitsune(file_path=chunk_file,
                limit=np.inf, # Process entire chunk file
                max_autoencoder_size=maxAE,
                FM_grace_period=FMgrace,
                AD_grace_period=ADgrace,
                pca_components=pca_components,
                pca_grace_period=pca_grace_period
               )

    # Prepare output files for this chunk
    chunk_base_name = os.path.splitext(os.path.basename(chunk_file))[0]
    rmse_file = os.path.join(output_dir, f"{chunk_base_name}_rmse.csv")
    ms_conf_file = os.path.join(output_dir, f"{chunk_base_name}_confidence_ms.csv")
    sec_conf_file = os.path.join(output_dir, f"{chunk_base_name}_confidence_sec.csv")

    current_batch = {'rmse': [], 'time': [], 'packet_idx': []}
    batch_counter = 0

    with open(rmse_file, 'w', newline='') as rmse_f, \
         open(ms_conf_file, 'w', newline='') as ms_conf_f, \
         open(sec_conf_file, 'w', newline='') as sec_conf_f:
        
        rmse_writer = csv.writer(rmse_f)
        ms_conf_writer = csv.writer(ms_conf_f)
        sec_conf_writer = csv.writer(sec_conf_f)

        rmse_writer.writerow(["packet_idx", "timestamp", "rmse"])
        ms_conf_writer.writerow(["packet_idx", "timestamp_ms", "confidence_score"])
        sec_conf_writer.writerow(["packet_idx", "timestamp_s", "confidence_score"])

        packet_idx = 0
        while True:
            feature_vectors_batch, timestamps_batch = K.FE.get_next_batch_vectors(1) # Process one packet at a time for offline processing of chunks

            if feature_vectors_batch.size == 0:
                break # End of chunk file

            RMSES_batch = K.proc_batch(feature_vectors_batch, timestamps_batch)

            for i_in_batch, RMSE in enumerate(RMSES_batch):
                current_timestamp = timestamps_batch[i_in_batch]
                
                # Filter out RMSEs that are -2 (PCA training in progress) for writing
                if RMSE is not None and RMSE != -2:
                    current_batch['rmse'].append(RMSE)
                    current_batch['time'].append(current_timestamp)
                    current_batch['packet_idx'].append(packet_idx)
                    batch_counter += 1
                
                packet_idx += 1 # Increment packet_idx for each processed packet

            # Write batch to files if it reaches a reasonable size (e.g., 100 packets)
            # This balances I/O overhead with memory usage.
            if batch_counter >= 100: # You can adjust this batch write size
                for i in range(len(current_batch['rmse'])):
                    # Calculate confidence scores
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

                # Clear batch after writing
                current_batch = {'rmse': [], 'time': [], 'packet_idx': []}
                batch_counter = 0 # Reset counter

        # Write any remaining data in the batch buffer to files before closing
        if current_batch['rmse']:
            for i in range(len(current_batch['rmse'])):
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
    
    # Explicitly delete the Kitsune instance and force garbage collection once for cleanup
    del K
    gc.collect() # <-- Strategic call after K is no longer needed

    logging.info(f"Finished processing chunk: {chunk_file}")
    return rmse_file, ms_conf_file, sec_conf_file

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # Example usage:
    # This block won't run directly during the main example.py execution,
    # as process_chunk is called via multiprocessing.Pool.
    # It's here for direct testing if needed.
    # You would need a dummy chunk file to test it standalone.
    logging.info("Running chunk_processor.py standalone example (for testing only).")
    # Dummy setup for testing
    # dummy_chunk_file = "dummy_chunk.tsv"
    # with open(dummy_chunk_file, 'w', newline='') as f:
    #     f.write("time_epoch\tprotocol\tsource_ip\tsource_port\tsource_mac\tdestination_ip\tdestination_port\tdestination_mac\tframe_len\n")
    #     f.write("1678886400.000\tTCP\t1.1.1.1\t1234\tAA:AA:AA:AA:AA:AA\t2.2.2.2\t80\tBB:BB:BB:BB:BB:BB\t100\n")
    #     f.write("1678886401.000\tUDP\t3.3.3.3\t5678\tCC:CC:CC:CC:CC:CC\t4.4.4.4\t53\tDD:DD:DD:DD:DD:DD\t50\n")
    # output_dir = "temp_output"
    # os.makedirs(output_dir, exist_ok=True)
    # process_chunk(dummy_chunk_file, 10, 5, 5, output_dir)
    # os.remove(dummy_chunk_file)
    # shutil.rmtree(output_dir) # Clean up
