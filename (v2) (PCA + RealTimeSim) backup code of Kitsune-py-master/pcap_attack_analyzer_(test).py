#!/usr/bin/env python3
"""
PCAP Attack Timeline Analyzer for SWaT Dataset
Analyzes pcap files to identify numbered attack sequences and generate start/stop events.
Now includes advanced Kitsune-based anomaly detection with multiprocessing,
progress bars, timestamped output, and resource checks.
"""

import pandas as pd
import numpy as np
from scapy.all import rdpcap, PcapReader, IP, TCP, UDP
from datetime import datetime, timedelta
import csv
import argparse
from collections import defaultdict
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import sys
import shutil # For check_system_resources and cleanup
import psutil  # For check_system_resources
from tqdm import tqdm # Import tqdm for progress bars

# Assuming Kitsune and KitNET are available in the same directory or Python path
# For Kitsune integration, we need to import functions from chunked_kitsune.py
try:
    from chunked_kitsune import process_in_chunks
except ImportError:
    print("Error: chunked_kitsune.py not found. Kitsune detection will not be available.")
    process_in_chunks = None # Set to None if import fails

# --- Helper Functions from example.py for Kitsune workflow ---
def check_system_resources(file_path, packet_limit=None):
    """Check if system has enough resources to process the file"""
    try:
        # First check if file exists
        if not os.path.exists(file_path):
            print(f"Error: Input file {file_path} not found!")
            return False

        # Check available disk space (need at least 3x file size for temporary chunks and outputs)
        file_size = os.path.getsize(file_path)
        # Using a higher multiplier for safety due to intermediate files
        required_disk_space = file_size * 5
        free_space = shutil.disk_usage(os.path.dirname(os.path.abspath(file_path))).free
        if free_space < required_disk_space:
            raise RuntimeError(f"Not enough disk space. Need at least {required_disk_space / (1024**3):.2f} GB, but only {free_space / (1024**3):.2f} GB available")

        # Calculate required RAM based on packet limit and chunk size
        # Assuming Kitsune might need more memory per packet for its internal structures
        packet_memory_estimate_kitsune = 2000 # bytes per packet, adjusted for Kitsune overhead
        
        # This estimate needs to align with how Kitsune processes chunks.
        # Kitsune processes individual packets within a chunk, so the key is the size of a chunk in memory.
        # Let's estimate based on a reasonable chunk size for Kitsune.
        # If chunked_kitsune splits into N chunks, each chunk is ~total_packets / N.
        cpu_count = os.cpu_count()
        num_kitsune_chunks = max(6, min(cpu_count * 2, 12)) # Estimate based on example.py
        
        # Get actual total packets if not limited, for a better RAM estimate
        total_packets_for_ram_est = 0
        if packet_limit:
            total_packets_for_ram_est = packet_limit
        else: # Try to get an estimate of total packets for RAM check if no limit
            try:
                # This can be slow, but it's for a pre-check
                with PcapReader(file_path) as pr:
                    total_packets_for_ram_est = sum(1 for _ in pr)
            except Exception:
                # If cannot read, assume a large number for safety in RAM check
                total_packets_for_ram_est = 5_000_000 # Default to 5 million packets for RAM check

        packets_per_kitsune_chunk = total_packets_for_ram_est // num_kitsune_chunks
        
        # Estimate RAM needed per process (for one chunk), plus some for Python overhead
        # This is a rough estimate; actual usage depends on Kitsune's internal data structures.
        required_ram = (packets_per_kitsune_chunk * packet_memory_estimate_kitsune) * 1.5 # 50% safety margin
        
        # Convert to GB for comparison
        required_ram_gb = required_ram / (1024**3)
        min_ram_gb = max(0.5, required_ram_gb) # At least 0.5GB free RAM required
        
        free_ram = psutil.virtual_memory().available
        if free_ram < min_ram_gb * (1024**3):  # Convert GB to bytes for comparison
            raise RuntimeError(f"Not enough RAM. Need at least {min_ram_gb:.2f} GB, but only {free_ram / (1024**3):.2f} GB available")

        print("System resource check passed.")
        return True
    except Exception as e:
        print(f"Resource check failed: {str(e)}")
        return False

# Function to process a chunk of packets, designed for multiprocessing (for statistical/label methods)
def process_packet_chunk(packets_chunk, start_packet_num):
    """
    Process a list of Scapy packets to extract relevant features.
    Designed to be run in a separate process.

    Args:
        packets_chunk (list): A list of Scapy packet objects.
        start_packet_num (int): The global starting packet number for this chunk.

    Returns:
        list: A list of dictionaries, where each dictionary contains features for a packet.
    """
    traffic_data = []
    for i, packet in enumerate(packets_chunk):
        current_packet_num = start_packet_num + i

        # Extract timestamp
        timestamp = datetime.fromtimestamp(float(packet.time))

        # Initialize packet info with default None values
        packet_info = {
            'packet_num': current_packet_num, # Add packet number for global context
            'timestamp': timestamp,
            'size': len(packet),
            'protocol': None,
            'src_ip': None,
            'dst_ip': None,
            'src_port': None,
            'dst_port': None,
            'flags': None
        }

        # Extract IP layer info
        if packet.haslayer(IP):
            packet_info['src_ip'] = packet[IP].src
            packet_info['dst_ip'] = packet[IP].dst
            packet_info['protocol'] = packet[IP].proto

        # Extract TCP layer info
        if packet.haslayer(TCP):
            packet_info['src_port'] = packet[TCP].sport
            packet_info['dst_port'] = packet[TCP].dport
            packet_info['flags'] = packet[TCP].flags
            packet_info['protocol'] = 'TCP'

        # Extract UDP layer info
        elif packet.haslayer(UDP):
            packet_info['src_port'] = packet[UDP].sport
            packet_info['dst_port'] = packet[UDP].dport
            packet_info['protocol'] = 'UDP'

        traffic_data.append(packet_info)
    return traffic_data

class PcapAttackAnalyzer:
    def __init__(self, pcap_file, labels_file=None, num_processes=None):
        self.pcap_file = pcap_file
        self.labels_file = labels_file
        self.packets_df = None # Store processed packets as DataFrame for easier access
        self.labels_df = None
        self.attack_timeline = []
        # Determine number of processes, capping at 8 to avoid excessive resource usage
        self.num_processes = num_processes if num_processes is not None else min(mp.cpu_count(), 8)

    def _get_total_packets_count(self, packet_limit):
        """Helper to count total packets in the PCAP file."""
        print(f"Counting total packets in '{self.pcap_file}' (this might take a while for large files)...")
        total_packets = 0
        try:
            # Use PcapReader for efficiency when just counting
            with PcapReader(self.pcap_file) as pcap_reader:
                for _ in tqdm(pcap_reader, desc="Counting Packets"):
                    total_packets += 1
                    if packet_limit and total_packets >= packet_limit:
                        break
        except Exception as e:
            print(f"Error counting packets: {e}")
            return 0
        print(f"Total packets found: {total_packets}")
        return total_packets

    def load_pcap_multiprocess(self, packet_limit=None):
        """Load and parse the pcap file using multiprocessing (for statistical/label methods)."""
        print(f"Loading PCAP file with {self.num_processes} processes: {self.pcap_file}")

        total_packets_in_file = self._get_total_packets_count(packet_limit)
        
        # Use actual total packets or the limit specified
        packets_to_process = min(total_packets_in_file, packet_limit) if packet_limit else total_packets_in_file

        if packets_to_process == 0:
            print("No packets to process.")
            return False

        # Calculate chunk size
        # Ensure at least 1 packet per chunk if total_packets is less than num_processes
        chunk_size = max(1, packets_to_process // self.num_processes) 
        
        chunks_data = [] # To store lists of packets for each chunk
        current_chunk_packets = []
        global_packet_offset = 0

        print(f"Distributing into {self.num_processes} chunks of approximately {chunk_size} packets each.")

        # Read packets and divide into in-memory chunks
        # This part will show progress of splitting the file
        pcap_reader_for_splitting = PcapReader(self.pcap_file)
        with tqdm(total=packets_to_process, desc="Splitting PCAP into chunks") as pbar_splitting:
            packets_read_for_splitting = 0
            current_packet_count_in_chunk = 0 # Track packets in current chunk
            for packet in pcap_reader_for_splitting:
                current_chunk_packets.append(packet)
                packets_read_for_splitting += 1
                current_packet_count_in_chunk += 1
                pbar_splitting.update(1)

                if current_packet_count_in_chunk >= chunk_size and len(chunks_data) < self.num_processes - 1:
                    chunks_data.append((current_chunk_packets, global_packet_offset))
                    global_packet_offset += len(current_chunk_packets)
                    current_chunk_packets = []
                    current_packet_count_in_chunk = 0 # Reset chunk counter
                
                if packet_limit and packets_read_for_splitting >= packet_limit:
                    break # Stop reading if packet limit is reached
        pcap_reader_for_splitting.close() # Close the PcapReader explicitly


        # Add any remaining packets as the last chunk
        if current_chunk_packets:
            chunks_data.append((current_chunk_packets, global_packet_offset))

        print(f"Prepared {len(chunks_data)} in-memory chunks for parallel processing.")

        start_time = time.time()
        all_processed_data = []

        # Use ProcessPoolExecutor to process chunks in parallel
        print("Processing chunks in parallel:")
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            futures = [executor.submit(process_packet_chunk, chunk, offset) for chunk, offset in chunks_data]
            
            for i, future in tqdm(enumerate(as_completed(futures)), total=len(futures), desc="Processing Chunks"):
                try:
                    processed_chunk_data = future.result()
                    all_processed_data.extend(processed_chunk_data)
                except Exception as e:
                    print(f"\nError processing chunk {i+1}: {e}")
                    # If a chunk fails, try to continue with other chunks, but report failure
                    return False # Indicate failure if any chunk fails

        elapsed_time = time.time() - start_time
        print(f"Parallel processing complete. Total packets processed: {len(all_processed_data)} in {elapsed_time:.2f} seconds.")
        if elapsed_time > 0:
            print(f"Processing rate: {len(all_processed_data) / elapsed_time:.2f} packets/second.")
        
        # Sort all_processed_data by 'packet_num' to ensure correct order
        all_processed_data.sort(key=lambda x: x['packet_num'])
        self.packets_df = pd.DataFrame(all_processed_data)
        print(f"Loaded {len(self.packets_df)} packets into DataFrame.")
        return True

    def load_pcap_single_process(self, packet_limit=None):
        """Fallback single-process loading method for PCAP (for statistical/label methods)."""
        print(f"Loading PCAP file (single process): {self.pcap_file}")
        traffic_data = []
        packet_count = 0
        start_time = time.time()

        total_packets_in_file = self._get_total_packets_count(packet_limit)
        packets_to_process = min(total_packets_in_file, packet_limit) if packet_limit else total_packets_in_file

        if packets_to_process == 0:
            print("No packets to process.")
            return False

        try:
            with PcapReader(self.pcap_file) as pcap_reader:
                with tqdm(total=packets_to_process, desc="Processing Packets (Single Process)") as pbar_single:
                    for packet in pcap_reader:
                        packet_count += 1
                        
                        # Extract timestamp
                        timestamp = datetime.fromtimestamp(float(packet.time))
                        
                        # Initialize packet info
                        packet_info = {
                            'packet_num': packet_count, # Add packet number
                            'timestamp': timestamp,
                            'size': len(packet),
                            'protocol': None,
                            'src_ip': None,
                            'dst_ip': None,
                            'src_port': None,
                            'dst_port': None,
                            'flags': None
                        }
                        
                        # Extract IP layer info
                        if packet.haslayer(IP):
                            packet_info['src_ip'] = packet[IP].src
                            packet_info['dst_ip'] = packet[IP].dst
                            packet_info['protocol'] = packet[IP].proto
                        
                        # Extract TCP layer info
                        if packet.haslayer(TCP):
                            packet_info['src_port'] = packet[TCP].sport
                            packet_info['dst_port'] = packet[TCP].dport
                            packet_info['flags'] = packet[TCP].flags
                            packet_info['protocol'] = 'TCP'
                        
                        # Extract UDP layer info
                        elif packet.haslayer(UDP):
                            packet_info['src_port'] = packet[UDP].sport
                            packet_info['dst_port'] = packet[UDP].dport
                            packet_info['protocol'] = 'UDP'
                        
                        traffic_data.append(packet_info)
                        
                        pbar_single.update(1)
                        
                        # Check packet limit
                        if packet_limit and packet_count >= packet_limit:
                            break
            
            elapsed = time.time() - start_time
            print(f"Single-process loading complete: {packet_count} packets in {elapsed:.2f}s")
            self.packets_df = pd.DataFrame(traffic_data)
            return True
            
        except Exception as e:
            print(f"\nError loading PCAP file (single process): {str(e)}")
            return False

    def load_pcap(self, use_multiprocessing=True, packet_limit=None):
        """
        Wrapper function to load PCAP file,
        choosing between multiprocessing and single-process for statistical/label methods.
        """
        if use_multiprocessing:
            success = self.load_pcap_multiprocess(packet_limit=packet_limit)
            if not success: # Fallback if multiprocessing fails
                print("Multiprocessing failed, falling back to single process...")
                success = self.load_pcap_single_process(packet_limit=packet_limit)
            return success
        else:
            return self.load_pcap_single_process(packet_limit=packet_limit)
        
    def load_labels(self):
        """Load the labels CSV file if provided"""
        if self.labels_file and os.path.exists(self.labels_file):
            try:
                self.labels_df = pd.read_csv(self.labels_file)
                self.labels_df['t_stamp'] = pd.to_datetime(self.labels_df['t_stamp'])
                print(f"Loaded labels file with {len(self.labels_df)} records")
                return True
            except Exception as e:
                print(f"Error loading labels file: {e}")
        return False
    
    def extract_traffic_features(self):
        """
        Returns the DataFrame of extracted packet features.
        This method assumes self.packets_df is already populated by load_pcap.
        """
        if self.packets_df is None:
            print("Packet data not loaded. Please load PCAP file first.")
            return pd.DataFrame() # Return empty DataFrame

        print("Traffic features already extracted during PCAP loading.")
        return self.packets_df
    
    def detect_anomalies_statistical(self, traffic_df):
        """Detect anomalies using statistical methods"""
        if traffic_df.empty:
            print("No traffic data to analyze for anomalies.")
            return pd.DataFrame()

        print("Analyzing traffic patterns and detecting anomalies (Statistical Method)...")

        # Group traffic by time windows (e.g., 30-second intervals)
        traffic_df['time_window'] = traffic_df['timestamp'].dt.floor('30S')
        
        # Calculate traffic metrics per time window
        window_stats = traffic_df.groupby('time_window').agg(
            packet_count=('size', 'count'), # Count packets
            total_bytes=('size', 'sum'),    # Sum of packet sizes
            avg_size=('size', 'mean'),      # Mean packet size
            std_size=('size', 'std'),       # Standard deviation of packet sizes
            unique_src_ips=('src_ip', 'nunique'),
            unique_dst_ips=('dst_ip', 'nunique'),
            unique_src_ports=('src_port', 'nunique'),
            unique_dst_ports=('dst_port', 'nunique')
        ).reset_index()
        
        # Fill NaN values (e.g., std_size might be NaN if only one packet in window)
        window_stats.fillna(0, inplace=True)
        
        # Calculate anomaly scores based on deviation from normal
        numeric_cols = [col for col in window_stats.columns if col != 'time_window']
        
        for col in numeric_cols:
            mean_val = window_stats[col].mean()
            std_val = window_stats[col].std()
            # Avoid division by zero for std_val
            if std_val > 0:
                window_stats[f'{col}_zscore'] = np.abs((window_stats[col] - mean_val) / std_val)
            else:
                window_stats[f'{col}_zscore'] = 0 # If std is 0, all values are same, z-score is 0

        # Identify anomalous windows (z-score > 2)
        zscore_cols = [col for col in window_stats.columns if col.endswith('_zscore')]
        if not zscore_cols:
            window_stats['anomaly_score'] = 0
        else:
            window_stats['anomaly_score'] = window_stats[zscore_cols].max(axis=1)

        window_stats['is_anomaly'] = window_stats['anomaly_score'] > 2
        
        return window_stats
    
    def detect_attacks_from_labels(self):
        """Detect attacks using the labels file"""
        if self.labels_df is None:
            return None
        
        print("Detecting attacks from labels file...")

        # Find periods where attack_label is not 0
        attack_periods = []
        current_attack = None
        attack_number = 0
        
        # Use tqdm for progress bar if labels_df is large
        for idx, row in tqdm(self.labels_df.iterrows(), total=len(self.labels_df), desc="Detecting from Labels"):
            if row['attack_label'] != 0:  # Attack detected
                if current_attack is None:  # Start of new attack
                    attack_number += 1
                    current_attack = {
                        'attack_number': attack_number,
                        'start_time': row['t_stamp'],
                        'start_idx': idx,
                        'attack_type': row['attack_label']
                    }
            else:  # Normal operation
                if current_attack is not None:  # End of attack
                    current_attack['end_time'] = self.labels_df.iloc[idx-1]['t_stamp']
                    current_attack['end_idx'] = idx - 1
                    attack_periods.append(current_attack)
                    current_attack = None
        
        # Handle case where attack continues until end of data
        if current_attack is not None:
            current_attack['end_time'] = self.labels_df.iloc[-1]['t_stamp']
            current_attack['end_idx'] = len(self.labels_df) - 1
            attack_periods.append(current_attack)
        
        return attack_periods
    
    def generate_attack_timeline(self, attack_periods):
        """Generate attack timeline in the required format for label-based detection"""
        timeline = []
        
        for attack in tqdm(attack_periods, desc="Generating Timeline"):
            # Start event
            start_event = {
                'attacks': attack['start_idx'], # Use index from labels as simulated attack ID
                'DATETIME': attack['start_time'].strftime('%m/%d/%Y %H:%M:%S'),
                'attack_number': f"{attack['attack_number']}-start"
            }
            timeline.append(start_event)
            
            # Stop event
            stop_event = {
                'attacks': attack['end_idx'], # Use index from labels as simulated attack ID
                'DATETIME': attack['end_time'].strftime('%m/%d/%Y %H:%M:%S'),
                'attack_number': f"{attack['attack_number']}-stop"
            }
            timeline.append(stop_event)
        
        return timeline
    
    def save_attack_timeline(self, timeline, output_file):
        """Save attack timeline to CSV file"""
        df = pd.DataFrame(timeline)
        df.to_csv(output_file, index=False)
        print(f"Attack timeline saved to: {output_file}")
    
    def group_consecutive_anomalies(self, anomaly_windows):
        """Group consecutive anomalous windows into attack periods"""
        if anomaly_windows.empty:
            return []
        
        print("Grouping consecutive anomalies into attack periods...")

        # Sort by time_window to ensure correct grouping
        anomaly_windows = anomaly_windows.sort_values(by='time_window').reset_index(drop=True)

        attack_periods = []
        current_attack = None
        attack_number = 0
        
        # Use tqdm for progress bar if anomaly_windows is large
        for idx, row in tqdm(anomaly_windows.iterrows(), total=len(anomaly_windows), desc="Grouping Anomalies"):
            if current_attack is None:
                # Start new attack period
                attack_number += 1
                current_attack = {
                    'attack_number': attack_number,
                    'start_time': row['time_window'],
                    'end_time': row['time_window']
                }
            else:
                # Check if this window is consecutive (within 2 minutes)
                time_diff = (row['time_window'] - current_attack['end_time']).total_seconds()
                
                # If the current window starts within 2 minutes (120 seconds) of the previous one ending,
                # it's considered part of the same attack.
                if time_diff <= 120:  
                    current_attack['end_time'] = row['time_window']
                else:
                    # Gap too large, end current attack and start new one
                    attack_periods.append(current_attack)
                    attack_number += 1
                    current_attack = {
                        'attack_number': attack_number,
                        'start_time': row['time_window'],
                        'end_time': row['time_window']
                    }
        
        # Add the last attack period
        if current_attack is not None:
            attack_periods.append(current_attack)
        
        return attack_periods
    
    def generate_attack_timeline_from_anomalies(self, attack_periods):
        """Generate attack timeline from anomaly-based attack periods"""
        timeline = []
        base_id = 1500 # A base ID for 'attacks' column when using statistical method
        
        for i, attack in tqdm(enumerate(attack_periods), total=len(attack_periods), desc="Generating Anomaly Timeline"):
            # Start event
            # Use a dummy packet number based on the attack period index for 'attacks' column
            start_event = {
                'attacks': base_id + i * 2, # Ensure unique 'attacks' IDs for start/stop
                'DATETIME': attack['start_time'].strftime('%m/%d/%Y %H:%M:%S'),
                'attack_number': f"{attack['attack_number']}-start"
            }
            timeline.append(start_event)
            
            # Stop event
            # For statistical anomalies, the 'end_time' is the last anomalous window.
            # We can define the stop event slightly after this for clarity, e.g., +30 seconds or +1 minute.
            stop_event = {
                'attacks': base_id + i * 2 + 1, # Ensure unique 'attacks' IDs for start/stop
                'DATETIME': (attack['end_time'] + timedelta(seconds=29)).strftime('%m/%d/%Y %H:%M:%S'), # End of the 30s window
                'attack_number': f"{attack['attack_number']}-stop"
            }
            timeline.append(stop_event)
        
        return timeline
    
    def analyze(self, output_file=None, method='statistical', use_multiprocessing=True, 
                packet_limit=None, maxAE=10, FMgrace=5000, ADgrace=10000, kitsune_chunks=None):
        """Main analysis function with different detection methods."""
        
        if method == 'kitsune':
            if process_in_chunks is None:
                print("Kitsune method selected but chunked_kitsune.py is not available. Exiting.")
                return False
            
            print(f"\n--- Running Kitsune-based Anomaly Detection ---")
            print(f"Kitsune Parameters: maxAE={maxAE}, FMgrace={FMgrace}, ADgrace={ADgrace}")

            # Define Kitsune-specific output folders
            kitsune_results_base = "Kitsune_Results"
            kitsune_anomalies_folder = os.path.join(kitsune_results_base, "anomalies")
            kitsune_confidence_folder = os.path.join(kitsune_results_base, "confidence")
            kitsune_rmse_folder = os.path.join(kitsune_results_base, "rmse")
            kitsune_thresholds_folder = os.path.join(kitsune_results_base, "thresholds")
            kitsune_logs_folder = os.path.join(kitsune_results_base, "logs")
            kitsune_chunks_folder = os.path.join(kitsune_results_base, "chunks") # Temporary chunks

            for folder in [kitsune_anomalies_folder, kitsune_confidence_folder, kitsune_rmse_folder,
                           kitsune_thresholds_folder, kitsune_logs_folder, kitsune_chunks_folder]:
                os.makedirs(folder, exist_ok=True)

            # Perform resource checks
            if not check_system_resources(self.pcap_file, packet_limit):
                print("System resource check failed. Aborting Kitsune analysis.")
                return False

            # Determine number of chunks for Kitsune
            kitsune_num_chunks = kitsune_chunks if kitsune_chunks is not None else max(6, min(mp.cpu_count() * 2, 12))
            print(f"Using {kitsune_num_chunks} chunks for Kitsune processing.")

            # Run Kitsune processing in chunks
            start_kitsune_processing_time = time.time()
            print("Starting Kitsune chunk processing...")
            try:
                combined_rmse_file, combined_ms_conf_file, combined_sec_conf_file = process_in_chunks(
                    self.pcap_file,
                    kitsune_num_chunks,
                    maxAE,
                    FMgrace,
                    ADgrace,
                    output_dir=kitsune_results_base, # This will be the base for combined outputs
                    chunks_dir=kitsune_chunks_folder,
                    packet_limit=packet_limit
                )
            except Exception as e:
                print(f"Error during Kitsune chunk processing: {e}")
                return False
            
            end_kitsune_processing_time = time.time()
            print(f"Kitsune chunk processing completed in {end_kitsune_processing_time - start_kitsune_processing_time:.2f} seconds.")

            # Load combined RMSEs and calculate thresholds
            print("\nCalculating Kitsune thresholds and detecting anomalies...")
            RMSEs = []
            timestamps = []
            
            batch_size = 100000 # For processing RMSE results in batches
            with open(combined_rmse_file, 'r') as f:
                reader = csv.reader(f)
                next(reader) # Skip header
                
                # Count total lines for progress bar
                total_rmse_lines = sum(1 for _ in f)
                f.seek(0)
                next(reader) # Skip header again
                
                with tqdm(total=total_rmse_lines, desc="Reading Kitsune RMSEs") as pbar_rmse:
                    for row in reader:
                        timestamps.append(float(row[1]))
                        RMSEs.append(float(row[2]))
                        pbar_rmse.update(1)

            # Calculate benign samples for thresholding
            benign_start_idx = FMgrace + ADgrace # This is based on packets, not 0-indexed RMSE list
            benign_rmses = []
            # Collect up to 500K benign samples if available
            for i in range(benign_start_idx, min(len(RMSEs), benign_start_idx + 500000)):
                benign_rmses.append(RMSEs[i])
            
            if not benign_rmses:
                print("Warning: Not enough benign samples to calculate Kitsune threshold effectively. Using default.")
                max_rmse_threshold = 0.1 # Default small threshold
                statistical_threshold = 0.1 # Default small threshold
            else:
                max_rmse_threshold = max(benign_rmses)
                
                benign_log_rmses = []
                for rmse_val in benign_rmses:
                    if rmse_val > 0:
                        benign_log_rmses.append(np.log(rmse_val))
                
                if benign_log_rmses:
                    statistical_threshold = np.exp(np.mean(benign_log_rmses) + 2 * np.std(benign_log_rmses))
                else:
                    statistical_threshold = max_rmse_threshold # Fallback if log transform fails

            print(f"Calculated Max RMSE threshold: {max_rmse_threshold:.4f}")
            print(f"Calculated Statistical (Log-Normal) threshold: {statistical_threshold:.4f}")

            # Choose which threshold to use (e.g., max_rmse_threshold is often stricter for Kitsune)
            anomaly_detection_threshold = max_rmse_threshold # Or statistical_threshold, depending on desired sensitivity

            # Save thresholds to file
            current_kitsune_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            threshold_filename = os.path.join(kitsune_thresholds_folder, f"kitsune_thresholds_{current_kitsune_timestamp}.csv")
            with open(threshold_filename, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["ThresholdType", "Value", "Description"])
                writer.writerow(["MaxRMSE", max_rmse_threshold, "Maximum RMSE from benign samples"])
                writer.writerow(["Statistical", statistical_threshold, "Log-normal distribution based threshold (mean + 2*std)"])
                writer.writerow(["BenignSamplesUsed", len(benign_rmses), "Number of benign samples for threshold calculation"])
                writer.writerow(["FMgrace", FMgrace, "Feature Mapping grace period (Kitsune)"])
                writer.writerow(["ADgrace", ADgrace, "Anomaly Detection grace period (Kitsune)"])
            print(f"Kitsune thresholds saved to: {threshold_filename}")

            # Detect anomalies based on chosen threshold
            kitsune_anomalies = []
            print(f"Detecting anomalies using threshold: {anomaly_detection_threshold:.4f}")
            for i in tqdm(range(len(RMSEs)), desc="Detecting Kitsune Anomalies"):
                # Skip grace period
                if i < benign_start_idx:
                    continue
                if RMSEs[i] > anomaly_detection_threshold:
                    # Convert timestamp back from float (Unix) to datetime object for consistency
                    anomaly_dt = datetime.fromtimestamp(timestamps[i])
                    kitsune_anomalies.append({
                        'packet_num': i + 1, # Packet numbers are 1-indexed
                        'timestamp': anomaly_dt,
                        'rmse': RMSEs[i]
                    })
            
            print(f"Found {len(kitsune_anomalies)} Kitsune anomalies.")

            # Convert Kitsune anomalies to attack timeline format
            if kitsune_anomalies:
                timeline = []
                current_attack = None
                attack_number = 0

                # Group consecutive anomalies into attack periods
                # A simple grouping logic: if two anomalies are within a certain time window, group them.
                # Here, we'll group if consecutive packets are anomalous, or within a small time window (e.g., 5 seconds)
                kitsune_anomalies.sort(key=lambda x: x['timestamp']) # Ensure sorted by time

                for anomaly_data in kitsune_anomalies:
                    if current_attack is None:
                        attack_number += 1
                        current_attack = {
                            'attack_number': attack_number,
                            'start_time': anomaly_data['timestamp'],
                            'end_time': anomaly_data['timestamp']
                        }
                    else:
                        time_diff = (anomaly_data['timestamp'] - current_attack['end_time']).total_seconds()
                        if time_diff <= 10: # If within 10 seconds, consider same attack burst
                            current_attack['end_time'] = anomaly_data['timestamp']
                        else:
                            # End previous attack, start new one
                            timeline.extend([
                                {
                                    'attacks': current_attack['start_time'].timestamp(), # Use timestamp as pseudo ID
                                    'DATETIME': current_attack['start_time'].strftime('%m/%d/%Y %H:%M:%S'),
                                    'attack_number': f"{current_attack['attack_number']}-start"
                                },
                                {
                                    'attacks': current_attack['end_time'].timestamp(),
                                    'DATETIME': current_attack['end_time'].strftime('%m/%d/%Y %H:%M:%S'),
                                    'attack_number': f"{current_attack['attack_number']}-stop"
                                }
                            ])
                            attack_number += 1
                            current_attack = {
                                'attack_number': attack_number,
                                'start_time': anomaly_data['timestamp'],
                                'end_time': anomaly_data['timestamp']
                            }
                # Add the last attack
                if current_attack is not None:
                    timeline.extend([
                        {
                            'attacks': current_attack['start_time'].timestamp(),
                            'DATETIME': current_attack['start_time'].strftime('%m/%d/%Y %H:%M:%S'),
                            'attack_number': f"{current_attack['attack_number']}-start"
                        },
                        {
                            'attacks': current_attack['end_time'].timestamp(),
                            'DATETIME': current_attack['end_time'].strftime('%m/%d/%Y %H:%M:%S'),
                            'attack_number': f"{current_attack['attack_number']}-stop"
                        }
                    ])
            else:
                print("No Kitsune anomalies to form a timeline.")

            # Cleanup temporary Kitsune chunks (handled by chunked_kitsune.py, but good to ensure)
            if os.path.exists(kitsune_chunks_folder):
                try:
                    shutil.rmtree(kitsune_chunks_folder)
                    print("Cleaned up temporary Kitsune chunk files.")
                except Exception as e:
                    print(f"Warning: Could not clean up Kitsune chunks folder: {e}")

        elif method == 'statistical':
            print(f"\n--- Running Statistical Anomaly Detection ---")
            # Load pcap file for statistical method
            if not self.load_pcap(use_multiprocessing=use_multiprocessing, packet_limit=packet_limit):
                print("Failed to load PCAP file. Exiting analysis.")
                return False
            
            traffic_df = self.packets_df 
            
            if traffic_df.empty:
                print("No traffic data available for statistical analysis.")
                return False

            anomalies = self.detect_anomalies_statistical(traffic_df)
            anomaly_windows = anomalies[anomalies['is_anomaly']].copy()
            
            if not anomaly_windows.empty:
                print(f"Found {len(anomaly_windows)} anomalous time windows.")
                attack_periods = self.group_consecutive_anomalies(anomaly_windows)
                timeline = self.generate_attack_timeline_from_anomalies(attack_periods)
            else:
                print("No significant anomalies detected via statistical method.")

        elif method == 'labels': # Renamed from 'manual' for clarity with 'swat_dec2019'
            print(f"\n--- Running Label-based Attack Detection ---")
            # Load pcap file for labels method (needed to get packet numbers/timestamps)
            if not self.load_pcap(use_multiprocessing=use_multiprocessing, packet_limit=packet_limit):
                print("Failed to load PCAP file. Exiting analysis.")
                return False

            # Load labels file
            labels_loaded = self.load_labels()
            
            if labels_loaded:
                attack_periods = self.detect_attacks_from_labels()
                if attack_periods:
                    print(f"Found {len(attack_periods)} attack periods from labels:")
                    for attack in attack_periods:
                        print(f"  Attack {attack['attack_number']}: {attack['start_time']} to {attack['end_time']}")
                    timeline = self.generate_attack_timeline(attack_periods)
                else:
                    print("No attacks found in labels file.")
            else:
                print("Labels file not available or invalid for label-based detection.")
                timeline = [] # Ensure timeline is empty if labels not loaded

        elif method == 'swat_dec2019':
            print(f"\n--- Running SWaT Dec 2019 Schedule-based Attack Detection ---")
            # Load pcap file (needed to map times to packet numbers)
            if not self.load_pcap(use_multiprocessing=use_multiprocessing, packet_limit=packet_limit):
                print("Failed to load PCAP file. Exiting analysis.")
                return False
            
            swat_attacks = self.get_swat_dec2019_attacks()
            
            # This part needs the actual packets to map timestamps to packet_num
            if not self.packets_df.empty:
                attack_periods = self.detect_attack_periods_manual_by_time(swat_attacks, self.packets_df)
                if attack_periods:
                    timeline = []
                    for attack in attack_periods:
                        timeline.extend([
                            {
                                'attacks': attack['start_packet'],
                                'DATETIME': attack['start_time'].strftime('%m/%d/%Y %H:%M:%S'),
                                'attack_number': f"{attack['attack_number']}-start"
                            },
                            {
                                'attacks': attack['end_packet'],
                                'DATETIME': attack['end_time'].strftime('%m/%d/%Y %H:%M:%S'),
                                'attack_number': f"{attack['attack_number']}-stop"
                            }
                        ])
                    print(f"Generated timeline for {len(attack_periods)} SWaT attacks.")
                else:
                    print("No SWaT Dec 2019 attacks found within the PCAP time range.")
            else:
                print("PCAP data not loaded for SWaT Dec 2019 analysis. Cannot map times to packets.")
                timeline = []

        else:
            print(f"Unknown detection method: {method}. Please choose 'statistical', 'labels', 'kitsune', or 'swat_dec2019'.")
            return False

        if timeline and output_file:
            self.save_attack_timeline(timeline, output_file)
            return timeline
        elif not timeline:
            print("No attack timeline generated for output.")
            return False
        return timeline # Return timeline even if not saved to file

    def detect_attack_periods_manual_by_time(self, known_attacks, packets_df):
        """Detect attack periods based on known attack times by mapping to packet numbers."""
        print("Mapping known attack times to packet numbers...")
        attack_periods = []
        if packets_df.empty:
            print("No packets loaded to map attack times.")
            return []

        # Ensure packets_df is sorted by timestamp for efficient searching
        packets_df = packets_df.sort_values(by='timestamp').reset_index(drop=True)
        
        for attack in tqdm(known_attacks, desc="Mapping SWaT Attacks"):
            start_time_dt = datetime.strptime(attack['start_time'], '%m/%d/%Y %H:%M:%S')
            end_time_dt = datetime.strptime(attack['end_time'], '%m/%d/%Y %H:%M:%S')

            # Find the first packet that is >= start_time
            start_packet_row = packets_df[packets_df['timestamp'] >= start_time_dt].head(1)
            # Find the last packet that is <= end_time
            end_packet_row = packets_df[packets_df['timestamp'] <= end_time_dt].tail(1)

            start_packet_num = None
            end_packet_num = None

            if not start_packet_row.empty:
                start_packet_num = start_packet_row['packet_num'].iloc[0]
            if not end_packet_row.empty:
                end_packet_num = end_packet_row['packet_num'].iloc[0]

            if start_packet_num is not None and end_packet_num is not None:
                attack_periods.append({
                    'attack_number': attack['attack_number'],
                    'start_packet': start_packet_num,
                    'end_packet': end_packet_num,
                    'start_time': start_time_dt,
                    'end_time': end_time_dt
                })
            # else:
            #     print(f"Warning: Could not find packet range for Attack {attack['attack_number']} ({start_time_dt} to {end_time_dt})")
        return attack_periods

    def get_swat_dec2019_attacks(self):
        """Get SWaT Dec 2019 specific attack timing"""
        # Based on the provided schedule
        base_date = "12/6/2019"
        
        attacks = [
            # Historian Data Exfiltration attacks (4 cycles)
            {"attack_number": 1, "start_time": f"{base_date} 10:30:00", "end_time": f"{base_date} 10:35:00"},
            {"attack_number": 2, "start_time": f"{base_date} 10:45:00", "end_time": f"{base_date} 10:50:00"},
            {"attack_number": 3, "start_time": f"{base_date} 11:00:00", "end_time": f"{base_date} 11:05:00"},
            {"attack_number": 4, "start_time": f"{base_date} 11:15:00", "end_time": f"{base_date} 11:20:00"},
            
            # Process Disruption attacks (5 cycles)
            {"attack_number": 5, "start_time": f"{base_date} 12:30:00", "end_time": f"{base_date} 12:33:00"},
            {"attack_number": 6, "start_time": f"{base_date} 12:43:00", "end_time": f"{base_date} 12:46:00"},
            {"attack_number": 7, "start_time": f"{base_date} 12:56:00", "end_time": f"{base_date} 12:59:00"},
            {"attack_number": 8, "start_time": f"{base_date} 13:09:00", "end_time": f"{base_date} 13:12:00"},
            {"attack_number": 9, "start_time": f"{base_date} 13:22:00", "end_time": f"{base_date} 13:25:00"},
        ]
        
        return attacks


def main():
    parser = argparse.ArgumentParser(description='Analyze PCAP file for attack timeline with multiprocessing.')
    parser.add_argument('pcap_file', nargs='?', help='Path to the PCAP file')
    parser.add_argument('--labels', help='Path to the labels CSV file (optional)')
    parser.add_argument('-o', '--output', help='Base name for the output CSV file (default: attack_timeline.csv). A timestamp will be appended.',
                       default='attack_timeline.csv')
    parser.add_argument('-m', '--method', choices=['statistical', 'labels', 'kitsune', 'swat_dec2019'],
                       default='statistical',
                       help='Detection method to use (default: statistical). '
                            'Choose "kitsune" for advanced ML-based anomaly detection.')
    parser.add_argument('-p', '--processes', type=int,
                       help=f'Number of processes to use for PCAP loading (default: {min(mp.cpu_count(), 8)}) for statistical/label methods.')
    parser.add_argument('--single-process', action='store_true',
                       help='Force single-process mode for PCAP loading (statistical/label methods).')
    parser.add_argument('-l', '--limit', type=int,
                       help='Limit the number of packets to process from the PCAP file.')
    
    # Kitsune specific arguments
    parser.add_argument('--maxae', type=int, default=10,
                       help='Kitsune: Maximum size for any autoencoder (default: 10).')
    parser.add_argument('--fmgrace', type=int, default=5000,
                       help='Kitsune: Feature Mapping grace period - number of packets to learn normal FM (default: 5000).')
    parser.add_argument('--adgrace', type=int, default=10000,
                       help='Kitsune: Anomaly Detection grace period - number of packets to learn normal AD (default: 10000).')
    parser.add_argument('--kitsune-chunks', type=int,
                       help=f'Kitsune: Number of chunks for Kitsune processing (default: {max(6, min(mp.cpu_count() * 2, 12))}).')

    args = parser.parse_args()
    
    pcap_file = args.pcap_file
    labels_file = args.labels
    base_output_filename = args.output
    
    # Handle hardcoded paths if no command-line argument is provided for pcap_file
    if pcap_file is None:
        print("No pcap file provided via command line. Using hardcoded paths...")
        pcap_file = "Dec2019_00000_20191206100500.pcap"  # Default PCAP file
        labels_file = labels_file if labels_file is not None else "Dec2019_with_labels.csv" # Default labels file
        base_output_filename = base_output_filename if base_output_filename != 'attack_timeline.csv' else "attack_timeline_from_pcap.csv" # Default output file

        print(f"  PCAP file: {pcap_file}")
        print(f"  Labels file: {labels_file}")
        print(f"  Base output filename: {base_output_filename}")
    
    # Check if input file exists
    if not os.path.exists(pcap_file):
        print(f"Error: PCAP file '{pcap_file}' not found!")
        sys.exit(1)

    # Generate timestamp for the output filename
    timestamp_str = datetime.now().strftime("_%Y%m%d_%H%M%S")
    output_dir, output_name = os.path.split(base_output_filename)
    name_without_ext, ext = os.path.splitext(output_name)
    final_output_file = os.path.join(output_dir, f"{name_without_ext}{timestamp_str}{ext}")

    # Determine whether to use multiprocessing for non-kitsune methods
    use_mp = not args.single_process
    
    # Create analyzer instance
    if args.method == 'kitsune':
        # Kitsune's chunking is managed internally by chunked_kitsune.py
        # and doesn't directly use PcapAttackAnalyzer's num_processes.
        # However, we still pass a default if needed by other parts later.
        analyzer = PcapAttackAnalyzer(pcap_file, labels_file, num_processes=args.processes)
    else:
        analyzer = PcapAttackAnalyzer(pcap_file, labels_file, num_processes=args.processes)

    # Record start time of analysis
    analysis_start_time = datetime.now()
    print(f"\nAnalysis started at: {analysis_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run analysis with specified method, multiprocessing, and packet limit
    timeline = analyzer.analyze(
        output_file=final_output_file,
        method=args.method,
        use_multiprocessing=use_mp,
        packet_limit=args.limit,
        maxAE=args.maxae,
        FMgrace=args.fmgrace,
        ADgrace=args.adgrace,
        kitsune_chunks=args.kitsune_chunks
    )
    
    # Record end time of analysis
    analysis_end_time = datetime.now()
    print(f"\nAnalysis completed at: {analysis_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total analysis duration: {analysis_end_time - analysis_start_time}")

    if timeline:
        print(f"\nGenerated attack timeline with {len(timeline)} events.")
        print(f"Results saved to: {final_output_file}")
        print("\nTimeline preview (first 10 events):")
        for event in timeline[:10]:
            print(f"  {event}")
    else:
        print("No attack timeline generated or analysis failed.")

if __name__ == "__main__":
    # Ensure multiprocessing starts correctly on Windows
    mp.freeze_support() 
    main()
