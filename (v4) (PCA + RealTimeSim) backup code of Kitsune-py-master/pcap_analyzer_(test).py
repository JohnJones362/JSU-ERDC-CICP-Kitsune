#!/usr/bin/env python3
"""
PCAP Attack Timeline Generator with Multiprocessing
Analyzes PCAP files and generates attack timing information in CSV format
Based on SWaT dataset attack patterns from Dec 2019
"""

import pandas as pd
import numpy as np
from scapy.all import rdpcap, PcapReader, wrpcap
import csv
import os
import argparse
from datetime import datetime, timedelta
import time
from collections import defaultdict
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile
import shutil
from functools import partial
import gc

def process_pcap_chunk(chunk_file, chunk_id, start_packet_num):
    """Process a single PCAP chunk - designed to run in separate process"""
    packets = []
    
    try:
        with PcapReader(chunk_file) as pcap_reader:
            packet_count = 0
            for packet in pcap_reader:
                packet_count += 1
                current_packet_num = start_packet_num + packet_count - 1
                
                # Extract timestamp
                timestamp = float(packet.time)
                
                # Store packet info
                packet_info = {
                    'packet_num': current_packet_num,
                    'timestamp': timestamp,
                    'datetime': datetime.fromtimestamp(timestamp),
                    'size': len(packet)
                }
                
                # Add protocol information if available
                if packet.haslayer('IP'):
                    packet_info['src_ip'] = packet['IP'].src
                    packet_info['dst_ip'] = packet['IP'].dst
                    packet_info['protocol'] = packet['IP'].proto
                else:
                    packet_info['src_ip'] = None
                    packet_info['dst_ip'] = None
                    packet_info['protocol'] = None
                
                if packet.haslayer('TCP'):
                    packet_info['src_port'] = packet['TCP'].sport
                    packet_info['dst_port'] = packet['TCP'].dport
                    packet_info['tcp_flags'] = packet['TCP'].flags
                elif packet.haslayer('UDP'):
                    packet_info['src_port'] = packet['UDP'].sport
                    packet_info['dst_port'] = packet['UDP'].dport
                    packet_info['tcp_flags'] = None
                else:
                    packet_info['src_port'] = None
                    packet_info['dst_port'] = None
                    packet_info['tcp_flags'] = None
                
                packets.append(packet_info)
        
        return chunk_id, packets, len(packets)
    
    except Exception as e:
        print(f"Error processing chunk {chunk_id}: {str(e)}")
        return chunk_id, [], 0

def split_pcap_file(pcap_file, num_chunks, temp_dir, packet_limit=None):
    """Split PCAP file into chunks for parallel processing"""
    print(f"Splitting PCAP file into {num_chunks} chunks...")
    
    # First pass: count total packets
    print("Counting total packets...")
    total_packets = 0
    with PcapReader(pcap_file) as pcap_reader:
        for packet in pcap_reader:
            total_packets += 1
            if packet_limit and total_packets >= packet_limit:
                break
    
    print(f"Total packets to process: {total_packets}")
    
    if total_packets == 0:
        return []
    
    # Calculate packets per chunk
    packets_per_chunk = max(1, total_packets // num_chunks)
    chunk_files = []
    
    # Second pass: split into chunks
    print(f"Creating chunks ({packets_per_chunk} packets per chunk)...")
    
    current_chunk = 0
    current_packet_count = 0
    current_chunk_packets = []
    
    with PcapReader(pcap_file) as pcap_reader:
        processed_packets = 0
        for packet in pcap_reader:
            current_chunk_packets.append(packet)
            current_packet_count += 1
            processed_packets += 1
            
            # Check if we should create a new chunk
            if (current_packet_count >= packets_per_chunk and current_chunk < num_chunks - 1) or \
               (packet_limit and processed_packets >= packet_limit):
                
                # Write current chunk
                chunk_file = os.path.join(temp_dir, f"chunk_{current_chunk:03d}.pcap")
                wrpcap(chunk_file, current_chunk_packets)
                
                chunk_info = {
                    'file': chunk_file,
                    'id': current_chunk,
                    'start_packet': sum(len(info['packets']) for info in chunk_files) + 1,
                    'packets': current_packet_count
                }
                chunk_files.append(chunk_info)
                
                print(f"Created chunk {current_chunk}: {current_packet_count} packets")
                
                # Reset for next chunk
                current_chunk += 1
                current_packet_count = 0
                current_chunk_packets = []
                
                if packet_limit and processed_packets >= packet_limit:
                    break
    
    # Handle remaining packets in last chunk
    if current_chunk_packets and current_chunk < num_chunks:
        chunk_file = os.path.join(temp_dir, f"chunk_{current_chunk:03d}.pcap")
        wrpcap(chunk_file, current_chunk_packets)
        
        chunk_info = {
            'file': chunk_file,
            'id': current_chunk,
            'start_packet': sum(info['packets'] for info in chunk_files) + 1,
            'packets': current_packet_count
        }
        chunk_files.append(chunk_info)
        print(f"Created chunk {current_chunk}: {current_packet_count} packets")
    
    print(f"Created {len(chunk_files)} chunks")
    return chunk_files

class PCAPAttackAnalyzer:
    def __init__(self, pcap_file, num_processes=None):
        self.pcap_file = pcap_file
        self.packets = []
        self.attack_periods = []
        self.num_processes = num_processes or min(mp.cpu_count(), 8)  # Cap at 8 processes
        self.temp_dir = None
        
    def load_pcap_multiprocess(self, packet_limit=None):
        """Load PCAP file using multiprocessing for speed"""
        print(f"Loading PCAP file with {self.num_processes} processes: {self.pcap_file}")
        
        try:
            # Create temporary directory for chunks
            self.temp_dir = tempfile.mkdtemp(prefix="pcap_chunks_")
            print(f"Temporary directory: {self.temp_dir}")
            
            # Split PCAP into chunks
            chunk_files = split_pcap_file(self.pcap_file, self.num_processes, self.temp_dir, packet_limit)
            
            if not chunk_files:
                print("No chunks created - file might be empty")
                return False
            
            # Process chunks in parallel
            print("Processing chunks in parallel...")
            start_time = time.time()
            
            all_results = []
            with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                # Submit all chunks for processing
                future_to_chunk = {}
                for chunk_info in chunk_files:
                    future = executor.submit(
                        process_pcap_chunk, 
                        chunk_info['file'], 
                        chunk_info['id'], 
                        chunk_info['start_packet']
                    )
                    future_to_chunk[future] = chunk_info
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_info = future_to_chunk[future]
                    try:
                        chunk_id, packets, packet_count = future.result()
                        all_results.append((chunk_id, packets))
                        print(f"Completed chunk {chunk_id}: {packet_count} packets")
                    except Exception as e:
                        print(f"Chunk {chunk_info['id']} generated an exception: {e}")
            
            # Sort results by chunk ID and combine
            all_results.sort(key=lambda x: x[0])
            total_packets = 0
            for chunk_id, packets in all_results:
                self.packets.extend(packets)
                total_packets += len(packets)
            
            elapsed = time.time() - start_time
            print(f"Parallel processing complete: {total_packets} packets in {elapsed:.2f}s")
            print(f"Processing rate: {total_packets/elapsed:.0f} packets/second")
            
            return True
            
        except Exception as e:
            print(f"Error in multiprocess loading: {str(e)}")
            return False
        finally:
            # Cleanup temporary files
            self.cleanup_temp_files()
    
    def load_pcap_single_process(self, packet_limit=None):
        """Fallback single-process loading method"""
        print(f"Loading PCAP file (single process): {self.pcap_file}")
        
        try:
            packet_count = 0
            start_time = time.time()
            
            with PcapReader(self.pcap_file) as pcap_reader:
                for packet in pcap_reader:
                    packet_count += 1
                    
                    # Extract timestamp
                    timestamp = float(packet.time)
                    
                    # Store packet info
                    packet_info = {
                        'packet_num': packet_count,
                        'timestamp': timestamp,
                        'datetime': datetime.fromtimestamp(timestamp),
                        'size': len(packet)
                    }
                    
                    # Add protocol information if available
                    if packet.haslayer('IP'):
                        packet_info['src_ip'] = packet['IP'].src
                        packet_info['dst_ip'] = packet['IP'].dst
                        packet_info['protocol'] = packet['IP'].proto
                    
                    if packet.haslayer('TCP'):
                        packet_info['src_port'] = packet['TCP'].sport
                        packet_info['dst_port'] = packet['TCP'].dport
                        packet_info['tcp_flags'] = packet['TCP'].flags
                    elif packet.haslayer('UDP'):
                        packet_info['src_port'] = packet['UDP'].sport
                        packet_info['dst_port'] = packet['UDP'].dport
                    
                    self.packets.append(packet_info)
                    
                    # Progress indicator
                    if packet_count % 50000 == 0:
                        elapsed = time.time() - start_time
                        rate = packet_count / elapsed if elapsed > 0 else 0
                        print(f"Processed {packet_count} packets in {elapsed:.2f}s ({rate:.0f} pkt/s)")
                    
                    # Check packet limit
                    if packet_limit and packet_count >= packet_limit:
                        print(f"Reached packet limit of {packet_limit}")
                        break
            
            elapsed = time.time() - start_time
            print(f"Single-process loading complete: {packet_count} packets in {elapsed:.2f}s")
            return True
            
        except Exception as e:
            print(f"Error loading PCAP file: {str(e)}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary chunk files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print("Temporary files cleaned up")
            except Exception as e:
                print(f"Warning: Could not clean up temporary files: {e}")
    
    def analyze_traffic_patterns(self):
        """Analyze traffic patterns to identify potential attack periods"""
        if not self.packets:
            print("No packets loaded. Please load PCAP file first.")
            return
        
        print("Analyzing traffic patterns...")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.packets)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Group by time windows (e.g., 1-minute intervals)
        df['time_window'] = df['datetime'].dt.floor('1min')
        
        # Calculate traffic metrics per time window
        traffic_stats = df.groupby('time_window').agg({
            'packet_num': ['count', 'min', 'max'],
            'size': ['sum', 'mean', 'std'],
            'src_ip': 'nunique',
            'dst_ip': 'nunique'
        }).reset_index()
        
        # Flatten column names
        traffic_stats.columns = ['time_window', 'packet_count', 'first_packet', 'last_packet',
                               'total_bytes', 'avg_size', 'size_std', 'unique_src_ips', 'unique_dst_ips']
        
        # Fill NaN values
        traffic_stats['size_std'] = traffic_stats['size_std'].fillna(0)
        
        return traffic_stats
    
    def detect_attack_periods_statistical(self, traffic_stats):
        """Detect attack periods using statistical analysis"""
        print("Detecting attack periods using statistical analysis...")
        
        # Calculate baseline statistics (assuming first portion is normal)
        baseline_size = min(len(traffic_stats) // 4, 60)  # Use first 25% or 60 minutes, whichever is smaller
        baseline = traffic_stats.head(baseline_size)
        
        # Calculate thresholds based on baseline
        packet_threshold = baseline['packet_count'].mean() + 2 * baseline['packet_count'].std()
        bytes_threshold = baseline['total_bytes'].mean() + 2 * baseline['total_bytes'].std()
        
        print(f"Baseline period: {baseline_size} minutes")
        print(f"Packet count threshold: {packet_threshold:.0f}")
        print(f"Bytes threshold: {bytes_threshold:.0f}")
        
        # Identify anomalous periods
        anomalous_periods = traffic_stats[
            (traffic_stats['packet_count'] > packet_threshold) |
            (traffic_stats['total_bytes'] > bytes_threshold)
        ].copy()
        
        return anomalous_periods
    
    def detect_attack_periods_manual(self, known_attacks):
        """Detect attack periods based on known attack times"""
        print("Using manual attack period detection...")
        
        # Parse known attack times
        attack_periods = []
        for attack in known_attacks:
            start_time = datetime.strptime(attack['start_time'], '%m/%d/%Y %H:%M:%S')
            end_time = datetime.strptime(attack['end_time'], '%m/%d/%Y %H:%M:%S')
            
            # Find packet numbers corresponding to these times
            start_packet = None
            end_packet = None
            
            for packet in self.packets:
                if start_packet is None and packet['datetime'] >= start_time:
                    start_packet = packet['packet_num']
                if packet['datetime'] <= end_time:
                    end_packet = packet['packet_num']
            
            if start_packet and end_packet:
                attack_periods.append({
                    'attack_number': attack['attack_number'],
                    'start_packet': start_packet,
                    'end_packet': end_packet,
                    'start_time': start_time,
                    'end_time': end_time
                })
        
        return attack_periods
    
    def generate_attack_csv(self, output_file, method='statistical', known_attacks=None):
        """Generate CSV file with attack timing information"""
        print(f"Generating attack CSV using {method} method...")
        
        attack_data = []
        
        if method == 'statistical':
            # Analyze traffic patterns
            traffic_stats = self.analyze_traffic_patterns()
            anomalous_periods = self.detect_attack_periods_statistical(traffic_stats)
            
            # Group consecutive anomalous periods into attacks
            attack_num = 1
            current_attack_start = None
            current_attack_end = None
            
            for idx, period in anomalous_periods.iterrows():
                if current_attack_start is None:
                    current_attack_start = period['first_packet']
                    current_attack_end = period['last_packet']
                else:
                    # Check if this period is continuous with the previous one
                    time_gap = (period['time_window'] - 
                              anomalous_periods.iloc[idx-1]['time_window']).total_seconds() / 60
                    
                    if time_gap <= 2:  # Within 2 minutes, consider it same attack
                        current_attack_end = period['last_packet']
                    else:
                        # End current attack and start new one
                        attack_data.extend([
                            [current_attack_start, 
                             traffic_stats[traffic_stats['first_packet'] <= current_attack_start]['time_window'].iloc[-1].strftime('%m/%d/%Y %H:%M:%S'),
                             f"{attack_num}-start"],
                            [current_attack_end,
                             traffic_stats[traffic_stats['last_packet'] >= current_attack_end]['time_window'].iloc[0].strftime('%m/%d/%Y %H:%M:%S'),
                             f"{attack_num}-stop"]
                        ])
                        attack_num += 1
                        current_attack_start = period['first_packet']
                        current_attack_end = period['last_packet']
            
            # Add the last attack
            if current_attack_start is not None:
                attack_data.extend([
                    [current_attack_start,
                     traffic_stats[traffic_stats['first_packet'] <= current_attack_start]['time_window'].iloc[-1].strftime('%m/%d/%Y %H:%M:%S'),
                     f"{attack_num}-start"],
                    [current_attack_end,
                     traffic_stats[traffic_stats['last_packet'] >= current_attack_end]['time_window'].iloc[0].strftime('%m/%d/%Y %H:%M:%S'),
                     f"{attack_num}-stop"]
                ])
        
        elif method == 'manual' and known_attacks:
            # Use provided attack times
            attack_periods = self.detect_attack_periods_manual(known_attacks)
            
            for attack in attack_periods:
                attack_data.extend([
                    [attack['start_packet'],
                     attack['start_time'].strftime('%m/%d/%Y %H:%M:%S'),
                     f"{attack['attack_number']}-start"],
                    [attack['end_packet'],
                     attack['end_time'].strftime('%m/%d/%Y %H:%M:%S'),
                     f"{attack['attack_number']}-stop"]
                ])
        
        elif method == 'swat_dec2019':
            # Use SWaT Dec 2019 specific attack times
            swat_attacks = self.get_swat_dec2019_attacks()
            attack_periods = self.detect_attack_periods_manual(swat_attacks)
            
            for attack in attack_periods:
                attack_data.extend([
                    [attack['start_packet'],
                     attack['start_time'].strftime('%m/%d/%Y %H:%M:%S'),
                     f"{attack['attack_number']}-start"],
                    [attack['end_packet'],
                     attack['end_time'].strftime('%m/%d/%Y %H:%M:%S'),
                     f"{attack['attack_number']}-stop"]
                ])
        
        # Write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['attacks', 'DATETIME', 'attack_number'])
            writer.writerows(attack_data)
        
        print(f"Attack CSV saved to: {output_file}")
        print(f"Total attack events: {len(attack_data)}")
        
        return output_file
    
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
    
    def print_summary(self):
        """Print analysis summary"""
        if not self.packets:
            print("No packets loaded.")
            return
        
        first_packet = self.packets[0]
        last_packet = self.packets[-1]
        
        print(f"\n=== PCAP Analysis Summary ===")
        print(f"File: {self.pcap_file}")
        print(f"Total packets: {len(self.packets)}")
        print(f"First packet: {first_packet['datetime']}")
        print(f"Last packet: {last_packet['datetime']}")
        print(f"Duration: {last_packet['datetime'] - first_packet['datetime']}")
        print(f"Total size: {sum(p['size'] for p in self.packets)} bytes")


def main():
    parser = argparse.ArgumentParser(description='Analyze PCAP files for attack periods with multiprocessing')
    parser.add_argument('pcap_file', help='Path to PCAP file')
    parser.add_argument('-o', '--output', default='attacks_detected.csv', 
                       help='Output CSV file (default: attacks_detected.csv)')
    parser.add_argument('-m', '--method', choices=['statistical', 'manual', 'swat_dec2019'], 
                       default='swat_dec2019',
                       help='Detection method (default: swat_dec2019)')
    parser.add_argument('-l', '--limit', type=int, 
                       help='Limit number of packets to process')
    parser.add_argument('-p', '--processes', type=int, 
                       help=f'Number of processes to use (default: {min(mp.cpu_count(), 8)})')
    parser.add_argument('--single-process', action='store_true',
                       help='Use single process instead of multiprocessing')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.pcap_file):
        print(f"Error: PCAP file '{args.pcap_file}' not found!")
        sys.exit(1)
    
    # Create analyzer
    analyzer = PCAPAttackAnalyzer(args.pcap_file, num_processes=args.processes)
    
    # Load PCAP (choose method based on user preference)
    if args.single_process:
        print("Using single-process mode...")
        success = analyzer.load_pcap_single_process(packet_limit=args.limit)
    else:
        print(f"Using multiprocessing mode with {analyzer.num_processes} processes...")
        success = analyzer.load_pcap_multiprocess(packet_limit=args.limit)
        
        # Fallback to single process if multiprocessing fails
        if not success:
            print("Multiprocessing failed, falling back to single process...")
            success = analyzer.load_pcap_single_process(packet_limit=args.limit)
    
    if not success:
        print("Failed to load PCAP file!")
        sys.exit(1)
    
    # Print summary
    analyzer.print_summary()
    
    # Generate attack CSV
    output_file = analyzer.generate_attack_csv(args.output, method=args.method)
    
    print(f"\nAnalysis complete! Results saved to: {output_file}")


if __name__ == "__main__":
    main()

# Use SWaT Dec 2019 attack schedule (recommended for your dataset)
#python pcap_analyzer.py your_file.pcap -m swat_dec2019 -o attacks.csv

# Use statistical detection
#python pcap_analyzer.py your_file.pcap -m statistical -o attacks.csv

# Limit processing to first 100,000 packets
#python pcap_analyzer.py your_file.pcap -l 100000

# Process entire PCAP file with SWaT Dec 2019 schedule
#python pcap_analyzer.py Dec2019_00000_20191206100500.pcap -m swat_dec2019 -o attacks.csv

# Process entire PCAP file with statistical detection
#python pcap_analyzer.py your_file.pcap -m statistical -o attacks.csv

# Process entire PCAP file with default settings
#python pcap_analyzer.py your_file.pcap

# Use multiprocessing (default - fastest)
#python pcap_analyzer.py your_file.pcap

# Specify number of processes
#python pcap_analyzer.py your_file.pcap -p 4

# Force single-process mode (if you have issues)
#python pcap_analyzer.py your_file.pcap --single-process

# Combine with other options
#python pcap_analyzer.py your_file.pcap -p 6 -m swat_dec2019 -o results.csv