# old code: 6/24/25 (9:19 AM)

from scapy.all import rdpcap, wrpcap, PcapReader
import os
import pandas as pd
import numpy as np
from typing import List, Tuple
import math
import csv
from datetime import datetime
import multiprocessing
from chunk_processor import process_chunk # Keep this import, as process_chunk is now the core logic
from Kitsune import Kitsune # Keep this import, though not directly used in execute_chunk anymore
from KitNET import KitNET # Keep this import, though not directly used in execute_chunk anymore
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import multiprocessing as mp
from multiprocessing import get_context
import pickle
from tqdm import tqdm
import psutil
import gc
import logging
import tempfile
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def split_pcap(input_file: str, num_chunks: int, output_dir: str = "chunks", packet_limit: int = None) -> List[str]:
    """Split a PCAP file into multiple chunks"""
    # Create output directory with timestamp
    temp_dir = os.path.join(output_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(temp_dir, exist_ok=True)

    # Count total packets
    total_packets = 0
    for _ in PcapReader(input_file):
        total_packets += 1
        if packet_limit and total_packets >= packet_limit:
            break

    packets_per_chunk = total_packets // num_chunks
    remaining_packets = total_packets % num_chunks

    logging.info(f"Splitting {os.path.basename(input_file)} into {num_chunks} chunks ({total_packets} total packets)")

    chunk_files = []
    current_chunk = 0
    packet_count = 0

    # Use tqdm for progress bar
    with tqdm(total=total_packets, desc="Splitting PCAP") as pbar:
        current_chunk_packets = []

        for packet in PcapReader(input_file):
            current_chunk_packets.append(packet)
            packet_count += 1
            pbar.update(1)

            # Write chunk when it reaches the target size
            if len(current_chunk_packets) >= packets_per_chunk + (1 if current_chunk < remaining_packets else 0):
                chunk_file = os.path.join(temp_dir, f"chunk_{current_chunk}.pcap")
                wrpcap(chunk_file, current_chunk_packets)
                chunk_files.append(chunk_file)

                # Clear memory
                current_chunk_packets = []
                gc.collect()

                current_chunk += 1

            if packet_limit and packet_count >= packet_limit:
                break

        # Write remaining packets if any
        if current_chunk_packets:
            chunk_file = os.path.join(temp_dir, f"chunk_{current_chunk}.pcap")
            wrpcap(chunk_file, current_chunk_packets)
            chunk_files.append(chunk_file)

    return chunk_files

def split_tsv(input_file: str, num_chunks: int, output_dir: str = "chunks", packet_limit: int = None) -> List[str]:
    """Split a TSV file into multiple smaller files"""
    chunk_files = []
    os.makedirs(output_dir, exist_ok=True)

    # Count total lines (up to limit if specified)
    total_lines = 0
    with open(input_file, 'r') as f:
        next(f)  # Skip header
        for _ in f:
            total_lines += 1
            if packet_limit and total_lines >= packet_limit:
                break

    # Adjust total_lines if limit is specified
    if packet_limit:
        total_lines = min(total_lines, packet_limit)

    lines_per_chunk = total_lines // num_chunks
    remaining_lines = total_lines % num_chunks

    logging.info(f"Splitting {input_file} into {num_chunks} chunks ({total_lines} total lines)")

    with open(input_file, 'r') as f:
        header = f.readline()  # Read header

        current_chunk = 0
        line_count = 0
        total_processed = 0
        current_lines = [header]  # Start with header

        with tqdm(total=total_lines, desc="Splitting TSV") as pbar:
            for line in f:
                if packet_limit and total_processed >= packet_limit:
                    break

                current_lines.append(line)
                line_count += 1
                total_processed += 1
                pbar.update(1)

                # Determine if we should write current chunk
                target_size = lines_per_chunk + (1 if current_chunk < remaining_lines else 0)
                if line_count >= target_size:
                    chunk_file = os.path.join(output_dir, f"chunk_{current_chunk}.tsv")
                    with open(chunk_file, 'w') as chunk_f:
                        chunk_f.writelines(current_lines)
                    chunk_files.append(chunk_file)

                    # Clear memory
                    current_lines = [header]
                    gc.collect()

                    current_chunk += 1
                    line_count = 0

            # Write any remaining lines
            if len(current_lines) > 1:  # If we have more than just the header
                chunk_file = os.path.join(output_dir, f"chunk_{current_chunk}.tsv")
                with open(chunk_file, 'w') as chunk_f:
                    chunk_f.writelines(current_lines)
                chunk_files.append(chunk_file)

    return chunk_files

def join_rmse_files(rmse_files: List[str], output_file: str):
    """Join multiple RMSE files into one, maintaining packet order"""
    # Rationale: This function combines RMSE outputs from parallel chunks.
    # It now correctly handles the 'packet_idx' column which is the first column,
    # ensuring global sequential packet indexing across all chunks.
    packet_idx_offset = 0

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['packet_idx', 'timestamp', 'rmse'])  # Write header

        for i, rmse_file in enumerate(rmse_files):
            with open(rmse_file, 'r') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header

                for row in reader:
                    # packet_idx is now the first column in row
                    current_packet_idx = int(row[0])
                    # Adjust packet_idx by the offset of previous chunks
                    writer.writerow([current_packet_idx + packet_idx_offset] + row[1:])

                # Update offset for next file based on the last packet_idx in the current file
                if i < len(rmse_files) - 1: # Only update offset if there are more files to process
                    # Get the last packet index from the current file
                    with open(rmse_file, 'r') as infile_last_line:
                        last_row = None
                        for line in infile_last_line:
                            last_row = line
                        if last_row:
                            last_packet_in_chunk = int(last_row.split(',')[0])
                            packet_idx_offset += last_packet_in_chunk

def join_confidence_files(confidence_files: List[str], output_file: str, is_millisecond: bool = True):
    """Join multiple confidence score files into one, maintaining packet order"""
    # Rationale: Similar to join_rmse_files, this function combines confidence
    # outputs from parallel chunks, ensuring global sequential packet indexing.
    packet_idx_offset = 0

    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        time_unit = "ms" if is_millisecond else "s"
        writer.writerow(['packet_idx', f'timestamp_{time_unit}', 'confidence_score'])

        for i, conf_file in enumerate(confidence_files):
            with open(conf_file, 'r') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header

                for row in reader:
                    # packet_idx is now the first column in row
                    current_packet_idx = int(row[0])
                    # Adjust packet_idx by the offset of previous chunks
                    writer.writerow([current_packet_idx + packet_idx_offset] + row[1:])

                # Update offset for next file based on the last packet_idx in the current file
                if i < len(confidence_files) - 1: # Only update offset if there are more files to process
                    with open(conf_file, 'r') as infile_last_line:
                        last_row = None
                        for line in infile_last_line:
                            last_row = line
                        if last_row:
                            last_packet_in_chunk = int(last_row.split(',')[0])
                            packet_idx_offset += last_packet_in_chunk


def execute_chunk(args):
    """
    Helper function to process a single chunk file with Kitsune.
    This function is designed to be run in a multiprocessing pool.
    """
    # Rationale: This function now directly calls process_chunk, which encapsulates
    # the Kitsune initialization and packet processing logic, including RMSE and
    # confidence score calculation. This simplifies execute_chunk and improves
    # modularity by delegating the core processing to process_chunk.
    chunk_file, maxAE, FMgrace, ADgrace, output_dir, pca_components, pca_grace_period = args
    return process_chunk(chunk_file, maxAE, FMgrace, ADgrace, output_dir, pca_components, pca_grace_period)


def process_in_chunks(input_file: str, num_chunks: int, maxAE: int, FMgrace: int, ADgrace: int,
                     output_dir: str = "results", chunks_dir: str = "chunks",
                     packet_limit: int = None,
                     pca_components: int = None, # Added parameter
                     pca_grace_period: int = None # Added parameter
                    ) -> Tuple[str, str, str]:
    """
    Process a large PCAP/TSV file in chunks using multiple processes.

    Args:
        input_file (str): Path to input PCAP/TSV file.
        num_chunks (int): Number of chunks to split into.
        maxAE (int): Maximum autoencoder size.
        FMgrace (int): Feature mapping grace period.
        ADgrace (int): Anomaly detection grace period.
        output_dir (str): Directory for final output files.
        chunks_dir (str): Directory for temporary chunk files.
        packet_limit (int, optional): Optional limit on number of packets to process.
        pca_components (int, optional): Number of components for PCA dimensionality reduction.
        pca_grace_period (int, optional): Number of packets to collect for PCA fitting.

    Returns:
        Tuple[str, str, str]: Tuple of (rmse_file, ms_confidence_file, sec_confidence_file) paths.
    """
    try:
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        temp_chunks_dir = os.path.join(chunks_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(temp_chunks_dir, exist_ok=True)

        # Split input file
        logging.info(f"Splitting input file into chunks{' (limited to ' + str(packet_limit) + ' packets)' if packet_limit else ''}...")
        if input_file.endswith('.pcap'):
            chunk_files = split_pcap(input_file, num_chunks, temp_chunks_dir, packet_limit)
        else:
            chunk_files = split_tsv(input_file, num_chunks, temp_chunks_dir, packet_limit)

        # Process chunks in parallel
        logging.info("Processing chunks in parallel...")
        chunk_args = []
        for chunk_file in chunk_files:
            # Rationale: Pass PCA configuration to each chunk processor.
            # This ensures that PCA is consistently applied across all processing
            # units, maintaining the intended dimensionality reduction strategy.
            chunk_args.append((chunk_file, maxAE, FMgrace, ADgrace, output_dir, pca_components, pca_grace_period))

        # Use half of available CPU cores to avoid overwhelming system
        n_cores = max(1, os.cpu_count() // 2)
        logging.info(f"Using {n_cores} CPU cores")

        with mp.Pool(n_cores) as pool:
            results = list(tqdm(
                pool.imap(execute_chunk, chunk_args),
                total=len(chunk_args),
                desc="Processing chunks"
            ))

        # Combine results
        logging.info("Combining results...")
        rmse_files = [r[0] for r in results]
        ms_conf_files = [r[1] for r in results]
        sec_conf_files = [r[2] for r in results]

        # Final output files
        final_rmse = os.path.join(output_dir, "combined_rmse.csv")
        final_ms_conf = os.path.join(output_dir, "combined_confidence_ms.csv")
        final_sec_conf = os.path.join(output_dir, "combined_confidence_sec.csv")

        join_rmse_files(rmse_files, final_rmse)
        join_confidence_files(ms_conf_files, final_ms_conf, is_millisecond=True)
        join_confidence_files(sec_conf_files, final_sec_conf, is_millisecond=False)

        return final_rmse, final_ms_conf, final_sec_conf

    except Exception as e:
        logging.error(f"Error in process_in_chunks: {str(e)}")
        raise

    finally:
        # Cleanup
        try:
            if os.path.exists(temp_chunks_dir):
                shutil.rmtree(temp_chunks_dir)
        except Exception as e:
            logging.warning(f"Failed to clean up temporary files: {str(e)}")

if __name__ == '__main__':
    multiprocessing.freeze_support()