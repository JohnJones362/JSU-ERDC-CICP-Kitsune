import os
import pandas as pd
import time

"""
To use this script to simulate real-time data streaming, include the following lines in your code:

import subprocess
import time

# Path to the simulation script
simulation_script = '/path/to/realtime_data_simulator/realtime_data_simulator.py'

# Start the simulation script as a separate process
subprocess.Popen(['python', simulation_script])

# Give the simulation some time to start
time.sleep(10)

# Datafile to import (should match the output file path in the simulation script):
'/path/to/simulated/datafile/simulated_data.csv'

"""

# User-configurable parameters
# IMPORTANT: Update this path to point to your ciniminer_sample_traffic_log.csv file
INPUT_FILE = 'test_traffic_log.csv' # Or '/path/to/your/test_traffic_log.csv'
OUTPUT_FILE = 'simulated_data_for_test.tsv' # Give it a unique name for testing
CHUNK_SIZE = 2 # Process very small chunks to see real-time behavior quickly
SLEEP_TIME = 0.5 # A noticeable delay between chunks

def simulate_data_pipeline(input_file, output_file, chunk_size, sleep_time):
    """
    Simulates a real-time data pipeline by reading an input CSV file in chunks
    and appending each chunk to an output file with a delay between chunks.

    Args:
        input_file (str): Path to the original data file.
        output_file (str): Path to the output file where simulated data will be saved.
        chunk_size (int): Number of rows to read and write at a time.
        sleep_time (int): Time in seconds to wait between writing chunks, simulating real-time data flow.
    """
    
    # Delete the output file if it exists to ensure a clean start
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"{output_file} has been deleted to start fresh simulation.")
    
    # Read the input file in chunks and simulate real-time data streaming
    # IMPORTANT: Set sep=',' for reading your comma-separated input file
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, sep=','): # Match this 'sep' to your test file
        # Append the chunk to the output file; write header only for the first chunk
        # Output to .tsv (tab-separated) as Kitsune's FeatureExtractor expects this by default
        chunk.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file), sep='\t')
        print(f"Chunk of {chunk_size} rows written to {output_file}. Sleeping for {sleep_time} seconds.")
        
        # Wait before processing the next chunk to simulate real-time data arrival
        time.sleep(sleep_time)

if __name__ == "__main__":
    simulate_data_pipeline(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE, SLEEP_TIME)

