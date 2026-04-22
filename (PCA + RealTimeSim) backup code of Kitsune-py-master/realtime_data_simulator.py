import os
import pandas as pd
import time
import subprocess

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
# IMPORTANT: Update these paths for your environment
INPUT_FILE = 'CTU-IoT-Malware-Capture-1-1conn.log.labeled.csv'  # Path to the original data file
OUTPUT_FILE = 'simulated_data.tsv'  # Path to the output file for simulated data (changed to .tsv)
CHUNK_SIZE = 5  # Number of rows per chunk to simulate data streaming
SLEEP_TIME = 1  # Delay (in seconds) between writing each chunk to the output file

def simulate_data_pipeline(input_file, output_file, chunk_size, sleep_time):
    """
    Simulates a real-time data pipeline by reading an input CSV file in chunks
    and appending each chunk to an output file with a delay between chunks.

    Args:
        input_file (str): Path to the original data file (expected pipe-delimited).
        output_file (str): Path to the output file where simulated data will be saved (will be tab-delimited).
        chunk_size (int): Number of rows to read and write at a time.
        sleep_time (int): Time in seconds to wait between writing chunks, simulating real-time data flow.
    """

    # Delete the output file if it exists to ensure a clean start
    if os.path.exists(output_file):
        os.remove(output_file)
        print(f"{output_file} has been deleted to start fresh simulation.")

    # Read the input file in chunks and simulate real-time data streaming
    # Read with pipe delimiter based on user's sample
    # Write with tab delimiter for compatibility with FeatureExtractor's TSV parsing
    first_chunk = True
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, sep='|'):
        # Append the chunk to the output file; write header only for the first chunk
        # Use 'a' mode for append, 'w' for the very first write if needed (but 'a' handles creating the file)
        chunk.to_csv(output_file, mode='a', index=False, header=first_chunk, sep='\t')
        print(f"Chunk of {chunk_size} rows written to {output_file}. Sleeping for {sleep_time} seconds.")
        first_chunk = False # After the first chunk, subsequent writes should not include header

        # Wait before processing the next chunk to simulate real-time data arrival
        time.sleep(sleep_time)
    print(f"Simulation complete for {input_file}.")


if __name__ == "__main__":
    # Start the simulation using the user-configurable parameters
    simulate_data_pipeline(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE, SLEEP_TIME)

