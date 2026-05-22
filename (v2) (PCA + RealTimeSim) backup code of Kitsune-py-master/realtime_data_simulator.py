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
CHUNK_SIZE = 1000 # Increased for faster simulation (e.g., from 5 to 1000)
SLEEP_TIME = 0.01 # Reduced for faster simulation (e.g., from 1 to 0.01 seconds)
                    # Set to 0 for maximum speed if you don't need real-time delays


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

    # We assume example.py has already deleted the file if it existed.
    # So, we just ensure the file exists and write header if it's a new file.
    first_chunk = not os.path.exists(output_file)

    try:
        with pd.read_csv(input_file, chunksize=chunk_size, sep='|') as reader:
            for i, chunk in enumerate(reader):
                chunk.to_csv(output_file, mode='a', index=False, header=first_chunk, sep='\t')
                print(f"Chunk of {len(chunk)} rows written to {output_file}. Sleeping for {sleep_time} seconds.")
                first_chunk = False # After the first chunk, subsequent writes should not include header

                if sleep_time > 0:
                    time.sleep(sleep_time)
        print(f"Simulation complete for {input_file}.")
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Please ensure it exists.")
    except pd.errors.EmptyDataError:
        print(f"Error: Input file '{input_file}' is empty.")
    except Exception as e:
        print(f"An error occurred during simulation: {e}")


if __name__ == "__main__":
    simulate_data_pipeline(INPUT_FILE, OUTPUT_FILE, CHUNK_SIZE, SLEEP_TIME)

