import pandas as pd

file_path = 'ciniminer_sample_traffic_log.csv'
delimiter = '|' # Your file is pipe-delimited

try:
    # Read a small chunk to quickly get the number of rows
    # The 'engine="c"' is generally faster for C-based parsing
    # Use chunksize to avoid loading the entire file into memory
    total_packets = 0
    with pd.read_csv(file_path, chunksize=100000, sep=delimiter, engine='c') as reader:
        for chunk in reader:
            total_packets += len(chunk)
    
    print(f"The file '{file_path}' has {total_packets} packets (counted by pandas).")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except pd.errors.EmptyDataError:
    print(f"Error: The file '{file_path}' is empty.")
except Exception as e:
    print(f"An error occurred: {e}")