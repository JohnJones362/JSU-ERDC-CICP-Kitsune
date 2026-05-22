# simple_test.py
from memory_profiler import profile
import numpy as np
import time
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@profile
def allocate_and_free_memory():
    """A simple function to allocate and then (implicitly) free some memory."""
    logging.info("Starting memory allocation test...")
    # Allocate a large NumPy array
    # An array of 1000x1000 floats will take 1000*1000*8 bytes (for float64) = 8,000,000 bytes ~ 7.63 MiB
    large_array = np.random.rand(1000, 1000)
    logging.info(f"Allocated a large array. Current array size: {large_array.nbytes / (1024 * 1024):.2f} MiB")
    time.sleep(1) # Simulate some work

    # Allocate another array
    another_array = np.random.rand(500, 500) # ~1.9 MiB
    logging.info(f"Allocated another array. Current array size: {another_array.nbytes / (1024 * 1024):.2f} MiB")
    time.sleep(1) # Simulate more work

    # Explicitly delete the first large array to see memory release
    del large_array
    logging.info("Deleted large_array. Python's garbage collector should eventually free this memory.")
    time.sleep(0.5) # Give some time for potential GC action (though usually not immediate)

    logging.info("Test function finished. Memory should be returning to baseline.")

if __name__ == '__main__':
    logging.info("Running simple_test.py to check memory_profiler functionality.")
    allocate_and_free_memory()
    logging.info("simple_test.py execution complete. Please check for 'simple_test.py.mprof' file.")

