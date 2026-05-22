import os
import sys
import multiprocessing
import concurrent.futures
import time
import psutil
from tqdm import tqdm

def print_info():
    """Print Python and environment information"""
    print("\nEnvironment Information:")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Number of CPU cores: {os.cpu_count()}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Operating System: {os.name} - {sys.platform}")
    print(f"Multiprocessing start method: {multiprocessing.get_start_method(allow_none=True)}")

def print_python_processes():
    """Print all running Python processes"""
    python_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                python_processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    print("\nCurrent Python processes:")
    for proc in python_processes:
        print(f"PID: {proc['pid']}, Name: {proc['name']}")
    print(f"Total Python processes: {len(python_processes)}")

def worker_function(seconds):
    """Simple worker function that sleeps for given seconds"""
    process_id = os.getpid()
    print(f"\nWorker process {process_id} starting...")
    print(f"Parent process ID: {os.getppid()}")
    
    # Sleep in smaller increments to show activity
    for i in range(seconds):
        print(f"Process {process_id}: Still working... ({i+1}/{seconds} seconds)")
        time.sleep(1)
    
    return process_id

def test_process_function():
    """Function for testing basic process creation"""
    pid = os.getpid()
    ppid = os.getppid()
    print(f"Test process running with PID: {pid}, Parent PID: {ppid}")
    time.sleep(5)  # Keep the process running longer

def test_multiprocessing():
    """Test multiprocessing functionality"""
    print("\nTesting multiprocessing...")
    n_workers = min(4, os.cpu_count() or 1)
    
    # Force 'spawn' method for Windows compatibility
    ctx = multiprocessing.get_context('spawn')
    
    print(f"Starting {n_workers} worker processes...")
    print("Main process ID:", os.getpid())
    
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx
    ) as executor:
        # Run workers for 30 seconds each
        futures = [executor.submit(worker_function, 30) for _ in range(n_workers)]
        
        # Print Python processes right after starting workers
        print_python_processes()
        
        print("\nProcesses are now running. Please check Task Manager.")
        print("Press Enter to continue...")
        input()
        
        # Use tqdm to show progress
        with tqdm(total=n_workers, desc="Completing workers") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    process_id = future.result()
                    print(f"\nWorker process {process_id} completed")
                    pbar.update(1)
                except Exception as e:
                    print(f"\nWorker failed with error: {str(e)}")
        
        # Print Python processes again after completion
        print_python_processes()

if __name__ == '__main__':
    # Set start method to 'spawn' for Windows compatibility
    multiprocessing.set_start_method('spawn')
    
    print_info()
    print_python_processes()
    
    print("\nTesting process creation...")
    test_process = multiprocessing.Process(target=test_process_function)
    test_process.start()
    test_process.join()
    
    test_multiprocessing()
    
    print("\nAll tests completed!")
    print_python_processes() 