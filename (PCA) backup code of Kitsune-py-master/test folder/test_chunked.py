from chunked_kitsune import run_chunked_kitsune
import os

if __name__ == '__main__':
    # Input parameters
    input_file = "Dec2019_00001_20191206102207.pcap"  # Your input file
    n_chunks = 4
    max_workers = os.cpu_count()  # Use all available CPU cores
    
    print(f"CPU cores available: {max_workers}")
    print(f"Starting chunked processing with {n_chunks} chunks...")
    
    output_file = run_chunked_kitsune(
        input_file=input_file,
        n_chunks=n_chunks,
        maxAE=10,
        FMgrace=5000,
        ADgrace=10000,
        learning_rate=0.1,
        hidden_ratio=0.75,
        max_workers=max_workers
    )
    
    print(f"Processing completed. Results saved to: {output_file}") 