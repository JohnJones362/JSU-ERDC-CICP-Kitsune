import os
import time
import logging
import multiprocessing
from chunked_kitsune import process_in_chunks
from datetime import datetime

# Set up logging
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - Process %(process)d - %(message)s',
        handlers=[
            logging.FileHandler('parallel_test.log'),
            logging.StreamHandler()
        ]
    )

def main():
    setup_logger()
    logger = logging.getLogger()
    
    # Test parameters
    num_chunks = multiprocessing.cpu_count()  # Use number of CPU cores
    maxAE = 10
    FMgrace = 5000
    ADgrace = 10000
    
    # Log system info
    logger.info(f"Number of CPU cores available: {multiprocessing.cpu_count()}")
    logger.info(f"Number of chunks to process: {num_chunks}")
    
    # Your input file path - replace with your actual file
    input_file = "Dec2019_00001_20191206102207.pcap"  # or your TSV file
    
    if not os.path.exists(input_file):
        logger.error(f"Input file {input_file} not found!")
        return
    
    # Process the file and measure time
    start_time = time.time()
    logger.info("Starting parallel processing...")
    
    try:
        # Process in chunks
        rmse_file = process_in_chunks(
            input_file=input_file,
            num_chunks=num_chunks,
            maxAE=maxAE,
            FMgrace=FMgrace,
            ADgrace=ADgrace,
            output_dir="test_results",
            chunks_dir="test_chunks"
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Processing completed in {total_time:.2f} seconds")
        logger.info(f"Output RMSE file: {rmse_file}")
        
        # Compare with sequential processing
        logger.info("Starting sequential processing (1 chunk)...")
        start_time_seq = time.time()
        
        rmse_file_seq = process_in_chunks(
            input_file=input_file,
            num_chunks=1,  # Sequential processing
            maxAE=maxAE,
            FMgrace=FMgrace,
            ADgrace=ADgrace,
            output_dir="test_results_seq",
            chunks_dir="test_chunks_seq"
        )
        
        end_time_seq = time.time()
        total_time_seq = end_time_seq - start_time_seq
        
        logger.info(f"Sequential processing completed in {total_time_seq:.2f} seconds")
        
        # Calculate speedup
        speedup = total_time_seq / total_time
        logger.info(f"Parallel speedup: {speedup:.2f}x")
        
        # Save summary
        summary_file = os.path.join("test_results", "parallel_test_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Test Summary - {datetime.now()}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Number of CPU cores: {multiprocessing.cpu_count()}\n")
            f.write(f"Number of chunks: {num_chunks}\n")
            f.write(f"Parallel processing time: {total_time:.2f} seconds\n")
            f.write(f"Sequential processing time: {total_time_seq:.2f} seconds\n")
            f.write(f"Speedup: {speedup:.2f}x\n")
            f.write("-" * 50 + "\n")
        
        logger.info(f"Test summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 