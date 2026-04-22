import numpy as np
from Kitsune import Kitsune
import os
import csv
from datetime import datetime
import concurrent.futures
from save_results import save_results

def run_kitsune_instance(instance_params):
    """
    Run a single Kitsune instance with given parameters and store results
    """
    instance_id = instance_params['instance_id']
    file_path = instance_params['file_path']
    packet_limit = instance_params['packet_limit']
    maxAE = instance_params['maxAE']
    FMgrace = instance_params['FMgrace']
    ADgrace = instance_params['ADgrace']
    learning_rate = instance_params.get('learning_rate', 0.1)
    hidden_ratio = instance_params.get('hidden_ratio', 0.75)
    
    # Create output directory if it doesn't exist
    output_dir = f"Results/instance_{instance_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Kitsune
    K = Kitsune(file_path, packet_limit, maxAE, FMgrace, ADgrace, learning_rate, hidden_ratio)
    
    # Lists to store results
    RMSEs = []
    anomalies = []
    time_buckets = {}
    second_buckets = {}
    
    print(f"Running Kitsune instance {instance_id}...")
    i = 0
    while True:
        i += 1
        if i % 1000 == 0:
            print(f"Instance {instance_id}: Processing packet {i}")
            
        rmse = K.proc_next_packet()
        if rmse == -1:
            break
            
        RMSEs.append(rmse)
        
        # Get packet timestamp
        packet_timestamp = K.FE.last_timestamp
        millisecond_bucket = int(packet_timestamp * 1000)
        second_bucket = int(packet_timestamp)
        
        # Update time buckets
        if millisecond_bucket not in time_buckets:
            time_buckets[millisecond_bucket] = {'total': 0, 'anomaly': 0}
        time_buckets[millisecond_bucket]['total'] += 1
        
        if second_bucket not in second_buckets:
            second_buckets[second_bucket] = {'total': 0, 'anomaly': 0}
        second_buckets[second_bucket]['total'] += 1
        
        # Check for anomalies
        if len(RMSEs) > FMgrace + ADgrace:
            threshold = np.mean(RMSEs) + 2 * np.std(RMSEs)
            if rmse > threshold:
                anomalies.append((i, packet_timestamp, rmse, K.FE.last_input))
                time_buckets[millisecond_bucket]['anomaly'] += 1
                second_buckets[second_bucket]['anomaly'] += 1
    
    # Save results
    import_file = os.path.basename(file_path)
    save_results(
        import_file,
        maxAE,
        FMgrace,
        ADgrace,
        packet_limit,
        anomalies,
        time_buckets,
        second_buckets,
        output_dir
    )
    
    # Save RMSE values
    rmse_file = os.path.join(output_dir, f"rmse_values_instance_{instance_id}.csv")
    with open(rmse_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["PacketIndex", "RMSE"])
        for idx, rmse in enumerate(RMSEs):
            writer.writerow([idx + 1, rmse])
            
    # Save model state
    model_file = os.path.join(output_dir, f"kitnet_state_instance_{instance_id}.pkl")
    K.AnomDetector.save_model(model_file)
    
    print(f"Instance {instance_id} completed. Results saved in {output_dir}")
    return instance_id

def run_parallel_kitsune(file_path, n_instances, packet_limit=None, maxAE=10, FMgrace=5000, ADgrace=10000,
                        learning_rates=None, hidden_ratios=None, max_workers=None):
    """
    Run N instances of Kitsune in parallel
    
    Args:
        file_path: Path to the pcap/tsv file
        n_instances: Number of Kitsune instances to run
        packet_limit: Maximum number of packets to process
        maxAE: Maximum size of autoencoder
        FMgrace: Feature mapping grace period
        ADgrace: Anomaly detection grace period
        learning_rates: List of learning rates for each instance (optional)
        hidden_ratios: List of hidden ratios for each instance (optional)
        max_workers: Maximum number of parallel workers (optional)
    """
    if learning_rates is None:
        learning_rates = [0.1] * n_instances
    if hidden_ratios is None:
        hidden_ratios = [0.75] * n_instances
        
    # Prepare parameters for each instance
    instance_params = []
    for i in range(n_instances):
        params = {
            'instance_id': i + 1,
            'file_path': file_path,
            'packet_limit': packet_limit,
            'maxAE': maxAE,
            'FMgrace': FMgrace,
            'ADgrace': ADgrace,
            'learning_rate': learning_rates[i],
            'hidden_ratio': hidden_ratios[i]
        }
        instance_params.append(params)
    
    # Run instances in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_kitsune_instance, params) for params in instance_params]
        concurrent.futures.wait(futures)
        
    print("All Kitsune instances completed!")

if __name__ == "__main__":
    # Example usage
    pcap_file = "Dec2019_00001_20191206102207.pcap"  # Replace with your input file
    n_instances = 3  # Number of instances to run
    packet_limit = 100000  # Limit the number of packets to process
    
    # Optional: Customize learning rates and hidden ratios for each instance
    learning_rates = [0.1, 0.05, 0.15]  # Different learning rates for each instance
    hidden_ratios = [0.75, 0.8, 0.7]    # Different hidden ratios for each instance
    
    run_parallel_kitsune(
        file_path=pcap_file,
        n_instances=n_instances,
        packet_limit=packet_limit,
        maxAE=10,
        FMgrace=5000,
        ADgrace=10000,
        learning_rates=learning_rates,
        hidden_ratios=hidden_ratios,
        max_workers=None  # Let the system decide based on CPU cores
    ) 