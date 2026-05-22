import pandas as pd
import numpy as np

def calculate_ttd(ground_truth, predicted_flags):
    """
    Calculates the Time-To-Detect (TTD) for each attack.

    TTD is the number of packets from the start of an attack until the first
    anomaly is detected.

    Args:
        ground_truth (np.array): A binary array where 1 indicates an attack packet.
        predicted_flags (np.array): A binary array where 1 indicates a detected anomaly.

    Returns:
        np.array: An array of TTD values (in number of packets) for each detected attack.
                  Returns an empty array if no attacks are detected.
    """
    ttd_values = []
    in_attack = False
    packets_since_attack_start = 0

    # Ensure both arrays have the same length
    if len(ground_truth) != len(predicted_flags):
        raise ValueError("Input arrays for TTD must have the same length.")

    for i in range(len(ground_truth)):
        # Check for the start of a new attack
        if ground_truth[i] == 1 and not in_attack:
            in_attack = True
            packets_since_attack_start = 0

        # If an attack is in progress, increment the counter
        if in_attack:
            packets_since_attack_start += 1
            # If an anomaly is predicted, record the TTD and reset for the next attack
            if predicted_flags[i] == 1:
                ttd_values.append(packets_since_attack_start)
                in_attack = False
    
    return np.array(ttd_values)


def calculate_metrics(ground_truth_file, predicted_anomalies_file, predicted_index_col=None):
    """
    Calculates various detection metrics using ground truth and predicted anomaly flags.

    Args:
        ground_truth_file (str): Path to the CSV file containing ground truth attack information.
                                 Expected columns: 'attacks' (packet index) and 'attack_number' (interval).
        predicted_anomalies_file (str): Path to the CSV file containing predicted anomaly information.
        predicted_index_col (str, optional): The name of the column in predicted_anomalies_file
                                             that contains the packet index. If None, the function
                                             will try 'PacketIndex' then 'packet_idx'.

    Returns:
        dict: A dictionary containing the calculated metrics. Returns None if a file is not found.
    """
    try:
        ground_truth_df_raw = pd.read_csv(ground_truth_file)
        predicted_df = pd.read_csv(predicted_anomalies_file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return None

    # Determine the correct packet index column for the predicted anomalies file
    if predicted_index_col:
        if predicted_index_col not in predicted_df.columns:
            raise ValueError(f"Specified predicted_index_col '{predicted_index_col}' not found in {predicted_anomalies_file}. Available columns: {predicted_df.columns.tolist()}")
        packet_index_col_name = predicted_index_col
    else:
        # Try common column names if not explicitly provided
        possible_cols = ['PacketIndex', 'packet_idx']
        found_col = None
        for col in possible_cols:
            if col in predicted_df.columns:
                found_col = col
                break
        if found_col:
            packet_index_col_name = found_col
        else:
            raise ValueError(f"Could not find a suitable packet index column in {predicted_anomalies_file}. "
                             f"Tried {possible_cols}. Please specify 'predicted_index_col' argument. Available columns: {predicted_df.columns.tolist()}")

    # Determine the maximum packet index from both ground truth and predictions
    # to ensure the ground truth array covers all packets.
    max_ground_truth_index = ground_truth_df_raw['attacks'].max() if not ground_truth_df_raw.empty else 0
    max_predicted_index = predicted_df[packet_index_col_name].max() if not predicted_df.empty else 0
    max_packet_index = int(max(max_ground_truth_index, max_predicted_index))

    # Create the ground truth flags based on attack intervals
    ground_truth = np.zeros(max_packet_index + 1)
    
    # Iterate through the attack intervals defined in the ground truth file
    attack_intervals = ground_truth_df_raw['attacks'].values
    # Ensure there are an even number of attack_intervals for start/stop pairs
    if len(attack_intervals) % 2 != 0:
        print("Warning: Uneven number of attack interval entries in ground truth file. Skipping last entry.")
        attack_intervals = attack_intervals[:-1]

    for i in range(0, len(attack_intervals), 2):
        start_index = attack_intervals[i]
        stop_index = attack_intervals[i+1]
        # Ensure indices are within bounds
        if start_index <= max_packet_index and stop_index <= max_packet_index:
            ground_truth[start_index : stop_index + 1] = 1
        else:
            print(f"Warning: Attack interval [{start_index}, {stop_index}] out of bounds for max_packet_index {max_packet_index}. Skipping.")


    # Create predicted flags
    predicted_anomalies = predicted_df[packet_index_col_name].values
    predicted_flags = np.zeros(max_packet_index + 1)
    # Filter predicted anomalies to be within the max_packet_index bounds
    predicted_anomalies = predicted_anomalies[predicted_anomalies <= max_packet_index]
    if len(predicted_anomalies) > 0:
        predicted_flags[predicted_anomalies] = 1

    # --- Debugging Information (Enhanced) ---
    print(f"\n--- Detailed Index Debugging Information ---")
    print(f"Ground Truth File: {ground_truth_file}")
    print(f"Predicted Anomalies File: {predicted_anomalies_file}")
    print(f"Using predicted index column: '{packet_index_col_name}'") # New debug print
    print(f"Max Packet Index derived: {max_packet_index}")
    print(f"Total packets considered in arrays: {len(ground_truth)}")
    
    print("\n--- Ground Truth Raw Data Sample (first 10 rows of 'attacks' column) ---")
    print(ground_truth_df_raw['attacks'].head(10))
    print(f"Min ground truth index: {ground_truth_df_raw['attacks'].min()}")
    print(f"Max ground truth index: {ground_truth_df_raw['attacks'].max()}")

    print("\n--- Ground Truth Array (Indices marked as 1) Sample ---")
    gt_attack_indices = np.where(ground_truth == 1)[0]
    if len(gt_attack_indices) > 0:
        print(f"First 10 ground truth attack indices: {gt_attack_indices[:10]}")
        print(f"Last 10 ground truth attack indices: {gt_attack_indices[-10:]}")
        print(f"Min ground truth attack index in array: {gt_attack_indices.min()}")
        print(f"Max ground truth attack index in array: {gt_attack_indices.max()}")
    else:
        print("No attack packets found in ground truth array.")
    print(f"Total actual attack packets (ground truth array sum): {np.sum(ground_truth)}")

    print("\n--- Predicted Anomalies Raw Data Sample (first 10 rows of '{packet_index_col_name}') ---")
    print(predicted_df[packet_index_col_name].head(10))
    print(f"Min predicted anomaly index: {predicted_df[packet_index_col_name].min()}")
    print(f"Max predicted anomaly index: {predicted_df[packet_index_col_name].max()}")

    print("\n--- Predicted Flags Array (Indices marked as 1) Sample ---")
    pred_anomaly_indices = np.where(predicted_flags == 1)[0]
    if len(pred_anomaly_indices) > 0:
        print(f"First 10 predicted anomaly indices: {pred_anomaly_indices[:10]}")
        print(f"Last 10 predicted anomaly indices: {pred_anomaly_indices[-10:]}")
        print(f"Min predicted anomaly index in array: {pred_anomaly_indices.min()}")
        print(f"Max predicted anomaly index in array: {pred_anomaly_indices.max()}")
    else:
        print("No predicted anomaly packets found in predicted flags array.")
    print(f"Total predicted anomaly packets (predicted flags array sum): {np.sum(predicted_flags)}")

    print(f"\nNumber of overlapping (TP) indices: {np.sum((ground_truth == 1) & (predicted_flags == 1))}")
    print(f"-------------------------------------------\n")
    # --- End Debugging Information ---


    # Calculate metrics
    true_positives = np.sum((ground_truth == 1) & (predicted_flags == 1))
    true_negatives = np.sum((ground_truth == 0) & (predicted_flags == 0))
    false_positives = np.sum((ground_truth == 0) & (predicted_flags == 1))
    false_negatives = np.sum((ground_truth == 1) & (predicted_flags == 0))

    total_packets = len(ground_truth)
    total_attacks = np.sum(ground_truth)

    accuracy = (true_positives + true_negatives) / total_packets if total_packets > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / total_attacks if total_attacks > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    tpr = recall
    fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
    tnr = true_negatives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0

    # Calculate TTD metrics
    ttd_values = calculate_ttd(ground_truth, predicted_flags)
    avg_ttd = np.mean(ttd_values) if len(ttd_values) > 0 else np.nan
    sttd = np.std(ttd_values) if len(ttd_values) > 1 else np.nan

    # Original metrics with less descriptive names (included for compatibility)
    sclf = np.sum(predicted_flags)
    s = total_attacks
    prt = total_packets

    return {
        'tp': true_positives,
        'fp': false_positives,
        'tn': true_negatives,
        'fn': false_negatives,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        'avg_ttd': avg_ttd,
        'sttd': sttd,
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'sclf': sclf,  # Number of predicted anomalies
        's': s,        # Total number of attack packets
        'prt': prt     # Total number of packets
    }

def main():
    """
    Main function to run the metric calculation.
    Defines file paths and prints the results.
    """
    # Use the correct ground truth file name as uploaded
    ground_truth_file = r"D:\Backup copies of code\(PCA) backup code of Kitsune-py-master - Claude2\A6-2015-12\df_attacks_with_network_attacks.csv"
    # Using a generic placeholder for the new anomalies file path.
    predicted_anomalies_file = r"D:\Backup copies of code\(PCA) backup code of Kitsune-py-master - Claude2\Results\rmse\rmse_Dec2019_00002_20191206103000.pcap_time=20250629_220912.csv"
    output_csv_file = "detection_metrics.csv"

    # Example of how to use the new parameter if you use the RMSE file:
    # predicted_anomalies_file = r"Results/rmse/rmse_Dec2019_00002_20191206103000.pcap_time=20250629_220912.csv"
    # metrics = calculate_metrics(ground_truth_file, predicted_anomalies_file, predicted_index_col='packet_idx')

    metrics = calculate_metrics(ground_truth_file, predicted_anomalies_file)

    if metrics:
        print("Detection Metrics:")
        # Print metrics in a clean format
        for key, value in metrics.items():
            print(f"- {key}: {value}")

        # Save metrics to a CSV file
        try:
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(output_csv_file, index=False)
            print(f"\nMetrics saved to {output_csv_file}")
        except Exception as e:
            print(f"Error saving metrics to CSV: {e}")
    else:
        print("Could not calculate metrics due to an error.")

if __name__ == "__main__":
    main()
