import pandas as pd
import numpy as np

def calculate_metrics(ground_truth_file, predicted_anomalies_file):
    """
    Calculates various detection metrics using ground truth and predicted anomaly flags.

    Args:
        ground_truth_file (str): Path to the CSV file containing ground truth attack information.
                                   Expected columns: 'packet_index' and 'attacks' (where 1 indicates an attack).
        predicted_anomalies_file (str): Path to the CSV file containing predicted anomaly information.
                                       Expected columns: 'PacketIndex' and potentially others indicating anomalies.
                                       We will infer predicted flags based on the presence of an entry for a packet index.

    Returns:
        dict: A dictionary containing the calculated metrics (tp, fp, tn, fn, accuracy, precision,
              recall, f1, avg_ttd, sttd, tpr, fpr, tnr, sclf, s, prt). Returns None if there's an error
              loading the data.
    """
    try:
        ground_truth_df = pd.read_csv(ground_truth_file)
        predicted_df = pd.read_csv(predicted_anomalies_file)
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return None

    # Create ground truth flags based on the 'attacks' column
    max_packet_index = int(ground_truth_df['packet_index'].max())
    ground_truth = np.zeros(max_packet_index + 1, dtype=int)
    for index, row in ground_truth_df.iterrows():
        if row['attacks'] == 1500:  # Assuming '1500' in 'attacks' column signifies an attack
            start_index = int(row['packet_index'])
            end_index = max_packet_index  # Initialize end_index to max in case of issues
            try:
                next_attack_row = ground_truth_df.iloc[index + 1]
                if pd.notna(next_attack_row['packet_index']):
                    end_index = int(next_attack_row['packet_index'])
            except IndexError:
                pass  # It's the last attack, end_index remains max_packet_index
            ground_truth[start_index:end_index + 1] = 1

    # Create predicted flags based on the presence of a PacketIndex in the predicted anomalies file
    predicted_flags = np.zeros(ground_truth.shape[0], dtype=int)
    predicted_indices = predicted_df['PacketIndex'].values.astype(int)
    predicted_flags[predicted_indices] = 1

    # Align the lengths if necessary (assuming predicted can have more packets)
    min_len = min(len(predicted_flags), len(ground_truth))
    predicted_flags = predicted_flags[:min_len]
    ground_truth = ground_truth[:min_len]

    tp = np.sum((predicted_flags == 1) & (ground_truth == 1))
    fp = np.sum((predicted_flags == 1) & (ground_truth == 0))
    tn = np.sum((predicted_flags == 0) & (ground_truth == 0))
    fn = np.sum((predicted_flags == 0) & (ground_truth == 1))

    accuracy = (tp + tn) / len(ground_truth) if len(ground_truth) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    # Compute Time To Detection (TTD)
    iteration_ttd = calculate_ttd(predicted_flags, ground_truth)
    avg_ttd = np.nanmean(iteration_ttd) if len(iteration_ttd) > 0 else np.nan
    sttd = 1 - avg_ttd if not np.isnan(avg_ttd) else np.nan

    # Compute additional rates
    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0

    sclf = (tpr + fpr) / 2
    s = 0.5 * sttd + 0.5 * sclf

    # Compute harmonic PRT (only if none of the denominators are zero)
    if precision > 0 and recall > 0 and sttd > 0:
        prt = 3 / (1/precision + 1/recall + 1/sttd) # correct, verified 2025/02/13
    else:
        prt = 0

    metrics = {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_ttd': avg_ttd,
        'sttd': sttd,
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'sclf': sclf,
        's': s,
        'prt': prt
    }
    return metrics

def calculate_ttd(predicted_flags, ground_truth):
    """
    Calculates the Time To Detection (TTD) for each attack instance.

    Args:
        predicted_flags (np.array): Array of predicted anomaly flags (1 for anomaly, 0 for normal).
        ground_truth (np.array): Array of ground truth anomaly flags.

    Returns:
        np.array: An array of TTD values (in number of packets) for each detected attack.
                  Returns an empty array if no attacks are detected.
    """
    ttd_values = []
    attack_started = False
    time_since_attack = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] == 1 and not attack_started:
            attack_started = True
            time_since_attack = 0
        if attack_started:
            time_since_attack += 1
            if predicted_flags[i] == 1:
                ttd_values.append(time_since_attack)
                attack_started = False
    return np.array(ttd_values)

if __name__ == "__main__":
    ground_truth_file = "df_attacks_with_network_attacks.csv"
    predicted_anomalies_file = "Results/kitsune_anomalies_for_Dec2019_00001_20191206102207_with_maxAE=10_FMgrace=5000_ADgrace=10000_packet_limit=500000.csv"
    output_csv_file = "detection_metrics.csv"

    metrics = calculate_metrics(ground_truth_file, predicted_anomalies_file)

    if metrics:
        print("Detection Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        # Save metrics to a CSV file
        metrics_df = pd.DataFrame([metrics])
        try:
            metrics_df.to_csv(output_csv_file, index=False)
            print(f"\nMetrics saved to '{output_csv_file}'")
        except Exception as e:
            print(f"\nError saving metrics to CSV: {e}")