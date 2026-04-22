# combined_metrics_script.py

import numpy as np
import csv
import pandas as pd

def calculate_ttd(predicted_flags, ground_truth):
    ttd = []
    attack_start_indices = [i for i, label in enumerate(ground_truth) if label == 1]
    for start_idx in attack_start_indices:
        attack_end_idx = next((i for i in range(start_idx, len(predicted_flags)) if predicted_flags[i] == 1), None)
        if attack_end_idx is not None:
            ttd.append(attack_end_idx - start_idx)
    return ttd

def calculate_metrics(predicted_flags, ground_truth):
    tp = sum((predicted_flags == 1) & (ground_truth == 1))
    fp = sum((predicted_flags == 1) & (ground_truth == 0))
    tn = sum((predicted_flags == 0) & (ground_truth == 0))
    fn = sum((predicted_flags == 0) & (ground_truth == 1))

    accuracy = (tp + tn) / len(ground_truth)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) != 0 else 0

    iteration_ttd = calculate_ttd(predicted_flags, ground_truth)
    avg_ttd = np.nanmean(iteration_ttd) if len(iteration_ttd) > 0 else np.nan
    sttd = 1 - avg_ttd if not np.isnan(avg_ttd) else np.nan

    sclf = (tpr + fpr) / 2
    s = 0.5 * sttd + 0.5 * sclf

    if precision > 0 and recall > 0 and sttd > 0:
        prt = 3 / (1/precision + 1/recall + 1/sttd)
    else:
        prt = 0

    return tp, fp, tn, fn, accuracy, precision, recall, f1, tpr, fpr, tnr, sttd, sclf, s, prt

def print_metrics(metrics):
    tp, fp, tn, fn, accuracy, precision, recall, f1, tpr, fpr, tnr, sttd, sclf, s, prt = metrics
    print(f"True Positives: {tp}")
    print(f"False Positives: {fp}")
    print(f"True Negatives: {tn}")
    print(f"False Negatives: {fn}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"True Positive Rate (TPR): {tpr:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"True Negative Rate (TNR): {tnr:.4f}")
    print(f"STTD (inverse time to detection): {sttd:.4f}")
    print(f"SCLF (score combining TPR and FPR): {sclf:.4f}")
    print(f"Final Score (s): {s:.4f}")
    print(f"Harmonic PRT: {prt:.4f}")

def save_metrics(metrics, filename):
    with open(filename, "w", newline='') as f_metrics:
        writer_metrics = csv.writer(f_metrics)
        writer_metrics.writerow([
            "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1",
            "TPR", "FPR", "TNR", "STTD", "SCLF", "FinalScore", "HarmonicPRT"
        ])
        writer_metrics.writerow(metrics)

# --- Configuration ---
GROUND_TRUTH_FILE = r'A6-2015-12\df_attacks_with_network_attacks.csv'
PREDICTION_FILE = r'Results\kitsune_anomalies_for_Dec2019_00001_20191206102207_with_maxAE=10_FMgrace=5000_ADgrace=10000_number_of_packets=1000000.csv'
ANOMALY_THRESHOLD = 0.1  # Adjust this threshold as needed
OUTPUT_METRICS_FILE = r'kitsune_metrics.csv'

# --- Load Ground Truth Data ---
try:
    df_ground_truth = pd.read_csv(GROUND_TRUTH_FILE)
except FileNotFoundError:
    print(f"Error: Ground truth file '{GROUND_TRUTH_FILE}' not found. Please make sure the file exists in the correct location.")
    exit()

# Initialize ground truth flags based on the 'attacks' column
max_packet_index = int(df_ground_truth['packet_index'].max()) if not pd.isna(df_ground_truth['packet_index'].max()) else 0
ground_truth = np.zeros(max_packet_index + 1, dtype=int)
for index, row in df_ground_truth.iterrows():
    if not pd.isna(row['packet_index']):
        start_index = int(row['packet_index']) # Ensure packet_index is treated as an integer
        # Assuming 'attacks' column has a non-zero value during attacks
        if row['attacks'] > 0:
            ground_truth[start_index] = 1

# --- Load Prediction Data ---
try:
    df_predictions = pd.read_csv(PREDICTION_FILE)
except FileNotFoundError:
    print(f"Error: Prediction file '{PREDICTION_FILE}' not found. Please make sure the file exists in the correct location.")
    exit()

# --- Convert Predictions to Flags ---
predicted_flags = np.zeros(ground_truth.shape[0], dtype=int)
for index, row in df_predictions.iterrows():
    packet_index = int(row['PacketIndex']) # Ensure PacketIndex is treated as an integer
    rmse_score = row['RMSE']
    if packet_index < len(predicted_flags):
        if rmse_score > ANOMALY_THRESHOLD:
            predicted_flags[packet_index] = 1

# --- Align Data (Important for correct metric calculation) ---
# Since the Kitsune output might not have entries for every single packet index,
# we need to make sure our predicted_flags and ground_truth arrays align.
# We've initialized 'predicted_flags' and 'ground_truth' based on the maximum
# packet index in the ground truth data. If the prediction data has indices
# beyond this, we might need to adjust. For now, we'll assume the Kitsune
# results are within the range of the ground truth packet indices.

# --- Calculate Metrics ---
metrics = calculate_metrics(predicted_flags, ground_truth)

# --- Print and Save Metrics ---
print("\n--- Calculated Metrics ---")
print_metrics(metrics)

save_metrics(metrics, OUTPUT_METRICS_FILE)
print(f"\nMetrics saved to '{OUTPUT_METRICS_FILE}'")