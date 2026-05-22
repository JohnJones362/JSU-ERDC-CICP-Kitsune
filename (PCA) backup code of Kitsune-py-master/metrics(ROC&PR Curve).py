# combined_metrics_script.py

import numpy as np
import csv
import pandas as pd
import os  # Import os for directory management
import matplotlib.pyplot as plt  # Importing matplotlib for plotting ROC curve
from sklearn.metrics import roc_curve, auc  # Import for ROC calculation

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
max_packet_index = int(df_ground_truth['attacks'].max()) if not pd.isna(df_ground_truth['attacks'].max()) else 0
ground_truth = np.zeros(max_packet_index + 1, dtype=int)
for index, row in df_ground_truth.iterrows():
    if not pd.isna(row['attacks']):
        start_index = int(row['attacks']) # Ensure packet_index is treated as an integer
        # Assuming 'attacks' column has a non-zero value during attacks
        if row['attacks'] > 0:
            ground_truth[start_index] = 1

# --- Load Prediction Data ---
try:
    df_predictions = pd.read_csv(PREDICTION_FILE)
except FileNotFoundError:
    print(f"Error: Prediction file '{PREDICTION_FILE}' not found. Please make sure the file exists in the correct location.")
    exit()

# Create an array to store RMSE scores aligned with ground truth packet indices
rmse_scores = np.zeros_like(ground_truth, dtype=float)

# Map RMSE from predictions to the corresponding indices
for index, row in df_predictions.iterrows():
    packet_index = int(row['PacketIndex'])
    if packet_index < len(rmse_scores):
        rmse_scores[packet_index] = row['RMSE']

# --- Align Data (Important for correct metric calculation) ---
# Since the Kitsune output might not have entries for every single packet index,
# we need to make sure our predicted_flags and ground_truth arrays align.
# We've initialized 'predicted_flags' and 'ground_truth' based on the maximum
# packet index in the ground truth data. If the prediction data has indices
# beyond this, we might need to adjust. For now, we'll assume the Kitsune
# results are within the range of the ground truth packet indices.

# --- Calculate Metrics ---
metrics = calculate_metrics(rmse_scores, ground_truth)

# --- Print and Save Metrics ---
print("\n--- Calculated Metrics ---")
# Printing calculated metrics
for metric_name, value in zip([
    "True Positives", "False Positives", "True Negatives", "False Negatives", 
    "Accuracy", "Precision", "Recall", "F1", "TPR", "FPR", "TNR", 
    "STTD", "SCLF", "FinalScore", "HarmonicPRT"
], metrics):
    print(f"{metric_name}: {value:.4f}")

# Save metrics to a CSV file
with open(OUTPUT_METRICS_FILE, "w", newline='') as f_metrics:
    writer_metrics = csv.writer(f_metrics)
    writer_metrics.writerow([
        "TP", "FP", "TN", "FN", "Accuracy", "Precision", "Recall", "F1",
        "TPR", "FPR", "TNR", "STTD", "SCLF", "FinalScore", "HarmonicPRT"
    ])
    writer_metrics.writerow(metrics)
print(f"\nMetrics saved to '{OUTPUT_METRICS_FILE}'")

# --- ROC Curve Calculation ---
# Prepare the true labels (ground_truth) and prediction scores (predicted_flags) for the ROC curve
fpr, tpr, roc_thresholds = roc_curve(ground_truth, rmse_scores)
roc_auc = auc(fpr, tpr)  # Calculate the area under the ROC curve

# Plot the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend(loc='lower right')
plt.grid(True)

# Save the ROC plot
roc_plot_filename = 'roc_curve.png'
plt.savefig(roc_plot_filename, dpi=300)
plt.show()
print(f"ROC curve saved to '{roc_plot_filename}'")

from sklearn.metrics import precision_recall_curve, average_precision_score

# Calculate precision, recall, and thresholds using RMSE scores
precision, recall, pr_thresholds = precision_recall_curve(ground_truth, rmse_scores)
avg_precision = average_precision_score(ground_truth, rmse_scores)

# Plot the Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {avg_precision:.4f})')
plt.title("Precision-Recall (PR) Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc='upper right')
plt.grid(True)

# Save and show the PR plot
plt.savefig('pr_curve.png', dpi=300)
plt.show()
print(f"Precision-Recall curve saved to 'pr_curve.png'")
