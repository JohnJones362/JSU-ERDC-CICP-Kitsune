import csv
import datetime
import os

def save_results(import_file, maxAE, FMgrace, ADgrace, packet_limit, anomalies, time_buckets, second_buckets, output_folder="Results"):
    # Ensure folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save anomalies to file
    anomaly_path = f"{output_folder}/kitsune_anomalies_for_{import_file}_with_maxAE={maxAE}_FMgrace={FMgrace}_ADgrace={ADgrace}_packet_limit={packet_limit}.csv"
    with open(anomaly_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["PacketIndex", "Attack Time", "RMSE", "Features"])
        for anomaly in anomalies:
            readable_time = datetime.datetime.fromtimestamp(anomaly[1]).strftime('%Y-%m-%d %H:%M:%S.%f')
            writer.writerow([anomaly[0], readable_time, anomaly[2], anomaly[3]])
    print(f"Anomalies saved to {anomaly_path}")

    # Save millisecond confidence scores
    ms_path = f"{output_folder}/kitsune_confidence_scores_ms_{import_file}_with_maxAE={maxAE}_FMgrace={FMgrace}_ADgrace={ADgrace}_packet_limit={packet_limit}.csv"
    with open(ms_path, "w", newline='') as f_ms:
        writer_ms = csv.writer(f_ms)
        writer_ms.writerow(["Time(ms)", "Total", "Anomalies", "ConfidenceScore"])
        for ts in sorted(time_buckets.keys()):
            total = time_buckets[ts]['total']
            anomalies_count = time_buckets[ts]['anomaly']
            confidence = anomalies_count / total if total else 0
            readable_time = datetime.datetime.fromtimestamp(ts / 1000.0).strftime('%Y-%m-%d %H:%M:%S.%f')
            writer_ms.writerow([readable_time, total, anomalies_count, f"{confidence:.4f}"])

    # Save second confidence scores
    sec_path = f"{output_folder}/kitsune_confidence_scores_sec_{import_file}_with_maxAE={maxAE}_FMgrace={FMgrace}_ADgrace={ADgrace}_packet_limit={packet_limit}.csv"
    with open(sec_path, "w", newline='') as f_sec:
        writer_sec = csv.writer(f_sec)
        writer_sec.writerow(["Time(s)", "Total", "Anomalies", "ConfidenceScore"])
        for ts in sorted(second_buckets.keys()):
            total = second_buckets[ts]['total']
            anomalies_count = second_buckets[ts]['anomaly']
            confidence = anomalies_count / total if total else 0
            readable_time = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            writer_sec.writerow([readable_time, total, anomalies_count, f"{confidence:.4f}"])
