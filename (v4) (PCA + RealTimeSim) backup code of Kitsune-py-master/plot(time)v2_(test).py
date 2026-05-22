import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import matplotlib.dates as mdates
import os  # Import the os module for directory operations

# Define the path to the anomalies CSV file
anomalies_file = os.path.join("Results", "kitsune_anomalies_for_Dec2019_00001_20191206102207_with_maxAE=10_FMgrace=5000_ADgrace=10000_number_of_packets=1000000.csv")

# Assume you have another file containing the RMSE values and their timestamps.
# The file should have two columns: 'timestamp' and 'rmse'.
rmse_file = "rmse_with_timestamps.npy"
try:
    rmse_data_with_timestamps = np.load(rmse_file)
    rmse_df = pd.DataFrame(rmse_data_with_timestamps, columns=['timestamp', 'rmse'])
    rmse_df['timestamp'] = pd.to_datetime(rmse_df['timestamp'])
    rmse_df = rmse_df.sort_values(by='timestamp').reset_index(drop=True)
    timestamps = rmse_df['timestamp'].tolist()
    rmse_values = rmse_df['rmse'].tolist()
except FileNotFoundError:
    print(f"Warning: '{rmse_file}' not found. Using dummy RMSE data with time.")
    FMgrace = 5000
    ADgrace = 10000
    total_packets = 1000000
    start_time_attack = datetime.datetime(2025, 5, 5, 10, 5, 0)
    end_time_attack = datetime.datetime(2025, 5, 5, 13, 30, 0)
    time_range = end_time_attack - start_time_attack
    timestamps = [start_time_attack + i * time_range / total_packets for i in range(total_packets)]
    rmse_values = np.random.rand(total_packets) * 0.1 + (np.array([(ts - start_time_attack).total_seconds() for ts in timestamps]) > (FMgrace + ADgrace) * 0.1) * 0.5 # Simulate increase over time

# Load the anomalies data to get the attack timestamps
attack_times = []
try:
    anomalies_df = pd.read_csv(anomalies_file)
    attack_times = pd.to_datetime(anomalies_df['Attack Time']).tolist()
except FileNotFoundError:
    print(f"Error: '{anomalies_file}' not found. Cannot plot attack boundaries.")
    attack_times = []

# Replace these values with your actual FMgrace and ADgrace from the training phase
FMgrace = 5000
ADgrace = 10000

# Calculate the threshold based on RMSE values after the grace periods
start_index_grace = FMgrace + ADgrace
if len(rmse_values) > start_index_grace:
    threshold = np.mean(rmse_values[start_index_grace:]) + 2 * np.std(rmse_values[start_index_grace:])
else:
    threshold = None
    print("Warning: Not enough RMSE data after grace periods to calculate threshold.")

# Prepare data for plotting in a 10-minute attack window in milliseconds
start_time_plot = datetime.datetime(2019, 5, 5, 10, 20, 0)
end_time_plot = datetime.datetime(2019, 5, 5, 10, 30, 0)

# Find indices that match the desired time range
indices_to_plot = [i for i, ts in enumerate(timestamps) if start_time_plot <= ts <= end_time_plot]
timestamps_to_plot = [timestamps[i] for i in indices_to_plot]
rmse_to_plot = [rmse_values[i] for i in indices_to_plot]

# Convert timestamps to milliseconds since start_time_plot
time_ms = [(ts - start_time_plot).total_seconds() * 1000 for ts in timestamps_to_plot]


# Create the plot
plt.figure(figsize=(16, 8))
plt.plot(time_ms, rmse_to_plot, color='violet', label='RMSE')

# Plot threshold line
if threshold is not None:
    plt.axhline(y=threshold, color='green', linestyle='--', linewidth=1, label='Threshold (mean + 2*SD)')

# Plot attack timestamps if within this window
first_attack = True
for attack_time in attack_times:
    if start_time_plot <= attack_time <= end_time_plot:
        x_ms = (attack_time - start_time_plot).total_seconds() * 1000
        plt.axvline(x=x_ms, color='orange', linestyle=':', linewidth=0.75,
                    label='Attack' if first_attack else "")
        first_attack = False

# Customize plot
plt.yscale("log")
plt.title("RMSE Anomaly Scores (10:20–10:30) in Milliseconds")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Time (ms since 10:20:00)")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# Format the x-axis to show time in smaller intervals
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.gca().xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
plt.xticks(rotation=45, ha='right')
plt.xlim(start_time_plot, end_time_plot) # Set the x-axis limits

plt.tight_layout()

# Create the 'Plots' folder if it doesn't exist
plots_folder = "Plots"
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Save the plot to the 'Plots' folder
plot_filename = os.path.join(plots_folder, f"rmse_plot_over_attack_time_of_{total_packets}packets.png")
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to '{plot_filename}'")