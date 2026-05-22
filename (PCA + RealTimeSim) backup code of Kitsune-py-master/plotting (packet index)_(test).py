import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os  # Import the os module for directory operations

# Define the path to the anomalies CSV file
anomalies_file = r"Results\kitsune_anomalies_for_Dec2019_00001_20191206102207_with_maxAE=10_FMgrace=5000_ADgrace=10000_number_of_packets=1000000.csv"

# Assume you have another file containing the RMSE values.
# Replace this with the actual path to your RMSE data.
# For demonstration, we'll create some dummy RMSE data.
rmse_file = "rmse_values.npy"
try:
    rmse_data = np.load(rmse_file)
except FileNotFoundError:
    print(f"Warning: '{rmse_file}' not found. Using dummy RMSE data.")
    # Replace these values with your actual FMgrace and ADgrace
    FMgrace = 5000
    ADgrace = 10000
    total_packets = 1000000
    rmse_data = np.random.rand(total_packets) * 0.1 + (np.arange(total_packets) > (FMgrace + ADgrace)) * 0.5

# Load the anomalies data to get the attack packet indices
try:
    anomalies_df = pd.read_csv(anomalies_file)
    attack_indices = anomalies_df['PacketIndex'].tolist()
except FileNotFoundError:
    print(f"Error: '{anomalies_file}' not found. Cannot plot attack boundaries.")
    attack_indices = []

# Replace these values with your actual FMgrace and ADgrace from the training phase
FMgrace = 5000
ADgrace = 10000

# Calculate the threshold (assuming RMSEs from the full dataset are available)
# For demonstration, we'll use the dummy data to calculate a threshold.
# In a real scenario, you should load the RMSEs calculated during the Kitsune run.
threshold = np.mean(rmse_data[FMgrace + ADgrace:]) + 2 * np.std(rmse_data[FMgrace + ADgrace:])

# Prepare x-axis values (packet indices after the grace periods)
x_values = range(FMgrace + ADgrace, len(rmse_data))
rmse_to_plot = rmse_data[FMgrace + ADgrace:]

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(x_values, rmse_to_plot, color='violet', label='RMSE')

# Plot the threshold line
plt.axhline(y=threshold, color='green', linestyle='--', linewidth=1, label=f'Threshold (mean + 2*SD)')

# Plot vertical lines for attack boundaries
for attack_index in attack_indices:
    if attack_index >= (FMgrace + ADgrace) and attack_index < len(rmse_data):
        plt.axvline(x=attack_index - (FMgrace + ADgrace), color='orange', linestyle=':', linewidth=0.75, label='Attack' if attack_index == attack_indices[0] else "")

# Customize the plot
plt.yscale("log")
plt.title("RMSE Anomaly Scores with Threshold and Attack Boundaries")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Packet Index (after grace periods)")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)

# Create the 'Plots' folder if it doesn't exist
plots_folder = "Plots"
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# Save the plot to the 'Plots' folder
plot_filename = os.path.join(plots_folder, f"rmse_plot_with_colored_threshold_and_attacks_of_{total_packets}packets.png")
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.show()

print(f"Plot saved to '{plot_filename}'")