import pandas as pd
import matplotlib.pyplot as plt

# === Load attack boundaries ===
df_attacks = pd.read_csv('A6-2015-12/df_attacks.csv', parse_dates=['DATETIME'])

# Split into start and stop rows
df_starts = df_attacks[df_attacks['attack_number'].str.contains('start')].reset_index(drop=True)
df_stops = df_attacks[df_attacks['attack_number'].str.contains('stop')].reset_index(drop=True)

# === Load anomaly data ===
df_data = pd.read_csv(
    'Results/kitsune_anomalies_for_Dec2019_00001_20191206102207_with_maxAE=10_FMgrace=5000_ADgrace=10000_number_of_packets=1000000.csv',
    parse_dates=['Attack Time']
)

# === Plot RMSE over time ===
plt.figure(figsize=(16, 5))
plt.plot(df_data['Attack Time'], df_data['RMSE'], label='RMSE (Anomaly Score)', color='blue')

# === Overlay shaded attack regions ===
for start, stop in zip(df_starts['DATETIME'], df_stops['DATETIME']):
    plt.axvspan(start, stop, color='red', alpha=0.3)

# === Final plot formatting ===
plt.xlabel('Time')
plt.ylabel('RMSE')
plt.title('Anomaly Scores with Attack Intervals (Shaded)')
plt.legend(['RMSE', 'Attack Interval'])
plt.grid(True)
plt.tight_layout()

# === Save the plot ===
plt.savefig('anomaly_with_attack_intervals.png', dpi=300)  # High resolution

plt.show()
