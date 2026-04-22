Kitsune: A Lightweight Online Network Intrusion Detection System 🦊
Kitsune is an advanced, lightweight, and online network intrusion detection system designed for real-time anomaly detection in network traffic. It leverages an ensemble of autoencoders and a feature attention mechanism to efficiently identify novel attack patterns with minimal computational overhead. This project is built upon the core KitNET algorithm and extends it with capabilities for large-scale data processing and real-time simulation.

🌟 Features
Online Anomaly Detection: Detects anomalies in network traffic as it arrives.

Lightweight: Designed for efficiency and minimal resource consumption.

Ensemble of Autoencoders: Utilizes a KitNET core for robust anomaly scoring.

Feature Attention Mechanism: Emphasizes important features for improved detection accuracy.

Chunked Processing: Processes large PCAP files in manageable chunks using multiprocessing.

Real-time Data Simulation: Simulates live network data streams from static datasets.

PCA Dimensionality Reduction: Integrates PCA as a preprocessing step for efficiency and stability.

Flexible Feature Extraction: Supports both tshark (for speed) and scapy for parsing PCAP files.

📁 Project Structure
This repository is organized into two main folders: KitNET (the core algorithm) and Kitsune (the application and feature extraction components).

KitNET/
__init__.py: Initializes the KitNET package.

corClust.py: Implements a correlation-based incremental clustering algorithm for feature mapping.

dA.py: Defines the Denoising Autoencoder (dA) model, the building block for KitNET's ensemble layer.

KitNET.py: Contains the main KitNET class, which manages the ensemble of autoencoders and the output layer.

utils.py: Provides utility functions, including activation functions and a rolling mean class.

Kitsune/
AfterImage.py: Implements statistical calculations for network flow features.

chunk_processor.py: Defines the process_chunk function, which handles the Kitsune anomaly detection for individual data chunks.

chunked_kitsune.py: Manages the splitting of input files into chunks and orchestrates their parallel processing.

example.py: The primary script to demonstrate and run the Kitsune system in various modes (chunked or real-time simulation).

feature_attention.py: Implements a FeatureAttention mechanism to weight input features.

FeatureExtractor.py: Extracts network traffic features from PCAP/TSV files.

Kitsune.py: The main Kitsune class, integrating feature extraction, attention, and the KitNET anomaly detector.

netStat.py: Gathers network statistics using the AfterImage module.

realtime_data_simulator.py: A script to simulate real-time data streaming from a static input file.

setup.py: Used for compiling Cython extensions (e.g., AfterImage_extrapolate.pyx).

🚀 Installation
To set up and run Kitsune, follow these steps:

Clone the Repository:

git clone <repository_url>
cd <repository_name>

Install Dependencies:
Kitsune relies on several Python libraries. It is recommended to use a virtual environment.

pip install numpy scipy scapy pandas tqdm psutil scikit-learn

Note: scapy might require additional system dependencies depending on your OS.

Cython Compilation (Optional but Recommended for Performance):
Some parts of the feature extraction (like AfterImage) can be accelerated using Cython. Compile the .pyx files using setup.py:

python setup.py build_ext --inplace

If you encounter issues, ensure you have a C compiler (e.g., GCC for Linux/macOS, MSVC for Windows).

TShark (Optional but Recommended for Feature Extraction Speed):
FeatureExtractor.py can use tshark (part of Wireshark) for faster PCAP parsing. If you want to leverage this, make sure Wireshark is installed on your system and tshark is in your system's PATH.

🏃‍♀️ Usage
The example.py script serves as the main entry point for running Kitsune. It supports two primary modes: processing static files in chunks or simulating real-time data.

1. Processing Static PCAP Files (Chunked Mode)
This mode is ideal for analyzing large historical PCAP files by splitting them into smaller chunks and processing them in parallel.

Edit example.py and set run_realtime_simulation = False.
You might need to adjust the INPUT_FILE path in example.py to point to your .pcap or .tsv file.

python example.py

The script will:

Split the input PCAP into chunks.

Process each chunk using Kitsune.

Combine the results into final RMSE and confidence score CSV files in the output directory.

Clean up temporary chunk files.

2. Real-time Data Simulation
This mode demonstrates Kitsune's online capabilities by simulating a live data stream from a static CSV.

Edit example.py and set run_realtime_simulation = True.
Ensure the INPUT_FILE and OUTPUT_FILE in realtime_data_simulator.py are correctly configured for your environment. You can also adjust CHUNK_SIZE and SLEEP_TIME to control the simulation speed.

python example.py

The example.py script will:

Start realtime_data_simulator.py as a subprocess, which will continuously write chunks of data to OUTPUT_FILE (default: simulated_data.tsv).

The main example.py process will then read from this OUTPUT_FILE in a streaming fashion, apply Kitsune, and generate anomaly scores.

Anomaly scores will be written to realtime_rmse_output.csv, realtime_ms_conf_output.csv, and realtime_sec_conf_output.csv.

📊 Output
Kitsune generates the following output files:

_rmse.csv: Contains the Root Mean Squared Error (RMSE) for each processed network instance. Higher RMSE values generally indicate higher anomaly scores.

Columns: packet_idx, timestamp, rmse_score

_confidence_ms.csv: Provides a millisecond-based confidence score, typically calculated as 1/
textRMSE.

Columns: packet_idx, timestamp, confidence_ms_score

_confidence_sec.csv: Provides a second-based confidence score, typically calculated as 1/(
textRMSE
times1000).

Columns: packet_idx, timestamp, confidence_sec_score

⚙️ Configuration
Key parameters can be adjusted in example.py or by modifying the respective class constructors:

Kitsune Parameters:

maxAE: Maximum size for any autoencoder in the ensemble layer.

FMgrace: Feature mapping grace period (number of packets to learn the feature map).

ADgrace: Anomaly detection grace period (number of packets to train the anomaly detector).

pca_components: Number of components for PCA dimensionality reduction.

pca_grace_period: Number of packets to collect before fitting the PCA model.

realtime_data_simulator.py Parameters:

INPUT_FILE: Path to the source data file for simulation.

OUTPUT_FILE: Path where simulated data will be written.

CHUNK_SIZE: Number of rows written per chunk during simulation.

SLEEP_TIME: Delay between writing chunks (simulates real-time latency).

📜 License
This project is open-sourced under the MIT License.

Copyright (c) 2017-2018 Yisroel Mirsky, Yusuke Sugomori

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
