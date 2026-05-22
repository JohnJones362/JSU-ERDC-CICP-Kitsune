Kitsune: A Lightweight Online Network Intrusion Detection System 🦊
Kitsune is an advanced, lightweight, and online network intrusion detection system (NIDS) designed to detect anomalies in real-time network traffic. It leverages an ensemble of autoencoders (KitNET) and various statistical methods, including feature attention and PCA, to efficiently learn and identify abnormal patterns.

🌟 Features
Online Learning: Adapts to new network behaviors in real-time.

Lightweight: Designed for efficient operation with minimal computational overhead.

Ensemble Autoencoders (KitNET): Utilizes a hierarchical autoencoder architecture for robust anomaly detection.

Feature Extraction (AfterImage): Extracts a comprehensive set of network flow features.

Feature Attention Mechanism: Adaptively emphasizes critical features and reduces high-dimensional noise.

PCA for Dimensionality Reduction: Improves efficiency and stability by reducing feature space.

Chunked Processing: Handles large PCAP/TSV files by splitting them into manageable chunks and processing them in parallel using multiprocessing.

Detailed Output: Generates RMSE values, confidence scores (millisecond and second), anomaly reports, and thresholds.

🛠️ Installation
Clone the Repository:
(Assuming you have the files locally, this step is for future reference or if moving the project.)

Prerequisites:
Ensure you have Python 3.x installed. The following libraries are required:

numpy

scipy

scapy

pandas

scikit-learn

tqdm

psutil

Cython (for AfterImage_extrapolate.pyx)

You can install most of them using pip:

pip install numpy scipy scapy pandas scikit-learn tqdm psutil cython

Note: For scapy, you might need to run pip install scapy or consult its documentation for platform-specific requirements.

Cython Compilation:
The AfterImage.py module utilizes a Cython extension (AfterImage_extrapolate.pyx) for performance. You need to compile this extension. Navigate to the project root directory (where setup.py is located) and run:

python setup.py build_ext --inplace

This command will compile the Cython file and create a shared library (e.g., .so or .pyd file) in the same directory, which Python can then import.

TShark (Optional but Recommended):
For faster PCAP parsing, Wireshark (which includes tshark.exe on Windows) is highly recommended. If tshark is not found in your system's PATH, the system will fall back to using scapy for parsing, which is significantly slower.

🚀 Usage
The primary entry point for running Kitsune is the example.py script.

Configuration
Open example.py and adjust the parameters in the main() function:

num_chunks: Number of parallel processes for chunking (scales with CPU cores).

packet_limit: (Optional) Limit the total number of packets to process. Set to None for unlimited.

import_file: The base name of your input traffic file (e.g., "ciniminer_sample_traffic_log" for ciniminer_sample_traffic_log.csv).

use_tsv: Set to True if your input file is a .tsv (pre-parsed by tshark), False if it's a .pcap or .pcapng.

KitNET Parameters:

maxAE: Maximum size for any autoencoder in the ensemble layer.

FMgrace: Feature mapping grace period (number of packets for initial feature clustering).

ADgrace: Anomaly detection grace period (number of packets for anomaly thresholding).

PCA Parameters (Optimization):

pca_components: Number of components for PCA. Set to None to disable PCA.

pca_grace_period: Number of packets to collect for PCA fitting. It's recommended to set this to be at least FMgrace.

Running the System
Execute the example.py script from your terminal:

python example.py

The script will:

Check system resources (disk space, RAM).

Split the input PCAP/TSV file into chunks.

Process each chunk in parallel using Kitsune.

Combine the results (RMSEs, confidence scores).

Calculate anomaly thresholds.

Detect anomalies based on the calculated threshold.

Save all results to the Results/ directory, organized by type (anomalies, confidence, rmse, thresholds, logs).

📂 Project Structure

│   ├── AfterImage.py               # Cython-optimized incremental statistics for feature extraction
│   ├── chunk_processor.py          # Processes individual data chunks with Kitsune
│   ├── chunked_kitsune.py          # Handles splitting files and managing chunk processing
│   ├── ciniminer_sample_traffic_log.csv # Sample network traffic log
│   ├── example.py                  # Main script to run the Kitsune IDS
│   ├── feature_attention.py        # Implements a lightweight feature attention mechanism
│   ├── FeatureExtractor.py         # Extracts network features from PCAP/TSV files
│   ├── Kitsune.py                  # Orchestrates feature extraction, attention, PCA, and KitNET
│   ├── netStat.py                  # Collects network statistics using AfterImage
│   └── setup.py                    # Setup script for Cython compilation
├── kitnet/
│   ├── __init__.py                 # Package initializer
│   ├── corClust.py                 # Correlation-based incremental clustering for feature mapping
│   ├── dA.py                       # Denosing Autoencoder implementation
│   └── KitNET.py                   # Ensemble of autoencoders for anomaly detection
│   └── utils.py                    # Utility functions (sigmoid, softmax, etc.)
└── README.md                       # This file

📊 Outputs
The Results/ directory will contain the following folders with timestamped output files:

anomalies/: CSV file listing detected anomalies (PacketIndex, Timestamp, RMSE).

confidence/: CSV files for millisecond and second-based confidence scores.

rmse/: CSV file containing all calculated Root Mean Square Errors (RMSEs).

thresholds/: CSV file detailing the calculated anomaly thresholds (Max RMSE and Statistical).

logs/: Text files with process completion summaries and any errors encountered.

chunks/: Temporary directory for intermediate chunk files (cleaned up after run).

📜 Credits & License
This project is based on the Kitsune research paper: "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection" by Yisroel Mirsky et al. (NDSS'18).

Portions of the dA.py code are adapted from Yusuke Sugomori's DeepLearning GitHub repository.

This project is open-sourced under the MIT License. Please see the individual file headers for specific copyright and licensing information.