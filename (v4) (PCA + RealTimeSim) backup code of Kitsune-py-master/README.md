Kitsune: A Lightweight Online Network Anomaly Detection System 🦊
Kitsune is an efficient, online anomaly detection algorithm based on an ensemble of autoencoders. Designed to operate in real-time on network traffic, it identifies intrusions and other anomalies with low latency and computational overhead.

This repository contains the core components of the Kitsune system, including the machine learning model (KitNET), feature extraction, and a demonstration script (example.py) for both real-time and offline processing.

Key Features ✨
Online, Incremental Learning: Learns and adapts to network behavior on a per-sample basis without requiring large historical datasets.

Ensemble of Autoencoders: Uses a lightweight ensemble of Denoising Autoencoders to learn the low-dimensional structure of network features.

Correlation-Based Feature Mapping: Features are dynamically clustered based on their statistical correlation, a key step handled by the corClust module.

Performance-Optimized: Integrates a Cython-compiled module (AfterImage) for fast statistical calculations and can leverage external tools like TShark for high-speed packet parsing.

Flexible Processing Modes: Supports both a real-time simulation mode for live data streams and an offline chunked mode for analyzing large PCAP files using multiprocessing.

Dimensionality Reduction: Includes built-in support for Principal Component Analysis (PCA) to reduce the input feature space and a lightweight FeatureAttention mechanism to dynamically weight feature importance.

Getting Started 🚀
Prerequisites
Python 3.x

NumPy

Pandas

Scapy

Tqdm

Cython

(Optional) TShark for faster packet parsing

Installation
Clone the repository:

Bash

git clone [repository_url]
cd [repository_folder]
Install dependencies:

Bash

pip install numpy pandas scapy tqdm Cython
Compile the Cython Module:
The netStat.py module uses a Cython-compiled component for performance. You must build it first:

Bash

python setup.py build_ext --inplace
Running the Example
The example.py script serves as the main entry point for the system. It demonstrates how to use Kitsune in both real-time and offline modes.

To run the script:

Bash

python example.py
Configuration ⚙️
All major parameters can be configured within the AppConfig class in example.py.

RUN_REALTIME_SIMULATION: Set to True for real-time simulation using realtime_data_simulator.py or False for offline, chunked processing.

INPUT_FILE: The path to the input data file (e.g., a .pcap or .tsv file).

OUTPUT_FILE: The path where the anomaly scores and confidence files will be saved.

PACKET_LIMIT: The maximum number of packets to process. Set to None for no limit.

MAX_AE_SIZE: The maximum number of neurons in any single autoencoder within the ensemble.

FM_GRACE_PERIOD: The number of packets used to learn the feature mapping clusters.

AD_GRACE_PERIOD: The number of packets used to train the autoencoder ensemble before anomaly scores are calculated.

PCA_COMPONENTS: The number of components for PCA dimensionality reduction. Set to None to disable.

PCA_GRACE_PERIOD: The number of packets to use for fitting the PCA model.

Project Structure 📁
.

│   ├── __init__.py
│   ├── AfterImage.py
│   ├── chunked_kitsune.py
│   ├── chunk_processor.py
│   ├── example.py
│   ├── feature_attention.py
│   ├── FeatureExtractor.py
│   ├── Kitsune.py
│   ├── netStat.py
│   ├── realtime_data_simulator.py
│   └── setup.py
├── kitnet/
│   ├── __init__.py
│   ├── corClust.py
│   ├── dA.py
│   ├── KitNET.py
│   └── utils.py
└── README.md
Citation 📝
This code is based on the research paper:

Yisroel Mirsky, Tomer Doitshman, Yuval Elovici, and Asaf Shabtai. "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection." In The 25th Annual Network and Distributed System Security Symposium (NDSS), 2018.

If you use this code in your research, please cite the original paper.

License
This project is licensed under the MIT License. See the LICENSE file for details.