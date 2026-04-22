# 6/27/2025
# file now incorporates the FeatureAttention module
# to preprocess feature vectors before they are fed
# into the KitNET autoencoders.

import numpy as np
from FeatureExtractor import *
from KitNET.KitNET import KitNET
# Import the new feature attention module
from feature_attention import FeatureAttention
from sklearn.decomposition import PCA # Added for PCA integration

# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class Kitsune:
    def __init__(self,file_path,limit,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75, pca_components: int = None, pca_grace_period: int = None, live_stream=False, input_delimiter='\t'): # Added live_stream and input_delimiter
        """
        Initializes the Kitsune system.

        Args:
            file_path (str): Path to the network traffic file (pcap or tsv/csv).
            limit (int): Limit on the number of packets to process.
            max_autoencoder_size (int): Maximum size for any autoencoder in KitNET.
            FM_grace_period (int): Feature mapping grace period (for KitNET's initial learning).
            AD_grace_period (int): Anomaly detection grace period (for KitNET's anomaly thresholding).
            learning_rate (float): Learning rate for autoencoders.
            hidden_ratio (float): Ratio of hidden to visible neurons in autoencoders.
            pca_components (int, optional): Number of components for PCA dimensionality reduction.
                                            If None, PCA is not applied.
            pca_grace_period (int, optional): Number of packets to collect for PCA fitting.
                                              Defaults to FM_grace_period if not provided.
            live_stream (bool): If True, the FeatureExtractor will continuously read from the file
                                as it's being appended to.
            input_delimiter (str): The delimiter character used in the input TSV/CSV file.
        """
        # Initialize the packet feature extractor (AfterImage).
        self.FE = FE(file_path,limit, live_stream=live_stream, delimiter=input_delimiter) # Pass live_stream and delimiter

        # Get the number of features after FE initialization
        num_features_from_fe = self.FE.get_num_features()
        print(f"Detected {num_features_from_fe} features from FeatureExtractor.")

        # Initialize the feature attention mechanism.
        # It's initialized here after FE so we can get the number of features.
        # 'hidden_size=4' is chosen based on the note "shallower attention layers (2–4)".
        self.feature_attention = FeatureAttention(num_features_from_fe, hidden_size=4)
        print(f"FeatureAttention initialized with {num_features_from_fe} input features and {self.feature_attention.hidden_size} hidden size.")


        self.pca = None
        self.pca_fitted = False
        self.pca_data_buffer = [] # To store samples for PCA fitting

        # Determine the number of features for KitNET after potential PCA
        num_features_for_kitnet = num_features_from_fe
        if pca_components is not None and pca_components > 0:
            # Rationale: Applying dimensionality reduction (PCA) can improve performance and
            # numerical stability by reducing the input dimensionality for KitNET's
            # autoencoders, as suggested by the research paper for robust IDS.
            # This also helps reduce noise and focus on most important variance.
            if pca_components > num_features_for_kitnet:
                print(f"Warning: PCA components ({pca_components}) cannot be greater than original features ({num_features_for_kitnet}). Setting PCA components to original features.")
                self.pca_components = num_features_for_kitnet
            else:
                self.pca_components = pca_components

            self.pca = PCA(n_components=self.pca_components)
            self.pca_grace_period = pca_grace_period if pca_grace_period is not None else FM_grace_period
            if self.pca_grace_period is None or self.pca_grace_period <= 0:
                raise ValueError("pca_grace_period must be a positive integer if pca_components is specified.")
            num_features_for_kitnet = self.pca_components # KitNET will receive PCA-transformed features
            print(f"PCA enabled with {self.pca_components} components. PCA grace period: {self.pca_grace_period} packets.")

        # Initialize the Kitsune anomaly detector with potentially reduced feature set
        # KitNET now receives features that have been processed by the attention mechanism
        # and potentially by PCA.
        self.AnomDetector = KitNET(num_features_for_kitnet,max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate,hidden_ratio)


    def proc_next_packet(self):
        """
        Processes the next packet from the input file.

        Returns:
            float: The RMSE (Reconstruction Error) from the anomaly detector,
                   -1 if no more packets are available or an error occurs,
                   or -2 if PCA is being fitted (no anomaly score yet).
        """
        # Create the raw feature vector from the packet.
        x = self.FE.get_next_vector()
        if len(x) == 0:
            return -1 # Error or no packets left

        # Process the raw feature vector through the attention mechanism.
        # This step weights the features, allowing the model to focus on
        # more relevant information and potentially reduce noise, as described
        # in the notes ("adaptively emphasize critical features and reduce high-dimensional noise").
        x_attended = self.feature_attention.process(x)

        # --- PCA Integration (Optimization from research paper) ---
        # Rationale: Applying dimensionality reduction (PCA) can improve performance
        # and numerical stability by reducing the input dimensionality for KitNET's
        # autoencoders, as suggested by the research paper for robust IDS.
        # This also helps reduce noise and focus on most important variance.
        x_processed = x_attended # Default to attended vector
        if self.pca is not None:
            if not self.pca_fitted:
                self.pca_data_buffer.append(x_attended)
                if len(self.pca_data_buffer) >= self.pca_grace_period:
                    # Fit PCA model after collecting enough samples
                    print(f"Fitting PCA model with {len(self.pca_data_buffer)} samples...")
                    try:
                        self.pca.fit(np.array(self.pca_data_buffer))
                        self.pca_fitted = True
                        print("PCA model fitted successfully.")
                        self.pca_data_buffer = None # Free up memory
                    except Exception as e:
                        print(f"Error fitting PCA: {e}. Disabling PCA.")
                        self.pca = None # Disable PCA if fitting fails
                        self.pca_data_buffer = None
                return -2 # Indicate PCA training in progress, no anomaly score yet

            # Transform the feature vector using the fitted PCA model
            # Ensure x_attended is 2D for PCA transform
            x_processed = self.pca.transform(x_attended.reshape(1, -1)).flatten()
        # --- End PCA Integration ---

        # Process the (potentially PCA-transformed) feature vector with KitNET.
        return self.AnomDetector.process(x_processed)

    def get_latest_packet_time(self):
        """
        Gets the timestamp of the last processed packet.

        Returns:
            float: Timestamp of the last packet.
        """
        return self.FE.get_latest_timestamp()

