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
import pickle # Added for saving/loading the model
import logging # Added for logging

# Set up logging for Kitsune.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define a model version for persistence
MODEL_VERSION = "1.1.0" # Increased from 1.0.0 to include PCA and versioning

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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER D...

class Kitsune:
    def __init__(self, file_path=None, limit=np.inf,
                 n_features=None, m_autoencoder_size=10,
                 FM_grace_period=10000, AD_grace_period=10000,
                 learning_rate=0.1, hidden_ratio=0.75,
                 feature_map=None,
                 pca_components=None, pca_grace_period=None,
                 live_stream=False, delimiter='\t'):
        """
        Initializes the Kitsune anomaly detection system.

        Args:
            file_path (str, optional): Path to the input pcap or tsv file. Required if not live_stream.
            limit (int, optional): Maximum number of packets to process. Defaults to infinity.
            n_features (int, optional): The total number of features in the input data.
                                        Required if starting with a pre-trained model or live stream.
            m_autoencoder_size (int): Max size for any autoencoder in the ensemble.
            FM_grace_period (int): Number of instances for Feature Mapping grace period.
            AD_grace_period (int): Number of instances for Anomaly Detection grace period.
            learning_rate (float): Learning rate for autoencoders.
            hidden_ratio (float): Ratio of hidden to visible neurons in autoencoders.
            feature_map (list, optional): Predefined feature map (list of lists of feature indices).
                                         If None, feature map is learned online.
            pca_components (int, optional): Number of components for PCA dimensionality reduction.
                                            If None, PCA is not used.
            pca_grace_period (int, optional): Number of packets to collect for PCA fitting.
                                             If None, defaults to AD_grace_period.
            live_stream (bool): If True, indicates live data stream, no file parsing.
            delimiter (str): Delimiter for TSV files. Defaults to tab.
        """
        self.FE = None
        if file_path or live_stream:
            self.FE = FeatureExtractor.FE(file_path=file_path, limit=limit, live_stream=live_stream, delimiter=delimiter)
            if n_features is None: # Determine n_features from FE if not provided explicitly
                headers = self.FE.get_headers()
                if headers:
                    n_features = len(headers)
                else:
                    raise ValueError("Could not determine number of features from file. Please provide n_features explicitly or ensure the input file has headers.")

        if n_features is None:
            raise ValueError("n_features must be provided for Kitsune initialization if no file_path or live_stream is given.")

        self.n_features = n_features
        self.m = m_autoencoder_size
        self.FM_grace_period = FM_grace_period
        self.AD_grace_period = AD_grace_period
        self.lr = learning_rate
        self.hr = hidden_ratio
        self.feature_map = feature_map # Pass to KitNET if provided

        self.AnomDetector = KitNET(n=self.n_features, m=self.m,
                                  AD_grace_period=self.AD_grace_period,
                                  FM_grace_period=self.FM_grace_period,
                                  learning_rate=self.lr,
                                  hidden_ratio=self.hr,
                                  feature_map=self.feature_map)

        # Feature Attention Mechanism
        self.feature_attention = FeatureAttention(num_features=self.n_features)
        self.use_attention = False # Currently not directly controlled by init param, assumed to be part of processing logic

        # PCA Integration
        self.pca_components = pca_components
        self.pca_grace_period = pca_grace_period if pca_grace_period is not None else AD_grace_period
        self.pca_data_buffer = [] # Buffer to collect data for PCA fitting
        self.pca = None
        self.pca_fitted = False
        if self.pca_components is not None and self.pca_components < self.n_features:
            self.pca = PCA(n_components=self.pca_components)
            logging.info(f"PCA initialized with {self.pca_components} components. Grace period: {self.pca_grace_period} samples.")
        elif self.pca_components is not None and self.pca_components >= self.n_features:
            logging.warning(f"PCA components ({self.pca_components}) must be less than n_features ({self.n_features}). PCA will be skipped.")
            self.pca_components = None # Disable PCA

    def get_next_vector_from_extractor(self):
        """
        Retrieves the next feature vector and timestamp from the FeatureExtractor.
        Returns (feature_vector, timestamp) or (None, None) if no more data.
        """
        if self.FE:
            return self.FE.get_next_vector()
        return None, None

    def _fit_pca(self):
        """
        Fits the PCA model using the collected data buffer.
        """
        if self.pca is not None and len(self.pca_data_buffer) >= self.pca_grace_period:
            logging.info(f"Fitting PCA with {len(self.pca_data_buffer)} samples...")
            try:
                self.pca.fit(np.array(self.pca_data_buffer))
                self.pca_fitted = True
                logging.info(f"PCA fitted. Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.2f}")
                # Clear buffer after fitting
                self.pca_data_buffer = []
            except Exception as e:
                logging.error(f"Error fitting PCA: {e}")
                self.pca_fitted = False # Mark as not fitted if error occurs

    def process_batch(self, input_batch=None):
        """
        Processes a batch of input samples through the Kitsune model.
        This method handles feature attention, PCA (if enabled), and passes data to KitNET.

        Args:
            input_batch (np.ndarray, optional): A 2D NumPy array where each row is a sample.
                                                Shape: (n_samples, n_features).
                                                If None, retrieves data from FeatureExtractor.

        Returns:
            np.ndarray: A 1D NumPy array of RMSE scores for each sample in the batch.
                        Returns -1.0 if not yet out of AD grace period, or -2.0 if in FM grace period,
                        or -3.0 if in PCA grace period.
        """
        if input_batch is None:
            # If no batch provided, pull from FeatureExtractor until batch is full or FE is exhausted
            vectors_from_extractor = []
            timestamps = []
            while len(vectors_from_extractor) < self.AD_grace_period * 2: # Arbitrary large enough batch size
                vec, ts = self.get_next_vector_from_extractor()
                if vec is None:
                    break
                vectors_from_extractor.append(vec)
                timestamps.append(ts)
            input_batch = np.array(vectors_from_extractor)
            if len(input_batch) == 0:
                return np.array([]) # No data to process

        num_samples = input_batch.shape[0]
        rmses_for_batch_output = np.full(num_samples, -1.0) # Default to -1 (AD grace period)

        # Separate samples for KitNET processing and PCA training if applicable
        processed_for_kitnet_batch = []
        kitnet_rmses_indices = []

        for i_in_batch in range(num_samples):
            current_vector = input_batch[i_in_batch, :]

            # 1. Apply Feature Attention (if enabled)
            x_attended_single = self.feature_attention.apply_attention(current_vector)

            # 2. Handle PCA (if configured)
            if self.pca is not None:
                if not self.pca_fitted:
                    self.pca_data_buffer.append(x_attended_single)
                    rmses_for_batch_output[i_in_batch] = -3.0 # Indicate PCA grace period
                    # Attempt to fit PCA if buffer is full
                    if len(self.pca_data_buffer) >= self.pca_grace_period:
                        self._fit_pca()
                    continue # Skip KitNET processing if still in PCA grace period or not fitted
                elif self.pca_fitted:
                    # Transform current sample using fitted PCA
                    x_processed_for_kitnet_single = self.pca.transform(x_attended_single.reshape(1, -1)).flatten()
            else:
                # If PCA is not enabled, the attended vector is the input for KitNET
                x_processed_for_kitnet_single = x_attended_single

            # Collect samples that are ready for KitNET
            processed_for_kitnet_batch.append(x_processed_for_kitnet_single)
            kitnet_rmses_indices.append(i_in_batch) # Store original index for mapping back

        # If all samples were for PCA training, return early with -3s
        if not processed_for_kitnet_batch: # This means all samples were -3
            return rmses_for_batch_output.tolist()

        # 3. Pass the (attended & PCA-processed) batch to KitNET
        kitnet_rmses_from_batch_processing = self.AnomDetector.process_batch(np.array(processed_for_kitnet_batch))

        # 4. Map KitNET's results back to the original batch structure
        for i, original_idx in enumerate(kitnet_rmses_indices): # Iterate over indices of processed samples
            rmses_for_batch_output[original_idx] = kitnet_rmses_from_batch_processing[i]


        return rmses_for_batch_output.tolist() # Convert back to list as example.py expects it


    def get_latest_packet_time(self):
        """
        Gets the timestamp of the last processed packet from the FeatureExtractor.
        """
        if self.FE:
            return self.FE.get_latest_timestamp()
        return None

    def get_weights(self):
        """
        Extracts all current weights and state from the Kitsune model.
        Returns a dictionary containing the model's state.
        """
        model_state = {
            "model_version": MODEL_VERSION, # Add model version
            "n_features": self.n_features,
            "m_autoencoder_size": self.m,
            "FM_grace_period": self.FM_grace_period,
            "AD_grace_period": self.AD_grace_period,
            "learning_rate": self.lr,
            "hidden_ratio": self.hr,
            "feature_map": self.feature_map, # This might be None if learned online
            "kitnet_state": self.AnomDetector.get_weights(),
            "feature_attention_W1": self.feature_attention.W1.tolist(),
            "feature_attention_b1": self.feature_attention.b1.tolist(),
            "feature_attention_W2": self.feature_attention.W2.tolist(),
            "feature_attention_b2": self.feature_attention.b2.tolist(),
            "pca_components_config": self.pca_components, # Original requested PCA components
            "pca_grace_period": self.pca_grace_period,
            "pca_fitted": self.pca_fitted,
        }
        if self.pca_fitted and self.pca is not None:
            model_state["pca_components_"] = self.pca.components_.tolist()
            model_state["pca_mean_"] = self.pca.mean_.tolist()
        else:
            # Explicitly set to None if PCA was not fitted or not used
            model_state["pca_components_"] = None
            model_state["pca_mean_"] = None

        return model_state

    def set_weights(self, model_state):
        """
        Restores the Kitsune model state from a dictionary.

        Args:
            model_state (dict): Dictionary containing the model's state.
        """
        # Basic version compatibility check
        if "model_version" not in model_state:
            logging.warning("Loading an old model state without versioning. Compatibility issues may arise.")
        elif model_state["model_version"] != MODEL_VERSION:
            logging.warning(f"Model version mismatch! Loaded: {model_state['model_version']}, Current: {MODEL_VERSION}. Compatibility issues may arise.")

        # Validate n_features - crucial for correct model loading
        if 'n_features' not in model_state or not isinstance(model_state["n_features"], int) or model_state["n_features"] <= 0:
            raise ValueError("Invalid model state: 'n_features' missing or invalid.")
        self.n_features = model_state['n_features']

        # Set basic parameters
        self.m = model_state.get("m_autoencoder_size", self.m)
        self.FM_grace_period = model_state.get("FM_grace_period", self.FM_grace_period)
        self.AD_grace_period = model_state.get("AD_grace_period", self.AD_grace_period)
        self.lr = model_state.get("learning_rate", self.lr)
        self.hr = model_state.get("hidden_ratio", self.hr)
        self.feature_map = model_state.get("feature_map", None)

        # Restore KitNET state
        if "kitnet_state" not in model_state:
            raise ValueError("Invalid model state: 'kitnet_state' missing.")
        # Ensure KitNET is re-initialized with potentially updated parameters before setting weights
        self.AnomDetector = KitNET(n=self.n_features, m=self.m,
                                  AD_grace_period=self.AD_grace_period,
                                  FM_grace_period=self.FM_grace_period,
                                  learning_rate=self.lr,
                                  hidden_ratio=self.hr,
                                  feature_map=self.feature_map)
        self.AnomDetector.set_weights(model_state["kitnet_state"])
        logging.info("KitNET state loaded.")

        # Restore Feature Attention weights
        try:
            self.feature_attention = FeatureAttention(num_features=self.n_features) # Reinitialize
            self.feature_attention.W1 = np.array(model_state["feature_attention_W1"])
            self.feature_attention.b1 = np.array(model_state["feature_attention_b1"])
            self.feature_attention.W2 = np.array(model_state["feature_attention_W2"])
            self.feature_attention.b2 = np.array(model_state["feature_attention_b2"])
            logging.info("FeatureAttention weights loaded.")
        except KeyError as e:
            logging.warning(f"Could not load FeatureAttention weights (missing key: {e}). Re-initializing with default.")
            self.feature_attention = FeatureAttention(num_features=self.n_features)


        # Restore PCA state
        self.pca_components = model_state.get("pca_components_config", None)
        self.pca_grace_period = model_state.get("pca_grace_period", self.AD_grace_period) # Fallback
        self.pca_fitted = model_state.get("pca_fitted", False)

        if self.pca_components is not None and self.pca_components < self.n_features:
            self.pca = PCA(n_components=self.pca_components)
            if self.pca_fitted:
                pca_components_loaded = model_state.get("pca_components_")
                pca_mean_loaded = model_state.get("pca_mean_")
                if pca_components_loaded is not None and pca_mean_loaded is not None:
                    self.pca.components_ = np.array(pca_components_loaded)
                    self.pca.mean_ = np.array(pca_mean_loaded)
                    logging.info("PCA components and mean loaded successfully.")
                else:
                    self.pca_fitted = False # Mark as not fitted if data is missing
                    logging.warning("PCA was marked as fitted, but components or mean are missing in the saved state. PCA will not be used.")
            else:
                logging.info("PCA initialized but not fitted (as per saved state). Will re-fit if grace period is met.")
        else:
            self.pca = None
            self.pca_fitted = False
            logging.info("PCA not used or components invalid as per saved state.")

        # FeatureExtractor is not part of the trained model state, it's for data input.
        # So, self.FE is not loaded/saved here.

    def save_model(self, filepath):
        """Saves the Kitsune model state to a pickle file."""
        model_state = self.get_weights()
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
            logging.info(f"Kitsune model saved to {filepath}")
        except Exception as e:
            logging.error(f"Error saving Kitsune model to {filepath}: {e}")

    @classmethod
    def load_model(cls, filepath):
        """
        Loads a Kitsune model state from a pickle file.
        Performs initial validation before creating the instance.
        """
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)

            # --- Pre-instantiation Validation ---
            if "n_features" not in model_state or not isinstance(model_state["n_features"], int) or model_state["n_features"] <= 0:
                raise ValueError(f"Model file '{filepath}' is corrupted or invalid: 'n_features' is missing or invalid.")
            if "kitnet_state" not in model_state:
                raise ValueError(f"Model file '{filepath}' is corrupted or invalid: 'kitnet_state' is missing.")

            # Create a dummy Kitsune instance (without FeatureExtractor) to load the state into
            # Provide n_features which is crucial for initialization.
            instance = cls(n_features=model_state['n_features'])
            instance.set_weights(model_state)
            logging.info(f"Kitsune model loaded from {filepath}. Version: {model_state.get('model_version', 'N/A')}")
            return instance

        except FileNotFoundError:
            logging.error(f"Model file not found at {filepath}")
            raise
        except (pickle.UnpicklingError, ValueError, KeyError) as e:
            logging.error(f"Error loading or parsing Kitsune model from {filepath}: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred during model loading: {e}")
            raise
