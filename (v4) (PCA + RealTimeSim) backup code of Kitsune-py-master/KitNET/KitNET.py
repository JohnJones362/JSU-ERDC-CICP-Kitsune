import numpy as np
# import cupy as np
import KitNET.dA as AE
import KitNET.corClust as CC
import pickle
import logging

# Set up logging for KitNET.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# This class represents a KitNET machine learner.
# KitNET is a lightweight online anomaly detection algorithm based on an ensemble of autoencoders.
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
# For licensing information, see the end of this document

# --- Docstring updated to raw string to avoid invalid escape sequence warning ---
r"""
KitNET machine learner. Optimized for batch processing of input packets,
while maintaining its online, per-sample learning and anomaly detection.

n: the number of features in your input dataset (i.e., x \in R^n)
m: the maximum size of any autoencoder in the ensemble layer
AD_grace_period: the number of instances the network will learn from before producing anomaly scores
FM_grace_period: the number of instances which will be taken to learn the feature mapping. If 'None', then FM_grace_period=AD_grace_period
learning_rate: the default stochastic gradient descent learning rate for all autoencoders in the KitNET instance.
hidden_ratio: the default ratio of hidden to visible neurons. E.g., 0.75 will cause roughly a 25% compression in the hidden layer.
feature_map: One may optionally provide a feature map instead of learning one. The map must be a list,
             where the i-th entry contains a list of the feature indices to be assingned to the i-th autoencoder in the ensemble.
             For example, [[2,5,3],[4,0...
"""
# --- End Docstring update ---

class KitNET:
    def __init__(self,n,m,AD_grace_period,FM_grace_period=None,learning_rate=0.1,hidden_ratio=0.75, feature_map = None):
        self.n = n # Total number of features
        self.m = m # Max size of an AE in the ensemble
        self.AD_grace_period = AD_grace_period # Samples for Anomaly Detector training
        self.FM_grace_period = FM_grace_period if FM_grace_period is not None else AD_grace_period # Samples for Feature Map training
        self.lr = learning_rate # Learning rate for AEs
        self.hr = hidden_ratio # Hidden ratio for AEs

        # KitNET autoencoder parameters
        self.v = n # number of features in vector (same as self.n)
        # self.H = max(1, round(self.hr * self.v)) # This is not directly used for individual AE sizes

        # Incremental feature clustering (Feature Map)
        self.FM = CC.corClust(self.v) # feature map: online correlation clustering
        if feature_map is None: # If no feature_map is provided, Kitsune learns it online
            self.FM_train_complete = False
            logging.info(f"Feature Map (FM) training started for {self.FM_grace_period} samples.")
        else: # If a feature_map is provided, load it
            self.FM_train_complete = True
            self.FM.n = self.n # Ensure FM object has correct total feature count
            self.FM.clusters = feature_map # Load predefined clusters
            
            # Create the ensemble layer based on the provided feature map
            self.ensembleLayer = []
            self.ensembleLayer_autoencoder_sizes = []
            for cluster in feature_map: # Create autoencoders for each feature cluster
                params = AE.dA_params(n_visible=len(cluster), n_hidden=max(1, round(self.hr * len(cluster))), lr=self.lr)
                self.ensembleLayer.append(AE.dA(params))
                self.ensembleLayer_autoencoder_sizes.append(len(cluster))
            logging.info(f"Feature Map (FM) pre-loaded with {len(self.FM.clusters)} clusters. Ensemble layer initialized.")


        # Anomaly detector: uses an autoencoder to learn normal behavior of network traffic
        self.AD_train_complete = False # while false, KitNET will only train the anomaly detector
        self.outputLayer = None # This will be the output layer autoencoder

    def __create_AE(self, params):
        # Helper to create a single autoencoder instance with given parameters
        return AE.dA(params)

    def __update_FM(self,X):
        """
        Updates the Feature Map (correlation clustering) with a single sample.
        This operation is inherently online and per-sample.
        """
        self.FM.update(X)
        if not self.FM_train_complete and self.FM.N == self.FM_grace_period:
            self.FM_train_complete = True
            self.FM.cluster(self.m) # Perform clustering once grace period is met
            logging.info(f"Feature Map (FM) training complete after {self.FM.N} instances. Formed {len(self.FM.clusters)} clusters.")
            # Ensemble layer is created in __update_ensembleLayer on its first call after FM training is complete

    def __update_ensembleLayer(self,X):
        """
        Updates each autoencoder in the ensemble layer with the relevant sub-vector from a single sample.
        Each AE training and execution is a vectorized operation across its features.
        """
        if len(self.ensembleLayer) == 0 and self.FM_train_complete: # Initialize ensemble layer after FM training
            self.ensembleLayer = []
            self.ensembleLayer_autoencoder_sizes = []
            # Create autoencoders for each feature cluster found by FM
            for cluster in self.FM.clusters:
                params = AE.dA_params(n_visible=len(cluster), n_hidden=max(1, round(self.hr * len(cluster))), lr=self.lr)
                self.ensembleLayer.append(self.__create_AE(params))
                self.ensembleLayer_autoencoder_sizes.append(len(cluster))
            logging.info(f"Ensemble Layer initialized with {len(self.ensembleLayer)} autoencoders based on FM clusters.")
        
        S_l = np.zeros(len(self.ensembleLayer)) # Reconstruction errors for each autoencoder
        for i, AE_inst in enumerate(self.ensembleLayer):
            # Pass the sub-vector corresponding to the current autoencoder's cluster
            X_cluster = X[self.FM.clusters[i]]
            AE_inst.train(X_cluster) # dA.train uses vectorized NumPy operations internally
            S_l[i] = AE_inst.execute(X_cluster) # dA.execute uses vectorized NumPy operations internally
        return S_l

    def __update_outputLayer(self,X_ensemble_errors):
        """
        Updates the output layer autoencoder with the ensemble errors (S_l) from a single sample.
        This AE's training and execution is also a vectorized operation.
        """
        if self.outputLayer is None and self.FM_train_complete: # Initialize output layer after ensemble layer is ready
            params = AE.dA_params(n_visible=len(X_ensemble_errors), n_hidden=max(1, round(self.hr * len(X_ensemble_errors))), lr=self.lr)
            self.outputLayer = self.__create_AE(params)
            logging.info(f"Output Layer AE initialized with {len(X_ensemble_errors)} visible units.")
        
        if self.outputLayer: # Only train if initialized
            self.outputLayer.train(X_ensemble_errors)
            if not self.AD_train_complete and self.outputLayer.N == self.AD_grace_period:
                self.AD_train_complete = True
                logging.info(f"Anomaly Detection (AD) training complete after {self.outputLayer.N} instances.")
            return self.outputLayer.execute(X_ensemble_errors) # Get reconstruction error
        return 0.0 # Return 0 if output layer not yet initialized (during grace period)


    def process_batch(self, input_batch):
        """
        Processes a batch of input samples through the Kitsune model.
        This method iterates through each sample in the batch sequentially,
        reflecting Kitsune's online processing paradigm.
        Internal operations on single samples are heavily vectorized.

        Args:
            input_batch (np.ndarray): A 2D NumPy array where each row is a sample.
                                     Shape: (n_samples, n_features).

        Returns:
            np.ndarray: A 1D NumPy array of RMSE scores for each sample in the batch.
                        Returns -1 if not yet out of AD grace period, or -2 if in FM grace period.
        """
        num_samples = input_batch.shape[0]
        # Initialize all RMSEs to -1 (AD grace period) or -2 (FM grace period) as default
        rmses = np.full(num_samples, -1.0) 

        for i in range(num_samples): # Loop through each sample in the input batch
            X = input_batch[i, :] # Get single sample (1D array)
            
            # Feature Mapping (FM) stage
            if not self.FM_train_complete:
                self.__update_FM(X)
                rmses[i] = -2.0 # Indicate FM grace period in progress
                continue

            # Ensemble Layer (EL) stage
            # This is also where the ensemble layer gets created for the first time after FM training.
            S_l = self.__update_ensembleLayer(X) # Get reconstruction errors from ensemble AEs
            
            # Anomaly Detection (AD) stage
            if not self.AD_train_complete:
                self.__update_outputLayer(S_l)
                rmses[i] = -1.0 # Indicate AD grace period in progress
                continue

            # Fully trained, output anomaly score
            # The outputLayer.execute call performs vectorized computations on S_l
            rmses[i] = self.__update_outputLayer(S_l) # Reconstruct from ensemble errors and get final RMSE
        
        return rmses


    def get_weights(self):
        """
        Extracts all current weights and state from the KitNET model.
        Returns a dictionary containing the model's state.
        """
        ensemble_weights = []
        for ae in self.ensembleLayer:
            ensemble_weights.append(ae.get_weights())
        
        output_weights = {}
        if self.outputLayer:
            output_weights = self.outputLayer.get_weights()
        
        fm_state = {
            'c': self.FM.c.tolist(),
            'c_r': self.FM.c_r.tolist(),
            'c_rs': self.FM.c_rs.tolist(),
            'C': self.FM.C.tolist(),
            'N': self.FM.N,
            'n': self.FM.n,
            'clusters': self.FM.clusters # Save clusters for re-initialization
        }

        return {
            'n': self.n,
            'm': self.m,
            'AD_grace_period': self.AD_grace_period,
            'FM_grace_period': self.FM_grace_period,
            'lr': self.lr,
            'hr': self.hr,
            'FM_train_complete': self.FM_train_complete,
            'AD_train_complete': self.AD_train_complete,
            'ensemble_weights': ensemble_weights,
            'output_weights': output_weights,
            'FM_state': fm_state
        }

    def set_weights(self, model_state):
        """
        Restores the KitNET model state from a dictionary.
        This method is crucial for loading pre-trained models.
        """
        self.n = model_state['n']
        self.m = model_state['m']
        self.AD_grace_period = model_state['AD_grace_period']
        self.FM_grace_period = model_state['FM_grace_period']
        self.lr = model_state['lr']
        self.hr = model_state['hr']
        self.FM_train_complete = model_state['FM_train_complete']
        self.AD_train_complete = model_state['AD_train_complete']
        
        # Reconstruct Feature Map and load state, then rebuild clusters
        self.FM = CC.corClust(self.n) # Reinitialize Feature Map
        self.FM.c = np.array(model_state['FM_state']['c'])
        self.FM.c_r = np.array(model_state['FM_state']['c_r'])
        self.FM.c_rs = np.array(model_state['FM_state']['c_rs'])
        self.FM.C = np.array(model_state['FM_state']['C'])
        self.FM.N = model_state['FM_state']['N']
        self.FM.n = model_state['FM_state']['n']
        self.FM.clusters = model_state['FM_state']['clusters'] # Load clusters
        logging.info(f"Loaded Feature Map (FM) state with {len(self.FM.clusters)} clusters.")
        
        # Reconstruct ensemble layer and load weights
        self.ensembleLayer = []
        for cluster in self.FM.clusters:
            params = AE.dA_params(n_visible=len(cluster), n_hidden=max(1, round(self.hr * len(cluster))), lr=self.lr)
            self.ensembleLayer.append(AE.dA(params))
        
        for i, ae_weights in enumerate(model_state['ensemble_weights']):
            self.ensembleLayer[i].set_weights(ae_weights)
        logging.info(f"Loaded Ensemble Layer with {len(self.ensembleLayer)} autoencoders.")
        
        # Reconstruct and load output layer weights
        if 'output_weights' in model_state and model_state['output_weights']:
            # The n_visible for the output layer is determined by the number of AEs in the ensemble
            output_n_visible = len(self.ensembleLayer)
            params = AE.dA_params(n_visible=output_n_visible, n_hidden=max(1, round(self.hr * output_n_visible)), lr=self.lr)
            self.outputLayer = AE.dA(params)
            self.outputLayer.set_weights(model_state['output_weights'])
            logging.info(f"Loaded Output Layer AE with {output_n_visible} visible units.")
        else:
            self.outputLayer = None # Ensure it's None if no output layer weights were saved
            logging.info("No output layer weights found in model state. Output layer remains uninitialized.")


    def save_model(self, filepath):
        """Saves the Kitsune model state to a pickle file."""
        model_state = self.get_weights()
        with open(filepath, 'wb') as f:
            pickle.dump(model_state, f)
        logging.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Loads a Kitsune model state from a pickle file."""
        with open(filepath, 'rb') as f:
            model_state = pickle.load(f)
        instance = cls(n=model_state['n'], m=model_state['m'],
                       FM_grace_period=model_state['FM_grace_period'],
                       AD_grace_period=model_state['AD_grace_period'],
                       learning_rate=model_state['lr'],
                       hidden_ratio=model_state['hr'],
                       feature_map=model_state['FM_state']['clusters'] # Pass loaded clusters to constructor for initialization
                      )
        instance.set_weights(model_state) # Call set_weights to restore full state
        return instance

