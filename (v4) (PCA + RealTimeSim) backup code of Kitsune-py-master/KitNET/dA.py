# Copyright (c) 2017 Yusuke Sugomori
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Portions of this code have been adapted from Yusuke Sugomori's code on GitHub: https://github.com/yusugomori/DeepLearning

import sys
import numpy as np
# import cupy as np # Uncomment if using CuPy for GPU acceleration
from KitNET.utils import *
import json
import logging

# Set up logging for dA.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class dA_params:
    def __init__(self, n_visible=5, n_hidden=3, lr=0.1, corruption_level=0.3, #corruption level is not used
                 activ_func='tanh', builder_type='dau'):
        self.n_visible = n_visible  # Number of visible units (input dimension)
        self.n_hidden = n_hidden    # Number of hidden units
        self.lr = lr                # Learning rate
        self.corruption_level = corruption_level # (Currently unused in this implementation)
        self.activ_func = activ_func # Activation function for hidden layer
        self.builder_type = builder_type # Type of autoencoder (e.g., 'dau' for Denoising Autoencoder)

class dA:
    """
    Denoising Autoencoder (dA) implementation.
    Optimized for single-sample, online updates using NumPy's vectorized operations.
    """
    def __init__(self, params):
        self.params = params
        self.n = self.params.n_visible # input dimension
        self.rng = np.random.RandomState(1234)

        # Initialize weights and biases
        # W maps visible to hidden, W_prime maps hidden back to visible
        # W is initialized with values from a uniform distribution [-sqrt(6./(n_visible+n_hidden)), sqrt(6./(n_visible+n_hidden))]
        # Ref: Yoshua Bengio, Xavier Glorot, "Understanding the difficulty of training deep feedforward neural networks"
        initializer_bound = 1. / self.params.n_visible # A common initialization for fan-in
        self.W = np.asarray(self.rng.uniform(low=-initializer_bound, high=initializer_bound,
                                             size=(self.params.n_visible, self.params.n_hidden)),
                            dtype=np.float64)
        self.hbias = np.zeros(self.params.n_hidden, dtype=np.float64) # Bias for hidden layer
        self.vbias = np.zeros(self.params.n_visible, dtype=np.float64) # Bias for visible (output) layer

        # Running average for normalization (for RMSE calculation)
        self.N = 0 # Number of samples processed
        self.mean = np.zeros(self.n)
        self.std = np.zeros(self.n)
        self.norm_min = np.full(self.n, np.inf) # Tracks minimum value for each feature
        self.norm_max = np.full(self.n, -np.inf) # Tracks maximum value for each feature

        # Select activation function
        if self.params.activ_func == 'sigmoid':
            self.activate = sigmoid
            self.deriv_activate = dsigmoid
        elif self.params.activ_func == 'tanh':
            self.activate = tanh
            self.deriv_activate = dtanh
        elif self.params.activ_func == 'relu':
            self.activate = ReLU
            self.deriv_activate = dReLU
        else:
            raise ValueError("Unsupported activation function: " + self.params.activ_func)

    def get_corrupted_input(self, x):
        # NOTE: Corruption is not explicitly implemented in train() or execute()
        # for simplicity as per the original KitNET design for online updates.
        # This method is a placeholder if corruption were to be added.
        return x

    def train(self, x, lr=None):
        """
        Trains the autoencoder with a single input sample using stochastic gradient descent.

        Args:
            x (np.ndarray): A 1D NumPy array representing a single input sample (n_visible features).
            lr (float, optional): Learning rate for this update. If None, uses self.params.lr.
        """
        assert x.ndim == 1, "Input 'x' must be a 1D array for single-sample training."
        if lr is None:
            lr = self.params.lr

        # Update normalization statistics (online mean/std for RMSE)
        self.N += 1
        new_mean = self.mean + (x - self.mean) / self.N
        new_std = np.sqrt(self.std**2 + (x - self.mean) * (x - new_mean) / self.N)
        self.mean = new_mean
        self.std = new_std

        # Update min/max for 0-1 normalization in execute()
        self.norm_min = np.minimum(self.norm_min, x)
        self.norm_max = np.maximum(self.norm_max, x)
        
        # Clip min/max to avoid division by zero or extreme values in normalization
        # Ensures that norm_max is always strictly greater than norm_min for each feature.
        self.norm_max = np.maximum(self.norm_max, self.norm_min + 1e-10)

        # Normalize the input (0-1 normalization using current running min/max)
        # This is a vectorized operation across all features of the single sample.
        x_normalized = (x - self.norm_min) / (self.norm_max - self.norm_min + 1e-10)

        # Forward pass to calculate hidden layer activation and reconstruction
        # These are vectorized matrix multiplications and element-wise operations.
        h = self.activate(np.dot(x_normalized, self.W) + self.hbias) # hidden layer activation
        y = self.activate(np.dot(h, self.W.T) + self.vbias) # reconstruction

        # Calculate errors (gradients) for backpropagation
        # These are vectorized element-wise operations.
        v_error = x_normalized - y # error at the visible layer
        h_error = np.dot(v_error, self.W) * self.deriv_activate(h) # error at the hidden layer

        # Update weights and biases using gradients (stochastic gradient descent)
        # These updates are also vectorized, applying to all weights/biases simultaneously.
        self.W += lr * (np.outer(x_normalized, h_error) + np.outer(v_error, h)) # Weight update
        self.hbias += lr * h_error # Hidden bias update
        self.vbias += lr * v_error # Visible bias update

    def reconstruct(self, x):
        """
        Reconstructs the input from its hidden representation.
        Assumes input 'x' is already 0-1 normalized.

        Args:
            x (np.ndarray): A 1D NumPy array representing a single 0-1 normalized sample.

        Returns:
            np.ndarray: The reconstructed 1D array.
        """
        # These are vectorized matrix multiplications and element-wise operations.
        h = self.activate(np.dot(x, self.W) + self.hbias)
        y = self.activate(np.dot(h, self.W.T) + self.vbias)
        return y

    def execute(self, input_sample):
        """
        Calculates the Reconstruction Mean Squared Error (RMSE) for a single input sample.

        Args:
            input_sample (np.ndarray): A 1D NumPy array representing a single sample.

        Returns:
            float: The RMSE for the given sample.
        """
        assert input_sample.ndim == 1, "Input must be a 1D array for single-sample execution."
        
        # 0-1 normalize the input sample using current min/max statistics.
        # This is a vectorized operation across all features.
        x_normalized = (input_sample - self.norm_min) / (self.norm_max - self.norm_min + 1e-10)
        
        # Reconstruct the normalized input
        z = self.reconstruct(x_normalized)
        
        # Calculate RMSE. This is a vectorized operation across all features.
        rmse = np.sqrt(np.mean((x_normalized - z) ** 2)) # RMSE for the single sample
        return rmse

    # The inGrace method is designed for single-sample checks.
    # It's not directly used for the batch processing logic anymore,
    # as KitNET manages grace periods externally based on batch size.
    # This method can be considered deprecated or its usage changed when batching KitNET fully.
    # For now, return False as individual dA instances don't have a grace period concept.
    def inGrace(self):
        return False # dA itself doesn't have a grace period. This is handled by KitNET.

    #######################
    def get_weights(self):
        """
        Returns the current weights and biases of the autoencoder, along with normalization stats.
        """
        return {
            'W': self.W.tolist(), # Convert to list for JSON compatibility if needed
            'hbias': self.hbias.tolist(),
            'vbias': self.vbias.tolist(),
            'norm_min': self.norm_min.tolist(),
            'norm_max': self.norm_max.tolist(),
            'n': self.n,
            'N': self.N, # Include N for consistent loading/saving of statistics
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'params': { # Save parameters for reconstruction
                'n_visible': self.params.n_visible,
                'n_hidden': self.params.n_hidden,
                'lr': self.params.lr,
                'corruption_level': self.params.corruption_level,
                'activ_func': self.params.activ_func,
                'builder_type': self.params.builder_type
            }
        }

    def set_weights(self, weights):
        """
        Sets the weights, biases, and normalization statistics of the autoencoder.
        """
        self.W = np.array(weights['W'], dtype=np.float64)
        self.hbias = np.array(weights['hbias'], dtype=np.float64)
        self.vbias = np.array(weights['vbias'], dtype=np.float64)
        self.norm_min = np.array(weights['norm_min'], dtype=np.float64)
        self.norm_max = np.array(weights['norm_max'], dtype=np.float64)
        self.n = weights['n']
        self.N = weights['N']
        self.mean = np.array(weights['mean'], dtype=np.float64)
        self.std = np.array(weights['std'], dtype=np.float64)

        # Re-initialize parameters and activation functions based on loaded state
        self.params = dA_params(**weights['params'])
        if self.params.activ_func == 'sigmoid':
            self.activate = sigmoid
            self.deriv_activate = dsigmoid
        elif self.params.activ_func == 'tanh':
            self.activate = tanh
            self.deriv_activate = dtanh
        elif self.params.activ_func == 'relu':
            self.activate = ReLU
            self.deriv_activate = dReLU
        else:
            raise ValueError("Unsupported activation function: " + self.params.activ_func)

