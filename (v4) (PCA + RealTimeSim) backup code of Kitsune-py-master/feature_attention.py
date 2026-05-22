# 6/27/2025
# This file defines the FeatureAttention class,
# which acts as a lightweight attention mechanism.
import numpy as np

class FeatureAttention:
    """
    A simplified attention mechanism to weight features.
    This class simulates a shallow neural network that learns feature weights.
    In a full implementation, these weights could be learned via a proper training
    phase (e.g., combining Random Forest-based importance and MLP-learned weights).
    For this streaming context, we'll use a simple MLP-like structure that applies
    and normalizes weights to the input features.

    The "shallower attention layers (2-4)" note is addressed by using a small
    'hidden_size' for the internal MLP.
    """
    def __init__(self, num_features, hidden_size=4):
        """
        Initializes the FeatureAttention module.

        Args:
            num_features (int): The number of input features.
            hidden_size (int): The size of the hidden layer in the attention MLP.
                               This corresponds to the "shallower attention layers" idea.
        """
        self.num_features = num_features
        self.hidden_size = hidden_size

        # Initialize weights and biases for a simple 2-layer MLP.
        # This MLP will learn to produce attention weights for the input features.
        # Input layer (num_features) -> Hidden layer (hidden_size) -> Output layer (num_features, for weighting).

        # Weights for the first layer (input to hidden)
        self.W1 = np.random.randn(num_features, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)

        # Weights for the second layer (hidden to output)
        self.W2 = np.random.randn(hidden_size, num_features) * 0.01
        self.b2 = np.zeros(num_features)

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-x))

    def _softmax(self, x):
        """
        Softmax function for attention weights.
        Handles both 1D (single vector) and 2D (batch of vectors) inputs.
        """
        e = np.exp(x - np.max(x, axis=-1, keepdims=True)) # Subtract max for numerical stability
        return e / np.sum(e, axis=-1, keepdims=True)

    def process(self, feature_vector):
        """
        Processes a feature vector (or a batch of feature vectors) through the
        attention mechanism to produce weighted feature vectors.

        Args:
            feature_vector (np.ndarray): The input feature vector (1D) or a batch
                                         of feature vectors (2D array: n_samples, n_features).

        Returns:
            np.ndarray: The feature vector(s) with attention weights applied.
                        If input was 1D, returns 1D. If input was 2D, returns 2D.
        """
        original_ndim = feature_vector.ndim
        if original_ndim == 1:
            feature_vector = feature_vector.reshape(1, -1) # Reshape to 2D for consistent matrix multiplication

        # Forward pass: Input features -> Hidden layer
        hidden_layer_input = np.dot(feature_vector, self.W1) + self.b1
        hidden_layer_output = self._sigmoid(hidden_layer_input)

        # Forward pass: Hidden layer -> Output (raw attention weights)
        attention_weights_raw = np.dot(hidden_layer_output, self.W2) + self.b2
        
        # Apply softmax to get normalized attention weights for each feature.
        # This ensures weights sum to 1 across features for each sample.
        attention_weights = self._softmax(attention_weights_raw)

        # Apply these attention weights to the original feature vector.
        # This element-wise multiplication emphasizes or de-emphasizes features.
        weighted_feature_vector = feature_vector * attention_weights

        # Return the weighted feature vector in the original dimensionality
        if original_ndim == 1:
            return weighted_feature_vector.flatten() # If input was 1D, return 1D
        else:
            return weighted_feature_vector # If input was 2D, return 2D (the batch)

