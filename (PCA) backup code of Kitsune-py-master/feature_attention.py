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
        # Initialized with small random values to break symmetry.
        self.W1 = np.random.uniform(-0.1, 0.1, (num_features, hidden_size))
        self.b1 = np.zeros(hidden_size)

        # Weights for the second layer (hidden to output - feature weights)
        # Initialized with small random values.
        self.W2 = np.random.uniform(-0.1, 0.1, (hidden_size, num_features))
        self.b2 = np.zeros(num_features)

    def _sigmoid(self, x):
        """Applies the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """
        Applies the softmax activation function to normalize attention weights.
        Ensures that the sum of weights for each input vector is 1,
        making them interpretable as probabilities or importance scores.
        """
        # Subtract max for numerical stability to prevent overflow with large exponents.
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def process(self, feature_vector):
        """
        Processes a feature vector through the attention mechanism to produce a
        weighted feature vector.

        Args:
            feature_vector (np.ndarray): The input feature vector (1D or 2D array).

        Returns:
            np.ndarray: The feature vector with attention weights applied, flattened to 1D.
        """
        # Ensure the feature_vector is 2D for consistent matrix multiplication,
        # even if a single 1D vector is passed.
        if feature_vector.ndim == 1:
            feature_vector = feature_vector.reshape(1, -1)

        # Forward pass: Input features -> Hidden layer
        hidden_layer_input = np.dot(feature_vector, self.W1) + self.b1
        hidden_layer_output = self._sigmoid(hidden_layer_input)

        # Forward pass: Hidden layer -> Output (raw attention weights)
        attention_weights_raw = np.dot(hidden_layer_output, self.W2) + self.b2
        
        # Apply softmax to get normalized attention weights for each feature.
        attention_weights = self._softmax(attention_weights_raw)

        # Apply these attention weights to the original feature vector.
        # This element-wise multiplication emphasizes or de-emphasizes features.
        weighted_feature_vector = feature_vector * attention_weights

        # Return the weighted feature vector as a 1D array.
        return weighted_feature_vector.flatten()
