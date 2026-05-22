import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, to_tree
import logging

# Set up logging for corClust.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# A helper class for KitNET which performs a correlation-based incremental clustering of the dimensions in X
# n: the number of dimensions in the dataset
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection
class corClust:
    """
    Performs correlation-based incremental clustering of features.

    n: The number of dimensions (features) in the dataset.
    """
    def __init__(self,n):
        # Parameters:
        self.n = n # Total number of features/dimensions

        # Internal variables for incremental statistics:
        # These are used to calculate means, variances, and covariances online.
        self.c = np.zeros(n, dtype=np.float64) # Linear sum of features
        self.c_r = np.zeros(n, dtype=np.float64) # Linear sum of feature residuals (x - mean)
        self.c_rs = np.zeros(n, dtype=np.float64) # Linear sum of squared feature residuals ((x - mean)^2)
        self.C = np.zeros((n,n), dtype=np.float64) # Partial correlation matrix (sum of outer products of residuals)
        self.N = 0 # Number of updates (samples) performed

        self.clusters = [] # Stores the final feature clusters (list of lists of feature indices)
        logging.info(f"corClust initialized for {n} features.")


    def update(self,x):
        """
        Updates the incremental statistics with a new feature vector.

        Args:
            x (np.ndarray): A 1D NumPy array representing a single feature vector of length n.
                            This method assumes x is already numerical.
        
        Optimization Notes:
        - All operations within this method (e.g., +, -, **, np.outer) are highly vectorized
          by NumPy, ensuring efficient computations in C/Fortran.
        - The np.outer operation for updating self.C scales quadratically with the number of features (n),
          i.e., O(n^2), which is inherent to computing pairwise relationships in a correlation matrix.
        """
        self.N += 1
        self.c += x # Sum of features
        
        # Calculate residual for the current sample: x_t - mean(x_t)
        # mean(x_t) is approximated by c/N incrementally
        c_rt = x - self.c/self.N 
        
        self.c_r += c_rt # Sum of residuals
        self.c_rs += c_rt**2 # Sum of squared residuals
        
        # Update partial correlation matrix C. This is the sum of outer products of residuals.
        # This is a key operation for calculating pairwise covariances.
        self.C += np.outer(c_rt,c_rt) # O(n^2) operation


    def corrDist(self):
        """
        Creates the current correlation distance matrix between the features.

        Optimization Notes:
        - All NumPy operations (np.sqrt, np.outer, division) are vectorized.
        - The core of this method involves matrix operations that inherently scale with O(n^2),
          where n is the number of features, due to the nature of correlation matrix calculation.
        - Adding a small epsilon (1e-10) to denominators to prevent division by zero in case of zero variance.
        """
        # Calculate the square root of the sum of squared residuals for each feature
        # (similar to standard deviation for normalization)
        c_rs_sqrt = np.sqrt(self.c_rs) # Vectorized operation

        # Compute the outer product for normalization of the C matrix
        C_rs_sqrt = np.outer(c_rs_sqrt,c_rs_sqrt) # O(n^2) operation

        # Calculate the correlation matrix: C_ij / (sqrt(C_ii) * sqrt(C_jj))
        # This is a vectorized element-wise division.
        # Add a small epsilon to the denominator to prevent division by zero.
        corr = self.C / (C_rs_sqrt + 1e-10) 
        
        # Clip values to ensure they are within [-1, 1] due to potential floating point inaccuracies
        corr = np.clip(corr, -1.0, 1.0) 

        # Convert correlation to correlation distance: 1 - |correlation|
        # This transforms similarity to distance.
        corDist = 1 - np.abs(corr) # Vectorized operation

        # Ensure diagonal is zero (distance of a feature to itself is 0)
        np.fill_diagonal(corDist, 0)

        # Return the upper triangle of the distance matrix (excluding diagonal) as a condensed vector
        # This is the format expected by scipy.cluster.hierarchy.linkage
        dist = corDist[np.triu_indices(self.n, k=1)]
        return dist


    def cluster(self, maxClust):
        """
        Performs hierarchical clustering on the features using the current correlation distance.
        The resulting clusters are stored in self.clusters.

        Args:
            maxClust (int): The maximum number of clusters to form.

        Optimization Notes:
        - This method relies on scipy.cluster.hierarchy functions (linkage, to_tree),
          which are highly optimized C/Fortran implementations.
        - For very large 'n' (number of features), hierarchical clustering can be
          computationally and memory intensive (e.g., O(n^3) or O(n^2) depending on linkage
          method and dataset size, and requires storing distance matrices).
        - While efficient for a given N, if this becomes a bottleneck for extremely high
          dimensional data, consideration of alternative clustering approaches or feature
          selection/dimensionality reduction prior to clustering might be necessary.
        """
        if self.n == 0:
            self.clusters = []
            return
        if self.n == 1: # Single feature, it's its own cluster
            self.clusters = [[0]]
            return

        # Ensure enough samples to calculate variance, otherwise clustering is meaningless
        if self.N < 2:
            logging.warning("Not enough samples for robust correlation clustering. Features will be treated as independent.")
            self.clusters = [[i] for i in range(self.n)] # Each feature is its own cluster
            return
        
        try:
            # Generate the hierarchical clustering linkage matrix
            # 'average' linkage is commonly used and is O(n^2 log n) or O(n^2) depending on scipy version.
            Z = linkage(self.corrDist(), method='average') 
            
            # Form flat clusters from the hierarchical clustering based on maxClust
            # This identifies the groups that meet the criteria.
            # Use maxclust criterion to cut the dendrogram to a maximum of maxClust clusters.
            flat_clusters = fcluster(Z, t=maxClust, criterion='maxclust')
            
            # Convert flat_clusters array into a list of lists, where each inner list
            # contains the original feature indices belonging to that cluster.
            temp_clusters = [[] for _ in range(maxClust)]
            for i, cluster_id in enumerate(flat_clusters):
                temp_clusters[cluster_id - 1].append(i) # cluster_id is 1-indexed

            # Remove any empty clusters if maxClust was too high or fcluster didn't fill all.
            self.clusters = [c for c in temp_clusters if c]
            
            if not self.clusters and self.n > 0: # Fallback if no valid clusters somehow
                self.clusters = [[i] for i in range(self.n)] # Each feature is its own cluster
                logging.warning("Clustering resulted in no valid clusters. Falling back to treating each feature as an independent cluster.")
            
            logging.info(f"Clustering complete. Formed {len(self.clusters)} clusters for {self.n} features.")

        except Exception as e:
            logging.error(f"Error during feature clustering: {e}. Falling back to treating each feature as an independent cluster.")
            self.clusters = [[i] for i in range(self.n)] # Fallback: each feature is its own cluster


    # This function is not directly used in the current clustering logic (since fcluster is used)
    # but is part of the original project structure, so adding docstrings for clarity.
    def __breakClust__(self,dendro,maxClust):
        """
        Recursively breaks a dendrogram tree into smaller clusters.
        (Primarily for internal or alternative clustering cut-off logic; fcluster is typically used).
        
        Args:
            dendro: A dendrogram node from scipy.cluster.hierarchy.to_tree.
            maxClust: The maximum number of elements a cluster should have (not max number of clusters).
        """
        if dendro.count <= maxClust: #base case: we found a minimal cluster, so mark it
            return [dendro.pre_order()] #return the original ids of the features in this cluster
        return self.__breakClust__(dendro.get_left(),maxClust) + self.__breakClust__(dendro.get_right(),maxClust)

# Copyright (c) 2017 Yisroel Mirsky
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
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
