
import numpy
from scipy.stats import norm
numpy.seterr(all='ignore') # Ignore numpy warnings (e.g., division by zero in some edge cases)

def pdf(x,mu,sigma): #normal distribution pdf
    """Calculates the Probability Density Function (PDF) of a normal distribution."""
    x = (x-mu)/sigma
    return numpy.exp(-x**2/2)/(numpy.sqrt(2*numpy.pi)*sigma)

def invLogCDF(x,mu,sigma): #normal distribution cdf
    """
    Calculates the inverse of the log cumulative distribution function (CDF) for a normal distribution.
    Note: Multiplies by -1 after normalization to better approximate 1-cdf behavior for anomaly scoring.
    """
    x = (x - mu) / sigma
    return norm.logcdf(-x) # This leverages scipy's optimized logcdf, numerically stable.

def sigmoid(x):
    """Sigmoid activation function. Fully vectorized using NumPy."""
    return 1. / (1 + numpy.exp(-x))

def dsigmoid(x):
    """Derivative of the sigmoid function. Fully vectorized using NumPy."""
    return x * (1. - x)

def tanh(x):
    """Hyperbolic tangent (tanh) activation function. Fully vectorized using NumPy."""
    return numpy.tanh(x)

def dtanh(x):
    """Derivative of the hyperbolic tangent (tanh) function. Fully vectorized using NumPy."""
    return 1. - x * x

def softmax(x):
    """
    Softmax activation function. Fully vectorized using NumPy.
    Handles both 1D and 2D inputs.
    """
    e = numpy.exp(x - numpy.max(x))  # prevent overflow by subtracting max
    if e.ndim == 1:
        return e / numpy.sum(e, axis=0)
    else:  
        return e / numpy.array([numpy.sum(e, axis=1)]).T  # For 2D, sum along axis 1 and reshape for broadcasting

def ReLU(x):
    """Rectified Linear Unit (ReLU) activation function. Fully vectorized using NumPy."""
    return x * (x > 0)

def dReLU(x):
    """Derivative of the Rectified Linear Unit (ReLU) function. Fully vectorized using NumPy."""
    return 1. * (x > 0)

class rollmean:
    """
    Implements a rolling mean (moving average) for a stream of data.
    This class maintains sums and counts to incrementally update the mean.
    """
    def __init__(self, size):
        self.size = size
        self.l = []
        self.ss = 0
    def add(self, x):
        self.l.append(x)
        self.ss += x
        if len(self.l) > self.size:
            self.ss -= self.l.pop(0)
    def get(self):
        if len(self.l) == 0:
            return 0
        return self.ss / float(len(self.l))

