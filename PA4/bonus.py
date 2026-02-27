import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from Task3 import LinearClassifier
import scipy.linalg.blas as blas

class Pegasos_opt(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
    """

    def __init__(self, n_iter=20, lambda_=None):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter
        self.lambda_ = lambda_

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.
        """

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        self.find_classes(Y)

        # Convert all outputs to +1 (for the positive class) or -1 (negative).
        Ye = self.encode_outputs(Y)

        # If necessary, convert the sparse matrix returned by a vectorizer
        # into a normal NumPy matrix.
        if not isinstance(X, np.ndarray):
            X = X.toarray()
            
        lam = self.lambda_ if self.lambda_ is not None else 1.0 / X.shape[0]


        # Initialize the weight vector to all zeros.
        n_samples = X.shape[0]
        self.w = np.zeros(X.shape[1])
        
        t = 0
        # X = np.ascontiguousarray(X, dtype=np.float64)
        # self.w = np.ascontiguousarray(self.w, dtype=np.float64)
        # Pegasos algorithm:
        for epoch in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            for i in indices:
                t += 1
                eta = 1.0 / (lam * t)
                
                # Compute the output score for this instance.
                x_i, y_i = X[i], Ye[i]
                
                
                #score = blas.ddot(self.w, x_i) #Slower for some reason?
                score = x_i.dot(self.w)
                
                factor1 = (1 - eta * lam)
                factor2 = eta * y_i
                
                blas.dscal(factor1, self.w)
                # If there was an error, update the weights.
                if y_i*score < 1:
                    #self.w = factor1 * self.w + factor2 * x_i
                    blas.daxpy(x_i, self.w, a=factor2)

        return self
    
    
class Pegasos_sparse_opt(LinearClassifier):
    """
    Sparse version of Pegasos_opt using x.indices / x.data helpers.
    Keeps the exact same update logic as Pegasos_opt.
    """

    def __init__(self, n_iter=20, lambda_=None):
        self.n_iter = n_iter
        self.lambda_ = lambda_

    def fit(self, X, Y):
        self.find_classes(Y)
        Ye = self.encode_outputs(Y)

        n_samples, n_features = X.shape
        lam = self.lambda_ if self.lambda_ is not None else 1.0 / n_samples

        # Dense weight vector, sparse input rows.
        self.w = np.zeros(n_features, dtype=np.float64)

        t = 0
        for epoch in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            for i in indices:
                t += 1
                eta = 1.0 / (lam * t)

                x_i, y_i = X[i], Ye[i]
                score = sparse_dense_dot(x_i, self.w)

                factor1 = 1.0 - eta * lam
                factor2 = eta * y_i
                
                self.w *= factor1
                
                if y_i * score < 1.0:
                    # self.w = factor1 * self.w + factor2 * x_i
                    add_sparse_to_dense(x_i, self.w, factor2)

        return self
    
def add_sparse_to_dense(x, w, factor):
    """
    Adds a sparse vector x, scaled by some factor, to a dense vector.
    This can be seen as the equivalent of w += factor * x when x is a dense
    vector.
    """
    w[x.indices] += factor * x.data

def sparse_dense_dot(x, w):
    """
    Computes the dot product between a sparse vector x and a dense vector w.
    """
    return np.dot(w[x.indices], x.data)

class Pegasos_sparse_scale_opt(LinearClassifier):
    """
    Sparse version of Pegasos_opt using x.indices / x.data helpers.
    Keeps the exact same update logic as Pegasos_opt.
    """

    def __init__(self, n_iter=20, lambda_=None):
        self.n_iter = n_iter
        self.lambda_ = lambda_

    def fit(self, X, Y):
        self.find_classes(Y)
        Ye = self.encode_outputs(Y)

        n_samples, n_features = X.shape
        lam = self.lambda_ if self.lambda_ is not None else 1.0 / n_samples

        # Dense weight vector, sparse input rows.
        self.w = np.zeros(n_features, dtype=np.float64)
        
        scaler = 1.0

        t = 0
        for epoch in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            for i in indices:
                t += 1
                eta = 1.0 / (lam * t)

                x_i, y_i = X[i], Ye[i]
                score = scaler * sparse_dense_dot(x_i, self.w)

                factor1 = 1.0 - eta * lam
                factor2 = (eta * y_i) / scaler
                
                # To prevent the weights from becoming too small, we can rescale them back up when the scaling factor drops below a certain threshold.
                if scaler * factor1 < 1e-8:
                    self.w *= scaler * factor1
                    scaler = 1.0
                else:
                    scaler *= factor1
                
                if y_i * score < 1.0:
                    # self.w = factor1 * self.w + factor2 * x_i
                    add_sparse_to_dense(x_i, self.w, factor2)

        self.w *= scaler
        return self
