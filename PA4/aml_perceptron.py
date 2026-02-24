
"""This file shows a couple of implementations of the perceptron learning
algorithm. It is based on the code from Lecture 3, but using the slightly
more compact perceptron formulation that we saw in Lecture 6.

There are two versions: Perceptron, which uses normal NumPy vectors and
matrices, and SparsePerceptron, which uses sparse vectors and matrices.
The latter may be faster when we have high-dimensional feature representations
with a lot of zeros, such as when we are using a "bag of words" representation
of documents.
"""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

class LinearClassifier(BaseEstimator):
    """
    General class for binary linear classifiers. Implements the predict
    function, which is the same for all binary linear classifiers. There are
    also two utility functions.
    """

    def decision_function(self, X):
        """
        Computes the decision function for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """
        return X.dot(self.w)

    def predict(self, X):
        """
        Predicts the outputs for the inputs X. The inputs are assumed to be
        stored in a matrix, where each row contains the features for one
        instance.
        """
        check_is_fitted(self, ['is_fitted_']) 
        
        scores = self.decision_function(X)

        out = np.select([scores >= 0.0, scores < 0.0],
                        [self.positive_class, self.negative_class],
                        default=self.negative_class) 
        return out

    def find_classes(self, Y):
        """
        Finds the set of output classes in the output part Y of the training set.
        If there are exactly two classes, one of them is associated to positive
        classifier scores, the other one to negative scores. If the number of
        classes is not 2, an error is raised.
        """
        classes = sorted(set(Y))
        if len(classes) != 2:
            raise Exception("this does not seem to be a 2-class problem")
        self.positive_class = classes[1]
        self.negative_class = classes[0]
        self.is_fitted_ = True

    def encode_outputs(self, Y):
        """
        A helper function that converts all outputs to +1 or -1.
        """
        return np.array([1 if y == self.positive_class else -1 for y in Y])


class Perceptron(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm.
    """

    def __init__(self, n_iter=20):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

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

        # Initialize the weight vector to all zeros.
        n_features = X.shape[1]
        self.w = np.zeros(n_features)

        # Perceptron algorithm:
        for i in range(self.n_iter):
            for x, y in zip(X, Ye):

                # Compute the output score for this instance.
                score = x.dot(self.w)

                # If there was an error, update the weights.
                if y*score <= 0:
                    self.w += y*x
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


class SparsePerceptron(LinearClassifier):
    """
    A straightforward implementation of the perceptron learning algorithm,
    assuming that the input feature matrix X is sparse.
    """

    def __init__(self, n_iter=20):
        """
        The constructor can optionally take a parameter n_iter specifying how
        many times we want to iterate through the training set.
        """
        self.n_iter = n_iter

    def fit(self, X, Y):
        """
        Train a linear classifier using the perceptron learning algorithm.

        Note that this will only work if X is a sparse matrix, such as the
        output of a scikit-learn vectorizer.
        """
        self.find_classes(Y)

        # First determine which output class will be associated with positive
        # and negative scores, respectively.
        Ye = self.encode_outputs(Y)

        # Initialize the weight vector to all zeros.
        self.w = np.zeros(X.shape[1])

        # Iteration through sparse matrices can be a bit slow, so we first
        # prepare this list to speed up iteration.
        XY = list(zip(X, Ye))

        for i in range(self.n_iter):
            for x, y in XY:

                # Compute the output score for this instance.
                # (This corresponds to score = x.dot(self.w) above.)
                score = sparse_dense_dot(x, self.w)

                # If there was an error, update the weights.
                if y*score <= 0:
                    # (This corresponds to self.w += y*x above.)
                    add_sparse_to_dense(x, self.w, y)
        return self


class PegasosSVC(LinearClassifier):
    def __init__(self, n_iter=10, lambda_reg=None):
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg

    def fit(self, X, Y):
        self.find_classes(Y)
        Ye = self.encode_outputs(Y)

        if not isinstance(X, np.ndarray):
            X = X.toarray()

        n_samples, n_features = X.shape
        lam = self.lambda_reg if self.lambda_reg is not None else 1.0 / n_samples

        self.w = np.zeros(n_features)
        t = 0  # global step counter

        for epoch in range(self.n_iter):
            # Shuffle training instances each epoch for better convergence.
            indices = np.random.permutation(n_samples)
            for i in indices:
                t += 1
                eta = 1.0 / (lam * t)   # learning rate schedule
                x_i, y_i = X[i], Ye[i]
                score = x_i.dot(self.w)

                # Hinge-loss gradient step.
                # Scale down first (regularisation shrinkage).
                self.w *= (1.0 - eta * lam)
                if y_i * score < 1:      # instance is inside or on wrong side of margin
                    self.w += eta * y_i * x_i

                # Optional projection onto the L2 ball of radius 1/sqrt(λ).
                norm_w = np.linalg.norm(self.w)
                radius = 1.0 / np.sqrt(lam)
                if norm_w > radius:
                    self.w *= radius / norm_w

        return self


class SparsePegasosSVC(LinearClassifier):
    """
    Sparse version of PegasosSVC for high-dimensional bag-of-words features.
    """

    def __init__(self, n_iter=10, lambda_reg=None):
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg

    def fit(self, X, Y):
        self.find_classes(Y)
        Ye = self.encode_outputs(Y)

        n_samples = X.shape[0]
        lam = self.lambda_reg if self.lambda_reg is not None else 1.0 / n_samples

        self.w = np.zeros(X.shape[1])
        XY = list(zip(X, Ye))
        t = 0

        for epoch in range(self.n_iter):
            np.random.shuffle(XY)
            for x_i, y_i in XY:
                t += 1
                eta = 1.0 / (lam * t)
                score = sparse_dense_dot(x_i, self.w)

                self.w *= (1.0 - eta * lam)
                if y_i * score < 1:
                    add_sparse_to_dense(x_i, self.w, eta * y_i)

                norm_w = np.linalg.norm(self.w)
                radius = 1.0 / np.sqrt(lam)
                if norm_w > radius:
                    self.w *= radius / norm_w

        return self


class LogisticRegressionSGD(LinearClassifier):
    """
    Logistic Regression trained with the Pegasos-style SGD.

    Replaces the hinge loss with the log loss:
        ℓ(w; x_i, y_i) = log(1 + exp(-y_i (w · x_i)))

    The gradient of the log loss with respect to w is:
        -y_i · σ(-y_i (w · x_i)) · x_i
    where σ(z) = 1 / (1 + exp(-z)) is the logistic sigmoid.

    Parameters
    ----------
    n_iter : int
        Number of passes (epochs) through the training set.
    lambda_reg : float
        Regularisation strength λ.
    """

    def __init__(self, n_iter=10, lambda_reg=None):
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg

    def fit(self, X, Y):
        self.find_classes(Y)
        Ye = self.encode_outputs(Y)

        if not isinstance(X, np.ndarray):
            X = X.toarray()

        n_samples, n_features = X.shape
        lam = self.lambda_reg if self.lambda_reg is not None else 1.0 / n_samples

        self.w = np.zeros(n_features)
        t = 0

        for epoch in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            for i in indices:
                t += 1
                eta = 1.0 / (lam * t)
                x_i, y_i = X[i], Ye[i]
                margin = y_i * x_i.dot(self.w)

                # σ(-margin) = 1 / (1 + exp(margin))
                # gradient of log-loss w.r.t. w = -y_i * σ(-margin) * x_i
                sigmoid_neg = 1.0 / (1.0 + np.exp(margin))  # numerically stable for large margin
                update_factor = y_i * sigmoid_neg            # η * (-grad of loss)

                self.w *= (1.0 - eta * lam)
                self.w += eta * update_factor * x_i

        return self


class SparseLogisticRegressionSGD(LinearClassifier):
    """
    Sparse version of LogisticRegressionSGD for high-dimensional features.
    """

    def __init__(self, n_iter=10, lambda_reg=None):
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg

    def fit(self, X, Y):
        self.find_classes(Y)
        Ye = self.encode_outputs(Y)

        n_samples = X.shape[0]
        lam = self.lambda_reg if self.lambda_reg is not None else 1.0 / n_samples

        self.w = np.zeros(X.shape[1])
        XY = list(zip(X, Ye))
        t = 0

        for epoch in range(self.n_iter):
            np.random.shuffle(XY)
            for x_i, y_i in XY:
                t += 1
                eta = 1.0 / (lam * t)
                margin = y_i * sparse_dense_dot(x_i, self.w)
                sigmoid_neg = 1.0 / (1.0 + np.exp(margin))
                update_factor = y_i * sigmoid_neg

                self.w *= (1.0 - eta * lam)
                add_sparse_to_dense(x_i, self.w, eta * update_factor)

        return self



