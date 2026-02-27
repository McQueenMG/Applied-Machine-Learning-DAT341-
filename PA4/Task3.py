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

class Pegasos(LinearClassifier):
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
        n_features, n_samples = X.shape[1], X.shape[0]
        self.w = np.zeros(n_features)
        
        t = 0
        # Pegasos algorithm:
        for epoch in range(self.n_iter):
            indices = np.random.permutation(n_samples)
            for i in indices:
                t += 1
                eta = 1.0 / (lam * t)
                
                # Compute the output score for this instance.
                x_i, y_i = X[i], Ye[i]
                score = x_i.dot(self.w)
                # If there was an error, update the weights.
                if y_i*score < 1:
                    self.w = (1 - eta * lam) * self.w + eta * y_i * x_i
                else:
                    self.w = (1 - eta * lam) * self.w

        return self
