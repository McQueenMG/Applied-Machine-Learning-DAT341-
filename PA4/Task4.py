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


class LogisticRegression(LinearClassifier):
    """
    Logistic Regression trained with Pegasos-style SGD (log loss).

    The update rule at step t is:
        eta  = 1 / (lambda * t)
        w   <- (1 - eta * lambda) * w  +  eta * y_i * sigma(-y_i * s_i) * x_i

    where s_i = w · x_i and sigma(-y_i*s_i) = 1 / (1 + exp(y_i * s_i)).

    Parameters
    ----------
    n_iter     : int   – number of epochs (passes through the training set)
    lambda_    : float – regularisation strength λ; defaults to 1/N
    """

    def __init__(self, n_iter=10, print_objective=True):
        self.n_iter = n_iter
        self.lambda_ = 0.1
        self.print_objective = print_objective


    def fit(self, X, Y, lambda_=None):
        self.find_classes(Y)
        Ye = self.encode_outputs(Y)

        # Convert sparse matrix (e.g. from TF-IDF) to dense array if needed.
        if not isinstance(X, np.ndarray):
            X = X.toarray()

        n_samples, n_features = X.shape
        lam = self.lambda_ if self.lambda_ is not None else 1.0 / n_samples
        self.w = np.zeros(n_features)
        t = 0  # global step counter

        for epoch in range(self.n_iter):
            # Shuffle each epoch for better convergence.
            loss_total = 0.0  # for tracking the total loss across epochs
            indices = np.random.permutation(n_samples)
            for i in indices:
                t += 1
                eta = 1.0 / (lam * t)
                x_i, y_i = X[i], Ye[i]
                margin = y_i * x_i.dot(self.w)

                # sigma(-margin) = 1 / (1 + exp(margin))
                sigmoid_neg = 1.0 / (1.0 + np.exp(margin))
                
                loss_total += np.log1p(np.exp(-margin))  # log-loss for this instance

                # L2 regularisation shrinkage + log-loss gradient step.
                self.w *= (1.0 - eta * lam)
                self.w += eta * y_i * sigmoid_neg * x_i

            # Objective: mean logistic loss + L2 regularizer
            if self.print_objective:
                mean_log_loss = loss_total / indices.size
                reg = 0.5 * lam * np.dot(self.w, self.w)
                obj = mean_log_loss + reg
                print(
                    f"Epoch {epoch + 1}/{self.n_iter} | "
                    f"objective≈{obj:.6f} (loss={mean_log_loss:.6f}, reg={reg:.6f})"
                )

        return self

