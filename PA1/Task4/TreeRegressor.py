from collections import Counter
from statistics import variance

from sklearn.base import RegressorMixin
from DecisionTree import DecisionTree
import numpy as np
import pandas as pd

class TreeRegressor(DecisionTree, RegressorMixin):

    def __init__(self, max_depth=10, criterion='varience_reduction', threshold=1.0e-5):
        super().__init__(max_depth)
        self.criterion = criterion
        self.threshold = threshold

    def fit(self, X, Y):
        if self.criterion == 'variance_reduction':
            self.criterion_function = variance_reduction_scorer
        super().fit(X, Y)

    # Select a default value that is going to be used if we decide to make a leaf.
    # We will select the most common value.
    def get_default_value(self, Y):
        return np.mean(Y)

    def is_homogeneous(self, Y):
        return np.var(Y) < self.threshold
        
    # Finds the best splitting point for a given feature. We'll keep frequency tables (Counters)
    # for the upper and lower parts, and then compute the impurity criterion using these tables.
    # In the end, we return a triple consisting of
    # - the best score we found, according to the criterion we're using
    # - the id of the feature
    # - the threshold for the best split
    def best_split(self, X, Y, feature):

        # Create a list of input-output pairs, where we have sorted
        # in ascending order by the input feature we're considering.
        sorted_indices = np.argsort(X[:, feature])        
        X_sorted = list(X[sorted_indices, feature])
        Y_sorted = list(Y[sorted_indices])

        n = len(Y)

        # Keep track of the best result we've seen so far.
        max_score = -np.inf
        max_i = None
        
        # Keep track of sum and sum of squares for low and high parts
        sum_low = 0.0
        sum_high = sum(Y_sorted)
        sumsq_low = 0.0
        sumsq_high = sum(y*y for y in Y_sorted)
        count_low = 0
        count_high = n

        # Go through all the positions (excluding the last position).
        for i in range(0, n-1):

            # Input and output at the current position.
            x_i = X_sorted[i]
            y_i = Y_sorted[i]


            # If the input is equal to the input at the next position, we will
            # not consider a split here.
            #x_next = XY[i+1][0]
            x_next = X_sorted[i+1]
            if x_i == x_next:
                continue
            
            # Update the sum and sum of squares for low and high parts.
            sum_low += y_i
            sum_high -= y_i
            sumsq_low += y_i * y_i
            sumsq_high -= y_i * y_i
            count_low += 1
            count_high -= 1
            
            
            # Update varience statistics
            varHigh = (sumsq_high / count_high) - (sum_high / count_high)**2
            varLow = (sumsq_low / count_low) - (sum_low / count_low)**2
            varTotal = (count_low * varLow) + (count_high * varHigh)

            # Compute the homogeneity criterion for a split at this position.
            score = -( (count_low / n) * varLow + (count_high / n) * varHigh )

            # If this is the best split, remember it.
            if score > max_score:
                max_score = score
                max_i = i

        # If we didn't find any split (meaning that all inputs are identical), return
        # a dummy value.
        if max_i is None:
            return -np.inf, None, None

        # Otherwise, return the best split we found and its score.
        split_point = 0.5*(X_sorted[max_i] + X_sorted[max_i+1])
        return max_score, feature, split_point

def variance_reduction_scorer(n, n_high, n_low, var_total, var_high, var_low):
    return var_total - (n_high/n)*var_high - (n_low/n)*var_low
    