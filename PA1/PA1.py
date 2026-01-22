import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
  
# Read the CSV file.
data = pd.read_csv('PA1\\CTG.csv', skiprows=1)

# Select the relevant numerical columns.
selected_cols = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV',
                 'MLTV', 'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean',
                 'Median', 'Variance', 'Tendency', 'NSP']
data = data[selected_cols].dropna()

# Shuffle the dataset.
data_shuffled = data.sample(frac=1.0, random_state=0)

# Split into input part X and output part Y.
X = data_shuffled.drop('NSP', axis=1)

# Map the diagnosis code to a human-readable label.
def to_label(y):
    return [None, 'normal', 'suspect', 'pathologic'][(int(y))]

Y = data_shuffled['NSP'].apply(to_label)

# Partition the data into training and test sets.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X.head())

# --------------------------------------------------------------------------------------------

from sklearn.dummy import DummyClassifier

clf = DummyClassifier(strategy='most_frequent')

from sklearn.model_selection import cross_val_score

crossArray = cross_val_score(clf, Xtrain, Ytrain)
print('Cross-validation scores: ', crossArray)

import numpy as np
dummyScore = np.mean(crossArray)
print('Dummy average cross-validation score: ', dummyScore)

#-----------------------------------------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

clf1 = DecisionTreeClassifier(random_state=0, max_depth=7)
clf2 = RandomForestClassifier(random_state=0)
clf3 = GradientBoostingClassifier(random_state=0)
clf4 = Perceptron(random_state=0)
clf5 = LogisticRegression(random_state=0)
clf6 = LinearSVC(random_state=0)
clf7 = MLPClassifier(random_state=0)

classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]
classifier_names = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Perceptron',
                    'Logistic Regression', 'Linear SVC', 'MLP Classifier']

for clf, name in zip(classifiers, classifier_names):
    crossArray = cross_val_score(clf, Xtrain, Ytrain)
    score = np.mean(crossArray)
    print(f'{name} average cross-validation score: {score}')
# --------------------------------------------------------------------------------------------

from sklearn.metrics import accuracy_score

clf3.fit(Xtrain, Ytrain)
Yguess = clf3.predict(Xtest)

print(f"Final Test Accuracy: {accuracy_score(Ytest, Yguess)}")
#----------------------------------------------------------------------------------------------


#  ------------ TASK 2  ------------ 
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from helper import TreeClassifier 

custom_depths = range(1, 15)
custom_scores = []

print("Tuning Custom TreeClassifier...")

for d in custom_depths:
    clf_custom = TreeClassifier(max_depth=d)
    
    # Run cross-validation
    scores = cross_val_score(clf_custom, Xtrain, Ytrain)
    custom_scores.append(np.mean(scores))
    print(f"Depth {d}: {np.mean(scores):.4f}")

# Find the winner
best_custom_score = max(custom_scores)
best_custom_depth = custom_depths[custom_scores.index(best_custom_score)]

print(f"\nWINNER: Best Depth for Custom Tree is {best_custom_depth} (Score: {best_custom_score:.4f})")


best_custom_tree = TreeClassifier(max_depth=best_custom_depth)
best_custom_tree.fit(Xtrain, Ytrain)

custom_test_acc = accuracy_score(Ytest, best_custom_tree.predict(Xtest))
print(f"Final Test Accuracy for Custom Tree: {custom_test_acc:.4f}")

#----------------------------------------------------------------------------------------------
# illustration purpose only
print("\n--- Visualizing a Simplified Tree (Depth 2) ---")
simple_tree = TreeClassifier(max_depth=2)
simple_tree.fit(Xtrain, Ytrain)
simple_tree.draw_tree() 
