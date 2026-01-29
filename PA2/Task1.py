
#--------Step-1-------------------------------------------#
import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

testdata = load_data('PA2\\adult_test.csv')
traindata = load_data('PA2\\adult_train.csv')

Ytest = testdata['target']
Ytrain = traindata['target']
Xtest = testdata.drop('target', axis=1)
Xtrain = traindata.drop('target', axis=1)

#--------Step-2-------------------------------------------#
Xtrain_dict = Xtrain.to_dict('records')
Xtest_dict = Xtest.to_dict('records')

from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()

Xtrain_encoded = dv.fit_transform(Xtrain_dict)
Xtest_encoded = dv.transform(Xtest_dict)

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
clf5 = LogisticRegression(random_state=0, max_iter=2000)
clf6 = LinearSVC(random_state=0, max_iter=2000)
clf7 = MLPClassifier(random_state=0, max_iter=500)

classifiers = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]
classifier_names = ['Decision Tree', 'Random Forest', 'Gradient Boosting', 'Perceptron',
                    'Logistic Regression', 'Linear SVC', 'MLP Classifier']

from sklearn.model_selection import cross_val_score
import numpy as np

def compare_classifiers(Xtrain_encoded, Ytrain, classifiers, classifier_names):
    for clf, name in zip(classifiers, classifier_names):
        crossArray = cross_val_score(clf, Xtrain_encoded, Ytrain)
        score = np.mean(crossArray)
        print(f'{name} average cross-validation score: {score}')
    
#compare_classifiers(Xtrain_encoded, Ytrain, classifiers, classifier_names)    

from sklearn.metrics import accuracy_score

# Gradient Boosting Classifier selected based on the best cross-validation score
clf3.fit(Xtrain_encoded, Ytrain)
Yguess = clf3.predict(Xtest_encoded)

print(f"Final Test Accuracy: {accuracy_score(Ytest, Yguess)}")
# This gave us an accuracy of 0.871199557766722

#--------Step-3-------------------------------------------#

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
  
pipeline = make_pipeline(
    DictVectorizer(), # Convert data to correct format
    StandardScaler(with_mean=False), # Scales features to better suit the model
    SelectKBest(f_classif, k=50), # Selects the best 50 features based on f_classif
    GradientBoostingClassifier() # Our chosen classifier
)

pipeline.fit(Xtrain_dict, Ytrain)
Yguess_pipeline = pipeline.predict(Xtest_dict)
print(f"Final Test Accuracy with Pipeline: {accuracy_score(Ytest, Yguess_pipeline)}")