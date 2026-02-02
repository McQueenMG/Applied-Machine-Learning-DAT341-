#----------------prestep----------------#
import pandas as pd
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

testdata = load_data('PA2\\adult_test.csv')
traindata = load_data('PA2\\adult_train.csv')

Ytest = testdata['target']
Ytrain = traindata['target']
Xtest = testdata.drop('target', axis=1)
Xtrain = traindata.drop('target', axis=1)
from sklearn.feature_extraction import DictVectorizer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

pipeline = make_pipeline(
    DictVectorizer(), # Convert data to correct format
    StandardScaler(with_mean=False), # Scales features to better suit the model
    DecisionTreeClassifier(max_depth=7) # Our chosen classifier
)

pipeline.fit(Xtrain.to_dict(orient='records'), Ytrain)

#----------------Task 3: Part 1 - Feature Importance----------------#

# Get feature names in Decision Tree after preprocessing 
feature_names = pipeline.steps[0][1].feature_names_
feature_importances = pipeline.steps[2][1].feature_importances_

features = list(zip(feature_names, feature_importances))
features.sort(key=lambda x: x[1], reverse=True)
print(features[:10])  # Print top 10 features