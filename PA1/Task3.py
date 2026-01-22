import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read the Excel file using Pandas.
alldata = pd.read_excel('PA1\\Hemnet_data.xlsx')

# # Convert the timestamp string to an integer representing the year.
alldata['year'] = pd.DatetimeIndex(alldata['Sold Date']).year

# Convert 'yes' to 1 and 'no' to 0
alldata['Balcony'] = alldata['Balcony'].map({'Yes': 1, 'No': 0})
alldata['Patio'] = alldata['Patio'].map({'Yes': 1, 'No': 0})
alldata['Lift'] = alldata['Lift'].map({'Yes': 1, 'No': 0})

# Select the 12 input columns and the output column.
selected_columns = ['Final Price (kr)', 'year',  'Num of Room', 'Living Area (m²)', 'Balcony', 'Patio','Current Floor', 'Total Floor', 'Lift', 'Built Year', 'Fee (kr/month)', 'Operating Fee (kr/year)']
alldata = alldata[selected_columns]
cols_to_clean = ['Final Price (kr)', 'Fee (kr/month)', 'Operating Fee (kr/year)']

# Cleaning...
for col in cols_to_clean:
    alldata[col] = alldata[col].astype(str).str.replace('kr', '', regex=False).str.replace(' ', '')
    alldata[col] = pd.to_numeric(alldata[col], errors='coerce')
alldata = alldata.dropna()

# Shuffle.
alldata_shuffled = alldata.sample(frac=1.0, random_state=0)

# Separate the input and output columns.
X = alldata_shuffled.drop('Final Price (kr)', axis=1)
# For the output, we'll use the log of the sales price.
Y = alldata_shuffled['Final Price (kr)'].apply(np.log)

# Split into training and test sets.
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate
m1 = DummyRegressor()
cross_validate(m1, Xtrain, Ytrain, scoring='neg_mean_squared_error')


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Decision Tree': DecisionTreeRegressor(random_state=0),
    'Random Forest': RandomForestRegressor(random_state=0),
    'Gradient Boosting': GradientBoostingRegressor(random_state=0),
    'MLP Regressor': MLPRegressor(random_state=0)
    }


for name, model in models.items():
    cv_results = cross_validate(model, Xtrain, Ytrain, scoring='neg_mean_squared_error')
    mean_mse = -np.mean(cv_results['test_score'])
    print(f"{name}: {mean_mse}")
    
#------------------------------------
from sklearn.metrics import mean_squared_error

print("\n--- Final Evaluation on Test Set ---")

best_model = models['Random Forest']
best_model.fit(Xtrain, Ytrain)
Yguess = best_model.predict(Xtest)
final_mse = mean_squared_error(Ytest, Yguess)

print(f"Final Test MSE: {final_mse:.4f}")